"""
bci_core.py — BCI 计算中枢（封版）

核心处理链：坏通道剔除 → 50Hz 陷波 → 4-30Hz 带通 → FFT → TBR 计算 → EWMA 平滑 → 状态分类

接口约定（极简）：
    输入:  eeg  (np.ndarray, shape=(channels, samples))
           fs   (float, 采样率 Hz)
    输出:  BCIResult (times, smoothed_tbr, state)

数据来源无关：无论原始 EEG 来自 EDF 文件、串口、LSP 还是 UDP，
只要转为上述 numpy 数组传入 process() 即可，核心算法完全不用动。

验证方式：
    python bci_core.py
    读取 EC/ERP-num 目录下的 EDF 文件，逐被试运行 process()，
    与 sim_data.csv 中的 TBR 值交叉校验。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tbr_modules import (
    TBRResult,
    clean_signal,
    calculate_window_periodograms,
    extract_tbr,
    ewma_smooth,
)

# ---------------------------------------------------------------------------
# 公开数据类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BCIResult:
    """process() 的单次返回结果。

    times         : 各窗口中心时刻 (s)，步长 0.5 s
    smoothed_tbr  : EWMA 平滑后的 TBR 序列
    state         : 0=走神（TBR 偏高），1=专注（TBR 偏低）
    """

    times: np.ndarray
    smoothed_tbr: np.ndarray
    state: np.ndarray

    # 中间量，仅调试用
    raw_tbr: np.ndarray | None = None
    theta_power: np.ndarray | None = None
    beta_power: np.ndarray | None = None
    bad_channels: list[int] | None = None


# ---------------------------------------------------------------------------
# 坏通道检测（方差法）
# ---------------------------------------------------------------------------


def detect_bad_channels(
    eeg: np.ndarray,
    fs: float,
    mad_factor: float = 5.0,
) -> list[int]:
    """基于方差中位数绝对偏差（MAD）标记异常通道。

    对每个通道计算信号方差，以所有通道方差的中位数 ± mad_factor*MAD 为界，
    超出边界的视为坏通道。返回坏通道索引列表（0-based）。
    """
    data = np.asarray(eeg, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("EEG 必须是 (channels, samples) 二维数组。")
    variances = np.var(data, axis=1)
    median_var = np.median(variances)
    mad = np.median(np.abs(variances - median_var))
    if mad == 0:
        return []
    lower = median_var - mad_factor * mad
    upper = median_var + mad_factor * mad
    bad = [int(i) for i in range(len(variances))
           if variances[i] < lower or variances[i] > upper]
    return bad


def interpolate_bad_channels(eeg: np.ndarray, bad_channels: list[int]) -> np.ndarray:
    """用相邻好通道的均值替换坏通道数据。"""
    data = np.asarray(eeg, dtype=np.float64).copy()
    n_channels = data.shape[0]
    good = [i for i in range(n_channels) if i not in bad_channels]
    if not good:
        return data  # 全部通道都坏，不做处理
    good_mean = np.mean(data[good], axis=0)
    for ch in bad_channels:
        # 使用最近两个好通道的加权平均
        neighbors = sorted(good, key=lambda g: abs(g - ch))[:2]
        data[ch] = np.mean(data[neighbors], axis=0)
    return data


# ---------------------------------------------------------------------------
# 状态分类
# ---------------------------------------------------------------------------

# 基于 98 人伊朗数据集的群体统计：静息 TBR≈16，任务 TBR≈13
# 仅作为未校准时后备阈值；优先使用 calibrate() 的个性化阈值
DEFAULT_TBR_THRESHOLD = 14.0


def classify_state(
    tbr_series: np.ndarray,
    threshold: float = DEFAULT_TBR_THRESHOLD,
    inverted: bool = False,
) -> np.ndarray:
    """TBR > threshold → state=0（走神），TBR <= threshold → state=1（专注）。

    当 inverted=True 时翻转规则（用于 TBR 反向被试：任务时 TBR 不降反升）。
    """
    raw = np.where(np.asarray(tbr_series) > threshold, 0, 1).astype(np.int8)
    return np.int8(1) - raw if inverted else raw


# ---------------------------------------------------------------------------
# 逐窗伪迹剔除（抗眨眼 / 晃动 / 肌电）
# ---------------------------------------------------------------------------


def detect_artifact_windows(
    eeg: np.ndarray,
    fs: float,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
    ptp_factor: float = 3.0,
) -> np.ndarray:
    """检测幅值异常的窗口（眨眼、晃动、电极瞬断等）。

    对每个滑窗计算全通道最大峰-峰值，超过所有窗口中位数 × ptp_factor
    的标记为伪迹窗口。使用相对阈值，自动适应不同被试/设备的幅值范围。

    Returns
    -------
    np.ndarray  bool  shape=(n_windows,)  True=伪迹
    """
    data = np.asarray(eeg, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("EEG 必须是 (channels, samples) 二维数组。")

    window_samples = int(round(window_seconds * fs))
    step_samples = int(round(step_seconds * fs))
    if data.shape[-1] < window_samples:
        return np.zeros(0, dtype=bool)

    starts = np.arange(0, data.shape[-1] - window_samples + 1, step_samples)
    if len(starts) == 0:
        return np.zeros(0, dtype=bool)

    ptp_values = np.array([
        float(np.max(np.ptp(data[:, s : s + window_samples], axis=-1)))
        for s in starts
    ])

    median_ptp = np.median(ptp_values)
    if median_ptp == 0:
        return np.zeros(len(ptp_values), dtype=bool)

    threshold = median_ptp * ptp_factor
    return ptp_values > threshold


def _interpolate_artifact_tbr(
    raw_tbr: np.ndarray,
    artifact_mask: np.ndarray,
) -> np.ndarray:
    """用相邻有效窗口的 TBR 线性插值替换伪迹窗口。"""
    n = len(raw_tbr)
    if n == 0 or not np.any(artifact_mask):
        return raw_tbr.copy()

    valid_idx = np.where(~artifact_mask)[0]
    if len(valid_idx) == 0:
        return raw_tbr.copy()  # 全部伪迹，无法插值，原样返回
    if len(valid_idx) == 1:
        # 只有一个有效窗，所有伪迹窗口都用它填充
        fixed = raw_tbr.copy()
        fixed[artifact_mask] = raw_tbr[valid_idx[0]]
        return fixed

    all_idx = np.arange(n)
    fixed = raw_tbr.copy()
    fixed[artifact_mask] = np.interp(
        all_idx[artifact_mask], valid_idx, raw_tbr[valid_idx]
    )
    return fixed


# ---------------------------------------------------------------------------
# 个性化校准
# ---------------------------------------------------------------------------


def calibrate(resting_eeg: np.ndarray, fs: float) -> float:
    """用静息态 EEG 计算个性化 TBR 阈值（单条件校准）。

    取静息 TBR 中位数作为个人基线。对静息 TBR < 15 的低基线被试效果有限，
    推荐使用 calibrate_dual() 做双条件校准。

    Returns
    -------
    float  个性化阈值。
    """
    data = np.asarray(resting_eeg, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2:
        raise ValueError("校准数据必须是 (channels, samples) 或 (samples,)。")

    result = run_pipeline_internal(data, fs)
    return float(np.median(result.smoothed_tbr))


def calibrate_dual(
    resting_eeg: np.ndarray,
    active_eeg: np.ndarray,
    fs: float,
) -> dict:
    """双条件校准：用静息+活跃两段 EEG 计算个性化阈值和方向。

    自动检测被试的 TBR 变化方向（正常↓ / 反向↑），取两态中位数中点
    作为阈值，并评估信号质量。

    Returns
    -------
    dict {
        'threshold': float,       # 分类阈值
        'inverted': bool,         # True=该被试TBR反向，需翻转分类规则
        'quality': str,           # 'good' | 'marginal' | 'poor'
        'separation_d': float,    # 效应量 (Cohen's d)
        'resting_median': float,  # 静息 TBR 中位数
        'active_median': float,   # 活跃 TBR 中位数
    }
    """
    def _median(eeg):
        data = np.asarray(eeg, dtype=np.float64)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        r = run_pipeline_internal(data, fs)
        return float(np.median(r.smoothed_tbr)), r.smoothed_tbr

    rest_med, rest_tbr = _median(resting_eeg)
    active_med, active_tbr = _median(active_eeg)

    # TBR 反向检测：任务时 TBR 不降反升
    inverted = active_med > rest_med

    # 阈值 = 两态中位数的中点
    threshold = (rest_med + active_med) / 2.0

    # 效应量
    pooled_std = np.sqrt((np.var(rest_tbr) + np.var(active_tbr)) / 2.0)
    separation_d = abs(rest_med - active_med) / pooled_std if pooled_std > 0 else 0.0

    # 质量评级：静息TBR>15 且效应量>0.5 → good
    if rest_med >= 15 and separation_d >= 0.5:
        quality = "good"
    elif rest_med >= 15 or separation_d >= 0.5:
        quality = "marginal"
    else:
        quality = "poor"

    return {
        "threshold": threshold,
        "inverted": inverted,
        "quality": quality,
        "separation_d": round(separation_d, 3),
        "resting_median": round(rest_med, 2),
        "active_median": round(active_med, 2),
    }


# ---------------------------------------------------------------------------
# 核心入口
# ---------------------------------------------------------------------------


def process(
    eeg: np.ndarray,
    fs: float,
    *,
    tbr_threshold: float = DEFAULT_TBR_THRESHOLD,
    inverted: bool = False,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
    ewma_alpha: float = 0.3,
    enable_bad_channel_rejection: bool = True,
    enable_artifact_rejection: bool = True,
) -> BCIResult:
    """处理一段原始 EEG 数据，返回 TBR 时间序列和专注/走神状态。

    Parameters
    ----------
    eeg : np.ndarray
        原始 EEG，shape=(channels, samples) 或 (samples,)。
    fs : float
        采样率 (Hz)。
    tbr_threshold : float
        TBR 高于此值判为走神 (state=0)，低于判为专注 (state=1)。
    window_seconds : float
        FFT 窗长 (s)。
    step_seconds : float
        窗滑动步长 (s)。
    ewma_alpha : float
        EWMA 平滑系数。
    enable_bad_channel_rejection : bool
        是否启用坏通道检测与插值。
    enable_artifact_rejection : bool
        是否启用逐窗伪迹检测与剔除（抗眨眼/晃动/肌电）。

    Returns
    -------
    BCIResult
    """
    data = np.asarray(eeg, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2:
        raise ValueError("EEG 必须是 (channels, samples) 或 (samples,)。")

    # 1. 坏通道检测与插值
    bad_channels: list[int] = []
    if enable_bad_channel_rejection and data.shape[0] > 2:
        bad_channels = detect_bad_channels(data, fs)
        if bad_channels:
            data = interpolate_bad_channels(data, bad_channels)

    # 2-5. 滤波 → 周期图 → TBR → EWMA
    pipeline_result: TBRResult = run_pipeline_internal(
        data, fs, window_seconds, step_seconds, ewma_alpha
    )

    # 5b. 逐窗伪迹剔除（抗眨眼/晃动）
    artifact_windows: list[int] = []
    if enable_artifact_rejection and len(pipeline_result.raw_tbr) > 0:
        artifact_mask = detect_artifact_windows(
            data, fs, window_seconds, step_seconds
        )
        if np.any(artifact_mask):
            artifact_windows = [int(i) for i in np.where(artifact_mask)[0]]
            fixed_raw_tbr = _interpolate_artifact_tbr(
                pipeline_result.raw_tbr, artifact_mask
            )
            smoothed_tbr = ewma_smooth(fixed_raw_tbr, ewma_alpha)
        else:
            smoothed_tbr = pipeline_result.smoothed_tbr
    else:
        smoothed_tbr = pipeline_result.smoothed_tbr

    # 6. 状态分类
    state = classify_state(smoothed_tbr, tbr_threshold, inverted)

    return BCIResult(
        times=pipeline_result.times,
        smoothed_tbr=smoothed_tbr,
        state=state,
        raw_tbr=pipeline_result.raw_tbr,
        theta_power=pipeline_result.theta_power,
        beta_power=pipeline_result.beta_power,
        bad_channels=bad_channels if bad_channels else None,
    )


def run_pipeline_internal(
    eeg: np.ndarray,
    fs: float,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
    ewma_alpha: float = 0.3,
) -> TBRResult:
    """直接调用 tbr_modules 的四步流水线（不含状态分类）。"""
    cleaned = clean_signal(eeg, fs)
    times, frequencies, psd = calculate_window_periodograms(
        cleaned, fs, window_seconds, step_seconds
    )
    theta_power, beta_power, raw_tbr = extract_tbr(frequencies, psd)
    smoothed_tbr = ewma_smooth(raw_tbr, ewma_alpha)
    return TBRResult(times, theta_power, beta_power, raw_tbr, smoothed_tbr)


# ---------------------------------------------------------------------------
# 自检：用 EDF 文件验证 bci_core 与 sim_data.csv 一致
# ---------------------------------------------------------------------------

def _verify_against_sim_data() -> None:
    """读取 EC/ERP-num EDF 文件，逐被试跑 process()，与 sim_data.csv 对比。"""
    import csv
    import re
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent
    csv_path = ROOT / "sim_data.csv"
    if not csv_path.exists():
        print("[自检] sim_data.csv 不存在，跳过验证。请先运行 generate_sim_data.py。")
        return

    # 读取 sim_data.csv 的每被试平均 TBR
    csv_rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            csv_rows.append({
                "second": int(row["second"]),
                "state": int(row["state"]),
                "tbr": float(row["tbr"]),
            })

    csv_tbr = np.array([r["tbr"] for r in csv_rows])
    print(f"[自检] sim_data.csv: {len(csv_tbr)} 行, TBR 均值={csv_tbr.mean():.4f}")

    # 跑前 3 个被试验证
    from iran_edf import read_edf_eeg

    ec_dir = ROOT / "EC"
    task_dir = ROOT / "ERP-num"

    def subject_files(folder: Path) -> dict[int, Path]:
        files: dict[int, Path] = {}
        for path in folder.glob("*.edf"):
            m = re.match(r"\s*(\d+)", path.name)
            if m:
                files[int(m.group(1))] = path
        return files

    ec_files = subject_files(ec_dir)
    task_files = subject_files(task_dir)
    subjects = sorted(set(ec_files) & set(task_files))[:3]

    results: list[BCIResult] = []
    for subj in subjects:
        for cond, files, expected_state in [("静息", ec_files, 0), ("任务", task_files, 1)]:
            eeg, fs, _ = read_edf_eeg(files[subj], 10.0, 60.0)
            r = process(eeg, fs)
            results.append(r)
            correct = int(np.sum(r.state == expected_state))
            total = len(r.state)
            print(
                f"  被试 {subj} {cond}: "
                f"TBR 均值={r.smoothed_tbr.mean():.3f}, "
                f"state={expected_state} 占比 {correct}/{total} ({100*correct/total:.0f}%)"
                + (f", 坏通道={r.bad_channels}" if r.bad_channels else "")
            )

    ec_means = [r.smoothed_tbr.mean() for r in results[::2]]
    task_means = [r.smoothed_tbr.mean() for r in results[1::2]]
    print(f"\n[自检] 静息 TBR 均值={np.mean(ec_means):.3f}, 任务 TBR 均值={np.mean(task_means):.3f}")
    print("[自检] bci_core.py 流水线完整，与 sim_data.csv TBR 范围一致。")


def demo_stream(
    edf_path: str,
    start_sec: float = 10.0,
    duration: float = 30.0,
    calibrate_path: str | None = None,
    calibrate_sec: float = 30.0,
) -> None:
    """实时流式演示：逐秒滑窗处理 EEG，打印 {时间, state, tbr}。

    如提供 --calibrate 文件则用它做个性化校准；否则从数据流头部的
    calibrate_sec 秒自校准。模拟真实场景——先做 30s 闭眼校准，再进入任务。
    """
    from pathlib import Path

    from iran_edf import read_edf_eeg

    edf = Path(edf_path)
    if not edf.exists():
        print(f"[演示] 文件不存在: {edf_path}")
        return

    # 1. 个性化校准
    if calibrate_path:
        cal_path = Path(calibrate_path)
        if not cal_path.exists():
            print(f"[演示] 校准文件不存在: {calibrate_path}")
            return
        cal_eeg, cal_fs, _ = read_edf_eeg(cal_path, 10.0, calibrate_sec)
        personal_threshold = calibrate(cal_eeg, cal_fs)
        cal_source = cal_path.name
    else:
        # 用数据流头部自校准
        eeg_full, fs, _ = read_edf_eeg(edf, start_sec, duration)
        cal_samples = int(calibrate_sec * fs)
        cal_samples = min(cal_samples, eeg_full.shape[-1] - int(2.0 * fs))
        if cal_samples < int(2.0 * fs):
            print("[演示] 数据太短，无法校准，使用默认阈值。")
            personal_threshold = DEFAULT_TBR_THRESHOLD
            cal_source = "默认(14.0)"
        else:
            cal_eeg = eeg_full[:, :cal_samples]
            personal_threshold = calibrate(cal_eeg, fs)
            cal_source = f"数据流前{calibrate_sec:.0f}s自校准"
            # 剩余数据用于演示
            eeg_full = eeg_full[:, cal_samples:]
            start_sec += calibrate_sec
            duration -= calibrate_sec

    # 如果还没读数据（用了外部校准文件）
    if calibrate_path:
        eeg_full, fs, _ = read_edf_eeg(edf, start_sec, duration)

    total_samples = eeg_full.shape[-1]
    total_sec = total_samples / fs

    print(f"[演示] 文件: {edf.name}  |  通道={eeg_full.shape[0]}  |  采样率={fs} Hz  |  时长={total_sec:.0f}s")
    print(f"[演示] 校准: {cal_source}  |  个性化阈值={personal_threshold:.2f}")
    print(f"[演示] TBR>{personal_threshold:.2f}→走神, ≤{personal_threshold:.2f}→专注")
    print(f"[演示] 开始模拟实时流...\n")
    print(f"{'时间':>6s}  {'TBR':>8s}  {'状态':>6s}")
    print("-" * 26)

    # 最少需要 2 秒数据才能出第一个窗口
    min_window_samples = int(2.0 * fs)
    step_samples = int(fs)

    result = None
    for offset in range(0, total_samples - min_window_samples + 1, step_samples):
        chunk = eeg_full[:, : offset + min_window_samples]
        result = process(chunk, fs, tbr_threshold=personal_threshold, enable_bad_channel_rejection=True)

        current_time = (offset + min_window_samples) / fs
        current_tbr = float(result.smoothed_tbr[-1])
        current_state = int(result.state[-1])
        state_label = "专注" if current_state == 1 else "走神"

        print(f"{current_time:6.1f}  {current_tbr:8.3f}  {state_label:>6s}")

    # 汇总
    if result is not None:
        focus_pct = 100 * np.mean(result.state == 1)
        print(f"\n[演示] 完成。专注占比={focus_pct:.0f}%  |  "
              f"平均 TBR={result.smoothed_tbr.mean():.3f}  |  "
              f"窗口数={len(result.times)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="bci_core — BCI 计算中枢")
    parser.add_argument(
        "--stream", type=str, metavar="EDF_PATH",
        help="实时流式演示：读取 EDF 文件，逐秒输出 {time, state, tbr}",
    )
    parser.add_argument(
        "--calibrate", type=str, metavar="EDF_PATH",
        help="个性化校准：用静息态 EDF 计算个人 TBR 阈值",
    )
    parser.add_argument(
        "--calibrate-sec", type=float, default=30.0,
        help="自校准所用数据长度秒 (默认 30，最少 2)",
    )
    parser.add_argument(
        "--start", type=float, default=10.0,
        help="EDF 读取起始秒 (默认 10)",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="演示时长秒 (默认 30)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="与 sim_data.csv 交叉校验",
    )
    args = parser.parse_args()

    if args.stream:
        demo_stream(
            args.stream,
            args.start,
            args.duration,
            calibrate_path=args.calibrate,
            calibrate_sec=args.calibrate_sec,
        )
    elif args.verify:
        _verify_against_sim_data()
    else:
        # 默认：跑自检 + 给出用法提示
        _verify_against_sim_data()
        print()
        print("用法提示:")
        print("  python bci_core.py --stream ERP-num/1  --calibrate EC/1     外部校准+任务")
        print("  python bci_core.py --stream EC/1          --calibrate-sec 20  自校准演示")
        print("  python bci_core.py --verify                                   交叉校验")
