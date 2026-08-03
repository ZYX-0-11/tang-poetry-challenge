"""运行一分钟冒烟测试及跨被试静息态 vs 数字任务 TBR 对比验证。"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

from iran_edf import read_edf_eeg
from tbr_modules import run_tbr_pipeline


ROOT = Path(__file__).resolve().parent
PYTHON_CMD = (
    r"C:\Users\huangna\.cache\codex-runtimes\codex-primary-runtime"
    r"\dependencies\python\python.exe"
)


def subject_files(folder: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in folder.glob("*.edf"):
        match = re.match(r"\s*(\d+)", path.name)
        if match:
            files[int(match.group(1))] = path
    return files


def analyze_file(path: Path, start_seconds: float, duration_seconds: float):
    eeg, fs, _ = read_edf_eeg(path, start_seconds, duration_seconds)
    return run_tbr_pipeline(eeg, fs)


def save_results(rows: list[dict[str, float | int]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["subject", "eyes_closed_mean_tbr", "numeric_task_mean_tbr", "difference"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _get_chinese_font():
    """查找系统可用的中文字体。"""
    import matplotlib.font_manager as fm
    for f in fm.fontManager.ttflist:
        if "CJK" in f.name and "SC" in f.name:
            return f.fname
    for f in fm.fontManager.ttflist:
        if "CJK" in f.name:
            return f.fname
    for f in fm.fontManager.ttflist:
        if any(k in f.name.lower() for k in ("noto", "wqy", "wenquan", "arphic")):
            return f.fname
    return None


def save_chart(ec_values: np.ndarray, task_values: np.ndarray, output: Path) -> None:
    means = np.array([ec_values.mean(), task_values.mean()])
    if ec_values.size > 1:
        sems = np.array([
            ec_values.std(ddof=1) / np.sqrt(ec_values.size),
            task_values.std(ddof=1) / np.sqrt(task_values.size),
        ])
    else:
        sems = np.zeros(2)
    font_path = _get_chinese_font()
    if font_path:
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(font_path)
        font_prop = {"family": fm.FontProperties(fname=font_path).get_name()}
        plt.rcParams["font.family"] = font_prop["family"]
    else:
        plt.rcParams["font.family"] = "sans-serif"
    paired_test = ttest_rel(ec_values, task_values)
    decrease_percent = 100.0 * (ec_values.mean() - task_values.mean()) / ec_values.mean()
    fig = plt.figure(figsize=(9.2, 6.6), dpi=180, facecolor="#F7F9FC")
    ax = fig.add_subplot(111, projection="3d", facecolor="#F7F9FC")

    # Two deliberately separated 3-D columns. The height always starts at zero,
    # while perspective and shading provide depth without changing the values.
    x = np.array([0.0, 1.8])
    y = np.zeros(2)
    z = np.zeros(2)
    dx = np.full(2, 0.82)
    dy = np.full(2, 0.72)
    colors = ["#4F83E7", "#F4B41A"]
    ax.bar3d(
        x, y, z, dx, dy, means,
        color=colors, edgecolor="#26364A", linewidth=0.75,
        shade=True, alpha=0.97, zsort="average",
    )

    upper = float(np.max(means + sems))
    z_limit = upper * 1.24 if upper > 0 else 1.0
    centers_x = x + dx / 2
    centers_y = y + dy / 2

    # Put mean ± SEM directly above each top face. Conventional 2-D whiskers
    # are easily occluded or visually distorted by a 3-D projection.
    for cx, cy, mean, sem in zip(centers_x, centers_y, means, sems):
        ax.text(cx, cy, mean + z_limit * 0.045, f"{mean:.3f}\n± {sem:.3f}",
                ha="center", va="bottom", fontsize=10.5, color="#17202A",
                linespacing=1.2)

    ax.set_xticks(centers_x)
    ax.set_xticklabels(["闭眼静息", "数字工作记忆"], fontsize=11)
    ax.set_yticks([])
    ax.set_zlim(0, z_limit)
    ax.set_zlabel("TBR 均值（静息与任务对比）", labelpad=11, fontsize=11)
    fig.suptitle(
        f"伊朗脑电数据集 TBR 跨被试验证（n={ec_values.size}）",
        fontsize=18, y=0.965, color="#17202A",
    )
    fig.text(
        0.5, 0.905,
        f"数字工作记忆较闭眼静息下降 {decrease_percent:.1f}%  ·  "
        f"配对 t 检验 p={paired_test.pvalue:.4f}",
        ha="center", va="center", fontsize=11.5,
        color="#536273",
    )

    # A balanced viewing angle keeps both top faces visible and labels readable.
    ax.view_init(elev=24, azim=-58)
    ax.set_proj_type("persp", focal_length=0.95)
    ax.set_box_aspect((2.5, 0.85, 1.75))
    # Place the vertical TBR axis, tick labels and axis title on the left side.
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.96, 0.97, 0.99, 1.0))
        axis.pane.set_edgecolor((0.78, 0.81, 0.86, 1.0))
    ax.zaxis._axinfo["grid"]["color"] = (0.65, 0.68, 0.73, 0.32)
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.7

    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.07, top=0.88)
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, default=10.0, help="start time in each EDF (s)")
    parser.add_argument("--duration", type=float, default=60.0, help="equal duration per condition (s)")
    parser.add_argument("--max-subjects", type=int, default=None, help="optional quick-test limit")
    args = parser.parse_args()

    ec_files = subject_files(ROOT / "EC")
    task_files = subject_files(ROOT / "ERP-num")
    subjects = sorted(set(ec_files) & set(task_files))
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
    if not subjects:
        raise RuntimeError("No matched EC and ERP-num EDF files found.")

    print("=== 冒烟测试（被试 1，闭眼静息） ===")
    smoke_subject = 1 if 1 in ec_files else subjects[0]
    smoke = analyze_file(ec_files[smoke_subject], args.start, args.duration)
    for time, tbr in zip(smoke.times[:5], smoke.smoothed_tbr[:5]):
        print(f"时间: {time:.1f}s, TBR: {tbr:.2f}")
    print(f"窗口数: {smoke.times.size}；无错误。\n")

    rows = []
    print("=== 配对被试条件验证 ===")
    for position, subject in enumerate(subjects, 1):
        ec_result = smoke if subject == smoke_subject else analyze_file(
            ec_files[subject], args.start, args.duration
        )
        task_result = analyze_file(task_files[subject], args.start, args.duration)
        ec_mean = float(ec_result.smoothed_tbr.mean())
        task_mean = float(task_result.smoothed_tbr.mean())
        rows.append({
            "subject": subject,
            "eyes_closed_mean_tbr": ec_mean,
            "numeric_task_mean_tbr": task_mean,
            "difference": ec_mean - task_mean,
        })
        print(f"[{position:03d}/{len(subjects):03d}] 被试 {subject:3d}: "
              f"静息={ec_mean:.3f}, 任务={task_mean:.3f}, 差值={ec_mean-task_mean:+.3f}")

    ec_values = np.array([row["eyes_closed_mean_tbr"] for row in rows], dtype=float)
    task_values = np.array([row["numeric_task_mean_tbr"] for row in rows], dtype=float)
    lower_count = int(np.sum(task_values < ec_values))
    paired_test = ttest_rel(ec_values, task_values)
    decrease_percent = 100.0 * (ec_values.mean() - task_values.mean()) / ec_values.mean()
    results_path = ROOT / "tbr_validation_results.csv"
    chart_path = ROOT / "tbr_rest_vs_numeric_task.png"
    save_results(rows, results_path)
    save_chart(ec_values, task_values, chart_path)

    print("\n=== 群体结果 ===")
    print(f"匹配被试数: {len(subjects)}")
    print(f"闭眼静息平均 TBR: {ec_values.mean():.4f}")
    print(f"数字任务平均 TBR: {task_values.mean():.4f}")
    print(f"平均差值（静息 - 任务）: {(ec_values-task_values).mean():+.4f}")
    print(f"任务态相对下降: {decrease_percent:.2f}%")
    print(f"任务 TBR 低于静息的被试: {lower_count}/{len(subjects)}")
    print(f"配对 t 检验: t={paired_test.statistic:.4f}, p={paired_test.pvalue:.6g}")
    print(f"CSV 文件: {results_path}")
    print(f"图表文件: {chart_path}")
    if task_values.mean() < ec_values.mean():
        print("验证通过：数字工作记忆 TBR 低于闭眼静息 TBR。")
    else:
        print("验证未通过；请检查数据伪迹或实验条件。")


if __name__ == "__main__":
    main()
