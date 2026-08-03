"""live_server.py — 实时 BCI 服务

串联三条链路：
  device_adapter.open_source()  →  收原始 EEG
  bci_core.process()            →  算 TBR + 状态
  WebSocket 广播                →  推 {"state":0|1, "tbr":12.34} 到前端

启动流程：
  1. 打开 EEG 数据源
  2. 采集 30 秒静息数据 → bci_core.calibrate() 计算个人阈值
  3. 启动 WebSocket，进入主循环：每秒采一帧 → 处理 → 广播

用法：
  python live_server.py                                # mock 模式
  python live_server.py --protocol mock --edf EC/1     # 指定 EDF
  python live_server.py --protocol lsl                 # LSL (待硬件)
  python live_server.py --calibrate-sec 20 --port 9999
"""

from __future__ import annotations

import asyncio
import json
import signal
import time
from pathlib import Path

import numpy as np
import websockets
from websockets.asyncio.server import serve

from bci_core import calibrate, calibrate_dual, process, DEFAULT_TBR_THRESHOLD
from device_adapter import open_source, EEGSource

ROOT = Path(__file__).resolve().parent

# 最少需要 2 秒数据才能出第一个 TBR 窗口
MIN_WINDOW_SECONDS = 2.0


def _build_config(args) -> dict:
    """根据命令行参数构造 device_adapter 配置。"""
    config: dict = {"protocol": args.protocol, "chunk_seconds": args.chunk_seconds}
    if args.protocol == "mock":
        edf = args.edf
        if not Path(edf).exists():
            edf = str(ROOT / edf)
        config["edf_path"] = edf
        config["edf_start_sec"] = args.edf_start
        config["edf_duration_sec"] = args.edf_duration
    elif args.protocol == "serial":
        config["com_port"] = args.com_port
        config["baudrate"] = args.baudrate
        config["fs"] = args.fs_override
        config["n_channels"] = args.n_channels
    elif args.protocol in ("lsl", "udp"):
        config["fs"] = args.fs_override
        config["n_channels"] = args.n_channels
    return config


async def _broadcast(websockets_set: set, msg: str) -> None:
    """向所有已连接客户端发送消息，断开连接的自动移除。"""
    dead = []
    for ws in websockets_set:
        try:
            await ws.send(msg)
        except websockets.exceptions.ConnectionClosed:
            dead.append(ws)
    for ws in dead:
        websockets_set.discard(ws)


async def run_server(args) -> None:
    """主服务入口。"""
    # ---- 第1步：打开数据源 ----
    config = _build_config(args)
    print(f"[live_server] 打开数据源: protocol={config['protocol']}")
    source: EEGSource = open_source(config)
    print(f"[live_server] 采样率={source.fs} Hz  通道数={source.n_channels}")

    try:
        # ---- 第2步：个性化校准 ----
        inverted = False
        calibration_quality = "unknown"
        if args.no_calibrate:
            threshold = DEFAULT_TBR_THRESHOLD
            print(f"[live_server] 跳过校准，使用默认阈值={threshold:.2f}")
        elif args.active_calibrate:
            # 双条件校准：静息 30s + 活跃 30s
            cal_sec = args.calibrate_sec
            cal_chunks = max(1, int(cal_sec * source.fs) // int(source.fs * args.chunk_seconds))

            print(f"[live_server] 阶段1/2：闭眼静息 {cal_sec}s ...")
            rest_buffer = [source.read_chunk() for _ in range(cal_chunks)]
            rest_eeg = np.concatenate(rest_buffer, axis=-1)

            print(f"[live_server] 阶段2/2：心算任务 {cal_sec}s（请被试做简单心算）...")
            active_buffer = [source.read_chunk() for _ in range(cal_chunks)]
            active_eeg = np.concatenate(active_buffer, axis=-1)

            cal = calibrate_dual(rest_eeg, active_eeg, source.fs)
            threshold = cal["threshold"]
            inverted = cal["inverted"]
            calibration_quality = cal["quality"]

            dir_label = "反向↑(任务TBR>静息)" if inverted else "正常↓(任务TBR<静息)"
            print(f"[live_server] 校准完成。阈值={threshold:.2f}  方向={dir_label}")
            print(f"[live_server] 质量={cal['quality']}  效应量d={cal['separation_d']}  "
                  f"静息中位数={cal['resting_median']}  活跃中位数={cal['active_median']}")
            if calibration_quality == "poor":
                print(f"[live_server] ⚠ 信号质量差（静息TBR<15且两态重叠），EEG权重将降低。")
        else:
            # 单条件校准（仅静息）
            cal_sec = args.calibrate_sec
            cal_samples = int(cal_sec * source.fs)
            cal_chunks = max(1, cal_samples // int(source.fs * args.chunk_seconds))
            print(f"[live_server] 开始校准：闭眼静息 {cal_sec}s，采集 {cal_chunks} 帧...")

            cal_buffer: list[np.ndarray] = []
            for i in range(cal_chunks):
                chunk = source.read_chunk()
                cal_buffer.append(chunk)
                print(f"\r  校准进度: {i+1}/{cal_chunks} 帧", end="", flush=True)
            print()

            cal_eeg = np.concatenate(cal_buffer, axis=-1)
            threshold = calibrate(cal_eeg, source.fs)
            print(f"[live_server] 校准完成。个性化阈值={threshold:.2f} "
                  f"(群体默认={DEFAULT_TBR_THRESHOLD})")

        # ---- 第3步：启动 WebSocket ----
        connected: set = set()
        stop = asyncio.get_running_loop().create_future()

        async def handler(ws):
            connected.add(ws)
            print(f"[live_server] 客户端连接 (当前 {len(connected)} 个)")
            try:
                await ws.wait_closed()
            finally:
                connected.discard(ws)
                print(f"[live_server] 客户端断开 (剩余 {len(connected)} 个)")

        async with serve(handler, args.host, args.port) as server:
            print(f"[live_server] WebSocket: ws://{args.host}:{args.port}")
            print(f"[live_server] 阈值={threshold:.2f}  "
                  f"格式={{\"state\":0|1, \"tbr\":float}}")
            print(f"[live_server] 主循环运行中，等待前端连接... (Ctrl+C 停止)\n")

            def shutdown():
                if not stop.done():
                    stop.set_result(None)

            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, shutdown)
            loop.add_signal_handler(signal.SIGTERM, shutdown)

            # ---- 第4步：主循环：采集 → 处理 → 广播 ----
            buffer: list[np.ndarray] = []
            min_buffer_samples = int(MIN_WINDOW_SECONDS * source.fs)
            seq = 0

            while not stop.done():
                chunk = source.read_chunk()
                buffer.append(chunk)

                # 保持 buffer 总量在当前采样率下约 10 秒（防止无限增长）
                total_samples = sum(b.shape[-1] for b in buffer)
                max_samples = int(10.0 * source.fs)
                while len(buffer) > 1 and total_samples - buffer[0].shape[-1] >= max_samples:
                    total_samples -= buffer[0].shape[-1]
                    buffer.pop(0)

                current_total = sum(b.shape[-1] for b in buffer)
                if current_total < min_buffer_samples:
                    continue  # 数据还不够出第一个窗

                eeg = np.concatenate(buffer, axis=-1)
                result = process(eeg, source.fs, tbr_threshold=threshold, inverted=inverted)

                state = int(result.state[-1])
                tbr = float(result.smoothed_tbr[-1])
                msg = json.dumps({"state": state, "tbr": round(tbr, 2)})

                seq += 1
                label = "专注" if state == 1 else "走神"
                print(f"  [{seq:04d}] state={state}({label})  tbr={tbr:.2f}  "
                      f"clients={len(connected)}  buffer={current_total/source.fs:.1f}s")

                if connected:
                    await _broadcast(connected, msg)

                # 等足 1 秒再采下一帧（减去处理耗时）
                await asyncio.sleep(args.chunk_seconds)

    finally:
        source.close()
        print("[live_server] 数据源已关闭。")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="live_server — 实时 BCI 桥接服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python live_server.py                                         # mock 默认 EDF
  python live_server.py --edf "EC/1 - rsEC.edf"                 # 指定静息 EDF
  python live_server.py --edf "ERP-num/1 - rsnum.edf"           # 任务 EDF
  python live_server.py --no-calibrate --edf "ERP-num/1 - rsnum.edf"
  python live_server.py --protocol lsl                          # LSL (待硬件)
  python live_server.py --protocol serial --com-port COM3       # 串口 (待硬件)
""",
    )

    # 协议
    p.add_argument("--protocol", default="mock",
                   choices=["mock", "lsl", "serial", "udp"],
                   help="数据源协议 (默认 mock)")
    # mock 模式参数
    p.add_argument("--edf", default="EC/1 - rsEC.edf",
                   help="Mock 模式用的 EDF 文件路径")
    p.add_argument("--edf-start", type=float, default=10.0,
                   help="EDF 读取起始秒")
    p.add_argument("--edf-duration", type=float, default=300.0,
                   help="EDF 最大读取秒数")
    # 真实设备参数
    p.add_argument("--com-port", default="COM3",
                   help="串口设备路径")
    p.add_argument("--baudrate", type=int, default=115200,
                   help="串口波特率")
    p.add_argument("--fs-override", type=float, default=250.0,
                   help="(非 mock) 手动指定采样率")
    p.add_argument("--n-channels", type=int, default=8,
                   help="(非 mock) 通道数")
    # 通用参数
    p.add_argument("--chunk-seconds", type=float, default=1.0,
                   help="每帧秒数")
    p.add_argument("--calibrate-sec", type=float, default=30.0,
                   help="校准阶段秒数")
    p.add_argument("--no-calibrate", action="store_true",
                   help="跳过校准，使用群体默认阈值 14.0")
    p.add_argument("--active-calibrate", action="store_true",
                   help="双条件校准：静息30s+心算30s，自动检测TBR方向")
    p.add_argument("--host", default="localhost",
                   help="WebSocket 监听地址")
    p.add_argument("--port", type=int, default=9999,
                   help="WebSocket 监听端口")

    args = p.parse_args()
    asyncio.run(run_server(args))
