"""device_adapter.py — 硬件数据源适配器

所有脑电数据源的唯一入口。无论 EEG 来自 EDF 文件、LSL 流、串口还是 UDP，
都封装为统一的 EEGSource 对象。下游 bci_core.py 完全不用动。

8月5日拿到真实脑电环后，只需：
  1. 在 open_source() 中添加对应协议的 elif 分支
  2. 将配置中的 protocol 从 "mock" 改为 "lsl" / "serial" / "udp"

接口约定：
  source = open_source(config)
  chunk = source.read_chunk()   # → np.ndarray, shape=(channels, samples)
  source.fs                      # → float, 采样率
  source.close()                 # → 释放资源
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class EEGSource:
    """EEG 数据源抽象基类。"""

    fs: float
    n_channels: int
    _closed: bool = False

    def read_chunk(self) -> np.ndarray:
        """读取下一帧 EEG 数据。

        Returns
        -------
        np.ndarray  shape=(n_channels, chunk_samples)，通常为 1 秒数据量。
        """
        raise NotImplementedError

    def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Mock 实现：从 EDF 文件模拟实时数据流
# ---------------------------------------------------------------------------


class MockEDFSource(EEGSource):
    """用 EDF 文件模拟实时脑电环，按 chunk_seconds 逐帧输出。

    8月5日拿到真实设备后，此类会被 LSLSource / SerialSource 等替换，
    或者保留作为离线测试工具。
    """

    def __init__(self, config: dict):
        from iran_edf import read_edf_eeg

        edf_path = Path(config["edf_path"])
        start = float(config.get("edf_start_sec", 10.0))
        duration = float(config.get("edf_duration_sec", 120.0))
        self._chunk_seconds = float(config.get("chunk_seconds", 1.0))

        if not edf_path.exists():
            raise FileNotFoundError(f"EDF 文件不存在: {edf_path}")

        data, self.fs, _ = read_edf_eeg(edf_path, start, duration)
        self.n_channels = data.shape[0]
        self._data = data
        self._chunk_samples = int(round(self._chunk_seconds * self.fs))
        self._cursor = 0
        self._total_samples = data.shape[-1]

    def read_chunk(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("数据源已关闭。")
        if self._cursor + self._chunk_samples > self._total_samples:
            self._cursor = 0  # 循环回绕，模拟持续数据流
        chunk = self._data[:, self._cursor : self._cursor + self._chunk_samples]
        self._cursor += self._chunk_samples
        # 模拟真实设备的采样间隔
        time.sleep(self._chunk_seconds * 0.01)
        return chunk


# ---------------------------------------------------------------------------
# LSL 预留（8月5日按硬件手册补齐）
# ---------------------------------------------------------------------------


class LSLSource(EEGSource):
    """Lab Streaming Layer 数据源 — 待实现。"""

    def __init__(self, config: dict):
        # import pylsl
        # self._inlet = pylsl.StreamInlet(...)
        # self.fs = self._inlet.info().nominal_srate()
        # self.n_channels = self._inlet.info().channel_count()
        raise NotImplementedError(
            "LSL 支持尚未实现。请根据硬件手册补齐 LSLSource.__init__ 和 read_chunk()。"
        )

    def read_chunk(self) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 串口预留（8月5日按硬件手册补齐）
# ---------------------------------------------------------------------------


class SerialSource(EEGSource):
    """蓝牙串口数据源 — 待实现。"""

    def __init__(self, config: dict):
        # import serial
        # self._port = serial.Serial(config['com_port'], baudrate=config.get('baudrate', 115200))
        # self.fs = config['fs']
        # self.n_channels = config['n_channels']
        raise NotImplementedError(
            "串口支持尚未实现。请根据硬件手册补齐 SerialSource.__init__ 和 read_chunk()。"
        )

    def read_chunk(self) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# UDP/TCP 预留（8月5日按硬件手册补齐）
# ---------------------------------------------------------------------------


class UDPSource(EEGSource):
    """UDP/TCP Socket 数据源 — 待实现。"""

    def __init__(self, config: dict):
        # import socket
        # self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self._sock.bind((config['host'], config['port']))
        # self.fs = config['fs']
        # self.n_channels = config['n_channels']
        raise NotImplementedError(
            "UDP 支持尚未实现。请根据硬件手册补齐 UDPSource.__init__ 和 read_chunk()。"
        )

    def read_chunk(self) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 工厂函数 —— 8月5日只改这里
# ---------------------------------------------------------------------------


def open_source(config: dict | None = None) -> EEGSource:
    """根据配置打开 EEG 数据源。

    config = {
        'protocol': 'mock',       # ★ 8月5日改这里: 'lsl' / 'serial' / 'udp'
        'edf_path': 'EC/1.edf',   # mock 模式用
        'chunk_seconds': 1.0,     # 每帧秒数
        ...
    }
    """
    if config is None:
        config = {}

    protocol = config.get("protocol", "mock")

    if protocol == "mock":
        return MockEDFSource(config)
    elif protocol == "lsl":
        return LSLSource(config)
    elif protocol == "serial":
        return SerialSource(config)
    elif protocol == "udp":
        return UDPSource(config)
    else:
        raise ValueError(f"未知协议: {protocol}。支持: mock, lsl, serial, udp")


# ---------------------------------------------------------------------------
# 快速自测
# ---------------------------------------------------------------------------


def _self_test():
    config = {
        "protocol": "mock",
        "edf_path": "EC/1 - rsEC.edf",
        "edf_start_sec": 10.0,
        "edf_duration_sec": 10.0,
        "chunk_seconds": 1.0,
    }
    source = open_source(config)
    print(f"[device_adapter 自测] 协议=mock  fs={source.fs} Hz  n_channels={source.n_channels}")

    for i in range(5):
        chunk = source.read_chunk()
        print(f"  第{i+1}帧: shape={chunk.shape}  mean={chunk.mean():.4f}  std={chunk.std():.4f}")

    source.close()
    print("[device_adapter 自测] 通过。")


if __name__ == "__main__":
    _self_test()
