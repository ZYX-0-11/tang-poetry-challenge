"""Small dependency-free EDF reader for the Iranian EEG dataset."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


DEFAULT_CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
)


def _text(raw: bytes) -> str:
    return raw.decode("ascii", errors="ignore").strip()


def read_edf_eeg(
    path: str | Path,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
) -> tuple[np.ndarray, float, tuple[str, ...]]:
    """Read selected EEG channels and return (channels x samples, fs, names)."""

    path = Path(path)
    with path.open("rb") as file:
        fixed = file.read(256)
        signal_count = int(_text(fixed[252:256]))
        record_count = int(_text(fixed[236:244]))
        record_duration = float(_text(fixed[244:252]))
        signal_header = file.read(256 * signal_count)

        def field(offset: int, width: int) -> list[str]:
            base = offset * signal_count
            return [
                _text(signal_header[base + i * width : base + (i + 1) * width])
                for i in range(signal_count)
            ]

        labels = field(0, 16)
        physical_mins = np.array(field(104, 8), dtype=float)
        physical_maxs = np.array(field(112, 8), dtype=float)
        digital_mins = np.array(field(120, 8), dtype=float)
        digital_maxs = np.array(field(128, 8), dtype=float)
        samples_per_record = np.array(field(216, 8), dtype=int)

        requested_labels = ["EEG " + channel for channel in channels]
        missing = [label for label in requested_labels if label not in labels]
        if missing:
            raise ValueError(f"Missing EEG channels in {path.name}: {missing}")
        selected = [labels.index(label) for label in requested_labels]
        selected_set = set(selected)

        selected_fs = samples_per_record[selected] / record_duration
        if not np.allclose(selected_fs, selected_fs[0]):
            raise ValueError("Selected EEG channels do not share one sampling frequency.")
        fs = float(selected_fs[0])
        first_sample = int(round(start_seconds * fs))
        total_samples = int(round(record_count * record_duration * fs))
        if duration_seconds is None:
            last_sample = total_samples
        else:
            last_sample = min(total_samples, first_sample + int(round(duration_seconds * fs)))
        if first_sample < 0 or first_sample >= last_sample:
            raise ValueError(f"Requested interval is outside {path.name}.")

        scales = (physical_maxs - physical_mins) / (digital_maxs - digital_mins)
        offsets = physical_mins - scales * digital_mins
        output = [[] for _ in selected]
        selected_to_output = {signal_index: out for out, signal_index in enumerate(selected)}
        sample_cursor = 0

        for _record in range(record_count):
            record_start = sample_cursor
            record_end = record_start + int(samples_per_record[selected[0]])
            intersects = record_end > first_sample and record_start < last_sample
            for signal_index, count in enumerate(samples_per_record):
                raw = file.read(2 * int(count))
                if intersects and signal_index in selected_set:
                    values = np.asarray(struct.unpack(f"<{count}h", raw), dtype=np.float64)
                    values = values * scales[signal_index] + offsets[signal_index]
                    local_start = max(0, first_sample - record_start)
                    local_end = min(int(count), last_sample - record_start)
                    output[selected_to_output[signal_index]].extend(values[local_start:local_end])
            sample_cursor = record_end
            if sample_cursor >= last_sample:
                break

    data = np.asarray(output, dtype=np.float64)
    if data.shape != (len(channels), last_sample - first_sample):
        raise RuntimeError(f"Unexpected EDF sample shape {data.shape} in {path.name}.")
    return data, fs, channels
