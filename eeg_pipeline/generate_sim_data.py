"""Process Iranian EEG EDF files through TBR pipeline → sim_data.csv.

EC (eyes-closed) → higher TBR → state=0 (走神)
ERP-num (numeric task) → lower TBR → state=1 (专注)

Output columns: second, state, tbr
Rows alternate per-subject between EC and task segments, yielding ~half state=0, half state=1.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from iran_edf import read_edf_eeg
from tbr_modules import run_tbr_pipeline

ROOT = Path(__file__).resolve().parent
START_SEC = 10.0
DURATION_SEC = 60.0


def subject_files(folder: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in folder.glob("*.edf"):
        match = re.match(r"\s*(\d+)", path.name)
        if match:
            files[int(match.group(1))] = path
    return files


def process_segment(path: Path) -> tuple[np.ndarray, np.ndarray]:
    eeg, fs, _ = read_edf_eeg(path, START_SEC, DURATION_SEC)
    result = run_tbr_pipeline(eeg, fs)
    return result.times, result.smoothed_tbr


def resample_1hz(times: np.ndarray, tbr: np.ndarray) -> np.ndarray:
    """Resample TBR from 0.5 s steps to one value per integer second."""
    target = np.arange(1, int(np.floor(times[-1])) + 1, dtype=float)
    return np.interp(target, times, tbr)


def main() -> None:
    ec_files = subject_files(ROOT / "EC")
    task_files = subject_files(ROOT / "ERP-num")
    subjects = sorted(set(ec_files) & set(task_files))
    if not subjects:
        raise RuntimeError("No matched EC and ERP-num subjects found.")

    print(f"Matched subjects: {len(subjects)}")

    rows: list[dict[str, float]] = []
    second = 0

    chunk = 5  # interleave in 5-second blocks for ~50/50 over any window

    for subject in subjects:
        ec_times, ec_tbr = process_segment(ec_files[subject])
        ec_tbr_1hz = resample_1hz(ec_times, ec_tbr)

        task_times, task_tbr = process_segment(task_files[subject])
        task_tbr_1hz = resample_1hz(task_times, task_tbr)

        n = min(len(ec_tbr_1hz), len(task_tbr_1hz))
        i = 0
        while i < n:
            end = min(i + chunk, n)
            for j in range(i, end):
                rows.append({"second": second, "state": 0, "tbr": round(float(ec_tbr_1hz[j]), 4)})
                second += 1
            for j in range(i, end):
                rows.append({"second": second, "state": 1, "tbr": round(float(task_tbr_1hz[j]), 4)})
                second += 1
            i = end

    output = ROOT / "sim_data.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["second", "state", "tbr"])
        writer.writeheader()
        writer.writerows(rows)

    states = np.array([r["state"] for r in rows], dtype=int)
    tbrs = np.array([r["tbr"] for r in rows], dtype=float)
    s0 = tbrs[states == 0]
    s1 = tbrs[states == 1]

    print(f"Total rows: {len(rows)} ({len(rows) / 3600:.2f} hours at 1 Hz)")
    print(f"State=0 (EC/drowsy):    {s0.size:5d} rows, mean TBR = {s0.mean():.3f}")
    print(f"State=1 (task/focused): {s1.size:5d} rows, mean TBR = {s1.mean():.3f}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
