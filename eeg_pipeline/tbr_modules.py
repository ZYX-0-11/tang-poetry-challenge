"""Label-independent building blocks for Theta/Beta Ratio (TBR) analysis.

All functions operate only on numerical EEG samples and sampling frequency.
The expected array layout is ``(n_channels, n_samples)``; one-dimensional
single-channel arrays are also accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, periodogram, sosfiltfilt, filtfilt


@dataclass(frozen=True)
class TBRResult:
    """Window-level output from the complete TBR pipeline."""

    times: np.ndarray
    theta_power: np.ndarray
    beta_power: np.ndarray
    raw_tbr: np.ndarray
    smoothed_tbr: np.ndarray


def _as_channel_matrix(eeg: np.ndarray) -> np.ndarray:
    data = np.asarray(eeg, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2:
        raise ValueError("EEG data must have shape (channels, samples) or (samples,).")
    if not np.all(np.isfinite(data)):
        raise ValueError("EEG data contains NaN or infinite values.")
    return data


def clean_signal(
    eeg: np.ndarray,
    fs: float,
    notch_hz: float = 50.0,
    notch_q: float = 30.0,
    band_hz: tuple[float, float] = (4.0, 30.0),
    bandpass_order: int = 4,
) -> np.ndarray:
    """Apply a 50 Hz notch followed by a zero-phase 4--30 Hz band-pass.

    Zero-phase forward/backward filtering avoids shifting EEG events in time.
    Although 50 Hz lies above the 30 Hz passband, the explicit notch is kept as
    a separate requested processing stage and improves rejection before the
    band-pass roll-off.
    """

    data = _as_channel_matrix(eeg)
    if fs <= 0:
        raise ValueError("Sampling frequency fs must be positive.")
    nyquist = fs / 2.0
    low_hz, high_hz = band_hz
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(f"Band-pass must satisfy 0 < low < high < Nyquist ({nyquist:g} Hz).")
    if not 0 < notch_hz < nyquist:
        raise ValueError(f"Notch frequency must be below Nyquist ({nyquist:g} Hz).")

    notch_b, notch_a = iirnotch(notch_hz, notch_q, fs=fs)
    notched = filtfilt(notch_b, notch_a, data, axis=-1)
    band_sos = butter(
        bandpass_order,
        (low_hz, high_hz),
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(band_sos, notched, axis=-1)


def calculate_window_periodograms(
    eeg: np.ndarray,
    fs: float,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate a SciPy periodogram in every overlapping time window.

    Returns ``(times, frequencies, psd)``. Times are window centers, so the
    first 0--2 s window is reported as Time=1.0 s. PSD has shape
    ``(n_windows, n_channels, n_frequencies)``.
    """

    data = _as_channel_matrix(eeg)
    if fs <= 0 or window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("fs, window_seconds and step_seconds must be positive.")
    window_samples = int(round(window_seconds * fs))
    step_samples = int(round(step_seconds * fs))
    if window_samples < 2 or step_samples < 1:
        raise ValueError("Window/step is too short for the sampling frequency.")
    if data.shape[-1] < window_samples:
        raise ValueError("EEG segment is shorter than one analysis window.")

    starts = np.arange(0, data.shape[-1] - window_samples + 1, step_samples)
    psd_windows = []
    frequencies = None
    for start in starts:
        frequencies, psd = periodogram(
            data[:, start : start + window_samples],
            fs=fs,
            window="hann",
            detrend="constant",
            scaling="density",
            axis=-1,
        )
        psd_windows.append(psd)

    times = (starts + window_samples / 2.0) / fs
    return times, frequencies, np.stack(psd_windows, axis=0)


def extract_tbr(
    frequencies: np.ndarray,
    psd: np.ndarray,
    theta_hz: tuple[float, float] = (4.0, 8.0),
    beta_hz: tuple[float, float] = (13.0, 30.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract mean theta/beta powers and compute TBR for every window.

    Formula::

        TBR = mean(PSD[4 <= f <= 8]) / mean(PSD[13 <= f <= 30])

    The mean also spans EEG channels, producing one robust whole-head value per
    window. No labels or condition information are used.
    """

    frequencies = np.asarray(frequencies, dtype=np.float64)
    spectra = np.asarray(psd, dtype=np.float64)
    if spectra.ndim != 3 or spectra.shape[-1] != frequencies.size:
        raise ValueError("PSD must have shape (windows, channels, frequencies).")
    theta_mask = (frequencies >= theta_hz[0]) & (frequencies <= theta_hz[1])
    beta_mask = (frequencies >= beta_hz[0]) & (frequencies <= beta_hz[1])
    if not np.any(theta_mask) or not np.any(beta_mask):
        raise ValueError("Frequency resolution does not contain the requested bands.")

    theta_power = spectra[..., theta_mask].mean(axis=(-2, -1))
    beta_power = spectra[..., beta_mask].mean(axis=(-2, -1))
    tiny = np.finfo(np.float64).tiny
    tbr = theta_power / np.maximum(beta_power, tiny)
    return theta_power, beta_power, tbr


def ewma_smooth(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Smooth TBR using EWMA: smooth[t] = 0.3*current + 0.7*previous."""

    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("EWMA input must be a one-dimensional sequence.")
    if not 0 < alpha <= 1:
        raise ValueError("EWMA alpha must be in (0, 1].")
    if series.size == 0:
        return series.copy()
    smoothed = np.empty_like(series)
    smoothed[0] = series[0]
    for index in range(1, series.size):
        smoothed[index] = alpha * series[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def run_tbr_pipeline(
    eeg: np.ndarray,
    fs: float,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
    ewma_alpha: float = 0.3,
) -> TBRResult:
    """Run the four label-independent modules in their intended order."""

    cleaned = clean_signal(eeg, fs)
    times, frequencies, psd = calculate_window_periodograms(
        cleaned, fs, window_seconds, step_seconds
    )
    theta_power, beta_power, raw_tbr = extract_tbr(frequencies, psd)
    smoothed_tbr = ewma_smooth(raw_tbr, ewma_alpha)
    return TBRResult(times, theta_power, beta_power, raw_tbr, smoothed_tbr)
