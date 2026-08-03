# TBR algorithm and Iranian EEG validation

The four reusable, label-independent math modules are in `tbr_modules.py`:

1. `clean_signal`: 50 Hz notch, then 4--30 Hz zero-phase band-pass.
2. `calculate_window_periodograms`: `scipy.signal.periodogram` in 2 s windows with a 0.5 s step.
3. `extract_tbr`: mean 4--8 Hz power divided by mean 13--30 Hz power.
4. `ewma_smooth`: `smooth[t] = 0.3 * current[t] + 0.7 * smooth[t-1]`.

`validate_tbr.py` uses folder-level condition labels only after the mathematical
pipeline has produced TBR values. It compares equal 60 s excerpts beginning at
10 s for every subject available in both `EC` and `ERP-num`.

Run from PowerShell in this folder:

```powershell
$env:PYTHONPATH = "D:\EEGdata\.python_packages"
& "C:\Users\huangna\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\validate_tbr.py
```

For a fast single-subject check, append `--max-subjects 1`.

Outputs:

- `tbr_validation_results.csv`: per-subject condition means.
- `tbr_rest_vs_numeric_task.png`: group mean bars with standard-error bars.
- `TBR_VALIDATION_REPORT.md`: methods and report-ready group findings.

Verified group result (98 matched subjects): eyes-closed mean TBR 16.0934,
numeric-task mean TBR 13.0147 (19.13% lower; 65/98 paired subjects show the
same direction).
