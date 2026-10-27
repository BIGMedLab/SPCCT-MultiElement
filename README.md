# Triad scoring, Random Forest, and SHAP analysis

This repository computes composite scores from the supplied feature workbook and performs two complementary analyses:

- **A — Triad-level analysis:** Aggregate observations across W values for each triad.
- **B — W = 18 analysis:** Analyze the 84 observations at W = 18.

The workflow provides **data-driven ranking with Random Forest and SHAP analysis**. It includes the original dataset, reproducible Python scripts, feature-importance figures, SHAP visualizations, and a numerical verification report.

## Quick start

Install Python 3.12, extract the ZIP file, and open a terminal in the directory containing `analyze.py`.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe analyze.py
```

### macOS / Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python analyze.py
```

The default input is `data/raw/triadW_features.xlsx`. Each run writes outputs into a new timestamped folder under `outputs/`. The `results/` directory contains previously generated, verified outputs.

To specify custom paths:

```powershell
.\.venv\Scripts\python.exe analyze.py --data "C:\Users\uesr\Downloads\triadW_features for ML_V2.xlsx" --output "C:\Users\uesr\Downloads\triad_results"
```

Add `--no-plots` to calculate scores, models, and SHAP values without saving PNG figures. Supported options are `--data`, `--output`, and `--no-plots`.

## Repository structure

| Path | Description |
| --- | --- |
| `analyze.py` | Complete scoring, Random Forest, SHAP, and plotting pipeline |
| `requirements.txt` | Python dependency versions used in verification |
| `data/raw/triadW_features.xlsx` | Unmodified copy of the supplied source workbook |
| `results/` | Verified scores, predictions, importance values, SHAP results, and figures |
| `verification/reference_original.py` | Independent reconstruction of the original numerical calculations |
| `verification/verify.py` | Reproducibility and numerical comparison script |
| `docs/VERIFICATION.md` | Verification methods, scope, and limitations |
| `docs/DATA_DICTIONARY.md` | Source variables and scoring definitions |
| `docs/GITHUB_GUIDE.md` | Instructions for publishing and running the repository |
| `.gitignore` | Excludes virtual environments, caches, and local generated runs |

Run `analyze.py` for the complete analysis; no separate notebook state or manually prepared intermediate tables are required.

## Analyses and outputs

The script produces scored observations and triad-level summaries, fits Random Forest models for both analysis groups, and exports feature importance, SHAP summary plots, and top-ranked waterfall plots.

**A — Triad-level:** 84 aggregated triads, with 6 model features.

**B — W = 18:** 84 observations at W = 18, with 6 model features.

See `docs/DATA_DICTIONARY.md` for the precise scoring conventions and `results/verified_run/` for the exported results.

## Verification

The cleaned workflow and an independently reconstructed version of the supplied original code were executed against the same real input workbook in the same environment. All **34 numerical, structural, and ordering checks passed**, with a maximum absolute numerical difference of **0.0**. An additional **11 saved Excel/CSV table checks** also passed.

These results establish agreement with the reconstructed original code **in the verification environment**; they do not independently compare against historical output files from another computer or environment.

Full details: [Verification report](docs/VERIFICATION.md) and [machine-readable report](results/verification_report.json).

## Reproducibility notes

The workbook contains 420 rows covering 84 triads and five W settings. The source data are included for reproducibility. Model explanations describe the fitted score-prediction model; they should not be interpreted as experimental causal effects. Results may depend on the specified package versions and environment.

## License and citation

No license is provided in this repository. If publishing publicly, add a license only after confirming permission to distribute the dataset and code. Cite the associated research manuscript when available.
