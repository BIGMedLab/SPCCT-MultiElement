# GitHub publishing and execution guide

## 1. Review the repository

The ZIP includes the input workbook, source code, documentation, previously verified results, and an independent verification implementation. Review the workbook and outputs for publication rights or confidential information before making the repository public.

The analysis requires only `analyze.py`, the included workbook, and the packages in `requirements.txt`. You do not need to manually prepare a DataFrame or open a notebook.

## 2. Create a GitHub repository

1. Sign in to GitHub and select **New repository**.
2. Use a descriptive name such as `triad-rf-shap-analysis`.
3. Choose **Private** initially if dataset-sharing permissions have not been confirmed.
4. Do not initialize with a new README when uploading this existing project.
5. Create the repository.

## 3. Upload the files

### Browser method

Extract the ZIP. In the new GitHub repository, choose **Add file → Upload files** and upload the contents of the project directory, preserving the folder structure. Commit the changes.

### Git command-line method

Open a terminal in the extracted project directory:

```bash
git init
git add .
git commit -m "Add reproducible triad RF and SHAP analysis"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/triad-rf-shap-analysis.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. If the repository is already initialized, adapt the commands accordingly. Authentication may require a GitHub credential manager or personal access token.

## 4. Install and run

Use Python 3.12.

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

Results are saved to a timestamped directory under `outputs/`. Existing verified figures and tables are also provided in `results/verified_run/`.

To change the input and output paths:

```bash
python analyze.py --data path/to/input.xlsx --output path/to/output_directory
```

Use `--no-plots` to skip PNG creation while retaining numeric calculations.

## 5. What the workflow does

The pipeline reads 420 records from the workbook, calculates directional transformations, standardized features, composite scores, and rankings. It then performs two Random Forest and SHAP analyses:

- **A:** Aggregated triad-level results across W values.
- **B:** Observations restricted to W = 18.

Each analysis exports predictions, feature importance, SHAP summaries, and waterfall explanations for ten top-ranked observations.

## 6. Check reproducibility

```bash
python verification/verify.py
```

The existing verification report documents 34 exact numerical/structural comparisons and 11 saved-table checks, all passing in the documented environment. This does not guarantee byte-identical figure rendering on every operating system.

## 7. Before manuscript submission

Confirm that all input data may be shared publicly, add an appropriate license if authorized, cite the source dataset and any associated publication, and include the exact software environment. Distinguish model-based feature importance from experimental or causal validation.
