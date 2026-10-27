"""Compare the restored original and cleaned pipeline on the same real workbook.

Run from the project root: python verification/verify.py
This is an independent regression check, not a model-performance evaluation.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze import analyze_data, score_data  # noqa: E402
from reference_original import compute_reference  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(data_path):
    """Return a report; every equivalence check requires exact numeric equality."""
    original = compute_reference(data_path)
    scored, summary, rank_by_w = score_data(pd.read_excel(data_path))
    cleaned = analyze_data(scored, summary)
    checks = []

    def record(name, passed, **details):
        checks.append({"check": name, "passed": bool(passed), **details})

    def arrays(name, expected, actual):
        a, b = np.asarray(expected), np.asarray(actual)
        same_shape = a.shape == b.shape
        numeric = np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)
        delta = None
        if numeric and same_shape and a.size:
            delta = float(np.max(np.abs(a.astype(float) - b.astype(float))))
        record(name, same_shape and np.array_equal(a, b), shape=list(a.shape),
               values=int(a.size), max_abs_difference=delta)

    def frames(name, expected, actual):
        try:
            pd.testing.assert_frame_equal(expected, actual, check_exact=True)
            passed, error = True, None
        except AssertionError as exc:
            passed, error = False, str(exc)[:500]
        numeric_cols = expected.select_dtypes(include="number").columns
        delta = None
        if expected.shape == actual.shape and all(c in actual for c in numeric_cols):
            a = expected[numeric_cols].to_numpy(dtype=float)
            b = actual[numeric_cols].to_numpy(dtype=float)
            if a.size:
                delta = float(np.max(np.abs(a - b)))
        record(name, passed, shape=list(expected.shape),
               max_abs_difference=delta, error=error)

    frames("all_scored_columns_and_original_row_order", original["df"], scored)
    arrays("direction_adjusted_features", original["df"][original["pos_feats"]],
           scored[original["pos_feats"]])
    arrays("standardized_features", original["df"][original["z_cols"]],
           scored[original["z_cols"]])
    arrays("composite_score_all_rows", original["df"]["composite_score"],
           scored["composite_score"])
    arrays("row_score_100", original["df"]["row_score_100"], scored["row_score_100"])
    frames("W_ranking_values_and_order", original["rank_by_W"], rank_by_w)
    frames("triad_summary_all_columns_and_order", original["triad_summary"], summary)
    frames("triad_summary_top10", original["top10_triad"], summary.head(10))

    additivity = {}
    for prefix, suffix, top_key in [
        ("A_triad_noW", "triad", "top10_triads"),
        ("B_W18", "w18", "top10_idx_w18"),
    ]:
        result = cleaned[prefix]
        x, y = result["x"], result["y"]
        frames(f"{prefix}.training_features_and_order", original[f"X_{suffix}"], x)
        frames(f"{prefix}.training_targets_and_order", original[f"y_{suffix}"].to_frame(),
               y.to_frame())
        params_match = original[f"rf_{suffix}"].get_params() == result["model"].get_params()
        record(f"{prefix}.RF_parameters", params_match)

        old_trees = original[f"rf_{suffix}"].estimators_
        new_trees = result["model"].estimators_
        states_equal = len(old_trees) == len(new_trees)
        for old, new in zip(old_trees, new_trees):
            a, b = old.tree_.__getstate__(), new.tree_.__getstate__()
            states_equal = states_equal and a.keys() == b.keys()
            states_equal = states_equal and all(np.array_equal(a[k], b[k]) for k in a)
        record(f"{prefix}.all_tree_structures_and_values", states_equal, trees=len(old_trees))

        arrays(f"{prefix}.predictions", original[f"pred_{suffix}"], result["predictions"])
        arrays(f"{prefix}.feature_importance", original[f"rf_{suffix}"].feature_importances_,
               result["model"].feature_importances_)
        frames(f"{prefix}.FI_sorted_table", original[f"imp_{suffix}"].reset_index(drop=True),
               result["importance"][["feature", "importance"]].reset_index(drop=True))
        arrays(f"{prefix}.all_SHAP_values", original[f"sv_{suffix}"], result["shap_values"])
        arrays(f"{prefix}.SHAP_base_value", np.asarray(original[f"base_{suffix}"]).reshape(-1),
               np.asarray(result["base_value"]).reshape(-1))
        arrays(f"{prefix}.mean_absolute_SHAP", np.abs(original[f"sv_{suffix}"]).mean(axis=0),
               np.abs(result["shap_values"]).mean(axis=0))
        arrays(f"{prefix}.waterfall_top10_ids_and_order", original[top_key], result["top_indices"])
        original_positions = original[f"X_{suffix}"].index.get_indexer(original[top_key])
        new_positions = x.index.get_indexer(result["top_indices"])
        arrays(f"{prefix}.waterfall_top10_features", original[f"X_{suffix}"].iloc[original_positions],
               x.iloc[new_positions])
        arrays(f"{prefix}.waterfall_top10_SHAP", original[f"sv_{suffix}"][original_positions],
               result["shap_values"][new_positions])
        additivity[prefix] = float(np.max(np.abs(
            result["base_value"] + result["shap_values"].sum(axis=1) - result["predictions"]
        )))

    dependencies = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, _ = line.split("==")
            dependencies[name] = version(name)
    numeric_differences = [c["max_abs_difference"] for c in checks
                           if c.get("max_abs_difference") is not None]
    exact_pass = all(c["passed"] for c in checks)
    additive_pass = all(v <= 1e-10 for v in additivity.values())
    report = {
        "status": "PASS" if exact_pass and additive_pass else "FAIL",
        "comparison": "restored_original_vs_cleaned_in_the_same_environment",
        "comparison_time_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": Path(data_path).name,
        "input_sha256": sha256(data_path),
        "cleaned_code_sha256": sha256(ROOT / "analyze.py"),
        "reference_code_sha256": sha256(ROOT / "verification/reference_original.py"),
        "verification_code_sha256": sha256(Path(__file__)),
        "requirements_sha256": sha256(ROOT / "requirements.txt"),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": dependencies},
        "counts": {"rows": len(scored), "triads": len(summary),
                   "W18_rows": int((scored["W"] == 18).sum()),
                   "features_per_model": 6, "trees_per_model": 800,
                   "SHAP_values_per_model": {name: int(result["shap_values"].size)
                                             for name, result in cleaned.items()}},
        "equivalence_rule": "exact equality; no rounding or tolerance used to pass a comparison",
        "equivalence_checks": len(checks),
        "passed_equivalence_checks": sum(c["passed"] for c in checks),
        "max_numeric_absolute_difference": max(numeric_differences),
        "SHAP_additivity_max_abs_error": additivity,
        "SHAP_additivity_absolute_tolerance": 1e-10,
        "checks": checks,
        "limits": [
            "No historical output files or historical Python environment were supplied.",
            "Equivalence is demonstrated on the included workbook and recorded environment.",
            "Figure pixels and workbook bytes are not required to be identical; plotted numbers are compared.",
            "This regression check does not measure held-out predictive performance or scientific validity."
        ],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/triadW_features.xlsx")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/verification_report.json")
    args = parser.parse_args()
    report = compare(args.data)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {report['passed_equivalence_checks']}/{report['equivalence_checks']} exact comparisons")
    print(f"Maximum numeric difference: {report['max_numeric_absolute_difference']}")
    print(f"Report: {args.report}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
