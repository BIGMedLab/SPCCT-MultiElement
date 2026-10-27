"""Reproduce the supplied triad scoring, random forest, and SHAP analysis.

Run: python analyze.py
Input: data/raw/triadW_features.xlsx. Results: a new folder under outputs/.
The random forests explain the calculated scores on their training data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
from io import BytesIO
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ROOT = Path(__file__).resolve().parent
RAW_FEATURES = ["S2", "S3", "S4", "L_over_6", "Riskx_geom", "Detect_geo"]
FEATURES = [f"_{name}_pos" for name in RAW_FEATURES]
Z_COLUMNS = [f"{name}_z" for name in FEATURES]
N_ESTIMATORS = 800
RANDOM_STATE = 42
TOP_N = 10


def score_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate scores with the original transforms, arithmetic, and sort order."""
    required = ["triad", "W", *RAW_FEATURES]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if raw.columns.duplicated().any() or not raw.index.is_unique:
        raise ValueError("Column names and input row indices must be unique.")
    if raw["triad"].isna().any() or raw["triad"].astype(str).str.strip().eq("").any():
        raise ValueError("triad contains missing or blank identifiers.")
    for name in ["W", *RAW_FEATURES]:
        if not pd.api.types.is_numeric_dtype(raw[name]) or not np.isfinite(raw[name]).all():
            raise ValueError(f"{name} must contain only finite numeric values.")

    # Keep the original row order and index, including when the W filter removes rows.
    df = raw[(raw["W"] >= 10) & (raw["W"] <= 26)].copy()
    if df.empty:
        raise ValueError("No rows remain after the inclusive W = 10..26 filter.")
    df["_S2_pos"] = df["S2"]
    df["_S3_pos"] = 1 - np.abs(df["S3"] - 1)
    df["_S4_pos"] = df["S4"]
    df["_L_over_6_pos"] = -df["L_over_6"]
    df["_Riskx_geom_pos"] = 1 - np.abs(df["Riskx_geom"] - 1)
    df["_Detect_geo_pos"] = df["Detect_geo"]

    # Fit one scaler to all filtered rows, not separately to each W or triad.
    df[Z_COLUMNS] = StandardScaler().fit_transform(df[FEATURES])
    df["composite_score"] = 0.0
    for column in Z_COLUMNS:
        # Preserve the original sequential, equally weighted sum (not an average).
        df["composite_score"] += df[column] * 1.0

    rank_by_w = (
        df.groupby(["W", "triad"], as_index=False)["composite_score"]
        .mean()
        .sort_values(["W", "composite_score"], ascending=[True, False])
    )
    summary = df.groupby("triad").agg(
        mean_score=("composite_score", "mean"),
        median_score=("composite_score", "median"),
        max_score=("composite_score", "max"),
        n_rows=("composite_score", "size"),
        mean_detect_geo=("Detect_geo", "mean"),
    )
    first_places = (
        rank_by_w.groupby("W").head(1).groupby("triad").size().rename("num_first_places")
    )
    summary = summary.join(first_places, how="left").fillna({"num_first_places": 0})
    summary = summary.sort_values(
        ["mean_score", "num_first_places", "max_score"], ascending=[False, False, False]
    )
    summary["final_score_100"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(
        summary[["mean_score"]]
    )
    df["row_score_100"] = MinMaxScaler(feature_range=(0, 100)).fit_transform(
        df[["composite_score"]]
    )
    return df, summary, rank_by_w


def normalize_shap(values, shape: tuple[int, int]) -> np.ndarray:
    """Accept the single-output SHAP array formats used by supported releases."""
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Expected SHAP values for one regression target.")
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"Invalid SHAP array: expected {shape}, received {values.shape}.")
    return values


def fit_analysis(x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                 top_indices: list) -> dict:
    """Fit and explain one analysis without reordering its training rows."""
    import shap

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(x, y)
    explainer = shap.TreeExplainer(model)
    values = normalize_shap(explainer.shap_values(x), x.shape)
    base = np.asarray(explainer.expected_value).reshape(-1)
    if base.size != 1 or not np.isfinite(base).all():
        raise ValueError("Expected one finite SHAP base value.")
    predictions = model.predict(x)
    if not np.allclose(float(base[0]) + values.sum(axis=1), predictions,
                       rtol=1e-5, atol=1e-7):
        raise ValueError("SHAP contributions do not reconstruct the model predictions.")
    importance = pd.DataFrame({
        "feature": x.columns, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return {
        "x": x, "y": y, "meta": meta, "top_indices": top_indices,
        "model": model, "explainer": explainer, "shap_values": values,
        "base_value": float(base[0]), "predictions": predictions, "importance": importance,
    }


def analyze_data(scored: pd.DataFrame, summary: pd.DataFrame) -> dict:
    """A: triad mean features/score. B: original rows/score at W=18."""
    triad_features = scored.groupby("triad")[FEATURES].mean()
    triad_target = summary["mean_score"].copy()
    common = triad_features.index.intersection(triad_target.index)
    x_triad = triad_features.loc[common].copy()
    y_triad = triad_target.loc[common].copy()
    top_triads = summary.sort_values("mean_score", ascending=False).head(TOP_N).index.tolist()
    top_triads = [triad for triad in top_triads if triad in x_triad.index]
    meta_triad = pd.DataFrame({"triad": x_triad.index}, index=x_triad.index)

    w18 = scored[scored["W"] == 18].copy()
    if w18.empty:
        raise ValueError("Analysis B requires at least one row at W=18.")
    top_w18 = w18.sort_values("composite_score", ascending=False).head(TOP_N).index.tolist()
    return {
        "A_triad_noW": fit_analysis(x_triad, y_triad, meta_triad, top_triads),
        "B_W18": fit_analysis(w18[FEATURES].copy(), w18["composite_score"].copy(),
                              w18[["triad", "W"]].copy(), top_w18),
    }


def safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")[:80] or "item"


def save_figure(fig, path: Path) -> None:
    """Finish rendering before writing a PNG to disk."""
    import matplotlib.pyplot as plt

    try:
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
            png = buffer.getvalue()
        if not png:
            raise OSError(f"PNG rendering returned no data: {path.name}")
        path.write_bytes(png)
    finally:
        plt.close(fig)


def save_plots(result: dict, prefix: str, directory: Path) -> None:
    """Save FI, two global SHAP plots, and target-score top-10 waterfalls."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    x, values = result["x"], result["shap_values"]
    importance = result["importance"]
    title = ("Triad-level, no W: mean_score" if prefix == "A_triad_noW"
             else "W=18 only: composite_score")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(importance["feature"], importance["importance"])
    ax.invert_yaxis()
    ax.set_title(f"Feature Importance ({title})")
    ax.set_xlabel("Random forest impurity-based importance")
    save_figure(fig, directory / f"{prefix}_feature_importance.png")

    for kind, name in [("dot", "beeswarm"), ("bar", "bar")]:
        # This seed controls only beeswarm jitter; the model has its own fixed seed.
        shap.summary_plot(values, x, plot_type=kind, show=False,
                          rng=np.random.default_rng(RANDOM_STATE))
        fig = plt.gcf()
        save_figure(fig, directory / f"{prefix}_SHAP_summary_{name}.png")

    for rank, index in enumerate(result["top_indices"], start=1):
        position = x.index.get_loc(index)
        explanation = shap.Explanation(
            values=values[position], base_values=result["base_value"],
            data=x.iloc[position].to_numpy(), feature_names=list(x.columns),
        )
        shap.plots.waterfall(explanation, show=False)
        fig = plt.gcf()
        meta = result["meta"].iloc[position]
        identifiers = "_".join(f"{key}_{safe_filename(value)}" for key, value in meta.items())
        save_figure(fig, directory / f"{prefix}_waterfall_rank{rank}_{identifiers}.png")


def export_results(scored: pd.DataFrame, summary: pd.DataFrame, rank_by_w: pd.DataFrame,
                   analyses: dict, directory: Path, make_plots: bool = True) -> None:
    """Keep the original score/FI tables and add the numeric values behind SHAP."""
    directory.mkdir(parents=True, exist_ok=True)
    original_columns = ["triad", "W", *RAW_FEATURES, "composite_score", "row_score_100"]
    scored[original_columns].to_excel(directory / "triad_scored_allrows.xlsx", index=False)
    with pd.ExcelWriter(directory / "triad_summary_scored.xlsx", engine="openpyxl") as writer:
        summary.reset_index().to_excel(writer, sheet_name="triad_summary_all", index=False)
        summary.head(TOP_N).reset_index().to_excel(writer, sheet_name="triad_top10", index=False)
        rank_by_w.to_excel(writer, sheet_name="rank_by_W", index=False)
    scored.to_csv(directory / "scored_features.csv", index=False, encoding="utf-8-sig")

    for prefix, result in analyses.items():
        subset_dir = directory / prefix
        subset_dir.mkdir(exist_ok=True)
        result["importance"].to_excel(
            subset_dir / f"{prefix}_feature_importance.xlsx", index=False
        )
        global_importance = result["importance"].copy()
        mean_abs = pd.Series(np.abs(result["shap_values"]).mean(axis=0), index=FEATURES)
        global_importance["mean_abs_SHAP"] = global_importance["feature"].map(mean_abs)
        global_importance.to_csv(subset_dir / "global_importance.csv", index=False)

        rows = result["meta"].copy()
        rows[result["y"].name] = result["y"]
        rows["predicted_score"] = result["predictions"]
        rows["residual_target_minus_prediction"] = result["y"].to_numpy() - result["predictions"]
        rows["SHAP_base_value"] = result["base_value"]
        rows["SHAP_additivity_error"] = (
            result["base_value"] + result["shap_values"].sum(axis=1) - result["predictions"]
        )
        selected_ranks = {index: rank for rank, index in enumerate(result["top_indices"], 1)}
        rows["waterfall_rank"] = [selected_ranks.get(index) for index in rows.index]
        for position, feature in enumerate(FEATURES):
            rows[feature] = result["x"][feature]
            rows[f"SHAP_{feature}"] = result["shap_values"][:, position]
        rows.to_csv(subset_dir / "predictions_and_SHAP.csv", index=False, encoding="utf-8-sig")
        if make_plots:
            save_plots(result, prefix, subset_dir)


def run_pipeline(data_path: Path, output_parent: Path, make_plots: bool = True) -> dict:
    """Run both analyses and record the environment and input checksum."""
    data_path = Path(data_path)
    if not data_path.is_file() or data_path.suffix.lower() != ".xlsx":
        raise ValueError(f"Input must be an existing .xlsx file: {data_path}")
    scored, summary, rank_by_w = score_data(pd.read_excel(data_path))
    print(f"Scored {len(scored)} rows / {len(summary)} triads. Fitting RF + SHAP...", flush=True)
    analyses = analyze_data(scored, summary)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = Path(output_parent) / f"run_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "created_utc": stamp,
        "input_filename": data_path.name,
        "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "versions": {"python": sys.version.split()[0], **{
            name: metadata.version(name) for name in
            ["numpy", "pandas", "scikit-learn", "shap", "matplotlib", "openpyxl"]
        }},
        "scoring": {"W_inclusive": [10, 26], "scaler": "StandardScaler on all filtered rows",
                    "composite_score": "Sequential sum of six z-scores, each weight = 1.0"},
        "model": {"type": "RandomForestRegressor", "n_estimators": N_ESTIMATORS,
                  "random_state": RANDOM_STATE, "other_parameters": "scikit-learn defaults"},
        "SHAP": {"explainer": "TreeExplainer", "background": None,
                 "feature_perturbation": "tree_path_dependent (no background data)"},
        "evaluation": "Training-data explanations; no held-out performance assessment",
        "waterfalls": "Top 10 observed calculated scores, using the original sorting rules",
        "filtered_rows": len(scored), "triads": len(summary), "plots_requested": make_plots,
        "analyses": {prefix: {
            "rows": len(result["x"]), "features": list(result["x"].columns),
            "target": result["y"].name, "SHAP_base_value": result["base_value"],
            "SHAP_additivity_max_abs_error": float(np.max(np.abs(
                result["base_value"] + result["shap_values"].sum(axis=1) - result["predictions"]
            ))),
        } for prefix, result in analyses.items()},
    }
    manifest_path = directory / "run_manifest.json"
    try:
        export_results(scored, summary, rank_by_w, analyses, directory, make_plots)
        manifest["status"] = "complete"
        manifest["PNG_count"] = len(list(directory.rglob("*.png")))
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                            allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved: {directory}", flush=True)
    return {"directory": directory, "scored": scored, "summary": summary,
            "rank_by_w": rank_by_w, "analyses": analyses, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/triadW_features.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-plots", action="store_true", help="Save numeric results without PNG files.")
    args = parser.parse_args()
    try:
        run_pipeline(args.data, args.output, make_plots=not args.no_plots)
    except (ValueError, ImportError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
