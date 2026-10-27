"""Independent numerical reference restored from the user's pasted original code.

Source: the two Python sections of the supplied pasted Markdown file.
Only analysis code was transcribed; unrelated text and URLs were excluded.

Permitted adaptations:
* Remove Markdown escaping, HTML indentation markers, and nonbreaking spaces.
* Accept the input workbook path as an argument instead of a Windows literal.
* Wrap the original calculations in a callable and return intermediate results.
* Omit file writes, printing, notebook display, and plotting.
* Allow SHAP to be skipped when unavailable; record predictions as a comparison
  output. Predictions do not change fitted models or training data.
* Retain the original SHAP list-to-array helper and expose expected values.

The scoring arithmetic, its order, grouping/sorting, model parameters/defaults,
training row order, and original TreeExplainer calls are unchanged. This module
does not import any code from the cleaned analysis implementation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _ensure_2d_shap(shap_values):
    if isinstance(shap_values, list):
        return shap_values[0]
    return shap_values


def compute_reference(input_path, include_shap=True):
    """Run the restored original calculations and return original variable names.

    With ``include_shap=False``, the ``sv_*`` and ``base_*`` results are ``None``;
    scoring, model fits, importance, predictions, and selection still execute.
    """
    df = pd.read_excel(input_path)

    required_cols = [
        'triad', 'W', 'S2', 'S3', 'S4', 'L_over_6', 'Riskx_geom', 'Detect_geo'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[(df['W'] >= 10) & (df['W'] <= 26)].copy()

    def closeness_to_1(x):
        return 1 - np.abs(x - 1)

    df['_S2_pos'] = df['S2']
    df['_S3_pos'] = closeness_to_1(df['S3'])
    df['_S4_pos'] = df['S4']
    df['_L_over_6_pos'] = -df['L_over_6']
    df['_Riskx_geom_pos'] = closeness_to_1(df['Riskx_geom'])
    df['_Detect_geo_pos'] = df['Detect_geo']

    pos_feats = [
        '_S2_pos', '_S3_pos', '_S4_pos', '_L_over_6_pos',
        '_Riskx_geom_pos', '_Detect_geo_pos'
    ]
    scaler = StandardScaler()
    z_cols = [f'{c}_z' for c in pos_feats]
    df[z_cols] = scaler.fit_transform(df[pos_feats])

    weights = {
        '_S2_pos_z': 1.0,
        '_S3_pos_z': 1.0,
        '_S4_pos_z': 1.0,
        '_L_over_6_pos_z': 1.0,
        '_Riskx_geom_pos_z': 1.0,
        '_Detect_geo_pos_z': 1.0
    }
    df['composite_score'] = 0.0
    for col, w in weights.items():
        if col not in df.columns:
            raise ValueError(
                f"Weight 컬럼이 dfnot found in: {col} (z-score generation needs verification)"
            )
        df['composite_score'] += df[col] * w

    rank_by_W = (
        df.groupby(['W', 'triad'], as_index=False)['composite_score']
          .mean()
          .sort_values(['W', 'composite_score'], ascending=[True, False])
    )

    triad_summary = (
        df.groupby('triad')
          .agg(
              mean_score=('composite_score', 'mean'),
              median_score=('composite_score', 'median'),
              max_score=('composite_score', 'max'),
              n_rows=('composite_score', 'size'),
              mean_detect_geo=('Detect_geo', 'mean')
          )
    )
    first_places = (
        rank_by_W.groupby('W').head(1)
                 .groupby('triad').size()
                 .rename('num_first_places')
    )
    triad_summary = triad_summary.join(
        first_places, how='left'
    ).fillna({'num_first_places': 0})
    triad_summary = triad_summary.sort_values(
        ['mean_score', 'num_first_places', 'max_score'],
        ascending=[False, False, False]
    )

    mm_triad = MinMaxScaler(feature_range=(0, 100))
    triad_summary['final_score_100'] = mm_triad.fit_transform(
        triad_summary[['mean_score']]
    )
    mm_row = MinMaxScaler(feature_range=(0, 100))
    df['row_score_100'] = mm_row.fit_transform(df[['composite_score']])

    top10_triad = triad_summary.head(10)
    df_out = df[
        ['triad', 'W', 'S2', 'S3', 'S4', 'L_over_6', 'Riskx_geom', 'Detect_geo',
         'composite_score', 'row_score_100']
    ].copy()

    # Original A: triad-level inputs and target alignment.
    triad_feat = df.groupby("triad")[pos_feats].mean()
    y_triad = triad_summary["mean_score"].copy()
    common_triads = triad_feat.index.intersection(y_triad.index)
    X_triad = triad_feat.loc[common_triads].copy()
    y_triad = y_triad.loc[common_triads].copy()

    rf_triad = RandomForestRegressor(n_estimators=800, random_state=42)
    rf_triad.fit(X_triad, y_triad)
    imp_triad = pd.DataFrame({
        "feature": X_triad.columns,
        "importance": rf_triad.feature_importances_
    }).sort_values("importance", ascending=False)

    if include_shap:
        import shap

        expl_triad = shap.TreeExplainer(rf_triad)
        sv_triad = _ensure_2d_shap(expl_triad.shap_values(X_triad))
        base_triad = expl_triad.expected_value
    else:
        sv_triad = None
        base_triad = None

    top10_triads = (
        triad_summary.sort_values("mean_score", ascending=False)
                     .head(10).index.tolist()
    )
    meta_triad = pd.DataFrame({"triad": X_triad.index}, index=X_triad.index)
    top10_triads = [t for t in top10_triads if t in X_triad.index]

    # Original B: W=18 inputs retain the workbook's original row ordering.
    df_w18 = df[df["W"] == 18].copy()
    X_w18 = df_w18[pos_feats].copy()
    y_w18 = df_w18["composite_score"].copy()

    rf_w18 = RandomForestRegressor(n_estimators=800, random_state=42)
    rf_w18.fit(X_w18, y_w18)
    imp_w18 = pd.DataFrame({
        "feature": X_w18.columns,
        "importance": rf_w18.feature_importances_
    }).sort_values("importance", ascending=False)

    if include_shap:
        expl_w18 = shap.TreeExplainer(rf_w18)
        sv_w18 = _ensure_2d_shap(expl_w18.shap_values(X_w18))
        base_w18 = expl_w18.expected_value
    else:
        sv_w18 = None
        base_w18 = None

    top10_idx_w18 = (
        df_w18.sort_values("composite_score", ascending=False)
              .head(10).index.tolist()
    )
    meta_w18 = df_w18[["triad", "W"]].copy()

    return {
        "df": df,
        "pos_feats": pos_feats,
        "z_cols": z_cols,
        "scaler": scaler,
        "weights": weights,
        "rank_by_W": rank_by_W,
        "triad_summary": triad_summary,
        "mm_triad": mm_triad,
        "mm_row": mm_row,
        "top10_triad": top10_triad,
        "df_out": df_out,
        "triad_feat": triad_feat,
        "X_triad": X_triad,
        "y_triad": y_triad,
        "rf_triad": rf_triad,
        "imp_triad": imp_triad,
        "sv_triad": sv_triad,
        "base_triad": base_triad,
        "pred_triad": rf_triad.predict(X_triad),
        "top10_triads": top10_triads,
        "meta_triad": meta_triad,
        "df_w18": df_w18,
        "X_w18": X_w18,
        "y_w18": y_w18,
        "rf_w18": rf_w18,
        "imp_w18": imp_w18,
        "sv_w18": sv_w18,
        "base_w18": base_w18,
        "pred_w18": rf_w18.predict(X_w18),
        "top10_idx_w18": top10_idx_w18,
        "meta_w18": meta_w18,
    }
