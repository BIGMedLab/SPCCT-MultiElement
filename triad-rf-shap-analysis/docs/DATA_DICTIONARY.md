# Data dictionary

## Sources and scope

The repository copy [data/raw/triadW_features.xlsx](../data/raw/triadW_features.xlsx) preserves the supplied `triadW_features for ML_V2.xlsx` byte for byte, including both worksheets and document properties. The scoring rules below come from the user's original preprocessing and RF/SHAP code. [analyze.py](../analyze.py) now runs those rules directly from this workbook.

| Worksheet | Content | Source cells |
| --- | --- | --- |
| `triadW_features` | 420 records, with 11 column headers | Headers `A1:K1`; data `A2:K421` |
| `Feature descriptions` (original worksheet: `Feature description`) | Seven original feature notes (Korean in the source workbook) | Labels `B1:B7`; descriptions `C1:C7` |

Neither worksheet contains formula cells. The provided code supplies the transformations and composite-score calculation, so a separate prepared-data file is unnecessary. The upstream construction of the raw scientific features is a separate matter: the workbook notes do not fully specify how to regenerate those features from primary measurements or matrices.

## Record structure

Each row is uniquely identified by `(triad, W)` in this workbook.

- There are 84 distinct triads and five W values: 10, 14, 18, 22, and 26.
- Every triad has exactly one record at each W, giving 420 records in total.
- Filtering to W=18 gives 84 records.
- Triad labels use Au, Bi, Ce, Gd, Hf, I, Ta, Tb, and Yb.
- No values are missing; numeric entries are finite; there are no duplicate rows or duplicate `(triad, W)` keys.

These observations describe the supplied file. They do not establish whether records are independent experimental measurements.

## Raw columns

The scientific descriptions in this table are from the workbook notes. The final column states the treatment in the supplied analysis code.

| Column | Excel column | Meaning supported by the source | Source note | Analysis treatment |
| --- | --- | --- | --- | --- |
| `Number` | A | Sequential record number, 1 through 420 | No separate note | Not a predictor or ranking score |
| `triad` | B | Three-element identifier, such as `Ce-Gd-I` | No separate note | Grouping and record identifier |
| `W` | C | Energy-bin width in keV | `Feature description!B1:C1` | Inclusive 10–26 filter; A aggregates across W, B selects W=18; not a predictor |
| `S2` | D | `1 - max_offdiag(M)`; larger values indicate lower worst cross leakage | `Feature description!B2:C2` | `_S2_pos = S2` |
| `S3` | E | `det(M)`; matrix separability/invertibility, with values closer to 1 preferred | `Feature description!B3:C3` | `_S3_pos = 1 - abs(S3 - 1)` |
| `S4` | F | `1 / cond(M)`; numerical stability, with larger values preferred | `Feature description!B4:C4` | `_S4_pos = S4` |
| `L_over_6` | G | Average signal leakage; smaller values preferred; note label is `L/6` | `Feature description!B5:C5` | `_L_over_6_pos = -L_over_6` |
| `R_max` | H | Numeric descriptor related to off-diagonal row sums; standalone definition absent | Related wording in `Feature description!C6` | Not selected as a predictor |
| `C_max` | I | Numeric descriptor related to off-diagonal column sums; standalone definition absent | Related wording in `Feature description!C6` | Not selected as a predictor |
| `Riskx_geom` | J | Crosstalk measure described using off-diagonal row and column sums; the note prefers lower values, closer to 1 | `Feature description!B6:C6` | `_Riskx_geom_pos = 1 - abs(Riskx_geom - 1)` |
| `Detect_geo` | K | Signal-scale descriptor; the note prefers a larger signal scale while discussing possible effects of mixing | `Feature description!B7:C7`; label written `Dect_geo` | `_Detect_geo_pos = Detect_geo` |

The workbook note leaves the combined scoring treatment of `Detect_geo` open. The provided code resolves its treatment **for this analysis**: identity transformation, global z-score standardization, then weight 1.0 in the score.

### Remaining upstream definitions

The workbook and scoring code do not specify the construction, dimensions, or normalization of `M`; the norm used in `cond(M)`; the exact calculation of `L_over_6`; complete definitions of `R_max` and `C_max`; or the formula and units of `Detect_geo`. This repository reproduces scoring and RF/SHAP from the supplied raw feature table; it does not reconstruct that table from primary measurements.

## Preprocessing and scoring

### 1. Filter before fitting any scaler

Only rows satisfying `10 <= W <= 26` are retained. All 420 rows in the supplied workbook satisfy this condition. The original Excel row order is preserved at this stage.

### 2. Direction-adjusted features

The six feature columns are created in this exact order:

| Position | Prepared feature | Exact transformation |
| --- | --- | --- |
| 1 | `_S2_pos` | `S2` |
| 2 | `_S3_pos` | `1 - np.abs(S3 - 1)` |
| 3 | `_S4_pos` | `S4` |
| 4 | `_L_over_6_pos` | `-L_over_6` |
| 5 | `_Riskx_geom_pos` | `1 - np.abs(Riskx_geom - 1)` |
| 6 | `_Detect_geo_pos` | `Detect_geo` |

The `_pos` suffix marks the intended preference direction, not a guarantee that values are positive. No clipping, logarithm, imputation, or additional transformation is applied. In particular, the negative `S2` values described below are retained unchanged.

### 3. Global standardization

A single default `StandardScaler()` is fitted to the six prepared columns across **all retained rows**. The fitted scope is neither each W separately nor each triad separately. Each output column adds the suffix `_z`, for example `_S2_pos_z`.

For each feature, the scaler subtracts its mean and divides by its population standard deviation over the retained rows. The same fitted transformation supplies the scores used in both A and B; B does not refit the scaler on W=18.

### 4. Composite score

`composite_score` starts at `0.0`. Each z-score is multiplied by `1.0` and added sequentially in the six-feature order above:

```python
composite_score = (
    _S2_pos_z + _S3_pos_z + _S4_pos_z
    + _L_over_6_pos_z + _Riskx_geom_pos_z + _Detect_geo_pos_z
)
```

This expression explains the formula. The implementation preserves the original sequential addition order when evaluating it. The score is the **sum of six equally weighted z-scores**, without division by six.

### 5. W-specific ranking and triad summary

For the W-specific ranking, the code groups by `(W, triad)`, takes the mean `composite_score`, and sorts by W ascending and score descending. Since each pair occurs once in the supplied workbook, these means equal the corresponding row scores.

The triad summary aggregates all retained rows belonging to each triad:

| Field | Definition |
| --- | --- |
| `mean_score` | Arithmetic mean of `composite_score` across the triad's retained rows |
| `median_score` | Median of those row scores |
| `max_score` | Maximum of those row scores |
| `n_rows` | Number of retained rows; 5 for every supplied triad |
| `mean_detect_geo` | Arithmetic mean of raw `Detect_geo` |
| `num_first_places` | Number of W values at which the triad is the first record in the sorted W-specific ranking; absent counts become 0 |
| `final_score_100` | MinMax scaling of `mean_score` over all triads to the range 0–100 |

The full summary is sorted by `mean_score`, then `num_first_places`, then `max_score`, all descending. Its first 10 rows form the original summary's top-10 table. The original selection and sort operations are preserved, including the handling of tied scores by those operations.

### 6. Display scores on a 0–100 scale

Two separate `MinMaxScaler(feature_range=(0, 100))` instances are fitted:

- `final_score_100`: fitted to the 84 triad-level `mean_score` values.
- `row_score_100`: fitted to the 420 row-level `composite_score` values.

Both fields are for presenting the calculated scores. Neither is a Random Forest target or predictor. Their numerical scales have separate minima and maxima and should not be treated as the same scale across row and triad tables.

## Random Forest and SHAP inputs

| Item | A: triad-level, no W predictor | B: W=18 rows |
| --- | --- | --- |
| Observations in this file | 84 triads | 84 rows |
| Predictors `X` | Triad means of the six `_pos` columns | Six `_pos` columns for W=18 |
| Target `y` | `triad_summary.mean_score` | `composite_score` |
| Feature standardization for the model | None beyond the direction transformations and triad averaging | None beyond the direction transformations |
| Model | `RandomForestRegressor(n_estimators=800, random_state=42)` | Same settings |
| Waterfall selection | Original summary sorted by `mean_score` descending, first 10 triads | Original W=18 subset sorted by `composite_score` descending, first 10 rows |

A uses the original pandas triad grouping and index alignment. B keeps the original workbook row order after filtering. These details matter when reproducing the same seeded Random Forest fit.

The predictors are **unstandardized `_pos` features**, not their `_z` versions. The z-scores are used only to construct the target. The two models are fitted separately; their FI and SHAP values describe different fitted models.

The original `TreeExplainer` is created from each fitted model without a separate background dataset. Each analysis produces impurity-based Random Forest feature importance, global SHAP beeswarm and mean-absolute-SHAP bar plots, and 10 local waterfalls. Waterfall contributions sum to the model prediction relative to the SHAP expected value, subject to numerical tolerance; they explain the prediction, which can differ from the calculated target score.

These are explanations of a model fitted to a score constructed from the same six features. They do not independently validate experimental outcomes or establish causal feature effects. See [VERIFICATION.md](VERIFICATION.md) for the actual comparison against the restored original calculations.

## Observed numeric ranges

These ranges come directly from `triadW_features!A2:K421`. The displayed precision is for readability; source data and calculations are not rounded.

| Column | Minimum | Maximum |
| --- | ---: | ---: |
| `Number` | 1 | 420 |
| `W` | 10 | 26 |
| `S2` | -0.00381963193442458 | 0.358304630394464 |
| `S3` | 0.0285409892538195 | 0.614812963520349 |
| `S4` | 0.0124341638352398 | 0.271681646137974 |
| `L_over_6` | 0.495490699910368 | 0.878296485349182 |
| `R_max` | 1.01696375456172 | 1.91069888792373 |
| `C_max` | 1.02761259328202 | 1.95139374518673 |
| `Riskx_geom` | 1.02337170108347 | 1.9309391131855 |
| `Detect_geo` | 5.93112499034626e9 | 8.75510108769445e10 |

### Negative S2 values

Seven rows have `S2 = -0.00381963193442458`: `Hf-I-Ta`, `Ce-Hf-Ta`, `Gd-Hf-Ta`, `Hf-Ta-Tb`, `Hf-Ta-Yb`, `Au-Hf-Ta`, and `Bi-Hf-Ta`, all at W=10. The original transformation is the identity, so these values remain negative in `_S2_pos`.

### Observed relationship among risk descriptors

Across the 420 rows, `Riskx_geom` approximately equals `sqrt(R_max * C_max)`, with a maximum absolute difference of approximately `8.44e-15`. This is an observation from the values, not a confirmed upstream feature-generation formula. The workbook note mentions a product of row and column sums without explicitly stating the square root. The scoring code reads `Riskx_geom` directly and does not recompute it from the other two columns.

### Source row order

Rows are ordered by decreasing `Detect_geo`, and `Number` follows that order. This is not the composite-score ranking. Neither `Number` nor row position is used as a score or predictor.
