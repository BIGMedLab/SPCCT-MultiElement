# Numerical equivalence verification

## Result

**PASS.** All 34 comparisons of numerical values, model structures, and ordering between the independently reconstructed original analysis and the cleaned workflow passed. Maximum absolute difference: **0.0**.

An additional 11 checks of exported Excel and CSV tables also passed.

- Verification timestamp: `2026-10-19T18:03:40.418778+00:00`
- Dataset: the complete 420-row workbook, identical byte-for-byte to the supplied source file
- No synthetic or sampled replacement dataset was used

**Important limitation:** This verifies the supplied original calculations against the cleaned code in the same environment. It does not compare with historical output files or a prior Python installation, which were not provided.

## Verification method

1. Extract the original preprocessing, scoring, Random Forest, and SHAP computations from the supplied Markdown code.
2. Restore escaped Markdown/HTML syntax and nonstandard whitespace into executable Python.
3. Preserve the original numerical computations in the independent `verification/reference_original.py`, with changes limited to parameterizing paths, organizing functions, returning comparison values, and omitting plotting/export operations.
4. Execute the reference implementation and `analyze.py` separately, without importing cleaned numerical functions into the reference.
5. Compare using exact `numpy.array_equal`, `pandas.testing.assert_frame_equal(check_exact=True)`, and Random Forest tree-state comparisons, without numerical tolerances.

## Scope of the 34 checks

| Area | Verification |
| --- | --- |
| Row-level scoring | Original row/column order, six direction transformations, six z-scores, composite score, and 0–100 row score |
| Rankings by W | Values and ordering for all 420 rows |
| Triad summaries | All summary columns for 84 triads, normalized scores, overall ranking, and top ten |
| Training data | Exact feature matrices, targets, indices, and row order for both 84 × 6 datasets |
| Random Forest | Hyperparameters and the complete structures and node values of all 800 trees per model |
| Predictions and feature importance | 84 predictions and six importance values per model, including ordering |
| SHAP | 84 × 6 SHAP values per model, expected values, and mean absolute SHAP |
| Waterfall examples | Identifiers, ranking, feature values, and SHAP contributions for the top ten in each analysis |

Together the models contain **1,600 trees** and **1,008 SHAP contributions**. See `results/verification_report.json` for the itemized checks, environment, and file hashes.

## Export verification

The generated files were read back and compared with independently produced reference tables.

| Output check | Tables | Result |
| --- | ---: | --- |
| Original Excel worksheets | 5 | Exact match |
| Additional W-based ranking worksheet | 1 | Exact match |
| CSV exports | 5 | Exact match |
| **Total** | **11** | **11/11 passed; maximum difference 0.0** |

Excel reference sheets were written in memory using the original export conventions and then read back. CSV files were compared using `float_precision="round_trip"`. Data values, types, and row/column ordering were checked.

## Re-run verification

```bash
python verification/verify.py
```

Install dependencies from `requirements.txt` first. Consult the JSON verification report for additional implementation-specific details.
