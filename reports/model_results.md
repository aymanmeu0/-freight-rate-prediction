# Model results

Every number here is written by `python run_pipeline.py` (see `docs/modeling.md` for the reasoning).
MAPE and MAE on clean rows decide; bias is the median signed error (below 0 = too low).

## CV folds (selection only)

Expanding window inside Jan-Aug. Clean-row MAPE per test month, then the mean over the four folds.
Candidates are listed in the order the stages scored them.

| Stage | Setup | May | Jun | Jul | Aug | Mean MAPE | Mean MAE |
|---|---|---|---|---|---|---|---|
| 1 drift handling | B1 baseline | 5.13% | 6.12% | 3.84% | 3.45% | 4.634% | $110.54 |
| 1 drift handling | ridge, log rate, geo, no drift | 1.72% | 3.19% | 1.83% | 2.54% | 2.320% | $55.88 |
| 1 drift handling | ridge, log rate, geo, trend+ramp damp 0.5 | 1.78% | 1.83% | 1.81% | 1.69% | 1.779% | $42.54 |
| 1 drift handling | xgb, log rate, geo, no drift, d6 n600 lr0.05 mcw1 | 1.65% | 3.19% | 1.76% | 2.60% | 2.301% | $54.53 |
| 1 drift handling | xgb, log rate, geo, no drift, recency hl30, d6 n600 lr0.05 mcw1 | 1.71% | 3.13% | 1.80% | 1.99% | 2.158% | $51.06 |
| 1 drift handling | xgb, log rate, geo, trend damp 1, d6 n600 lr0.05 mcw1 | 1.55% | 2.25% | 3.19% | 1.90% | 2.222% | $51.75 |
| 1 drift handling | xgb, log rate, geo, trend damp 0.5, d6 n600 lr0.05 mcw1 | 1.53% | 2.40% | 2.88% | 1.80% | 2.152% | $50.15 |
| 1 drift handling | xgb, log rate, geo, trend damp 0, d6 n600 lr0.05 mcw1 | 1.55% | 2.56% | 2.59% | 1.72% | 2.106% | $49.11 |
| 1 drift handling | xgb, log rate, geo, trend+ramp damp 1, d6 n600 lr0.05 mcw1 | 1.54% | 1.75% | 1.70% | 1.51% | 1.626% | $37.41 |
| 1 drift handling | xgb, log rate, geo, trend+ramp damp 0.5, d6 n600 lr0.05 mcw1 | 1.57% | 1.77% | 1.62% | 1.48% | 1.610% | $37.12 |
| 1 drift handling | xgb, log rate, geo, trend+ramp damp 0, d6 n600 lr0.05 mcw1 | 1.62% | 1.81% | 1.57% | 1.46% | 1.612% | $37.28 |
| 1 drift handling | xgb, log rate, geo, trend+ramp damp 1, recency hl60, d6 n600 lr0.05 mcw1 | 1.56% | 1.75% | 1.72% | 1.52% | 1.639% | $37.75 |
| 2 target | xgb, log rpm, geo, trend+ramp damp 0.5, d6 n600 lr0.05 mcw1 | 1.44% | 1.63% | 1.53% | 1.38% | 1.495% | $35.32 |
| 3 features | xgb, log rpm, base, trend+ramp damp 0.5, d6 n600 lr0.05 mcw1 | 1.47% | 1.65% | 1.55% | 1.40% | 1.517% | $35.97 |
| 3 features | xgb, log rpm, geo_dow, trend+ramp damp 0.5, d6 n600 lr0.05 mcw1 | 1.41% | 1.64% | 1.53% | 1.37% | 1.488% | $35.18 |
| 4 damping re-check | xgb, log rpm, geo_dow, trend+ramp damp 1, d6 n600 lr0.05 mcw1 | 1.38% | 1.63% | 1.61% | 1.41% | 1.509% | $35.60 |
| 4 damping re-check | xgb, log rpm, geo_dow, trend+ramp damp 0, d6 n600 lr0.05 mcw1 | 1.46% | 1.68% | 1.47% | 1.35% | 1.489% | $35.27 |
| 5 XGBoost grid | xgb, log rpm, geo_dow, trend+ramp damp 0.5, d4 n2000 lr0.03 mcw1 | 1.37% | 1.61% | 1.52% | 1.35% | 1.465% | $34.79 |
| 5 XGBoost grid | xgb, log rpm, geo_dow, trend+ramp damp 0.5, d5 n2000 lr0.03 mcw1 | 1.37% | 1.61% | 1.50% | 1.34% | 1.454% | $34.48 |
| 5 XGBoost grid | xgb, log rpm, geo_dow, trend+ramp damp 0.5, d6 n2000 lr0.03 mcw1 | 1.39% | 1.62% | 1.49% | 1.35% | 1.461% | $34.63 |
| 5 XGBoost grid | xgb, log rpm, geo_dow, trend+ramp damp 0.5, d5 n1000 lr0.05 mcw1 | 1.38% | 1.62% | 1.51% | 1.35% | 1.466% | $34.82 |
| 5 XGBoost grid | xgb, log rpm, geo_dow, trend+ramp damp 0.5, d5 n2000 lr0.03 mcw20 | 1.37% | 1.62% | 1.50% | 1.34% | 1.459% | $34.60 |
| 6 lane correction | xgb, log rpm, geo_dow, trend+ramp damp 0.5, d5 n2000 lr0.03 mcw1, lane directed p10 **(chosen)** | 1.34% | 1.59% | 1.46% | 1.30% | 1.423% | $33.76 |
| 6 lane correction | xgb, log rpm, geo_dow, trend+ramp damp 0.5, d5 n2000 lr0.03 mcw1, lane undirected p10 | 1.35% | 1.60% | 1.48% | 1.33% | 1.440% | $34.18 |

Stage winners:

| Stage | Winner | Mean clean MAPE |
|---|---|---|
| 1 drift handling | xgb, log rate, geo, trend+ramp damp 0.5, d6 n600 lr0.05 mcw1 | 1.610% |
| 2 target | xgb, log rpm, geo, trend+ramp damp 0.5, d6 n600 lr0.05 mcw1 | 1.495% |
| 3 features | xgb, log rpm, geo_dow, trend+ramp damp 0.5, d6 n600 lr0.05 mcw1 | 1.488% |
| 4 damping re-check | xgb, log rpm, geo_dow, trend+ramp damp 0.5, d6 n600 lr0.05 mcw1 | 1.488% |
| 5 XGBoost grid | xgb, log rpm, geo_dow, trend+ramp damp 0.5, d5 n2000 lr0.03 mcw1 | 1.454% |
| 6 lane correction | xgb, log rpm, geo_dow, trend+ramp damp 0.5, d5 n2000 lr0.03 mcw1, lane directed p10 | 1.423% |

## Main holdout (fit Jan-Aug, test Sep-Oct, scored once after selection)

Fit rows 37,944; test rows 9,523 (144 with corrupted labels).

| Setup | Clean MAE | Clean MAPE | Clean RMSE | Clean bias | All MAE | All MAPE | All RMSE |
|---|---|---|---|---|---|---|---|
| chosen | $38.97 | 1.64% | $56.50 | -0.76% | $94.98 | 4.16% | $629.91 |
| chosen fit, damp 0 at prediction | $42.40 | 1.78% | $61.09 | -1.04% | $98.36 | 4.29% | $630.79 |
| chosen fit, damp 0.5 at prediction | $38.97 | 1.64% | $56.50 | -0.76% | $94.98 | 4.16% | $629.91 |
| chosen fit, damp 1 at prediction | $36.26 | 1.53% | $52.86 | -0.48% | $92.31 | 4.06% | $629.13 |
| B1 baseline | $77.34 | 3.47% | $106.22 | -0.77% | $133.05 | 5.99% | $638.01 |
| xgb, log rate, geo, no drift, d6 n600 lr0.05 mcw1 | $105.80 | 4.42% | $138.05 | -4.27% | $160.90 | 6.81% | $648.40 |
| ridge, log rate, geo, trend+ramp damp 0.5 | $45.06 | 1.95% | $65.86 | -0.87% | $101.04 | 4.47% | $631.71 |
| chosen without the ramp (trend only) | $51.86 | 2.14% | $77.59 | -1.69% | $107.70 | 4.63% | $633.70 |

By month:

| Setup | Clean MAE | Clean MAPE | Clean RMSE | Clean bias | All MAE | All MAPE | All RMSE |
|---|---|---|---|---|---|---|---|
| chosen, Sep | $36.41 | 1.54% | $54.44 | +0.08% | $87.67 | 3.60% | $609.93 |
| chosen, Oct | $41.44 | 1.74% | $58.42 | -1.42% | $102.01 | 4.69% | $648.55 |
| B1 baseline, Sep | $77.97 | 3.46% | $107.96 | -0.59% | $129.02 | 5.50% | $621.57 |
| B1 baseline, Oct | $76.73 | 3.48% | $104.51 | -0.93% | $136.92 | 6.46% | $653.44 |

Drift learned on Jan-Aug: slope 0.000177 log units per day (0.53% per 30 days), quarter-end ramp 0.0407 (4.16% by the last day of the month).

## Unseen-city check

Cities held out: Birmingham, Dallas, Detroit, Las Vegas, Louisville, New Orleans, St. Louis, Washington. Fit rows 33,819; test rows 1,003 Sep-Oct rows touching them (9 with corrupted labels).

| Setup | Clean MAE | Clean MAPE | Clean RMSE | Clean bias | All MAE | All MAPE | All RMSE |
|---|---|---|---|---|---|---|---|
| chosen, cities unseen | $38.84 | 1.84% | $57.67 | -0.61% | $74.08 | 3.13% | $499.17 |
| chosen main holdout model (cities seen) | $35.95 | 1.69% | $52.94 | -0.70% | $71.20 | 2.98% | $498.22 |
| B1, cities unseen | $69.26 | 3.52% | $97.11 | -0.99% | $104.30 | 4.80% | $505.87 |

Penalty for an unseen city: +0.15 points of clean MAPE, +2.90 $ of clean MAE.

## Final model (fit on all clean Jan-Oct rows)

Fit rows 47,323. Drift: slope 0.000198 per day, ramp 0.0401.

| Output | Min | Median | Mean | Max |
|---|---|---|---|---|
| Validation, all 12,000 rows | $189.42 | $2,070.64 | $2,389.37 | $7,072.66 |
| Validation, November | $189.42 | $2,055.18 | $2,364.50 | $6,840.16 |
| Validation, December | $190.16 | $2,082.96 | $2,412.92 | $7,072.66 |
| December chart, 31 days | $832.14 | $858.66 | $859.37 | $883.85 |

December chart (Lexington to Fort Wayne, 360 mi, Dry Van, 32,000 lb):

| Date | Weekday | market_index | Predicted rate |
|---|---|---|---|
| 2025-12-01 | Mon | 0.8349 | $832.14 |
| 2025-12-02 | Tue | 0.9147 | $840.49 |
| 2025-12-03 | Wed | 0.9941 | $846.31 |
| 2025-12-04 | Thu | 1.0240 | $850.99 |
| 2025-12-05 | Fri | 0.9751 | $849.21 |
| 2025-12-06 | Sat | 0.8892 | $845.34 |
| 2025-12-07 | Sun | 0.8306 | $840.23 |
| 2025-12-08 | Mon | 0.8401 | $841.48 |
| 2025-12-09 | Tue | 0.9173 | $848.95 |
| 2025-12-10 | Wed | 0.9989 | $855.57 |
| 2025-12-11 | Thu | 1.0272 | $859.55 |
| 2025-12-12 | Fri | 0.9814 | $857.85 |
| 2025-12-13 | Sat | 0.8966 | $854.56 |
| 2025-12-14 | Sun | 0.8382 | $850.05 |
| 2025-12-15 | Mon | 0.8478 | $850.95 |
| 2025-12-16 | Tue | 0.9242 | $858.25 |
| 2025-12-17 | Wed | 1.0021 | $864.55 |
| 2025-12-18 | Thu | 1.0340 | $869.02 |
| 2025-12-19 | Fri | 0.9887 | $867.92 |
| 2025-12-20 | Sat | 0.9018 | $864.36 |
| 2025-12-21 | Sun | 0.8441 | $858.66 |
| 2025-12-22 | Mon | 0.8599 | $860.34 |
| 2025-12-23 | Tue | 0.9304 | $868.06 |
| 2025-12-24 | Wed | 1.0147 | $875.06 |
| 2025-12-25 | Thu | 1.0445 | $880.26 |
| 2025-12-26 | Fri | 0.9972 | $877.00 |
| 2025-12-27 | Sat | 0.9145 | $873.35 |
| 2025-12-28 | Sun | 0.8580 | $869.68 |
| 2025-12-29 | Mon | 0.8696 | $870.38 |
| 2025-12-30 | Tue | 0.9440 | $875.91 |
| 2025-12-31 | Wed | 1.0273 | $883.85 |
