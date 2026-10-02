# EDA findings and data decisions

Owner: Senior Data Scientist. Every number here is printed by `python src/eda.py` (run from the project root; it takes about 15 seconds and gives the same output on every run). Figures are in `reports/figures/`.

This is the decision document for the Data Engineer (cleaning, data contract, December inputs) and the ML Engineer (features, validation, model choice). Where it says "must", implement it exactly. Where it says "test", the choice is open and the CV spec in section 5 decides it.

## 1. Key findings

1. `quote_signal` leaks the label in training and is pure noise in validation. Its meaning changes by month:

   | Months (2025) | What `quote_signal` is | Evidence |
   |---|---|---|
   | Jan, Feb, Mar, Jun, Sep | a copy of rate per mile | qs / rpm median 0.9999, sd of log ratio 0.0104; corr with actual rpm 0.996 to 0.997 |
   | Apr, May, Jul, Oct | a mirror of it: qs = 4.149 - 0.9995 x rpm | residual sd 0.030; corr -0.993 to -0.994 |
   | Aug | noise, unrelated to anything | corr with rpm -0.020, with market_index +0.020 |
   | Nov, Dec (validation) | same as August | mean 2.051 vs 2.053, sd 0.220 vs 0.223, KS p = 0.88; corr with typical rpm -0.003 and -0.017 |

   A quick model shows the trap (random 80/20 split inside Jan-Aug, so time drift does not muddy it). Without `quote_signal`: 2.08% MAPE. With it: 1.79% on the test rows as given, but 2.61% once `quote_signal` looks like the validation file. With `quote_signal` and month: 1.58% as given, 2.78% with validation-like values. Figure `02_quote_signal_regimes.png`.

2. 677 rates (1.41%) are corrupted by a random factor. A clean gap separates them: no row has a log ratio to its expected rate between 0.25 and 0.70. The rule flags the same 677 rows at every threshold from 0.3 to 0.7, whether the expected rate comes from band medians, from Jan-Aug only, or from a log-linear model. In the 9 months where `quote_signal` holds the clean rate, it recovers the factor exactly: high rows are x2.21 to x5.10, low rows x0.178 to x0.427, spread evenly with no spikes at 2, 3, 4 or 5. These are random errors, not unit or scale slips. Figure `01_rate_outliers.png`.

3. Price structure. Rate per mile falls smoothly with distance (elasticity of rate to distance 0.87, the same within a lane, 0.88). Reefer costs about 13% and Flatbed about 8% more than Dry Van at every distance band (ratios 1.12 to 1.14 and 1.08 to 1.10). Figure `06_rate_per_mile_by_distance.png`.

4. Weight matters and the negatives are sign errors. Rate rises about 8% from the lightest to the heaviest loads in an S shape: flat below 15,000 lb, steep from 20,000 to 40,000 lb, flat above 45,000 lb. The slope is the same for all three equipment types (+0.030 log units per 10,000 lb). Negative weights have the same distribution as positive ones (KS p = 0.70 in train, 0.33 in validation), and their absolute value predicts the rate just as well (corr 0.585 vs 0.595). Figure `03_weight.png`.

5. `market_index` is a daily market factor with a strong weekly cycle. The within-day sd is 0.025 against 0.167 across days. The index peaks on Thursday and bottoms on Sunday and Monday. A 10% higher index means about 1.3% higher rates (elasticity 0.139), and the index carries the whole weekday effect in rates (what is left by weekday is within +/-0.3%). Figures `04_market_index_and_rate_level.png`, `05_weekday_pattern.png`.

6. Rates drift upward over the year beyond what the index explains. On daily levels, index alone gives R2 0.47; index plus a linear day trend gives R2 0.84 (trend +0.00021 log units per day, about +0.6% a month). Sep and Oct sit 3.2% and 2.4% above the index-only fit. A model fitted on Jan-Aug with the index under-predicts Sep-Oct by 4.3% (median signed error). This month-level bias is the largest error source we found (section 5.6).

7. City effects are small and smooth in space. After a smooth distance curve, equipment, day and weight, each end of a load moves the rate by -3.6% (Syracuse) to +3.0% (Oklahoma City). The effect follows latitude (corr -0.84): the south and Texas cost more, the north-east and Great Lakes less. A city has the same effect as pickup and as delivery (corr 0.991), and direction does not matter (A to B minus B to A: mean -0.0005, sd 0.016). From lat/lon, the 3 nearest cities predict a left-out city's effect with RMSE 0.0055, against 0.0180 for the overall mean. So lat/lon handle the 8 unseen cities well. Lanes add a little on top: lane means spread about 1.1% beyond the two city effects. What is left after all of this has an sd of about 2% (0.0195 in log units). Figure `07_city_effects_map.png`.

8. Validation looks like Sep-Oct. Validation `market_index` averages 0.919 (Nov) and 0.935 (Dec), against 0.926 for train Sep-Oct and 1.083 for all of train, and the weekday shape matches Sep-Oct. 12.1% of validation rows (1,447) touch one of 8 cities absent from training. No row has both ends unseen.

9. The data is synthetic but clean. There are no duplicates, no bad dates, no spelling variants, ids are sequential, and every city has exactly one coordinate pair. Coordinates are shifted from the real cities and look clipped (Boston and Providence both at lon -69.5, Laredo at lat 25.5). Listed distance is 1.18x the straight-line distance between the given coordinates (corr 0.9995) with a 70-mile floor. Figure `08_distance_vs_haversine.png`.

## 2. Data quality issues

| # | Issue | Count (train / val) | Evidence | Decision |
|---|---|---|---|---|
| Q1 | Corrupted `posted_rate` | 677 (340 high, 337 low) / n.a. | Empty gap in log ratio from 0.25 to 0.70; factors x2.2 to x5.1 and x0.18 to x0.43; even across months (57 to 81 per month) and equipment (1.35% to 1.51%) | Drop from training (rule C5). Correction is possible for 609 rows through `quote_signal`, but it would tie cleaning to a leaking column and recover only 1.3% of rows. |
| Q2 | Negative `weight` | 292 / 145 | Same distribution as positive weights (KS p 0.70 / 0.33); same link to rate | Take the absolute value (rule C3). |
| Q3 | Zero `weight` | 0 / 0 | None exist. The "292 rows <= 0" are all negative. | Defensive rule only: treat 0 as missing. |
| Q4 | Missing `weight` | 300 (0.62%) / 165 (1.38%) | Random: chi-square p by month 0.43 / 0.84, equipment 0.77 / 0.44, pickup city 0.93 / 0.14; mean rate residual +0.0011 | Fill with equipment median of training rows (rule C3). No missing flag. |
| Q5 | Missing `market_index` | 374 (0.78%) / 249 (2.08%) | Random: p by month 0.83 / 0.47, equipment 0.58 / 0.31, city 0.25 / 0.25; mean residual -0.0037 | Fill with same-date mean, train and val pooled (rule C4). Leave-one-out error of that fill: RMSE 0.025, against 0.083 for a month mean and 0.166 for a global mean. No missing flag. |
| Q6 | `quote_signal` leaks the label | all rows in 9 of 10 months | Section 1, finding 1 | Never a model feature (rule C6). |
| Q7 | Weight capped | 1,191 at +47,500 and 13 at -47,500 (2.5%) / 297 and 4 | Hard cap; the rate curve is flat above 45,000 lb anyway | Keep as is. Range after rule C3 is 5,000 to 47,500. |
| Q8 | Distance floor | 48 rows at 70 mi / 21 | Short lanes (New Orleans to Shreveport, New York to Allentown) clipped to 70; no other ratio above 1.6 when haversine > 100 mi | Keep as is. Rate follows the listed distance. |
| Q9 | Validation has more missing values | weight 1.38% vs 0.62%, index 2.08% vs 0.78% | Still random | Covered by C3 and C4. |
| Q10 | Negative-weight rows have more rate outliers | 9 of 292 (3.1%) vs 1.4% overall, binomial p = 0.024 | Weak link, few rows | No action. Rule C5 already removes them. |
| Q11 | Duplicates, spellings, impossible values | 0 / 0 | No duplicate ids or rows, no whitespace or case variants, no non-positive distance, index, `quote_signal` or rate | No action; QA asserts it. |

## 3. Cleaning rules

Apply the same function to train and validation unless a rule says "train only". Rules are in execution order.

1. C1, types. Parse `date` with format `%Y-%m-%d`. Assert no failures, train dates 2025-01-01..2025-10-31, validation dates 2025-11-01..2025-12-31.
2. C2, categories. Strip whitespace on `pickup`, `delivery`, `equipment` (a no-op today). Assert `equipment` is in {Dry Van, Reefer, Flatbed}.
3. C3, weight. `weight = abs(weight)`; set `weight == 0` to missing; fill missing with the median of `abs(weight)` for the same equipment, computed on the training rows of the current fit only. Full-train values are Dry Van 31,444, Flatbed 31,532.5, Reefer 31,577. Assert 5,000 <= weight <= 47,500 after.
4. C4, market_index. Fill missing with the mean of non-missing `market_index` on the same `date`, pooling `train_test.csv` and `validation.csv`. Pooling is fine because the index is an input, not the target, and each date sits in only one file. Every date has at least 114 observed values. Fallback, never triggered today: the mean of the nearest 3 dates on each side. Assert no missing values after.
5. C5, rate outliers, train only.
   - `rpm = posted_rate / distance`.
   - `band = pd.cut(distance, [0, 100, 150, 200, 250, 300, 400, 500, 600, 750, 900, 1100, 1300, 1550, 1800, 2100, 2500, 2900, 4000])` (right-closed).
   - `expected_rpm` = median `rpm` per (equipment, band) over the labeled rows being cleaned. 54 cells; the smallest holds 48 rows.
   - `is_outlier = abs(log(rpm / expected_rpm)) > 0.5`, which means outside x0.61 to x1.65.
   - Expected: 677 flagged in all of train (533 in Jan-Aug, 144 in Sep-Oct). Medians from Jan-Aug only give the identical set.
   - Drop flagged rows from every training set. Keep the `is_outlier` column on holdout rows for scoring only. It is never a feature.
6. C6, quote_signal. Exclude from model features. Keep the raw column in the data so QA can test the regimes.
7. C7, distance and coordinates. No changes. Build one city table (city, lat, lon) from pickup and delivery columns of both files; assert exactly one pair per city (72 cities).
8. C8, keys. Assert `load_id` unique. Validation output keeps all 12,000 rows in file order; never drop a validation row.

## 4. Feature recommendations

Target: `log(posted_rate)`. Predict with `exp(prediction)`. Train with squared error on the log target.

Use:

| Feature | Why |
|---|---|
| `distance`, `log(distance)` | main driver (corr 0.91 with rate); rate per mile falls with distance |
| `equipment` (categorical) | constant premiums: Reefer about 13%, Flatbed about 8% over Dry Van |
| `weight` after C3 | about 8% effect, S-shaped; trees handle the shape |
| `market_index` after C4, row level | elasticity 0.139; row-level deviations carry the same elasticity, so use the row value, not the daily mean |
| `pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon` | city effects are smooth in space and cover unseen cities |
| `haversine` miles and `distance / haversine` (optional) | cheap geometry; test in CV |
| `day_of_week` (optional) | after the index only +/-0.3% is left; test in CV |

Do not use:

| Feature | Why not |
|---|---|
| `quote_signal` | leaks the label in train, noise in validation (finding 1) |
| `month` or any month-level category | Nov and Dec never appear in training, and month lets the model learn the `quote_signal` regimes |
| a raw day number as a tree feature | trees cannot extrapolate it; Nov-Dec would all get the last October value. Handle the trend as in 5.6 instead. |
| city names as one-hot or raw city ids | 8 validation cities have no training rows |
| `load_id` | an identifier |

Optional, test in CV: an out-of-fold, smoothed lane mean of the log residual (shrink toward 0 with a prior of about 10 loads). Lane means spread 0.0118 after city effects against 0.0050 from sampling noise alone, so there is a real lane effect of about 1.1%. 64.7% of validation rows sit on a training lane with at least 10 loads. Unseen-city lanes must get 0.

## 5. Split and validation spec

### 5.1 Main time holdout

- Fit: 2025-01-01..2025-08-31, rows with `is_outlier == False` (37,944 of 38,477).
- Test: 2025-09-01..2025-10-31, 9,523 rows (144 flagged).
- Why: it copies the real task (predict the next two months), and Sep-Oct market conditions match Nov-Dec (index 0.926 against 0.919 and 0.935, same weekday shape).
- Use it once, for the final comparison of 2 or 3 finished candidates. Never tune on it.

### 5.2 Time-series CV for tuning (expanding window, inside Jan-Aug only)

| Fold | Fit months | Test month | Clean fit rows | Test rows (flagged) | Baseline B1, clean MAE / MAPE |
|---|---|---|---|---|---|
| 1 | Jan-Apr | May | 18,835 | 4,913 (62) | $124.10 / 5.13% |
| 2 | Jan-May | Jun | 23,686 | 4,783 (63) | $153.17 / 6.12% |
| 3 | Jan-Jun | Jul | 28,406 | 4,912 (65) | $89.46 / 3.84% |
| 4 | Jan-Jul | Aug | 33,253 | 4,759 (68) | $75.41 / 3.45% |

Pick hyperparameters and the trend handling by mean clean MAPE over folds 1 to 4. Report each fold too: the spread across folds is the honest error bar.

### 5.3 Unseen-city holdout

- Hold out the 8 lowest-volume training cities: Birmingham, Dallas, Detroit, Las Vegas, Louisville, New Orleans, St. Louis, Washington. They are spread across regions, and their counts (569 to 758 appearances) are the closest training has to the unseen cities (170 to 195 in validation).
- Fit on Jan-Aug clean rows with neither end in this set (4,194 Jan-Aug rows removed).
- Test on the 1,003 Sep-Oct rows with at least one end in the set (18 have both). That is 10.5% of Sep-Oct, close to the 12.1% of validation that touches unseen cities.
- Report the unseen-city penalty: MAPE of this model minus MAPE of the main holdout model on the same 1,003 rows.
- Laredo is a harder case than any city in this test. It sits 3.4 degrees from its nearest training city and south of every training city.

### 5.4 Metrics

Report MAE ($), MAPE (%) and RMSE ($) for each test set, twice: on clean rows (`is_outlier == False`) and on all rows. Model selection uses clean MAPE, with clean MAE as the tie-break.

Why clean rows decide: the 1.4% corrupted rows are off by x2 to x6 with a random factor that no feature predicts. On all rows they swamp the comparison. B1 RMSE goes from $106 (clean) to $638 (all) on Sep-Oct, and MAE from $77 to $133, whatever the model does. Spotter's labels probably carry the same corruption, so the all-rows numbers tell us roughly what score to expect. Do not shift predictions to chase RMSE on those rows.

### 5.5 Baselines to beat (fit on clean Jan-Aug, scored on Sep-Oct)

| Baseline | Clean MAE | Clean MAPE | Clean RMSE | All-rows MAE | All-rows MAPE | All-rows RMSE |
|---|---|---|---|---|---|---|
| B0: global median $/mile x distance | $203.69 | 9.27% | $295.92 | $256.95 | 11.63% | $684.25 |
| B1: median $/mile by equipment x distance band, x distance | $77.34 | 3.47% | $106.22 | $133.05 | 5.99% | $638.01 |

B1 is the baseline to beat, and it is harder than it looks. It ignores the index, so it does not suffer the level drift. A plain XGBoost (400 trees, depth 6, the features above, no trend handling) scored 4.46% clean MAPE on Sep-Oct, worse than B1. B1 on the unseen-city test rows: clean MAE $69.26, MAPE 3.52%.

### 5.6 Trend handling (the main modelling choice)

Quick XGBoost checks, clean rows, MAPE with the median signed error in brackets:

| Fit months to test | Plain | Recency weights (half-life 30 days) | Linear detrend |
|---|---|---|---|
| 1-4 to May | 1.84 (-0.77) | 1.88 (-0.78) | 1.73 (+0.26) |
| 1-5 to Jun | 3.28 (-2.80) | 3.21 (-2.74) | 2.35 (-1.42) |
| 1-6 to Jul | 1.88 (-0.72) | 1.97 (+0.01) | 3.33 (+3.04) |
| 1-7 to Aug | 2.66 (-2.36) | 2.09 (-1.10) | 2.07 (+1.52) |
| 1-8 to Sep-Oct | 4.46 (-4.29) | 2.78 (-2.26) | 2.11 (-1.19) |
| Mean of 5 | 2.82 | 2.39 | 2.32 |

Recency weight = `0.5 ** ((last training day - day) / 30)`. Linear detrend: fit OLS of log rate on log distance, its square, equipment dummies, log index, weight/10,000 and day number; take the day slope b; train the trees on `log(rate) - b x day`; add `b x day` back at prediction. The slope varied from 0.00017 to 0.00045 per day across folds.

Both beat plain. Neither wins every fold, and when the level is right the model reaches about 1.7% to 1.9%. Decide between them, or a blend, with folds 1 to 4 only, then confirm on the Sep-Oct holdout. The last row above already uses Sep-Oct and is here as evidence only.

### 5.7 Final fit

Refit the chosen setup on all clean Jan-Oct rows (47,323) and predict validation. The cleaning medians (C3, C5) come from all of train at this point.

## 6. December chart inputs recipe

Fixed lane: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb, one row per day 2025-12-01..2025-12-31. The file has no coordinates, `market_index` or `quote_signal`.

| Model input | Source | Value |
|---|---|---|
| `pickup_lat`, `pickup_lon` | city table (C7) | 36.99152, -84.99876 |
| `delivery_lat`, `delivery_lon` | city table (C7) | 41.31561, -85.36206 |
| `market_index` | mean of non-missing `market_index` in `validation.csv` on the same date (159 to 221 values per day; all 31 dates present) | 0.8306 to 1.0445, mean 0.9344; the per-day values are printed by `src/eda.py` section 9 |
| `quote_signal` | not a model input. If shared code needs the column, fill it with the same-date validation mean (2.009 to 2.089); it must not change predictions. | n.a. |
| `weight` | file value through C3 (unchanged) | 32,000 |
| `distance`, `equipment` | file values | 360, Dry Van |

Write `predicted_rate` back into the file and keep the original 7 columns in their order (`pickup, delivery, distance, equipment, weight, date, predicted_rate`). `score.py` rejects anything else.

Evidence and sanity checks:

- Train has 32 loads on this lane (21 Dry Van, 8 Reefer, 3 Flatbed); none of the Dry Van ones is flagged. Dry Van rates run $757.93 to $934.37, median $807.89 ($2.233 a mile), distances 350.9 to 378.8. The 4 Sep-Oct Dry Van loads, scaled to 360 miles, average $812.27. Figure `09_december_lane.png`.
- The December index follows the weekly cycle (Thursday high, Sunday and Monday low). Through the 0.139 elasticity that alone means a swing of about 3.2% peak to trough, about $26 on an $808 load. A good chart shows that weekly saw-tooth around roughly $800 to $850, with no jump at month start or on Dec 25. Training holidays showed no special effect (mean unexplained level -0.006 against a daily sd of 0.023).
- All 31 values must be positive and should sit inside or near the training range for this lane. A flat line means the model ignores the index; jumps of more than about 5% between days need a look.

## 7. Risks

1. Level drift. The month-level bias moved between -4.3% and +3.0% across folds in our checks. Nobody can verify the Nov-Dec level from this data. Trend handling (5.6) cuts the risk; it does not remove it.
2. `quote_signal` misuse. Adding it improves every holdout that contains copy or mirror months and hurts the real submission. QA should assert that it is not in the feature list and that the per-month correlations in section 1 still hold.
3. Noisy labels at Spotter. If their labels have about 1.4% corrupted rows like ours, their RMSE will be dominated by rows no model can predict. MAE and MAPE are less affected.
4. Unseen cities. 12.1% of validation rows touch a new city. Seven of the eight have a training city within 1.4 degrees. Laredo does not: it lies 3.4 degrees from Corpus Christi and below the training latitude range (28.36 to 44.30), so its predictions rely on extrapolation.
5. A strong simple baseline. B1 reaches 3.47% clean MAPE on Sep-Oct. A model without trend handling lost to it (4.46%). Always report the model next to B1.
6. More missing values in validation (weight 1.38%, index 2.08%). The fills are accurate (index fill RMSE 0.025), so the effect is small, but the pipeline must never drop or leave missing a validation row.
7. Synthetic data. Coordinates are shifted and clipped, and `quote_signal` follows calendar-month regimes. We read the generator's rules from the data; Spotter's real scoring set may hold surprises that the training data does not show.
