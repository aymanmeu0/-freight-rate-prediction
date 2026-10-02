# Model: choice, evidence and limits

Owner: Senior ML Engineer. Every number here is printed by `python run_pipeline.py` and saved in `reports/metrics.json` and `reports/model_results.md`, unless the text says it comes from a side check. The full run takes about 6 minutes on our laptop. The code is in `src/features.py`, `src/model.py`, `src/train.py` and `src/predict.py`, and it uses the Data Engineer's cleaning in `src/data.py` unchanged.

## 1. What we chose

An XGBoost model of log(rate per mile), with three extra pieces:

1. A level-drift term, learned by least squares on the training rows only. It has two parts: a straight-line trend in time, and a quarter-end ramp that climbs through March, June, September and December and drops back on the 1st of the next month. The trees learn the target with this level taken out, and the level is added back when we predict. Past the last training day the trend continues at half its slope (damping 0.5).
2. A lane correction: the average out-of-fold error of each pickup-to-delivery lane, shrunk toward zero with a prior of 10 loads, then added to the prediction. A lane never seen in training gets no correction.
3. XGBoost settings: depth 5, 2,000 trees, learning rate 0.03, subsample 0.8, column sample 0.8, seed 42, 4 threads.

Features: distance, log distance, pickup and delivery lat/lon, the change in lat and lon, the bearing (as sine and cosine), the straight-line distance, distance divided by straight-line distance, equipment (three 0/1 columns), weight, market_index and day of week.

How it compares with the B1 baseline (median $/mile by equipment and distance band):

| Test | Rows | Chosen model | B1 baseline |
|---|---|---|---|
| CV, mean of 4 folds (selection) | clean | MAPE 1.423%, MAE $33.76 | MAPE 4.634%, MAE $110.54 |
| Holdout Sep-Oct | clean (9,379) | MAPE 1.64%, MAE $38.97, RMSE $56.50 | MAPE 3.47%, MAE $77.34, RMSE $106.22 |
| Holdout Sep-Oct | all (9,523) | MAPE 4.16%, MAE $94.98, RMSE $629.91 | MAPE 5.99%, MAE $133.05, RMSE $638.01 |
| Unseen cities (1,003 rows) | clean | MAPE 1.84%, MAE $38.84 | MAPE 3.52%, MAE $69.26 |

On clean rows the chosen model halves the baseline's error. On all rows the 144 corrupted labels dominate RMSE for every model ($629.91 against $56.50 on clean rows), as the findings doc warned.

## 2. The quarter-end ramp

The findings doc showed that rates drift upward beyond what market_index explains, and that neither recency weights nor a straight-line detrend won every fold. Before choosing between them we looked at the daily rate level more closely. We fit log rate on distance, equipment, weight and location (no index, no time), took the residual, averaged it per day, and regressed that daily level on the log of the daily index and the day number:

| Daily level explained by | R2 | Residual sd |
|---|---|---|
| index only | 0.482 | 0.0222 |
| index + day number | 0.843 | 0.0122 |
| index + day number + quarter-end ramp | 0.976 | 0.0048 |

The ramp is 0 outside the last month of a quarter. Inside it, it rises in a straight line from 0 on day 1 to 1 on the last day. Its coefficient is 0.0402, so rates end each quarter about 4% above where they started that month. After the index and the trend, the first five days of March, June and September sit at -0.07%, +0.12% and -0.70%, and the last five days at +3.11%, +3.56% and +3.40%. The first five days of the next month fall back to -1.19%, -0.67% and -0.68%. Figure `15_quarter_end_ramp.png` shows all three ramps.

The ramp is stable however much history the fit sees. Fit on months 1-4 (March is the only ramp), its coefficient is 0.0375; on months 1-5, 1-6, 1-7, 1-8 and 1-10 it is 0.0350, 0.0363, 0.0399, 0.0407 and 0.0402. It also explains why the plain detrend was unstable. Without the ramp, a fit that ends in June reads the June climb as trend: the daily day slope jumps to 0.000445, and the straight-line detrend then over-predicts July by 2.25% to 2.99% (median signed error, CV fold 3). With the ramp, the daily day slope stays between 0.000177 and 0.000274 in every window.

December is the last month of a quarter. The model therefore expects December rates to climb about 4% from December 1 to December 31.

This is a calendar effect, so here is how it squares with the "no month feature" rule:

- The trees never see the date, the month, the day number or the ramp. `features.assert_allowed` raises if any column in `config.NEVER_FEATURES` (load_id, quote_signal, posted_rate, is_outlier, date, month) or a drift-only column reaches a learner.
- The ramp lives only in the drift term, as one coefficient learned on the training rows of each fit. It does not use `quote_signal`, so it cannot learn the leaking regimes.
- A month category cannot say anything about November or December, because training has neither. The ramp can: December is a quarter-end month, like March, June and September.

We found the ramp by plotting daily levels over all of Jan-Oct, so September was in view when we designed it. The holdout result for the ramp is therefore a little optimistic. The CV evidence does not depend on September: fold 2 predicts June from a ramp learned on March alone, and its MAPE fell from 3.19% with no drift handling to between 1.75% and 1.81% with the ramp.

## 3. Target and features

Target: log(rate / distance). On the CV folds it beat log(rate) by 0.115 points of MAPE (1.495% against 1.610%), and it was better on all four folds. The trees no longer have to rebuild the strong distance scaling from splits.

Feature sets compared on CV (log rate-per-mile target, trend and ramp, damping 0.5, default XGBoost):

| Set | Mean clean MAPE |
|---|---|
| base: distance, log distance, lat/lon, equipment, weight, market_index | 1.517% |
| geo: base + change in lat/lon, bearing, straight-line distance, circuity | 1.495% |
| geo_dow: geo + day of week | 1.488% |

Day of week adds very little (0.007 points). It is near zero in the gain importance (figure `13_feature_importance.png`), which fits the findings doc: market_index already carries the weekly cycle. We kept it because CV picked it.

No feature uses city names, so the 8 cities that appear only in validation get predictions from their coordinates.

## 4. Drift handling

All options were compared on the four CV folds with the same default XGBoost (depth 6, 600 trees, learning rate 0.05) on log rate and the geo features:

| Drift handling | May | Jun | Jul | Aug | Mean |
|---|---|---|---|---|---|
| none | 1.65% | 3.19% | 1.76% | 2.60% | 2.301% |
| recency weights, half-life 30 days | 1.71% | 3.13% | 1.80% | 1.99% | 2.158% |
| straight-line trend, damping 1 | 1.55% | 2.25% | 3.19% | 1.90% | 2.222% |
| straight-line trend, damping 0.5 | 1.53% | 2.40% | 2.88% | 1.80% | 2.152% |
| straight-line trend, damping 0 | 1.55% | 2.56% | 2.59% | 1.72% | 2.106% |
| trend + ramp, damping 1 | 1.54% | 1.75% | 1.70% | 1.51% | 1.626% |
| trend + ramp, damping 0.5 | 1.57% | 1.77% | 1.62% | 1.48% | 1.610% |
| trend + ramp, damping 0 | 1.62% | 1.81% | 1.57% | 1.46% | 1.612% |
| trend + ramp, damping 1, recency half-life 60 | 1.56% | 1.75% | 1.72% | 1.52% | 1.639% |

Trend plus ramp has by far the best mean and wins June, July and August. In May, a month without a ramp, the straight-line trend is slightly better (1.53% to 1.55% against 1.54% to 1.62%). Recency weights help a model without drift handling but add nothing once the ramp is in.

How the drift term works: least squares of log rate on a distance curve, equipment, a weight curve, location terms, log index, the day number and the ramp, fit on the training rows of each fold only. That gives a slope b per day and a ramp size c. The trees learn log(rate per mile) minus b x day minus c x ramp. At prediction time we add back c x ramp and the trend. Up to the last training day the trend is b x day. After it, the trend grows at damping x b per day.

Extrapolating a trend is a bet, so damping got its own check. The evidence is mixed. On CV, damping 0, 0.5 and 1 score 1.489%, 1.488% and 1.509% (with the final target and features). Damping 1 wins May and June, damping 0 wins July and August. We chose 0.5: it has the best mean, it sits between the two, and a one-month CV horizon cannot separate the options. The holdout, scored after the choice and never used for it, leans toward a full trend: with the same fitted model, damping 0, 0.5 and 1 give 1.78%, 1.64% and 1.53% clean MAPE. So the upward trend kept going through October. In the final model the choice moves predictions by at most 0.5 x 0.000198 x 61 days, about 0.6%, on December 31.

## 5. Lane correction

After the drift term and the trees, each lane still has a small level of its own (the findings doc measured about 1.1%). We estimate it without leaking: a 5-fold split of the training rows, each fold predicted by a lighter XGBoost (the default settings) fit on the other four. The lane correction is the sum of those out-of-fold errors divided by (number of loads + 10). Directed lanes (A to B separate from B to A) scored 1.423% on CV and undirected lanes 1.440%, against 1.454% with no correction. The directed correction helped on all four folds. Validation rows on a lane never seen in training, including every row that touches one of the 8 new cities, get no correction.

## 6. Tuning

A small grid on the CV folds, around the winner of the earlier stages:

| XGBoost settings | Mean clean MAPE | Mean clean MAE |
|---|---|---|
| depth 6, 600 trees, learning rate 0.05 (default) | 1.488% | $35.18 |
| depth 4, 2,000 trees, 0.03 | 1.465% | $34.79 |
| depth 5, 2,000 trees, 0.03 | 1.454% | $34.48 |
| depth 6, 2,000 trees, 0.03 | 1.461% | $34.63 |
| depth 5, 1,000 trees, 0.05 | 1.466% | $34.82 |
| depth 5, 2,000 trees, 0.03, min child weight 20 | 1.459% | $34.60 |

The differences are small. In a side check (not part of the pipeline), three other seeds moved the mean CV MAPE of one setup by about 0.002 points, so differences of 0.01 points or more are larger than seed noise. Selection runs in six stages (drift, target, features, damping re-check, grid, lane). Each stage varies one thing around the previous winner and keeps the lowest mean clean MAPE, with clean MAE as the tie-break. `src/train.py` runs this automatically; the full table is in `reports/model_results.md`.

A ridge regression on the same inputs, with the same trend and ramp, reached 1.779% on CV and 1.95% on the holdout, so the trees earn their place.

## 7. Holdout and unseen-city results

Main holdout: fit on 37,944 clean Jan-Aug rows (Cleaner fit on Jan-Aug only), test on 9,523 Sep-Oct rows. Scored once, after selection.

| Setup | Clean MAE | Clean MAPE | Clean RMSE | Clean bias | All MAE | All MAPE | All RMSE |
|---|---|---|---|---|---|---|---|
| Chosen model | $38.97 | 1.64% | $56.50 | -0.76% | $94.98 | 4.16% | $629.91 |
| B1 baseline | $77.34 | 3.47% | $106.22 | -0.77% | $133.05 | 5.99% | $638.01 |
| XGBoost, no drift handling | $105.80 | 4.42% | $138.05 | -4.27% | $160.90 | 6.81% | $648.40 |
| Ridge, trend + ramp | $45.06 | 1.95% | $65.86 | -0.87% | $101.04 | 4.47% | $631.71 |
| Chosen model without the ramp | $51.86 | 2.14% | $77.59 | -1.69% | $107.70 | 4.63% | $633.70 |

Bias is the median signed error; below 0 means predictions are too low. By month, the chosen model scores 1.54% in September (bias +0.08%) and 1.74% in October (bias -1.42%). Figure `12_holdout_residual_by_date.png` shows the error by day. September sits close to zero, so the ramp tracked the September climb. October runs about 1.4% low. Damping explains part of that: with damping 1 the holdout bias moves from -0.76% to -0.48% (section 4).

Unseen cities: we removed every Jan-Aug row touching Birmingham, Dallas, Detroit, Las Vegas, Louisville, New Orleans, St. Louis or Washington (33,819 fit rows left) and tested on the 1,003 Sep-Oct rows that touch them.

| Setup, same 1,003 rows | Clean MAE | Clean MAPE | Clean bias | All MAE | All MAPE |
|---|---|---|---|---|---|
| Chosen model, cities never seen | $38.84 | 1.84% | -0.61% | $74.08 | 3.13% |
| Chosen model from the main holdout (cities seen) | $35.95 | 1.69% | -0.70% | $71.20 | 2.98% |
| B1, cities never seen | $69.26 | 3.52% | -0.99% | $104.30 | 4.80% |

An unseen city costs 0.15 points of clean MAPE ($2.90 of MAE). These rows also lose the lane correction, which the main model can apply to them.

## 8. Final model and outputs

The final model is the chosen setup, refit on all 47,323 clean Jan-Oct rows with the Cleaner fit on all of train. The drift term it learned has slope 0.000198 per day and ramp 0.0401. It writes `validation_predictions.csv` (also copied to `outputs/`), `data/december_chart_inputs.csv` and `outputs/december_predictions.csv` through the pipeline writers. `score.py` validates both files and draws `scorer_results/candidate_december.png`.

Checks on the 12,000 validation predictions: all finite and positive, from $189.42 to $7,072.66 (median $2,070.64), next to $184.55 to $7,498.05 (median $2,032.37) for the clean training rates. The median predicted rate per mile for Dry Van is 2.045 in November and 2.097 in December, against 2.051 for actual Sep-Oct loads. Reefer is 2.298 and 2.369 against 2.318; Flatbed is 2.211 and 2.272 against 2.239. Against the Sep-Oct median for the same equipment and distance band, the median prediction is 0.995 times in November and 1.020 times in December. November sits slightly below Sep-Oct because September carried a ramp and November does not. December is about 2% higher because of its ramp. The 1,447 rows touching an unseen city sit at 1.011 times that median, against 1.007 for the other rows.

December chart (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb): $832.14 to $883.85, mean $859.37, median $858.66. The largest change from one day to the next is 1.00%. The line has two parts:

- A weekly saw-tooth from market_index. Each Thursday is a local high and each Sunday or Monday a local low (for example $832.14 on Monday December 1, $850.99 on Thursday December 4, $840.23 on Sunday December 7).
- A climb of about 4% from the quarter-end ramp, from a first-week mean of $843.53 to a last-week mean of $875.78.

There is no jump on December 25. The model has no holiday effect, and the findings doc found none in training.

All 31 values sit inside the lane's training range ($757.93 to $934.37). They are above the findings doc's rough guess of $800 to $850, for reasons we can trace. Priced on each Sep-Oct day with that day's index, the same December row (360 miles, 32,000 lb) averages $842.27 in September and $837.79 in October, so the model already put this load near $840 before December. The lane's 21 past Dry Van loads come out at a median of 1.0027 times the model's in-sample prediction, so the model is not off on this lane. The past loads were lighter (the four Sep-Oct ones weighed 17,917 to 26,829 lb) and most came earlier in the year. The drift term stands at +6.18% over January 1 on October 31, +6.51% on December 1 and +11.20% on December 31 (trend plus ramp). The lane correction adds +0.31%. Figure `14_december_vs_history.png` puts the December line next to the lane's past loads.

## 9. Limits and risks

1. The December ramp is a bet based on three quarter-ends. If December does not ramp (year-end could behave differently), December predictions will be about 2% too high on average and about 4% too high by month end. If it ramps as in March, June and September, leaving the ramp out would cost the same in the other direction.
2. Trend damping. The holdout favoured the full trend (1.53% against 1.64% at damping 0.5), and October ran 1.42% low. If the trend continues at full speed, our Nov-Dec predictions run up to about 0.6% low by December 31, on top of any level surprise.
3. Optimism. The ramp was designed after looking at all of Jan-Oct, September included, so the holdout result for it is somewhat optimistic. CV (Jan-Aug only) supports it on its own.
4. Corrupted labels. If Spotter's labels carry about 1.4% corrupted rates like ours, their RMSE will be near $630 whatever the model does. On our holdout, all-rows MAPE is 4.16% against 1.64% on clean rows.
5. Unseen cities. The measured penalty is 0.15 points of MAPE, but Laredo sits south of every training city, so its rows rely on extrapolation beyond what this test covers.
6. CV horizon. Each CV fold predicts one month ahead; the real task predicts two. The holdout covers two months, and October (the second month) was the harder one.
7. Reproducibility. Results are identical across runs with the fixed seed and 4 threads; two full runs gave byte-identical prediction files. With a different thread count XGBoost gives slightly different predictions (in a side check, up to about $36 on single training rows), so `model.N_JOBS` must stay fixed.
8. Small gains. Day of week (0.007 points) and the grid choices (about 0.01 points) are small. The big gains come from the drift term (2.30% to 1.61%), the rate-per-mile target (to 1.50%) and the lane correction (1.454% to 1.423%).
