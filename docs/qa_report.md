# QA report

Owner: Senior QA Data Engineer. Date: 2 October 2026.

Verdict: the solution is correct, free of leakage, reproducible and ready to submit. Every number below comes from code run during this review. One small defect was found and fixed (it changed no output), and three judgment calls are reviewed in section 6.

## 1. What was tested

The suite lives in `tests/` and runs with `python -m pytest -q` from the project root. It takes about 25 seconds on our laptop, slow tests included. Tests that need the data skip with a clear reason when `data/` is empty, and the synthetic tests always run. The 4 slow tests (marked `slow`, registered in `pytest.ini`) share one fit of the full final model, which takes about 13 seconds.

| File | Tests | What it checks |
|---|---|---|
| `test_data_contract.py` | 35 | The loaders accept the real files. They reject 15 kinds of broken train file (reordered, missing or extra columns, unknown equipment, bad or out-of-range date, repeated id, text in a number column, missing or infinite values, zero price, negative distance, latitude out of range, missing city), a missing file, a validation file with labels or October dates, and a bad template or December file. The writers reject short, NaN, infinite, zero or negative predictions and repeated or foreign ids, and they write rows in template order. |
| `test_cleaning.py` | 12 | Negative weights flipped, zeros and gaps filled with the equipment median, missing index filled from the same date, corrupted labels flagged and dropped only on request. Counts match the docs: 292 / 300 / 374 / 677 (340 high, 337 low) on train, 145 / 165 / 249 on validation, 47,323 rows kept, and the Jan-Aug numbers 233 / 235 / 301 / 533 and 59 / 65 / 73 / 144. Validation keeps all 12,000 rows in file order. A Cleaner fit on Jan-Aug learns the same values when every Sep-Oct row is wrecked, and CV fold 1 is unchanged when May-Oct rows are wrecked. |
| `test_splits.py` | 4 | CV folds are expanding, strictly time-ordered and disjoint, never reach Sep-Oct, and have the documented sizes. The holdout is exactly Jan-Aug to Sep-Oct (37,944 / 9,523 / 144). The unseen-city split has no training row touching the 8 cities, and every test row touches one (33,819 / 1,003). |
| `test_leakage.py` | 23 | No feature set contains a banned or drift-only column, and `assert_allowed` raises on each one. The feature matrix does not change when quote_signal, load_id, posted_rate or is_outlier change. A model fit with quote_signal replaced by noise gives identical drift, lane table and predictions. Changing quote_signal, labels, is_outlier or load_id of the rows being predicted changes nothing. The out-of-fold lane correction never uses a row's own label. The lane table is the shrunk mean of the out-of-fold residuals, and unseen lanes get 0. In a CV fold the drift equals an OLS fit on that fold's fit rows (last day 30 April), and wrecking the test-month labels changes nothing. The same checks are repeated on the full final model (slow). |
| `test_outputs.py` | 11 | `validation_predictions.csv` has the header `load_id,predicted_rate`, 12,000 rows, the template ids in template order, and finite positive prices in cents. The December file keeps the 7 original columns and 31 dates, and its first six columns are byte-identical to the original file. Both files pass the validators in `score.py`, and those validators do reject broken copies. The `outputs/` copies are byte-identical, and the input files in `data/` equal the originals at the root. |
| `test_sanity.py` | 4 | December sits inside the lane's history ($757.93 to $934.37, 21 loads), moves at most 1% a day, is not flat, and follows the index once the monthly climb is taken out (correlation above 0.8). In all 54 cells of equipment and distance band, the median predicted rate per mile is within 10% of the training median (actual range 0.995 to 1.042). The 1,447 rows touching a new city are priced like other rows in their cell. |
| `test_reproducibility.py` | 4 | The saved setup equals `model.FINAL_SPEC`, with seed 42 and 4 threads. Refitting the final model reproduces both prediction files to the cent, and the drift and summary numbers in `metrics.json` (slow). |

`tests/qa_audit.py` is the independent audit in section 3. It is a script, so pytest does not collect it: `python tests/qa_audit.py`, about one minute.

## 2. Results

| Check | Result |
|---|---|
| `python -m pytest -q` | 93 passed, 0 failed, 0 skipped (23 seconds) |
| Independent audit, 21 headline numbers | all equal to six decimal places |
| `python run_pipeline.py` from scratch | pass, exit 0 (3 min 33 s; a later run took 7 min 18 s, with every stage about twice as slow and no single stall, which points to machine load or power state) |
| `python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv` | pass: "Validated 12,000 final predictions. Validated 31 fixed December predictions." |
| Outputs byte-identical to the files before QA | pass (see section 4) |
| `python src/report.py` builds the report | pass (built to a scratch folder so the real report files were not touched) |

## 3. Independent audit

The B1 baseline was rebuilt from the raw CSV with plain pandas. The audit uses its own outlier flags, its own distance bands typed in from the findings doc, and its own metric code, with nothing imported from `src/`. The chosen model was refit on Jan-Aug with the project's functions, and every metric was computed by hand from its predictions.

| Number | Recomputed | Reported |
|---|---|---|
| B1 holdout, clean MAE / MAPE / RMSE | $77.34 / 3.47% / $106.22 | $77.34 / 3.47% / $106.22 |
| B1 holdout, all rows MAE / MAPE / RMSE | $133.05 / 5.99% / $638.01 | $133.05 / 5.99% / $638.01 |
| B1 CV mean clean MAPE | 4.634% | 4.634% |
| B1 unseen-city clean MAPE / MAE | 3.52% / $69.26 | 3.52% / $69.26 |
| Chosen holdout, clean MAPE / MAE / RMSE / bias | 1.64% / $38.97 / $56.50 / -0.76% | 1.64% / $38.97 / $56.50 / -0.76% |
| Chosen holdout, all rows MAPE / MAE / RMSE | 4.16% / $94.98 / $629.91 | 4.16% / $94.98 / $629.91 |
| Chosen, Sep / Oct clean MAPE | 1.54% / 1.74% | 1.54% / 1.74% |
| Same fit, damping 0 / 1 at prediction | 1.78% / 1.53% | 1.78% / 1.53% |
| Chosen without the ramp | 2.14% | 2.14% |

The largest difference over all 21 numbers is 0 at six decimal places. The audit's own flags also give the documented counts: 533 flagged in Jan-Aug, 144 in Sep-Oct, 33,819 fit rows and 1,003 test rows in the unseen-city split.

## 4. Run from scratch and hashes

The output files were hashed (SHA-256) before any run, and the pipeline was then run twice from scratch.

| File | SHA-256 (first 16) | After both runs |
|---|---|---|
| `validation_predictions.csv` (and `outputs/` copy) | `68935f7f9cf81534` | identical |
| `data/december_chart_inputs.csv` (and `outputs/december_predictions.csv`) | `23c65b4972a14528` | identical |
| `scorer_results/candidate_december.png` | `876a9f0cd7a55813` | identical |
| `reports/model_results.md` | `d018e03e0c2dc51e` | identical |
| `reports/figures/01` to `15` | (15 files) | identical |
| `reports/metrics.json` | `316f798e0cf36365` before, `2d61bf9c05e4a252` now | identical except `evaluation_seconds` |

`metrics.json` stores the wall-clock time of the evaluation, so it changes on every run. Putting the old value (257.3) back into the new file gives the old hash exactly, so nothing else changed. The second run used the code with the fix from section 8, and its outputs are identical too.

## 5. The 30-minute stall

Cause: the laptop went into Windows Modern Standby, which pauses desktop programs. The code is not at fault.

- The Windows event log shows standby from 08:57:47 to 09:31:21 this morning (33.5 minutes), while the ML Engineer's runs were going (the code was saved at 08:55 and the outputs written at 09:37).
- It happened again during this review. Our first full run started at 18:37:45. The system had entered standby at 18:37:02 (screen off), went into deep standby at 18:41:56, and woke at 19:07:23. The run log stopped at 18:41, in stage 1, and the candidate just before it took 52 seconds against the usual 5.
- With the machine awake, neither of the two full runs nor any test-suite run stalled. Stage 6 took 42 and 83 seconds.
- Thread oversubscription is unlikely. The only BLAS pool loaded is MKL (8 threads), and the code never runs NumPy least squares and XGBoost at the same time.

Advice for reviewers: keep the machine plugged in and awake during `python run_pipeline.py`. We kept it awake for our runs with a small script that asks Windows not to sleep while it runs. No settings were changed.

## 6. Verdicts on the risky decisions

### a. Quarter-end ramp in the drift layer: accept, and keep it on the risk list

- The effect is real and specific to quarter ends. Our own regression (daily level after the log index and a straight-line trend) gives a climb from day 1 to the last day of +3.85% in March, +3.94% in June and +4.52% in September. The other seven months range from -0.91% to +1.26%. The last five days sit at +3.09%, +3.53% and +3.43%, in line with the Team Lead's 3%. Every other month ends below its trend (-1.46% to -0.18%).
- December gets exactly the same treatment. The ramp input is 0 on 1 December and 1 on 31 December, as on day 1 and the last day of March, June and September. It is 0 for every November row and back to 0 on 1 January. In the final model, the December 31 offset minus the December 1 offset is 0.043093, which is exactly the ramp (0.040123) plus half the daily slope times 30 days.
- No leakage. The ramp is one OLS coefficient fitted on the training rows of each fit, and it only uses the date. Tests prove that test-month labels and quote_signal cannot move it. CV fold 2 predicts June from a ramp learned on March alone.
- Size of the bet. Without the ramp, the December chart averages $842.22 instead of $859.37, and December validation predictions fall by 1.97% on average. If December does not ramp, our December predictions run about 2% high. If it ramps like the other three quarters and we had dropped it, they would run about 2% low.

### b. Trend damping 0.5: accept

On CV, damping 0, 0.5 and 1 score 1.489%, 1.488% and 1.509%, a tie. On the holdout, with the same fit, they score 1.78%, 1.64% and 1.53%, so the trend did keep going through October. The choice moves the submission very little: against 0.5, a full trend raises November by 0.15% and December by 0.46% on average (+0.61% on 31 December), and no trend lowers them by the same amounts. 0.5 followed the selection protocol (CV only). Switching to 1 now would use the holdout for a choice, and it would change December by under half a percent.

### c. Thread count fixed at 4: accept, it is required

We refit the final model with 1, 2, 4, 8 and again 4 XGBoost threads. Both 4-thread fits reproduce the files to the cent. With 1, 2 or 8 threads, nearly every validation prediction changes: the mean change is $4.73 to $4.97, the largest $57.30 (1.32% on one row), and December moves by up to $4.38. These are equally valid models, but only 4 threads gives the submitted files. `model.N_JOBS = 4` must stay fixed, and `metrics.json` records it. We could not test whether a machine with a different CPU gives the same bytes with 4 threads.

## 7. Review of `src/report.py`

It builds without errors (DOCX only, to a scratch folder). Every table value is read from `reports/metrics.json` or recomputed through `src/data.py`, so the tables cannot drift from the pipeline. We checked the hard-coded statements against the data:

- Figure 2 caption ("without drift handling XGBoost runs 2% to 8% low; the chosen model stays close to zero"): true. On the holdout the daily median error of the no-drift model runs from -7.57% to -2.35%, and the chosen model stays within -1.99% to +0.49% every day.
- "No other month shows a rise": true for the last five days (every other month ends below trend). April does climb 1.26% within the month, but it starts from -1.26% after the March drop and still ends below trend.
- quote_signal regimes, the 0.91 correlation, the 13% and 8% premiums, the 0.6% monthly drift, the 0.025 and 0.17 spread of the index, the 47,323 kept rows and the $757.93 to $934.37 lane range all match the findings and our checks.
- Two wording notes (low): the data-quality table says "distance errors: 0", while the findings doc notes 48 train and 21 validation rows clipped at a 70-mile floor (kept as is). The sentence "every cleaning number is learned on the training rows of each split" is true for the medians, but the market_index fill uses the same-date mean pooled over all files. That is an input, not a label, and the findings doc explains why it is safe.

## 8. Defects

| # | Severity | Status | Defect |
|---|---|---|---|
| D1 | Low | Fixed | The lane-correction memo in `RateModel._fit_lane_correction` was keyed on the load ids, the sum of the target and the settings, but not on the feature values. A second fit in the same process with the same ids and labels but different features (for example a cleaning change) silently reused stale residuals. Shown by `test_oof_memo_is_keyed_on_the_features_too` (failed before, passes after). Fix: the key now holds a SHA-256 of the exact bytes of X, the target and the weights. It never triggered in the pipeline: both prediction files and all figures are byte-identical after the fix. |
| D2 | Info | Open (machine) | Runs pause when Windows enters Modern Standby (section 5). Not a code issue. Worth one line in the README. |
| D3 | Info | Open | `metrics.json` holds `evaluation_seconds`, so the file changes on every full run even when every result is identical. Harmless; just do not expect a byte-identical `metrics.json`. |
| D4 | Low | Open (report text) | Two wording notes in `src/report.py`, listed in section 7. |

No defect touches the predictions. Nothing was found that needs a model change.

## 9. Files changed by QA

- Added: `pytest.ini`, `tests/conftest.py`, `tests/test_*.py` (7 files), `tests/qa_audit.py`, `docs/qa_report.md`, `reports/qa_summary.json`.
- Changed: `src/model.py` (fix D1), `docs/WORKLOG.md` (step 4).
- Rewritten by the pipeline with identical content: the prediction files, figures, `model_results.md` and `scorer_results/candidate_december.png`. In `metrics.json` only `evaluation_seconds` changed.
