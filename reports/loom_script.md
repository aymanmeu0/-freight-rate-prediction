# Loom script (about 2.5 minutes)

Read at a normal pace. Each part lists what to show on screen. The five topics the assessment asks for are covered in order.

## 0:00 to 0:15 | Intro
Show: the GitHub repo README.

"Hi, I'm Ayman. This is my freight rate prediction project. The goal is to predict the price of 12,000 truck loads in November and December 2025, using 48,000 loads from January to October. On a two-month forecast test, the model is off by 1.6% on average, against 3.5% for a simple baseline."

## 0:15 to 0:50 | Key findings from exploring the data
Show: `reports/figures/02_quote_signal_regimes.png`, then `15_quarter_end_ramp.png`.

"Distance drives most of the price, and Reefer and Flatbed cost more per mile than Dry Van. The most important finding was a trap. The column quote_signal copies the price in some months and mirrors it in others, but in November and December it's pure noise. A model that uses it looks great in testing and fails on the real task, so I never use it, and the model never sees the month either. The second finding: prices drift up slowly over the year, and in March, June and September they climb about 4% through the month and reset on the 1st. December is also a quarter-end month."

## 0:50 to 1:15 | Data quality issues and fixes
Show: section 3 of `reports/report.pdf`.

"I found 677 prices, 1.4%, that are wrong by a factor of 2 to 6. Nothing normal comes near them, so I removed them from training and report scores with and without them. 292 weights were negative with the same shape as the positive ones, so the sign was flipped and I use the absolute value. Missing weights get the typical weight for the equipment, and missing market index values get that day's average, because the index is one number per day."

## 1:15 to 1:45 | Training and validation, and how the data was split
Show: section 4 of the report.

"Because the task is a forecast, I never split at random. I chose the model on four monthly folds, training on January to April and testing on May, and so on up to August. Then I scored the final choice once on September and October, which it had never seen. I also hid eight cities from training to check new cities, since eight cities in the November-December file never appear in training. The cost was only 0.15 points."

## 1:45 to 2:10 | Why this model
Show: the CV table in `reports/model_results.md`.

"The model is XGBoost on the log of the price per mile. A plain XGBoost guessed September and October about 4% too low because of the drift. So under the trees I fit a small market layer, a trend plus the quarter-end ramp, on training rows only. That took the tuning error from 2.3% to 1.6%. A small per-lane correction brought it to 1.42%."

## 2:10 to 2:35 | Code walkthrough
Show: `run_pipeline.py`, then `src/data.py` (the Cleaner class), then `src/features.py` (assert_allowed), then the December chart.

"One command, python run_pipeline.py, runs everything. src/data.py checks every file and learns the cleaning on training rows only. src/features.py builds the features and stops the run if a banned column like quote_signal ever reaches the model. src/train.py runs the folds, and src/predict.py fits the final model and writes both files. The tests in the tests folder check for leakage and confirm the outputs rebuild exactly. Here's the December chart from score.py: a weekly pattern from the market index, on a climb into quarter end. Thanks for watching."
