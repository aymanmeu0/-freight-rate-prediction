# Freight Rate Prediction

Predicts `posted_rate`, the price in dollars to move one truck load, for 12,000 loads dated November and December 2025. The model learns from 48,000 labeled loads dated January to October 2025.

| Test (clean rows) | This model | Simple baseline |
|---|---|---|
| Sep-Oct holdout, MAPE | 1.64% | 3.47% |
| Sep-Oct holdout, MAE | $38.97 | $77.34 |
| Tuning folds May-Aug, mean MAPE | 1.42% | 4.63% |
| Cities never seen in training, MAPE | 1.84% | 3.52% |

The baseline is the median rate per mile for the equipment type and distance band, times the distance. Full numbers are in [`reports/model_results.md`](reports/model_results.md) and the report.

## Deliverables

| File | What it is |
|---|---|
| [`validation_predictions.csv`](validation_predictions.csv) | `load_id,predicted_rate` for all 12,000 validation loads |
| [`outputs/december_predictions.csv`](outputs/december_predictions.csv) | the filled December chart inputs (copy of `data/december_chart_inputs.csv`) |
| [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png) | December chart produced by `score.py` |
| [`reports/report.pdf`](reports/report.pdf) / [`.docx`](reports/report.docx) | report: data findings, cleaning, split and validation, model, results, December chart |
| [`reports/loom_script.md`](reports/loom_script.md) | script for the 2-3 minute walkthrough video |

## Setup

Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt
```

The data is not in this repo. Put the four files from the assessment pack in `data/` with these names:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Run

```bash
python run_pipeline.py              # everything: checks, CV selection, holdout, final fit, both files, score.py (about 5 minutes)
python run_pipeline.py --skip-eval  # final fit, both files and score.py only (under 1 minute)
```

Then, as in the assessment README:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Other commands:

```bash
python -m pytest -q          # tests: data contract, cleaning, leakage, split, outputs, reproducibility
python src/data.py           # data pipeline self-check and cleaning counts
python src/eda.py            # exploration numbers and figures 01-09
python src/report.py         # rebuild reports/report.docx (and the PDF when Microsoft Word is installed)
```

Runs are deterministic: seed 42 and a fixed XGBoost thread count give byte-identical prediction files.

On a laptop, keep the machine awake during the full run. Sleep or standby pauses the run, though it does not change the results.

## Approach in brief

- Split by time, never at random, because the task is a two-month forecast. The model is chosen on four expanding monthly folds inside Jan-Aug (train Jan-Apr, test May; and so on up to Aug), then scored once on Sep-Oct. A second check hides eight cities from training to measure how new cities are priced.
- `quote_signal` is never used. It copies the rate in some training months, mirrors it in others, and is noise in Nov-Dec. The month is never a feature either.
- Cleaning, learned on the training rows of each split only: negative weights are sign flips (absolute value); missing weight is filled with the equipment median; missing `market_index` is filled with the same-date mean; 677 corrupted rates (2x to 6x off) are removed from training.
- Model: XGBoost on log(rate per mile) with distance, map-position, lane-direction, equipment, weight, `market_index` and day-of-week features. Underneath, a small market layer fitted on training rows captures a slow upward drift and a price rise through each quarter-end month. A shrunk per-lane correction is added on top.

## Project layout

```
run_pipeline.py        one command for the whole pipeline
score.py               scorer from the assessment pack (unchanged)
src/config.py          paths, constants, split definition
src/data.py            loading, data contract checks, fold-safe cleaning, output writers
src/eda.py             data exploration and figures
src/features.py        feature builder and the banned-column guard
src/model.py           model definition (market layer, XGBoost, lane correction)
src/train.py           CV selection, holdout and unseen-city evaluation
src/predict.py         final fit and both prediction files
src/report.py          builds the report from reports/metrics.json
tests/                 pytest suite
docs/WORKLOG.md        every step explained in plain English
docs/eda_findings.md   data findings and cleaning decisions
docs/modeling.md       model choice and reasoning
docs/qa_report.md      quality checks and results
```
