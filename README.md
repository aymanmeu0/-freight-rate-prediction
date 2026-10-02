# Freight Rate Prediction

Predicts `posted_rate`, the price in dollars to move one truck load, for 12,000 loads in Nov-Dec 2025. The model learns from 48,000 labeled loads from Jan-Oct 2025.

Work in progress. Full run instructions will be added when the pipeline is finished.

## Data

The data is not in this repo. Put the four files from the assessment pack in `data/` with these names:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Project notes

Every step is explained in plain English in [`docs/WORKLOG.md`](docs/WORKLOG.md).
