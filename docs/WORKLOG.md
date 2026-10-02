# Work log

This file explains every step of the project in plain English: what we did, why, and what came out of it. Steps are added in the order they happen.

## Step 0: project setup

What we did:
- Made the folders: `data/` (input files), `src/` (code), `tests/` (checks), `reports/` (report and figures), `docs/` (notes like this one) and `outputs/` (prediction files).
- Copied the four data files into `data/` and renamed them to match the names the assessment README expects (`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`). We checked each copy byte for byte against the original, and they are identical.
- Wrote `requirements.txt` with the libraries the scorer needs (pandas, numpy, matplotlib) plus the ones our solution needs (scikit-learn, xgboost, python-docx, pytest).
- Wrote `.gitignore` so the data never goes to GitHub. The repo is public and the data belongs to Spotter. Only our code, our predictions and our report are uploaded.

Why: a fixed layout lets anyone clone the repo, drop the data into `data/` and run one command. Keeping the data out of git protects the company's files.
