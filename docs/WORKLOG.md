# Work log

This file explains every step of the project in plain English: what we did, why, and what came out of it. Steps are added in the order they happen.

## Step 0: project setup

What we did:
- Made the folders: `data/` (input files), `src/` (code), `tests/` (checks), `reports/` (report and figures), `docs/` (notes like this one) and `outputs/` (prediction files).
- Copied the four data files into `data/` and renamed them to match the names the assessment README expects (`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`). We checked each copy byte for byte against the original, and they are identical.
- Wrote `requirements.txt` with the libraries the scorer needs (pandas, numpy, matplotlib) plus the ones our solution needs (scikit-learn, xgboost, python-docx, pytest).
- Wrote `.gitignore` so the data never goes to GitHub. The repo is public and the data belongs to Spotter. Only our code, our predictions and our report are uploaded.

Why: a fixed layout lets anyone clone the repo, drop the data into `data/` and run one command. Keeping the data out of git protects the company's files.

## Step 1: data exploration (Senior Data Scientist)

What we did:
- Wrote `src/eda.py`. Running `python src/eda.py` prints every number we rely on and saves 9 charts to `reports/figures/`. It reads the data and never changes it, and it gives the same output every time.
- Checked every column of the training file (48,000 loads, Jan to Oct) and the validation file (12,000 loads, Nov to Dec): missing values, odd values, duplicates, spelling, cities, dates.
- Wrote `docs/eda_findings.md`. It holds the exact cleaning rules, the feature list, how we split the data to test the model, and how to fill the December chart file.

What we found:
- One column is a trap. `quote_signal` is a copy of the price per mile in five months, a mirror image of it in four months, and random noise in August. In Nov and Dec, the months we must predict, it is noise like August. A model that uses it looks better in testing and does worse on the real task, so we leave it out.
- 677 prices (1.4%) are wrong by a random factor of 2 to 6, too high or too low. They are easy to spot because no normal price comes anywhere near them. We remove them from training.
- 292 weights are negative. They look exactly like the positive weights, so the sign was simply flipped and we use the absolute value. There are no zero weights. Heavier loads cost up to about 8% more.
- A few weights and market index values are missing, at random. We fill weight with the typical weight for that equipment, and the market index with that day's average, because the index is one number per day.
- Distance drives the price. Reefer costs about 13% more than Dry Van and Flatbed about 8% more. The market index moves prices too, with a weekly cycle (high on Thursday, low on Sunday and Monday).
- Prices crept up over the year by more than the market index explains. A simple model trained on Jan to Aug guessed Sep and Oct about 4% too low. This is the biggest source of error we found, and the model step has to deal with it.
- Location matters a little: southern and Texas cities cost a bit more, northern cities a bit less, and nearby cities behave alike. That lets us price the 8 cities that only appear in the Nov to Dec file, using their map position.
- For the December chart (Lexington to Fort Wayne, Dry Van) we have 21 past loads on that exact route, priced $758 to $934. The missing inputs come from the city table (map position) and from that day's average market index in the Nov to Dec file.

How we will test the model:
- Train on Jan to Aug and test on Sep to Oct, like predicting the next two months for real.
- Tune on month by month tests inside Jan to Aug, so Sep to Oct stays untouched until the end.
- Hide 8 cities from training to see how well the model prices cities it has never seen.
- Score with average dollar error (MAE), average percent error (MAPE) and RMSE, with and without the wrong prices. The simple rule to beat (typical price per mile for the equipment and distance) already gets within 3.5% on Sep to Oct.

Why it matters: these checks decide what the model is allowed to see. Leaving out the trap column and the broken prices protects the real score. The time drift and the strong simple baseline tell the next step where the risk is.
