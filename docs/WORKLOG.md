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

## Step 2: data pipeline (Senior Data Engineer)

What we did:
- Wrote `src/config.py`. It keeps every path and fixed number in one place: the file paths, the column lists, the three equipment types, the date ranges, the 18 distance bands, the outlier threshold of 0.5 and the random seed. Paths are built from the code's own location, so the code runs from any folder.
- Wrote `src/data.py`, the data pipeline. It loads each file, checks it, cleans it and writes the two output files. The top of the file shows how the model step calls it. Running `python src/data.py` runs the whole pipeline on the full data and prints the cleaning log.
- Every loader checks the file first. The columns must be exactly the expected ones, in order. Numbers must be numbers and never infinite, dates must be `YYYY-MM-DD` and inside the expected months, equipment must be Dry Van, Reefer or Flatbed, and no `load_id` may repeat. If anything is wrong, the loader stops and lists every problem it found.
- Cleaning is learned on training rows only, then applied to any set of rows. The `Cleaner` learns two numbers from the rows it is fitted on: the typical weight for each equipment type, and the typical price per mile for each equipment type and distance band. When we test on later months, those months never feed these numbers, so nothing leaks from the test rows into training.
- The cleaning steps, in order:
  - Weight: take the absolute value, treat 0 as missing, and fill missing weights with the typical weight for that equipment. We add no "was missing" column, because the gaps are random.
  - Market index: fill a gap with that day's average, using both the training and the validation file. This is safe because the index is an input, not the price we predict, and each day sits in only one file.
  - Wrong prices: flag a price that is more than 1.65 times or less than 0.61 times the typical price per mile for its equipment and distance band. We drop flagged rows from training rows only. Test rows keep the flag, so we can score them with and without the broken prices. Validation rows are never dropped.
  - `quote_signal`, distance and coordinates pass through unchanged. `quote_signal` stays in the data so QA can check it, and `config.NEVER_FEATURES` lists it with the other columns the model must never use.
- Built the city table from the pickup and delivery columns of both files: 72 cities, each with exactly one map position.
- Built the December chart rows. The file has no map positions and no market index, so the positions come from the city table and the index from that day's average in the validation file. All 31 days have values, from 0.8306 to 1.0445 (mean 0.9344). The rows get the same columns as the validation file, so the model handles them the same way.
- Wrote the two output writers. The validation writer puts the predictions in template order with exactly the columns `load_id,predicted_rate`, and checks that all 12,000 ids are there once and every price is a real number above zero. The December writer fills only `predicted_rate` and copies the other six columns as the original text, so 360, 32000 and the dates stay exactly as given. We tested both writers on copies in a temporary folder, and both files passed the checks in `score.py`. The real output files are written later, by the model step.

Counts of every fix, with cleaning learned on all of train (the final fit):

| Fix | Train (Jan to Oct) | Validation (Nov to Dec) |
|---|---|---|
| Negative weights flipped | 292 | 145 |
| Zero weights set to missing | 0 | 0 |
| Missing weights filled | 300 | 165 |
| Missing market index filled | 374 | 249 |
| Broken prices flagged (high / low) | 677 (340 / 337) | not labeled |
| Rows dropped | 677 | 0 |
| Rows left | 47,323 | 12,000 |

Counts with cleaning learned on Jan to Aug only (the main time holdout):

| Fix | Jan to Aug (training) | Sep to Oct (test) | Validation |
|---|---|---|---|
| Negative weights flipped | 233 | 59 | 145 |
| Missing weights filled | 235 | 65 | 165 |
| Missing market index filled | 301 | 73 | 249 |
| Broken prices flagged (high / low) | 533 (267 / 266) | 144 (73 / 71) | not labeled |
| Rows dropped | 533 | 0 (flag kept for scoring) | 0 |
| Rows left | 37,944 | 9,523 | 12,000 |

What we checked:
- Every count matches `docs/eda_findings.md`. Typical weights learned on all of train are 31,444 lb for Dry Van, 31,577 for Reefer and 31,532.5 for Flatbed. On Jan to Aug only they are 31,418.5, 31,540 and 31,521.
- Price bands learned on Jan to Aug flag the same 677 rows as bands learned on all of train.
- The four month-by-month test splits give the same row counts as the findings doc: 18,835, 23,686, 28,406 and 33,253 clean training rows, with 62, 63, 65 and 68 flagged test rows.
- After cleaning, no weight or market index is missing, every weight is between 5,000 and 47,500 lb, and `load_id`, `quote_signal`, distance and coordinates are unchanged. The validation rows keep their file order.
- The loaders reject broken copies of the data: unknown equipment, a bad date, a date outside the range, a repeated id, text in a number column, a missing or infinite value, a zero price, and wrong or reordered columns. The writers reject the wrong number of predictions, repeated or missing ids, and missing, infinite or non-positive prices.
- The input files in `data/` are unchanged, byte for byte.

One change from the findings doc. If a day had no market index at all, the doc said to use the average of the 3 nearest days on each side. That window covers almost a full week, so it averages away the weekly cycle. Leaving out one day at a time over all 365 days, it misses by 0.084 (RMSE), about as much as a month average. We use the same weekday one week before and after instead, which misses by 0.024. If those are missing too, we go two weeks out, then up to four, and as a last resort the nearest day with data. Today every day has at least 114 values, so this fallback never runs and no number changes.

Why it matters: the same code cleans every test split, the final fit, the validation file and the December rows. The scores we report in testing therefore come from the same cleaning the submission gets.
