"""Independent audit of the headline numbers in reports/model_results.md.

    python tests/qa_audit.py          # about 1 minute; prints a table of recomputed vs reported values

Part 1 rebuilds the B1 baseline from the raw CSV with plain pandas: its own
outlier flags, its own distance bands (typed in from docs/eda_findings.md),
its own metrics. It imports nothing from src/.
Part 2 refits the chosen model on Jan-Aug with the project's functions, then
computes every metric by hand from the raw predictions.
Part 3 measures the quarter-end ramp month by month with a separate regression.
Not collected by pytest (the file name does not start with test_).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDGES = [0, 100, 150, 200, 250, 300, 400, 500, 600, 750, 900, 1100, 1300, 1550, 1800, 2100, 2500, 2900, 4000]
HIDDEN = ["Birmingham", "Dallas", "Detroit", "Las Vegas", "Louisville", "New Orleans", "St. Louis", "Washington"]


def scores(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {"MAE": np.abs(e).mean(), "MAPE": 100 * (np.abs(e) / y).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "bias": 100 * np.median(e / y), "n": len(y)}


# --------------------------------------------------------------------------
# Part 1: B1 from scratch
# --------------------------------------------------------------------------
def band(d):
    return pd.cut(d, EDGES, right=True, labels=False)


def flags(rows, ref):
    """|ln(rpm / median rpm of ref in the same equipment x band)| > 0.5."""
    med = (ref["posted_rate"] / ref["distance"]).groupby([ref["equipment"], band(ref["distance"])]).median()
    exp = med.reindex(pd.MultiIndex.from_arrays([rows["equipment"], band(rows["distance"])])).to_numpy()
    return np.abs(np.log(rows["posted_rate"].to_numpy() / rows["distance"].to_numpy() / exp)) > 0.5


def part1():
    raw = pd.read_csv(ROOT / "data" / "train_test.csv", parse_dates=["date"])
    out = {}
    ja, so = raw[raw["date"] <= "2025-08-31"], raw[raw["date"] >= "2025-09-01"]
    # Holdout: flags from Jan-Aug medians (raw, as the Cleaner does).
    test_clean = ~flags(so, ja)
    fit = ja[~flags(ja, ja)]
    med = (fit["posted_rate"] / fit["distance"]).groupby([fit["equipment"], band(fit["distance"])]).median()
    pred = med.reindex(pd.MultiIndex.from_arrays([so["equipment"], band(so["distance"])])).to_numpy() * so["distance"].to_numpy()
    y = so["posted_rate"].to_numpy()
    out["B1 holdout clean"] = scores(y[test_clean], pred[test_clean])
    out["B1 holdout all"] = scores(y, pred)
    out["holdout flagged"] = int((~test_clean).sum())
    out["Jan-Aug flagged"] = int(flags(ja, ja).sum())
    # CV folds.
    m = raw["date"].dt.month
    mape, mae = [], []
    for last in (4, 5, 6, 7):
        f, t = raw[m <= last], raw[m == last + 1]
        fc = f[~flags(f, f)]
        md = (fc["posted_rate"] / fc["distance"]).groupby([fc["equipment"], band(fc["distance"])]).median()
        p = md.reindex(pd.MultiIndex.from_arrays([t["equipment"], band(t["distance"])])).to_numpy() * t["distance"].to_numpy()
        c = ~flags(t, f)
        s = scores(t["posted_rate"].to_numpy()[c], p[c])
        mape.append(s["MAPE"])
        mae.append(s["MAE"])
    out["B1 CV mean clean MAPE"] = float(np.mean(mape))
    out["B1 CV mean clean MAE"] = float(np.mean(mae))
    # Unseen-city B1.
    touch = lambda d: d["pickup"].isin(HIDDEN) | d["delivery"].isin(HIDDEN)  # noqa: E731
    f, t = ja[~touch(ja)], so[touch(so)]
    fc = f[~flags(f, f)]
    md = (fc["posted_rate"] / fc["distance"]).groupby([fc["equipment"], band(fc["distance"])]).median()
    p = md.reindex(pd.MultiIndex.from_arrays([t["equipment"], band(t["distance"])])).to_numpy() * t["distance"].to_numpy()
    c = ~flags(t, f)
    out["B1 unseen-city clean"] = scores(t["posted_rate"].to_numpy()[c], p[c])
    out["unseen-city fit rows"] = int(len(fc))
    out["unseen-city test rows"] = int(len(t))
    return out


# --------------------------------------------------------------------------
# Part 2: chosen model on the holdout, metrics by hand
# --------------------------------------------------------------------------
def part2():
    sys.path.insert(0, str(ROOT))
    from src import data
    from src import model as mdl
    from src import train as T

    ds = data.load_all()
    split = T.holdout_split(ds)
    spec = mdl.FINAL_SPEC.with_(name="chosen")
    model = mdl.RateModel(spec).fit(split.fit)
    y = split.test["posted_rate"].to_numpy()
    clean = ~split.test["is_outlier"].to_numpy()
    month = split.test["date"].dt.month.to_numpy()
    out = {}
    p = model.predict(split.test)
    out["chosen holdout clean"] = scores(y[clean], p[clean])
    out["chosen holdout all"] = scores(y, p)
    out["chosen Sep clean"] = scores(y[clean & (month == 9)], p[clean & (month == 9)])
    out["chosen Oct clean"] = scores(y[clean & (month == 10)], p[clean & (month == 10)])
    for d in (0.0, 1.0):
        pd_ = np.exp(model.predict_log(split.test, damp=d))
        out[f"chosen damp {d:g} clean"] = scores(y[clean], pd_[clean])
    noramp = mdl.RateModel(spec.with_(drift="linear")).fit(split.fit).predict(split.test)
    out["chosen without ramp clean"] = scores(y[clean], noramp[clean])
    out["drift Jan-Aug"] = {"slope": model.drift_.slope_, "ramp": model.drift_.ramp_}
    return out


# --------------------------------------------------------------------------
# Part 3: the ramp month by month (own regression)
# --------------------------------------------------------------------------
def part3():
    raw = pd.read_csv(ROOT / "data" / "train_test.csv", parse_dates=["date"])
    raw = raw[~flags(raw, raw)].copy()
    raw["weight"] = raw["weight"].abs()
    raw["weight"] = raw["weight"].fillna(raw.groupby("equipment")["weight"].transform("median"))
    raw["market_index"] = raw["market_index"].fillna(raw.groupby("date")["market_index"].transform("mean"))
    ld = np.log(raw["distance"])
    w = raw["weight"] / 1e4
    X = np.column_stack([np.ones(len(raw)), ld, ld ** 2, ld ** 3, raw["equipment"].eq("Reefer"), raw["equipment"].eq("Flatbed"),
                         w, w ** 2] + [raw[c] for c in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon")])
    yv = np.log(raw["posted_rate"].to_numpy())
    beta, *_ = np.linalg.lstsq(X.astype(float), yv, rcond=None)
    raw["r"] = yv - X.astype(float) @ beta
    d = raw.groupby("date").agg(r=("r", "mean"), mi=("market_index", "mean")).reset_index()
    d["t"] = (d["date"] - pd.Timestamp("2025-01-01")).dt.days
    A = np.column_stack([np.ones(len(d)), np.log(d["mi"]), d["t"]])
    b, *_ = np.linalg.lstsq(A, d["r"].to_numpy(), rcond=None)
    d["e"] = d["r"] - A @ b
    d["month"], d["day"], d["dim"] = d["date"].dt.month, d["date"].dt.day, d["date"].dt.days_in_month
    d["pos"] = (d["day"] - 1) / (d["dim"] - 1)
    rows = []
    for mth, g in d.groupby("month"):
        slope = np.polyfit(g["pos"], g["e"], 1)[0]
        rows.append({"month": int(mth), "climb_first_to_last_day_pct": 100 * slope,
                     "first5_pct": 100 * g.loc[g["day"] <= 5, "e"].mean(),
                     "last5_pct": 100 * g.loc[g["day"] > g["dim"] - 5, "e"].mean()})
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 200)
    reported = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    h = reported["holdout"]["results"]
    p1 = part1()
    p2 = part2()
    rows = [
        ("B1 holdout clean MAE", p1["B1 holdout clean"]["MAE"], h["B1 baseline"]["clean"]["MAE"]),
        ("B1 holdout clean MAPE", p1["B1 holdout clean"]["MAPE"], h["B1 baseline"]["clean"]["MAPE"]),
        ("B1 holdout clean RMSE", p1["B1 holdout clean"]["RMSE"], h["B1 baseline"]["clean"]["RMSE"]),
        ("B1 holdout all MAE", p1["B1 holdout all"]["MAE"], h["B1 baseline"]["all"]["MAE"]),
        ("B1 holdout all MAPE", p1["B1 holdout all"]["MAPE"], h["B1 baseline"]["all"]["MAPE"]),
        ("B1 holdout all RMSE", p1["B1 holdout all"]["RMSE"], h["B1 baseline"]["all"]["RMSE"]),
        ("B1 CV mean clean MAPE", p1["B1 CV mean clean MAPE"],
         next(c["mean_clean_MAPE"] for c in reported["cv"]["candidates"] if c["name"] == "B1 baseline")),
        ("B1 unseen-city clean MAPE", p1["B1 unseen-city clean"]["MAPE"],
         reported["unseen_city"]["results"]["B1, cities unseen"]["clean"]["MAPE"]),
        ("B1 unseen-city clean MAE", p1["B1 unseen-city clean"]["MAE"],
         reported["unseen_city"]["results"]["B1, cities unseen"]["clean"]["MAE"]),
        ("Chosen holdout clean MAPE", p2["chosen holdout clean"]["MAPE"], h["chosen"]["clean"]["MAPE"]),
        ("Chosen holdout clean MAE", p2["chosen holdout clean"]["MAE"], h["chosen"]["clean"]["MAE"]),
        ("Chosen holdout clean RMSE", p2["chosen holdout clean"]["RMSE"], h["chosen"]["clean"]["RMSE"]),
        ("Chosen holdout clean bias", p2["chosen holdout clean"]["bias"], h["chosen"]["clean"]["median_signed_pct"]),
        ("Chosen holdout all MAPE", p2["chosen holdout all"]["MAPE"], h["chosen"]["all"]["MAPE"]),
        ("Chosen holdout all MAE", p2["chosen holdout all"]["MAE"], h["chosen"]["all"]["MAE"]),
        ("Chosen holdout all RMSE", p2["chosen holdout all"]["RMSE"], h["chosen"]["all"]["RMSE"]),
        ("Chosen Sep clean MAPE", p2["chosen Sep clean"]["MAPE"], reported["holdout"]["by_month"]["chosen, Sep"]["clean"]["MAPE"]),
        ("Chosen Oct clean MAPE", p2["chosen Oct clean"]["MAPE"], reported["holdout"]["by_month"]["chosen, Oct"]["clean"]["MAPE"]),
        ("Damp 0 clean MAPE", p2["chosen damp 0 clean"]["MAPE"], h["chosen fit, damp 0 at prediction"]["clean"]["MAPE"]),
        ("Damp 1 clean MAPE", p2["chosen damp 1 clean"]["MAPE"], h["chosen fit, damp 1 at prediction"]["clean"]["MAPE"]),
        ("Without ramp clean MAPE", p2["chosen without ramp clean"]["MAPE"], h["chosen without the ramp (trend only)"]["clean"]["MAPE"]),
    ]
    table = pd.DataFrame(rows, columns=["number", "recomputed", "reported"])
    table["abs diff"] = (table["recomputed"] - table["reported"]).abs()
    print(table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\ncounts: holdout flagged {p1['holdout flagged']} (doc 144), Jan-Aug flagged {p1['Jan-Aug flagged']} (doc 533), "
          f"unseen-city fit rows {p1['unseen-city fit rows']} (doc 33,819), test rows {p1['unseen-city test rows']} (doc 1,003)")
    print(f"drift learned on Jan-Aug: {p2['drift Jan-Aug']}")
    print("\nQuarter-end check, own regression (daily level after log index and a straight-line trend):")
    print(part3().round(2).to_string(index=False))
    print(f"\nmax abs diff over all headline numbers: {table['abs diff'].max():.2e}")


if __name__ == "__main__":
    main()
