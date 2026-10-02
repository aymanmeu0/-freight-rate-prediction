"""Exploratory data analysis for the freight-rate prediction task.

Run from the project root:

    python src/eda.py

The script only reads files under ``data/``. It prints every number quoted in
``docs/eda_findings.md`` and saves the figures to ``reports/figures/``.
It is deterministic: the only random step uses a fixed seed.

Sections
    1. Integrity: shapes, ids, dates, duplicates, spellings, cities
    2. Outlier rates (the cleaning rule)
    3. Distance vs straight-line (haversine) distance
    4. Weight: sign errors, caps, effect on rate
    5. Missing values and market_index imputation
    6. quote_signal: three regimes and the leakage trap
    7. Time: market_index, weekday, month, holidays, trend
    8. Geography: city effects and unseen cities
    9. December chart lane and inputs
   10. Validation design: holdout, CV folds, baselines
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"

SEED = 42

# Distance bands (miles) used for the robust "expected rate per mile".
# Right-closed intervals, as produced by pd.cut. Every (equipment, band) cell
# holds at least 48 training rows.
DIST_BANDS = [0, 100, 150, 200, 250, 300, 400, 500, 600, 750, 900,
              1100, 1300, 1550, 1800, 2100, 2500, 2900, 4000]

# A row is a rate outlier when |ln(rate per mile / expected rate per mile)| > 0.5,
# i.e. the rate is more than 1.65x or less than 0.61x the typical rate for its
# equipment and distance band.
OUTLIER_LOG_THRESHOLD = 0.5

# quote_signal regimes found in section 6 (calendar months of 2025).
QS_COPY_MONTHS = [1, 2, 3, 6, 9]      # quote_signal ~= rate per mile
QS_MIRROR_MONTHS = [4, 5, 7, 10]      # quote_signal ~= 4.15 - rate per mile
QS_NOISE_MONTHS = [8]                 # quote_signal unrelated to the rate

# Unseen-city holdout: the 8 lowest-volume training cities (see section 10).
HOLDOUT_CITIES_N = 8

# US federal holidays that fall in the labeled period.
HOLIDAYS_2025 = ["2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
                 "2025-07-04", "2025-09-01", "2025-10-13"]

# Figure style: one quiet, consistent look for every chart.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dd", "#fcfcfb"
BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9a9893"
EQUIP_COLORS = {"Dry Van": BLUE, "Reefer": ORANGE, "Flatbed": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRAY, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5,
    "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def expected_rpm_table(frame: pd.DataFrame) -> pd.Series:
    """Median rate per mile for each (equipment, distance band) cell."""
    return frame.groupby(["equipment", "dist_band"], observed=True)["rpm"].median().rename("exp_rpm")


def attach_expected(frame: pd.DataFrame, table: pd.Series) -> pd.DataFrame:
    """Join the expected rate per mile and the log ratio to it."""
    out = frame.drop(columns=["exp_rpm", "log_ratio"], errors="ignore").join(
        table, on=["equipment", "dist_band"])
    if "rpm" in out:
        out["log_ratio"] = np.log(out["rpm"] / out["exp_rpm"])
    return out


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {"MAE": np.mean(np.abs(err)),
            "MAPE%": 100 * np.mean(np.abs(err) / y_true),
            "RMSE": np.sqrt(np.mean(err ** 2))}


def fmt_metrics(m: dict) -> str:
    return f"MAE ${m['MAE']:8.2f} | MAPE {m['MAPE%']:6.2f}% | RMSE ${m['RMSE']:8.2f}"


def savefig(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  [figure] {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train_test.csv")
    val = pd.read_csv(DATA / "validation.csv")
    dec = pd.read_csv(DATA / "december_chart_inputs.csv")
    for df in (train, val):
        df["dt"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df["month"] = df["dt"].dt.month
        df["weekday"] = df["dt"].dt.dayofweek
        df["dist_band"] = pd.cut(df["distance"], DIST_BANDS)
        df["hav"] = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    train["rpm"] = train["posted_rate"] / train["distance"]
    return train, val, dec


# --------------------------------------------------------------------------
# 1. Integrity
# --------------------------------------------------------------------------
def section_integrity(train, val, dec):
    header("1. INTEGRITY")
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    for name, df, prefix in (("train", train, "TR-"), ("val", val, "TE-")):
        raw_cols = [c for c in df.columns if c not in
                    ("dt", "month", "weekday", "dist_band", "hav", "rpm", "load_id")]
        expected_ids = [f"{prefix}{i:06d}" for i in range(1, len(df) + 1)]
        print(f"{name}: {df.shape[0]:,} rows | dates {df.date.min()}..{df.date.max()} "
              f"({df.date.nunique()} days, none skipped: "
              f"{df.date.nunique() == (df.dt.max() - df.dt.min()).days + 1})")
        print(f"  ids sequential {prefix}000001..: {(df.load_id == expected_ids).all()} | "
              f"duplicate ids {df.load_id.duplicated().sum()} | "
              f"duplicate rows ignoring id {df.duplicated(subset=raw_cols).sum()}")
        print(f"  loads per day: min {df.groupby('date').size().min()}, "
              f"median {df.groupby('date').size().median():.0f}, max {df.groupby('date').size().max()}")
        for col in ("pickup", "delivery", "equipment"):
            s = df[col]
            print(f"  {col:9s}: {s.nunique()} values | untrimmed {int((s != s.str.strip()).sum())} | "
                  f"case variants {s.nunique() - s.str.strip().str.lower().nunique()}")
        print(f"  equipment counts: {df.equipment.value_counts().to_dict()}")
        print(f"  non-positive distance {(df.distance <= 0).sum()}, market_index {(df.market_index <= 0).sum()}, "
              f"quote_signal {(df.quote_signal <= 0).sum()}; pickup == delivery {(df.pickup == df.delivery).sum()}")
        print(f"  missing per column: { {k: int(v) for k, v in df[raw_cols].isna().sum().items() if v} }")
    print(f"train posted_rate: min {train.posted_rate.min()}, max {train.posted_rate.max()}, "
          f"non-positive {(train.posted_rate <= 0).sum()}")
    print(f"template ids == validation ids in order: {(template.load_id.values == val.load_id.values).all()}")

    # Cities and coordinates
    def city_coords(df):
        a = df[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        b = df[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        return pd.concat([a, b])
    coords = pd.concat([city_coords(train), city_coords(val)])
    per_city = coords.groupby("city")[["lat", "lon"]].nunique()
    print(f"cities with more than one lat/lon pair: {int((per_city.max(axis=1) > 1).sum())} "
          f"| coordinate pairs shared by two cities: "
          f"{int((coords.drop_duplicates().groupby(['lat', 'lon']).city.nunique() > 1).sum())}")
    train_cities = set(train.pickup) | set(train.delivery)
    val_cities = set(val.pickup) | set(val.delivery)
    unseen = sorted(val_cities - train_cities)
    touch = val.pickup.isin(unseen) | val.delivery.isin(unseen)
    both = val.pickup.isin(unseen) & val.delivery.isin(unseen)
    print(f"train cities {len(train_cities)}, val cities {len(val_cities)}, val-only cities {len(unseen)}: {unseen}")
    print(f"val rows touching an unseen city: {touch.sum():,} ({touch.mean():.1%}); both ends unseen: {both.sum()}")
    lanes = set(zip(train.pickup, train.delivery))
    print(f"train lanes {len(lanes):,} of {len(train_cities) * (len(train_cities) - 1):,} possible; "
          f"val rows on a lane seen in train: {np.mean([l in lanes for l in zip(val.pickup, val.delivery)]):.1%}")
    print(f"December file: {dec.shape}, columns {list(dec.columns)}, dates "
          f"{dec.date.min()}..{dec.date.max()}, unique (pickup, delivery, distance, equipment, weight): "
          f"{dec[['pickup', 'delivery', 'distance', 'equipment', 'weight']].drop_duplicates().shape[0]}")
    city_table = coords.drop_duplicates("city").set_index("city")[["lat", "lon"]]
    edge = city_table[(city_table.lon == city_table.lon.max()) | (city_table.lat == city_table.lat.min())]
    print(f"cities sitting on a coordinate bound (looks clipped): {edge.to_dict('index')}")
    print(f"corr(distance, posted_rate) in train: {np.corrcoef(train.distance, train.posted_rate)[0, 1]:.3f}")
    return unseen, city_table


# --------------------------------------------------------------------------
# 3. Distance
# --------------------------------------------------------------------------
def section_distance(train, val):
    header("3. DISTANCE vs HAVERSINE")
    for name, df in (("train", train), ("val", val)):
        ratio = df.distance / df.hav
        q = ratio.quantile([0.001, 0.01, 0.5, 0.99, 0.999])
        print(f"{name}: corr(distance, haversine) {np.corrcoef(df.distance, df.hav)[0, 1]:.4f} | "
              f"ratio quantiles 0.1%/1%/50%/99%/99.9%: {q.round(3).tolist()} | max {ratio.max():.2f}")
        print(f"  distance == 70 (the floor): {(df.distance == 70).sum()} | min {df.distance.min()} | "
              f"ratio > 1.6 with haversine > 100 mi: {((ratio > 1.6) & (df.hav > 100)).sum()}")
    big = val[(val.distance / val.hav) > 5][["pickup", "delivery", "hav", "distance"]].drop_duplicates(["pickup", "delivery"])
    print("  largest ratios are short lanes clipped to the 70-mile floor:\n" + big.round(2).to_string(index=False))

    # Within a lane, does the rate follow the row's own distance? (elasticity)
    clean = train[train.is_outlier.eq(False)].copy()
    clean["lane"] = clean.pickup + ">" + clean.delivery
    clean["ld"] = np.log(clean.distance)
    clean["lrate"] = np.log(clean.posted_rate)
    g = clean.groupby("lane")
    clean = clean[g.ld.transform("size") >= 5]
    dd = clean.ld - clean.groupby("lane").ld.transform("mean")
    rr = clean.lrate - clean.groupby("lane").lrate.transform("mean")
    print(f"within-lane: sd of log distance {dd.std():.4f}; elasticity of rate to distance "
          f"{np.polyfit(dd, rr, 1)[0]:.3f} (across all rows: "
          f"{np.polyfit(np.log(train.distance[~train.is_outlier]), np.log(train.posted_rate[~train.is_outlier]), 1)[0]:.3f})")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.scatter(train.hav, train.distance / train.hav, s=3, alpha=0.25, color=BLUE, label="train", rasterized=True)
    ax.scatter(val.hav, val.distance / val.hav, s=3, alpha=0.25, color=ORANGE, label="validation", rasterized=True)
    hv = np.linspace(0.8, 60, 300)
    ax.plot(hv, 70 / hv, color=INK_2, lw=1.2, ls="--", label="distance = 70 mi floor")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Straight-line (haversine) distance between the given coordinates, miles")
    ax.set_ylabel("Listed distance / haversine")
    ax.set_title("Listed distance is about 1.18x the straight line; short lanes hit a 70-mile floor")
    ax.legend(loc="upper right", markerscale=4)
    savefig(fig, "08_distance_vs_haversine.png")


# --------------------------------------------------------------------------
# 2. Outlier rates
# --------------------------------------------------------------------------
def section_outliers(train):
    header("2. OUTLIER RATES")
    print(f"rate per mile quantiles 0.1/0.5/1/50/99/99.5/99.9%: "
          f"{train.rpm.quantile([.001, .005, .01, .5, .99, .995, .999]).round(3).tolist()}")
    print(f"median rate per mile by equipment: {train.groupby('equipment').rpm.median().round(3).to_dict()}")

    table = expected_rpm_table(train)
    df = attach_expected(train, table)
    print(f"(equipment x distance band) cells: {len(table)}, smallest cell "
          f"{int(df.groupby(['equipment', 'dist_band'], observed=True).size().min())} rows")
    lr = df.log_ratio
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        print(f"  |log ratio| > {t}: {(lr.abs() > t).sum()} rows "
              f"(high {(lr > t).sum()}, low {(lr < -t).sum()})")
    core, flagged = lr[lr.abs() <= OUTLIER_LOG_THRESHOLD], lr[lr.abs() > OUTLIER_LOG_THRESHOLD]
    print(f"gap: largest |log ratio| kept {core.abs().max():.3f}, smallest flagged {flagged.abs().min():.3f}; "
          f"rows with 0.25 < |log ratio| < 0.70: {((lr.abs() > 0.25) & (lr.abs() < 0.70)).sum()}")
    print(f"core log-ratio sd {core.std():.4f} (about {100 * core.std():.1f}% typical spread)")
    df["is_outlier"] = lr.abs() > OUTLIER_LOG_THRESHOLD
    n_out = int(df.is_outlier.sum())
    print(f"RULE -> {n_out} outlier rows ({n_out / len(df):.2%}): "
          f"{int((lr > OUTLIER_LOG_THRESHOLD).sum())} high, {int((lr < -OUTLIER_LOG_THRESHOLD).sum())} low")

    # Same rule with medians from Jan-Aug only (no peeking at holdout labels).
    early = df[df.month <= 8]
    df_early = attach_expected(df, expected_rpm_table(early))
    same = (df_early.log_ratio.abs() > OUTLIER_LOG_THRESHOLD) == df.is_outlier
    print(f"medians from Jan-Aug only flag the identical set: {same.all()} "
          f"(Sep-Oct flagged: {int(df[df.month >= 9].is_outlier.sum())})")

    # A global log-linear model gives the same flags too (robustness check).
    X = np.column_stack([np.ones(len(df)), np.log(df.distance), np.log(df.distance) ** 2,
                         pd.get_dummies(df.equipment, drop_first=True).to_numpy(float)])
    y = np.log(df.posted_rate.to_numpy())
    keep = np.ones(len(df), bool)
    for _ in range(3):
        beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        res = y - X @ beta
        keep = np.abs(res) < OUTLIER_LOG_THRESHOLD
    print(f"log-linear model residual rule flags {(~keep).sum()} rows; agreement with band rule: "
          f"{((~keep) == df.is_outlier.to_numpy()).mean():.4%}")

    # Is the error a clean multiple (2x, 10x, per-mile) or a random factor?
    # quote_signal recovers the clean rate per mile in 9 of 10 months (section 6),
    # so it tells us the exact factor that hit each corrupted row.
    clean_rpm = np.where(df.month.isin(QS_COPY_MONTHS), df.quote_signal,
                         np.where(df.month.isin(QS_MIRROR_MONTHS), 4.15 - df.quote_signal, np.nan))
    factor = (df.rpm / clean_rpm)[df.is_outlier & ~np.isnan(clean_rpm)]
    hi, lo = factor[factor > 1], factor[factor < 1]
    print(f"recovered corruption factor ({len(factor)} rows outside August): "
          f"high x{hi.min():.2f}..x{hi.max():.2f} (median x{hi.median():.2f}); "
          f"low x{lo.min():.3f}..x{lo.max():.3f} (median x{lo.median():.3f}, i.e. 1/{1 / lo.median():.1f})")
    counts, edges = np.histogram(hi, bins=np.arange(2.0, 5.51, 0.5))
    print(f"  high factors per 0.5 bin from 2.0: {counts.tolist()} -> flat, no spikes at 2,3,4,5")
    ok = df[~df.is_outlier]
    ok_err = np.log(ok.rpm / pd.Series(clean_rpm, index=df.index)[ok.index]).dropna()
    print(f"  for clean rows the recovered rate matches posted rate within sd {ok_err.std():.4f}")
    print(f"outlier share by equipment: {df.groupby('equipment').is_outlier.mean().round(4).to_dict()}")
    print(f"outlier count by month: {df.groupby('month').is_outlier.sum().to_dict()}")
    print(f"outlier share: negative weight rows {df[df.weight < 0].is_outlier.mean():.3f} "
          f"({int(df[df.weight < 0].is_outlier.sum())}/{(df.weight < 0).sum()}, binomial p="
          f"{stats.binomtest(int(df[df.weight < 0].is_outlier.sum()), int((df.weight < 0).sum()), df.is_outlier.mean()).pvalue:.3f}), "
          f"missing weight {df[df.weight.isna()].is_outlier.mean():.3f}, "
          f"missing market_index {df[df.market_index.isna()].is_outlier.mean():.3f}")
    ex = df[df.is_outlier].sort_values("log_ratio")
    print("examples (lowest and highest):")
    print(pd.concat([ex.head(3), ex.tail(3)])[["load_id", "pickup", "delivery", "distance", "equipment",
                                               "posted_rate", "exp_rpm", "log_ratio"]].round(3).to_string(index=False))

    # Figure: the gap that makes the rule safe.
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    bins = np.arange(-2.0, 2.01, 0.04)
    ax.hist(lr, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
    ax.set_yscale("log")
    for t in (-OUTLIER_LOG_THRESHOLD, OUTLIER_LOG_THRESHOLD):
        ax.axvline(t, color=RED, ls="--", lw=1.2)
    ax.text(0.52, ax.get_ylim()[1] * 0.3, "flag if |log ratio| > 0.5", color=INK_2)
    ax.set_xlabel("ln(rate per mile / median for same equipment and distance band)")
    ax.set_ylabel("Loads (log scale)")
    ax.set_title(f"An empty gap separates {n_out} outliers from the rest")
    ax = axes[1]
    ok = df[~df.is_outlier].sample(8000, random_state=SEED)
    ax.scatter(ok.distance, ok.posted_rate, s=3, color=GRAY, alpha=0.35, label="normal (sample of 8,000)", rasterized=True)
    bad = df[df.is_outlier]
    ax.scatter(bad.distance, bad.posted_rate, s=9, color=RED, alpha=0.8, label=f"flagged ({n_out})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Distance, miles"); ax.set_ylabel("Posted rate, $")
    ax.set_title("Flagged rates are 2x to 6x too high or too low")
    ax.legend(loc="upper left", markerscale=2)
    savefig(fig, "01_rate_outliers.png")
    return df


# --------------------------------------------------------------------------
# 4. Weight
# --------------------------------------------------------------------------
def section_weight(train, val):
    header("4. WEIGHT")
    for name, df in (("train", train), ("val", val)):
        w = df.weight
        print(f"{name}: missing {w.isna().sum()}, zero {(w == 0).sum()}, negative {(w < 0).sum()}, "
              f"min {w.min():.0f}, max {w.max():.0f}, smallest positive {w[w > 0].min():.0f}, "
              f"at +47,500 {(w == 47500).sum()} ({(w == 47500).mean():.2%}), at -47,500 {(w == -47500).sum()}, "
              f"non-integer {(w.dropna() % 1 != 0).sum()}")
        ks = stats.ks_2samp(w[w > 0], -w[w < 0])
        print(f"  positive vs |negative| weights: medians {w[w > 0].median():.0f} vs {(-w[w < 0]).median():.0f}, "
              f"KS stat {ks.statistic:.3f}, p={ks.pvalue:.2f}")
        print(f"  negative count by equipment {df[w < 0].equipment.value_counts().to_dict()}")
    ks = stats.ks_2samp(train.weight[train.weight > 0], val.weight[val.weight > 0])
    print(f"train vs val positive weights: KS p={ks.pvalue:.2f}")
    print(f"median |weight| by equipment (train): "
          f"{train.weight.abs().groupby(train.equipment).median().to_dict()}")

    # Effect on rate: residual after equipment x distance band and the day's level.
    c = train[~train.is_outlier].copy()
    c["res"] = c.log_ratio - c.groupby("date").log_ratio.transform("mean")
    c["aw"] = c.weight.abs()
    edges = [0, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 47499, 47500]
    c["wb"] = pd.cut(c.aw, edges)
    tab = c.groupby("wb", observed=True).res.agg(["mean", "count"])
    print("mean log residual by |weight| band:\n" + tab.round(4).to_string())
    has = c.aw.notna()
    print(f"corr(|weight|, residual) {np.corrcoef(c.aw[has], c.res[has])[0, 1]:.3f}; "
          f"negative rows only (using |weight|): "
          f"{np.corrcoef(c.aw[c.weight < 0], c.res[c.weight < 0])[0, 1]:.3f}; "
          f"rows with missing weight have mean residual {c.res[c.weight.isna()].mean():+.4f}")
    for e in EQUIP_COLORS:
        x = c[(c.equipment == e) & has]
        print(f"  {e:8s}: slope {np.polyfit(x.aw / 1e4, x.res, 1)[0]:+.4f} log-units per 10,000 lb")
    print(f"corr(|weight|, distance) {c[['aw', 'distance']].corr().iloc[0, 1]:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    bins = np.arange(5000, 49001, 1500)          # last bin holds the 47,500 cap
    pos, neg = train.weight[train.weight > 0], -train.weight[train.weight < 0]
    ax.hist(pos, bins=bins, weights=np.full(len(pos), 100 / len(pos)), color=BLUE, alpha=0.55,
            label=f"positive weights ({len(pos):,})")
    ax.hist(neg, bins=bins, weights=np.full(len(neg), 100 / len(neg)), histtype="step", lw=2,
            color=ORANGE, label=f"negative weights, sign flipped ({len(neg)})")
    ax.set_xlabel("Weight, lb"); ax.set_ylabel("Share of loads per 1,500 lb bin, %")
    ax.set_title("Negative weights mirror the positive ones: a sign error")
    ax.legend(loc="upper left")
    ax = axes[1]
    mids = [(iv.left + iv.right) / 2 if iv.right < 47500 else 47500 for iv in tab.index]
    mids[0] = 8000
    ax.plot(mids, 100 * (np.exp(tab["mean"]) - 1), color=BLUE, lw=2, marker="o", ms=6)
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xlabel("|Weight|, lb (band midpoint)")
    ax.set_ylabel("Rate vs same-day, same-band typical, %")
    ax.set_title("Heavier loads cost more: about 8% from lightest to heaviest")
    savefig(fig, "03_weight.png")


# --------------------------------------------------------------------------
# 5. Missing values
# --------------------------------------------------------------------------
def section_missing(train, val):
    header("5. MISSING VALUES")
    for name, df in (("train", train), ("val", val)):
        for col in ("weight", "market_index"):
            mis = df[col].isna()
            p_month = stats.chi2_contingency(pd.crosstab(df.month, mis)).pvalue
            p_eq = stats.chi2_contingency(pd.crosstab(df.equipment, mis)).pvalue
            p_city = stats.chi2_contingency(pd.crosstab(df.pickup, mis)).pvalue
            per_day = mis.groupby(df.date).sum()
            print(f"{name} {col:12s}: {mis.sum():4d} missing ({mis.mean():.2%}) | chi-square p by month {p_month:.2f}, "
                  f"equipment {p_eq:.2f}, pickup city {p_city:.2f} | max per day {per_day.max()}, "
                  f"days with none observed {int((df.groupby('date')[col].count() == 0).sum())}")
        both = (df.weight.isna() & df.market_index.isna()).sum()
        expect = df.weight.isna().mean() * df.market_index.isna().mean() * len(df)
        print(f"  both missing: {both} (expected {expect:.1f} if independent)")

    # market_index behaves like one number per day plus small noise.
    pooled = pd.concat([train, val], ignore_index=True)
    daily = pooled.groupby("date").market_index
    print(f"market_index (train): mean within-day sd {train.groupby('date').market_index.std().mean():.4f}, "
          f"sd of daily means {train.groupby('date').market_index.mean().std():.4f}; "
          f"val within-day sd {val.groupby('date').market_index.std().mean():.4f}")
    dev = pooled.market_index - daily.transform("mean")
    print(f"  within-day deviation vs equipment means {dev.groupby(pooled.equipment).mean().round(4).to_dict()}, "
          f"sd of pickup-city means {dev.groupby(pooled.pickup).mean().std():.4f}, "
          f"corr with distance {np.corrcoef(dev.dropna(), pooled.distance[dev.notna()])[0, 1]:+.4f}")
    # Leave-one-out test of the same-date fill.
    s, n = daily.transform("sum"), daily.transform("count")
    loo = (s - pooled.market_index) / (n - 1)
    err = (pooled.market_index - loo).dropna()
    month_mean = pooled.groupby(pooled.dt.dt.to_period("M")).market_index.transform("mean")
    print(f"  leave-one-out same-date mean fill: MAE {err.abs().mean():.4f}, RMSE {np.sqrt((err ** 2).mean()):.4f} "
          f"| month-mean fill RMSE {np.sqrt(((pooled.market_index - month_mean) ** 2).mean()):.4f} "
          f"| global-mean fill RMSE {pooled.market_index.std():.4f} | fewest observed values on a day {n.min()}")
    c = train[~train.is_outlier]
    res = c.log_ratio - c.groupby("date").log_ratio.transform("mean")
    print(f"  rows with missing market_index: mean rate residual {res[c.market_index.isna()].mean():+.4f} "
          f"(n={c.market_index.isna().sum()})")


# --------------------------------------------------------------------------
# 6. quote_signal
# --------------------------------------------------------------------------
def section_quote_signal(train, val):
    header("6. QUOTE_SIGNAL")
    c = train[~train.is_outlier]
    val_e = attach_expected(val, expected_rpm_table(c))
    rows = []
    for m in range(1, 13):
        df = (c if m <= 10 else val_e)
        x = df[df.month == m]
        rows.append({"month": m,
                     "corr(qs, expected rpm)": np.corrcoef(x.quote_signal, x.exp_rpm)[0, 1],
                     "corr(qs, actual rpm)": np.corrcoef(x.quote_signal, x.rpm)[0, 1] if m <= 10 else np.nan,
                     "qs mean": x.quote_signal.mean(), "qs sd": x.quote_signal.std(),
                     "qs sd within day": x.groupby("date").quote_signal.std().mean(),
                     "sd of daily qs mean": x.groupby("date").quote_signal.mean().std()})
    tab = pd.DataFrame(rows).set_index("month")
    print(tab.round(3).to_string())

    copy = c[c.month.isin(QS_COPY_MONTHS)]
    mirror = c[c.month.isin(QS_MIRROR_MONTHS)]
    noise = c[c.month.isin(QS_NOISE_MONTHS)]
    lq = np.log(copy.quote_signal / copy.rpm)
    print(f"COPY months {QS_COPY_MONTHS}: qs / rpm median {np.exp(lq.median()):.4f}, sd of log ratio {lq.std():.4f}, "
          f"share within 4%: {(lq.abs() < 0.04).mean():.4f}")
    b = np.polyfit(mirror.rpm, mirror.quote_signal, 1)
    r = mirror.quote_signal - np.polyval(b, mirror.rpm)
    print(f"MIRROR months {QS_MIRROR_MONTHS}: qs = {b[1]:.3f} {b[0]:+.4f} * rpm, residual sd {r.std():.4f}; "
          f"qs + rpm mean {(mirror.quote_signal + mirror.rpm).mean():.3f}")
    for m in QS_MIRROR_MONTHS:
        x = mirror[mirror.month == m]
        bm = np.polyfit(x.rpm, x.quote_signal, 1)
        print(f"   month {m}: intercept {bm[1]:.3f}, slope {bm[0]:+.4f}")
    print(f"NOISE month {QS_NOISE_MONTHS}: corr(qs, rpm) {np.corrcoef(noise.quote_signal, noise.rpm)[0, 1]:+.4f}, "
          f"corr(qs, market_index) {noise[['quote_signal', 'market_index']].corr().iloc[0, 1]:+.4f}")
    ks = stats.ks_2samp(noise.quote_signal, val.quote_signal)
    print(f"validation (Nov-Dec) qs vs August qs: means {val.quote_signal.mean():.3f} vs {noise.quote_signal.mean():.3f}, "
          f"sd {val.quote_signal.std():.3f} vs {noise.quote_signal.std():.3f}, KS p={ks.pvalue:.2f}")
    print(f"validation: corr(qs, log distance) {np.corrcoef(val.quote_signal, np.log(val.distance))[0, 1]:+.4f}, "
          f"corr(qs, market_index) {val[['quote_signal', 'market_index']].corr().iloc[0, 1]:+.4f}, "
          f"corr(qs, weight) {val[['quote_signal', 'weight']].corr().iloc[0, 1]:+.4f}")
    print("-> in Nov-Dec quote_signal looks like the August noise regime: it carries no information about the rate.")

    # Leakage demonstration with a quick, untuned gradient-boosting model.
    leakage_demo(train)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    samp = lambda d: d.sample(min(len(d), 4000), random_state=SEED)
    for ax, d, title in ((axes[0], copy, "Jan-Mar, Jun, Sep: copy of the rate"),
                         (axes[1], mirror, "Apr, May, Jul, Oct: mirror image"),
                         (axes[2], noise, "Aug: pure noise")):
        s = samp(d)
        ax.scatter(s.rpm, s.quote_signal, s=3, alpha=0.3, color=BLUE, rasterized=True)
        ax.set_xlim(1.4, 3.4); ax.set_ylim(0.6, 3.7)
        ax.set_xlabel("Actual rate per mile, $")
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("quote_signal")
    ax = axes[3]
    colors = [BLUE if m in QS_COPY_MONTHS else ORANGE if m in QS_MIRROR_MONTHS else GRAY for m in tab.index]
    ax.bar(tab.index, tab["corr(qs, expected rpm)"], color=colors, width=0.7)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.axvline(10.5, color=INK_2, lw=1, ls="--")
    ax.text(10.65, 0.75, "validation\n(Nov-Dec)", color=INK_2, fontsize=8.5)
    ax.set_xticks(range(1, 13)); ax.set_xlabel("Month of 2025")
    ax.set_ylabel("corr(quote_signal, typical rate per mile)")
    ax.set_title("Nov-Dec match the noise month", fontsize=10.5)
    fig.suptitle("quote_signal changes meaning by month: it leaks the label in train and is noise in validation",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, "02_quote_signal_regimes.png")


def model_frame(train: pd.DataFrame) -> pd.DataFrame:
    """Clean rows with the minimal features used by the quick model checks."""
    c = train[~train.is_outlier].copy()
    c["mi"] = c.market_index.fillna(c.groupby("date").market_index.transform("mean"))
    c["aw"] = c.weight.abs()
    c["eq"] = c.equipment.map({"Dry Van": 0, "Reefer": 1, "Flatbed": 2})
    c["t"] = (c.dt - pd.Timestamp("2025-01-01")).dt.days
    c["y"] = np.log(c.posted_rate)
    return c


BASE_FEATURES = ["distance", "eq", "aw", "mi", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def quick_xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=SEED, n_jobs=4)


def leakage_demo(train):
    """Show what quote_signal does to a model when it turns into noise.

    A random 80/20 split inside Jan-Aug keeps the time drift (section 7) out of
    the comparison. Each model is scored twice: with quote_signal as given, and
    with quote_signal replaced by noise drawn like the validation file
    (mean 2.05, sd 0.22). The second score is what a submission would get."""
    c = model_frame(train)
    c = c[c.month <= 8]
    rng = np.random.default_rng(SEED)
    test_mask = rng.random(len(c)) < 0.2
    fit, test = c[~test_mask], c[test_mask]
    test_noise = test.copy()
    test_noise["quote_signal"] = rng.normal(2.05, 0.22, len(test))
    print("leakage demo (XGBoost 400 trees, depth 6, lr 0.05, log target; random 80/20 inside Jan-Aug, clean rows):")
    for name, cols in (("without quote_signal", BASE_FEATURES),
                       ("with quote_signal", BASE_FEATURES + ["quote_signal"]),
                       ("with quote_signal + month", BASE_FEATURES + ["quote_signal", "month"])):
        model = quick_xgb().fit(fit[cols], fit.y)
        scores = [f"as given {metrics(test.posted_rate, np.exp(model.predict(test[cols])))['MAPE%']:.2f}% MAPE"]
        if "quote_signal" in cols:
            noisy = metrics(test_noise.posted_rate, np.exp(model.predict(test_noise[cols])))["MAPE%"]
            scores.append(f"with validation-like quote_signal {noisy:.2f}% MAPE")
        print(f"  {name:26s} | " + " | ".join(scores))


# --------------------------------------------------------------------------
# 7. Time
# --------------------------------------------------------------------------
def section_time(train, val):
    header("7. TIME: MARKET_INDEX, WEEKDAY, MONTH, HOLIDAYS, TREND")
    c = train[~train.is_outlier]
    day = c.groupby("dt").agg(level=("log_ratio", "mean"), mi=("market_index", "mean"))
    day["lmi"] = np.log(day.mi)
    day["t"] = (day.index - pd.Timestamp("2025-01-01")).days
    print(f"daily rate level (mean log ratio to equipment x band median): sd {day.level.std():.4f}; "
          f"corr with daily market_index {day[['level', 'mi']].corr().iloc[0, 1]:.3f}")

    def ols(frame, cols):
        X = np.column_stack([np.ones(len(frame))] + [frame[k] for k in cols])
        beta, *_ = np.linalg.lstsq(X, frame.level, rcond=None)
        return beta

    def pred(frame, cols, beta):
        return np.column_stack([np.ones(len(frame))] + [frame[k] for k in cols]) @ beta

    for cols in (["lmi"], ["lmi", "t"]):
        beta = ols(day, cols)
        r = day.level - pred(day, cols, beta)
        print(f"  level ~ {' + '.join(cols):8s}: coefs {np.round(beta, 5).tolist()}, R2 {1 - r.var() / day.level.var():.3f}, "
              f"residual sd {r.std():.4f}, lag-1 autocorr of residual {r.autocorr(1):.2f}")
    beta = ols(day, ["lmi"])
    day["res"] = day.level - pred(day, ["lmi"], beta)
    print(f"  elasticity of rate to market_index ~ {beta[1]:.3f} "
          f"(a 10% higher index -> about {100 * (1.1 ** beta[1] - 1):.1f}% higher rate)")

    mon = day.groupby(day.index.month).agg(level=("level", "mean"), mi=("mi", "mean"), res=("res", "mean"))
    val_mi = val.groupby("month").market_index.mean()
    print("by month (train): level, market_index, level not explained by index:\n" + mon.round(4).to_string())
    print(f"validation monthly market_index: {val_mi.round(4).to_dict()} (train Sep-Oct "
          f"{train[train.month >= 9].market_index.mean():.4f}, train all {train.market_index.mean():.4f})")
    wd = day.groupby(day.index.dayofweek).agg(level=("level", "mean"), mi=("mi", "mean"), res=("res", "mean"))
    vwd = val.groupby("weekday").market_index.mean()
    wd["val_mi"] = vwd
    print("by weekday (0=Mon): level, train index, unexplained, val index:\n" + wd.round(4).to_string())
    hol = day.loc[pd.to_datetime(HOLIDAYS_2025), ["level", "mi", "res"]]
    print(f"holidays: mean unexplained level {hol.res.mean():+.4f} vs all days sd {day.res.std():.4f}\n"
          + hol.round(4).to_string())

    # Backtest: can a trend term predict next month's level? Daily-level regression.
    day["m"] = day.index.month
    print("next-month level error (mean log error) for three simple level models:")
    errs = {"index only": [], "index + linear trend": [], "index + last-30-day offset": []}
    for k in range(3, 10):
        a, b = day[day.m <= k], day[day.m == k + 1]
        e1 = (b.level - pred(b, ["lmi"], ols(a, ["lmi"]))).mean()
        e2 = (b.level - pred(b, ["lmi", "t"], ols(a, ["lmi", "t"]))).mean()
        ba = ols(a, ["lmi"]); last = a.iloc[-30:]
        off = (last.level - pred(last, ["lmi"], ba)).mean()
        e3 = (b.level - pred(b, ["lmi"], ba) - off).mean()
        for key, e in zip(errs, (e1, e2, e3)):
            errs[key].append(e)
        print(f"  train months 1-{k} -> month {k + 1}: {e1:+.4f} | {e2:+.4f} | {e3:+.4f}")
    for key, e in errs.items():
        print(f"  {key:28s}: mean |error| {np.mean(np.abs(e)):.4f}")
    jan_aug = day[day.m <= 8]
    so = day[day.m >= 9]
    for cols in (["lmi"], ["lmi", "t"]):
        e = so.level - pred(so, cols, ols(jan_aug, cols))
        print(f"  Jan-Aug -> Sep-Oct, level ~ {' + '.join(cols)}: mean error {e.mean():+.4f}")

    # Figure 4: index and rate level over time (two panels, shared x; no twin axes).
    allday = pd.concat([train, val]).groupby("dt").market_index.agg(["mean", "std"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True, gridspec_kw={"hspace": 0.18})
    ax = axes[0]
    tr_d, va_d = allday[allday.index <= "2025-10-31"], allday[allday.index >= "2025-11-01"]
    for d, col, lab in ((tr_d, BLUE, "train"), (va_d, ORANGE, "validation")):
        ax.plot(d.index, d["mean"], color=col, lw=1.2, label=f"{lab}: daily mean")
        ax.fill_between(d.index, d["mean"] - d["std"], d["mean"] + d["std"], color=col, alpha=0.15, lw=0)
    ax.plot(allday.index, allday["mean"].rolling(28, center=True, min_periods=7).mean(), color=INK, lw=1.6,
            label="28-day average")
    ax.set_ylabel("market_index")
    ax.set_title("market_index is one number per day (band = within-day sd) with a strong weekly cycle")
    ax.legend(loc="upper right", ncol=3)
    ax = axes[1]
    ax.plot(day.index, 100 * (np.exp(day.level) - 1), color=BLUE, lw=1, alpha=0.6, label="daily rate level")
    ax.plot(day.index, 100 * (np.exp(day.level.rolling(28, center=True, min_periods=7).mean()) - 1),
            color=INK, lw=1.6, label="28-day average")
    fitted = pred(day, ["lmi"], ols(day, ["lmi"]))
    ax.plot(day.index, 100 * (np.exp(pd.Series(fitted, index=day.index).rolling(28, center=True, min_periods=7).mean()) - 1),
            color=ORANGE, lw=1.6, ls="--", label="28-day average explained by market_index")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_ylabel("Rate vs typical, %")
    ax.set_title("Rates follow the index but drift upward: Sep-Oct sit 2-3% above what the index alone implies")
    ax.legend(loc="upper left", ncol=3)
    ax.set_xlabel("Date (2025)")
    savefig(fig, "04_market_index_and_rate_level.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x = np.arange(7)
    ax = axes[0]
    ax.plot(x, wd.mi, color=BLUE, marker="o", lw=2, label="train (Jan-Oct)")
    ax.plot(x, train[train.month >= 9].groupby("weekday").market_index.mean(), color=AQUA, marker="s", lw=2,
            label="train Sep-Oct")
    ax.plot(x, wd.val_mi, color=ORANGE, marker="o", lw=2, label="validation (Nov-Dec)")
    ax.set_xticks(x, names); ax.set_ylabel("Mean market_index")
    ax.set_title("Weekly cycle in the index; Nov-Dec look like Sep-Oct")
    ax.legend(loc="lower center")
    ax = axes[1]
    ax.plot(x, 100 * (np.exp(wd.level) - 1), color=BLUE, marker="o", lw=2, label="rate level")
    ax.plot(x, 100 * (np.exp(wd.res) - 1), color=GRAY, marker="o", lw=2, label="left after market_index")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(x, names); ax.set_ylabel("Rate vs typical, %")
    ax.set_title("The weekday effect in rates is carried by the index")
    ax.legend(loc="lower center")
    savefig(fig, "05_weekday_pattern.png")


# --------------------------------------------------------------------------
# 8. Geography
# --------------------------------------------------------------------------
def section_geography(train, val, unseen, city_table):
    header("8. GEOGRAPHY")
    c = train[~train.is_outlier].copy()
    # Rate per mile vs distance by equipment.
    tab = c.pivot_table(index="dist_band", columns="equipment", values="rpm", aggfunc="median", observed=True)
    tab["Reefer/Dry Van"] = tab["Reefer"] / tab["Dry Van"]
    tab["Flatbed/Dry Van"] = tab["Flatbed"] / tab["Dry Van"]
    print("median rate per mile by distance band:\n" + tab.round(3).to_string())

    # City effect: residual of log rate per mile after a smooth distance curve
    # (cubic in log distance) with equipment shifts, the day's level and a weight
    # curve. Band medians would leave within-band distance slope in the residual.
    # Each load has a pickup and a delivery end; both are fit jointly by least squares.
    ld = np.log(c.distance)
    X = np.column_stack([np.ones(len(c)), ld, ld ** 2, ld ** 3,
                         c.equipment.eq("Reefer"), c.equipment.eq("Flatbed")]).astype(float)
    beta, *_ = np.linalg.lstsq(X, np.log(c.rpm), rcond=None)
    c["res"] = np.log(c.rpm) - X @ beta
    c["res"] -= c.groupby("date").res.transform("mean")
    aw = c.weight.abs().fillna(c.weight.abs().median())
    c["res"] -= np.polyval(np.polyfit(aw, c.res, 3), aw)
    print(f"residual sd after smooth distance curve, equipment, day and weight: {c.res.std():.4f}")
    cities = sorted(set(c.pickup) | set(c.delivery))
    idx = {k: i for i, k in enumerate(cities)}
    M = np.zeros((len(c), len(cities)))
    M[np.arange(len(c)), c.pickup.map(idx)] += 1
    M[np.arange(len(c)), c.delivery.map(idx)] += 1
    eff, *_ = np.linalg.lstsq(M, c.res.to_numpy(), rcond=None)
    ce = pd.DataFrame({"effect": eff}, index=cities).join(city_table)
    pick = c.groupby("pickup").res.mean()
    deliv = c.groupby("delivery").res.mean()
    print(f"city effect (per end): sd {ce.effect.std():.4f}, range {ce.effect.min():+.4f}..{ce.effect.max():+.4f}; "
          f"corr with latitude {ce[['effect', 'lat']].corr().iloc[0, 1]:+.3f}, longitude "
          f"{ce[['effect', 'lon']].corr().iloc[0, 1]:+.3f}")
    print(f"same city as pickup vs as delivery: corr {np.corrcoef(pick[cities], deliv[cities])[0, 1]:.3f} "
          f"(no head-haul / back-haul asymmetry)")
    print("cheapest cities: " + ", ".join(f"{k} {v:+.3f}" for k, v in ce.effect.nsmallest(5).items()))
    print("dearest cities:  " + ", ".join(f"{k} {v:+.3f}" for k, v in ce.effect.nlargest(5).items()))
    dlat, dlon = c.delivery_lat - c.pickup_lat, c.delivery_lon - c.pickup_lon
    c["direction"] = np.where(dlon.abs() > dlat.abs(), np.where(dlon > 0, "east", "west"),
                              np.where(dlat > 0, "north", "south"))
    print(f"mean residual by direction: {c.groupby('direction').res.mean().round(4).to_dict()}")
    lane = c.groupby(["pickup", "delivery"]).res.mean()
    pairs = [(a, b) for a, b in lane.index if a < b and (b, a) in lane.index]
    diff = np.array([lane[(a, b)] - lane[(b, a)] for a, b in pairs])
    print(f"A->B minus B->A residual over {len(pairs)} lane pairs: mean {diff.mean():+.4f}, sd {diff.std():.4f}")
    # Is there a lane effect beyond the two city effects?
    c["res_city"] = c.res - M @ eff
    lane_g = c.groupby(["pickup", "delivery"]).res_city.agg(["mean", "size"])
    lane_g = lane_g[lane_g["size"] >= 10]
    print(f"residual sd after city effects too: {c.res_city.std():.4f}")
    print(f"lanes with >= 10 loads: {len(lane_g)}; sd of lane mean after city effects {lane_g['mean'].std():.4f} "
          f"vs {(c.res_city.std() / np.sqrt(lane_g['size'])).mean():.4f} expected from sampling noise alone "
          f"-> real lane spread about {np.sqrt(max(lane_g['mean'].var() - ((c.res_city.std() / np.sqrt(lane_g['size'])) ** 2).mean(), 0)):.4f}")
    print(f"validation rows on a training lane with >= 10 loads: "
          f"{val.set_index(['pickup', 'delivery']).index.isin(lane_g.index).mean():.1%}")

    # Can lat/lon predict the effect of a city we have never seen? Leave one city out.
    def quad(d):
        return np.column_stack([np.ones(len(d)), d.lat, d.lon, d.lat ** 2, d.lon ** 2, d.lat * d.lon])
    e_q, e_nn, e_0 = [], [], []
    for k in ce.index:
        rest, one = ce.drop(k), ce.loc[[k]]
        bq, *_ = np.linalg.lstsq(quad(rest), rest.effect, rcond=None)
        e_q.append(one.effect.iloc[0] - (quad(one) @ bq)[0])
        dist = np.hypot(rest.lat - one.lat.iloc[0], rest.lon - one.lon.iloc[0])
        e_nn.append(one.effect.iloc[0] - rest.loc[dist.nsmallest(3).index, "effect"].mean())
        e_0.append(one.effect.iloc[0] - rest.effect.mean())
    rmse = lambda e: np.sqrt(np.mean(np.square(e)))
    print(f"leave-one-city-out RMSE of city effect: quadratic lat/lon {rmse(e_q):.4f}, "
          f"3 nearest cities {rmse(e_nn):.4f}, overall mean {rmse(e_0):.4f}")
    print(f"train latitude range {ce.lat.min():.2f}..{ce.lat.max():.2f}, longitude {ce.lon.min():.2f}..{ce.lon.max():.2f}")
    print("unseen validation cities, their coordinates and 3 nearest training cities (degrees):")
    for k in unseen:
        la, lo = city_table.loc[k]
        dist = np.hypot(ce.lat - la, ce.lon - lo).nsmallest(3)
        print(f"  {k:10s} ({la:6.2f}, {lo:8.2f}) rows {int((val.pickup == k).sum() + (val.delivery == k).sum()):3d} -> "
              + ", ".join(f"{n} {v:.2f}" for n, v in dist.items()))

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for e, col in EQUIP_COLORS.items():
        g = c[c.equipment == e].groupby("dist_band", observed=True).agg(d=("distance", "median"), r=("rpm", "median"))
        ax.plot(g.d, g.r, color=col, lw=2, marker="o", ms=6, label=e)
    ax.set_xscale("log")
    ax.set_xlabel("Distance, miles (log scale; band medians)")
    ax.set_ylabel("Median rate per mile, $")
    ax.set_title("Rate per mile falls with distance; Reefer is ~13% and Flatbed ~8% above Dry Van at every distance")
    ax.legend(loc="upper right")
    savefig(fig, "06_rate_per_mile_by_distance.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    lim = np.abs(ce.effect).max()
    sc = ax.scatter(ce.lon, ce.lat, c=100 * ce.effect, cmap="RdBu_r", vmin=-100 * lim, vmax=100 * lim,
                    s=110, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for k, r in ce.iterrows():
        ax.annotate(k, (r.lon, r.lat), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK_2)
    un = city_table.loc[unseen]
    ax.scatter(un.lon, un.lat, s=130, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=4,
               label="validation-only city (no training rows)")
    for k, r in un.iterrows():
        ax.annotate(k, (r.lon, r.lat), xytext=(4, -10), textcoords="offset points", fontsize=8,
                    color=INK, fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("City effect on rate, % (each end of the load)")
    ax.set_xlabel("Longitude (as given)"); ax.set_ylabel("Latitude (as given)")
    ax.set_title("City effects are smooth in space (south dearer, north cheaper), so lat/lon can cover new cities")
    ax.legend(loc="upper left")
    savefig(fig, "07_city_effects_map.png")


# --------------------------------------------------------------------------
# 9. December chart
# --------------------------------------------------------------------------
def section_december(train, val, dec, city_table):
    header("9. DECEMBER CHART: LEXINGTON -> FORT WAYNE, DRY VAN, 360 MI, 32,000 LB")
    lane = train[(train.pickup == "Lexington") & (train.delivery == "Fort Wayne")]
    dv = lane[lane.equipment == "Dry Van"]
    print(f"train loads on the lane: {len(lane)} ({lane.equipment.value_counts().to_dict()}); "
          f"reverse lane: {((train.pickup == 'Fort Wayne') & (train.delivery == 'Lexington')).sum()}")
    print(f"Dry Van loads: {len(dv)}, flagged outliers {int(dv.is_outlier.sum())}, rate ${dv.posted_rate.min():.2f}.."
          f"${dv.posted_rate.max():.2f}, median ${dv.posted_rate.median():.2f}, "
          f"median rate per mile ${dv.rpm.median():.3f}, distance {dv.distance.min()}..{dv.distance.max()}")
    print(dv[["date", "distance", "weight", "market_index", "posted_rate"]].to_string(index=False))
    so = dv[dv.month >= 9]
    print(f"Sep-Oct Dry Van loads on the lane: {len(so)}, mean rate ${so.posted_rate.mean():.2f}, "
          f"mean rate scaled to 360 mi ${(so.posted_rate * (360 / so.distance) ** 0.87).mean():.2f}")
    vl = val[(val.pickup == "Lexington") & (val.delivery == "Fort Wayne")]
    print(f"validation loads on the lane: {len(vl)} ({vl.equipment.value_counts().to_dict()})")
    print(f"coordinates from the city table: Lexington {tuple(city_table.loc['Lexington'])}, "
          f"Fort Wayne {tuple(city_table.loc['Fort Wayne'])}")
    d = val[val.month == 12].groupby("date").agg(n=("load_id", "size"), mi_n=("market_index", "count"),
                                                 market_index=("market_index", "mean"),
                                                 quote_signal=("quote_signal", "mean"))
    print(f"December dates in validation: {len(d)} of 31; fewest loads with market_index on a day: {d.mi_n.min()}")
    print(f"December daily market_index from validation: min {d.market_index.min():.4f}, max {d.market_index.max():.4f}, "
          f"mean {d.market_index.mean():.4f}; daily quote_signal mean range "
          f"{d.quote_signal.min():.3f}..{d.quote_signal.max():.3f}")
    print(d.round(4).to_string())
    print(f"December file dates match validation December dates: {set(dec.date) == set(d.index)}")
    # Section 7 found a rate elasticity to market_index of about 0.139.
    swing = 0.139 * np.log(d.market_index.max() / d.market_index.min())
    print(f"expected weekly swing on the chart from market_index alone: about {100 * swing:.1f}% peak to trough "
          f"(~${swing * dv.posted_rate.median():.0f} on a ${dv.posted_rate.median():.0f} load)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.scatter(dv.dt, dv.posted_rate, color=BLUE, s=36, zorder=3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Posted rate, $"); ax.set_xlabel("Date (2025)")
    ax.set_title(f"Lexington to Fort Wayne, Dry Van: {len(dv)} training loads")
    ax = axes[1]
    dd = pd.to_datetime(d.index)
    ax.plot(dd, d.market_index, color=ORANGE, marker="o", ms=4, lw=1.8)
    ax.set_ylabel("Daily mean market_index (validation)")
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 5, 9, 13, 17, 21, 25, 29]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    ax.set_xlim(dd.min() - pd.Timedelta(days=1), dd.max() + pd.Timedelta(days=1))
    ax.set_xlabel("Day of December 2025")
    ax.set_title("December inputs: the index cycles weekly")
    savefig(fig, "09_december_lane.png")


# --------------------------------------------------------------------------
# 10. Validation design and baselines
# --------------------------------------------------------------------------
def section_validation(train, val):
    header("10. VALIDATION DESIGN AND BASELINES")
    tr_all, te_all = train[train.month <= 8], train[train.month >= 9]
    tr = tr_all[~tr_all.is_outlier]
    print(f"main holdout: train Jan-Aug {len(tr_all):,} rows ({len(tr):,} after dropping {tr_all.is_outlier.sum()} outliers); "
          f"test Sep-Oct {len(te_all):,} rows, of which {te_all.is_outlier.sum()} flagged")

    def report(name, pred_fn, fit_rows, test_rows):
        p = pred_fn(fit_rows, test_rows)
        clean = ~test_rows.is_outlier.to_numpy()
        print(f"  {name:44s} all rows   {fmt_metrics(metrics(test_rows.posted_rate, p))}")
        print(f"  {'':44s} clean rows {fmt_metrics(metrics(test_rows.posted_rate[clean], p[clean]))}")

    def global_median(fit, test):
        return fit.rpm.median() * test.distance.to_numpy()

    def band_median(fit, test):
        t = expected_rpm_table(fit)
        return attach_expected(test.drop(columns=["rpm"]), t).exp_rpm.to_numpy() * test.distance.to_numpy()

    for k in range(4, 8):
        a = train[(train.month <= k) & ~train.is_outlier]
        b = train[train.month == k + 1]
        clean = ~b.is_outlier.to_numpy()
        m_c = metrics(b.posted_rate[clean], band_median(a, b)[clean])
        print(f"  CV fold {k - 3}: train months 1-{k} ({len(a):,} clean rows) -> test month {k + 1} "
              f"({len(b):,} rows, {int(b.is_outlier.sum())} flagged) | B1 clean MAE ${m_c['MAE']:.2f}, "
              f"MAPE {m_c['MAPE%']:.2f}%")
    print("baselines fitted on clean Jan-Aug, scored on Sep-Oct:")
    report("B0 global median $/mile x distance", global_median, tr, te_all)
    report("B1 median $/mile by equipment x band x dist", band_median, tr, te_all)

    # Unseen-city holdout: the 8 lowest-volume training cities.
    vol = (train.pickup.value_counts() + train.delivery.value_counts()).sort_values()
    hold = sorted(vol.index[:HOLDOUT_CITIES_N])
    print(f"unseen-city holdout cities (8 lowest volume): {hold}")
    print(f"  their volumes: {vol[hold].to_dict()} (validation-only cities have 170-195 rows each)")
    touch_tr = tr_all.pickup.isin(hold) | tr_all.delivery.isin(hold)
    touch_te = te_all.pickup.isin(hold) | te_all.delivery.isin(hold)
    both_te = te_all.pickup.isin(hold) & te_all.delivery.isin(hold)
    print(f"  Jan-Aug rows removed from training: {touch_tr.sum():,}; Sep-Oct test rows touching them: "
          f"{touch_te.sum():,} (both ends: {both_te.sum()}); Sep-Oct rows with seen cities: {(~touch_te).sum():,}")
    print(f"  validation share touching an unseen city is 12.1%; this holdout's test share is {touch_te.mean():.1%}")
    tr_u = tr_all[~touch_tr & ~tr_all.is_outlier]
    report("B1 on Sep-Oct rows touching held-out cities", band_median, tr_u, te_all[touch_te])

    level_drift_demo(train)


def level_drift_demo(train):
    """The biggest error source is the month-level drift (section 7). Compare
    three cheap ways to handle it with the same quick model on every
    expanding-window fold, clean rows only:
      plain     - no time handling
      recency   - sample weight 0.5 ** (days before the last training day / 30)
      detrend   - a linear model (log distance, its square, equipment, log index,
                  weight, day number) gives a slope per day; the trees fit
                  log(rate) minus slope * day, and the slope * day is added back.
    Exploratory only: the ML Engineer re-runs the choice inside the CV spec."""
    c = model_frame(train)
    c["aw"] = c.aw.fillna(c.aw.median())

    def linear_slope(f):
        X = np.column_stack([np.ones(len(f)), np.log(f.distance), np.log(f.distance) ** 2,
                             f["eq"].eq(1), f["eq"].eq(2), np.log(f.mi), f.aw / 1e4, f.t]).astype(float)
        beta, *_ = np.linalg.lstsq(X, f.y, rcond=None)
        return beta[-1]

    print("level drift check (quick XGBoost, clean rows): MAPE % (median signed error %)")
    totals = {"plain": [], "recency": [], "detrend": []}
    for k in range(4, 9):
        fit = c[c.month <= k]
        test = c[c.month == k + 1] if k < 8 else c[c.month >= 9]
        out = []
        for name in totals:
            w = 0.5 ** ((fit.t.max() - fit.t) / 30) if name == "recency" else None
            slope = linear_slope(fit) if name == "detrend" else 0.0
            model = quick_xgb().fit(fit[BASE_FEATURES], fit.y - slope * fit.t, sample_weight=w)
            pred = np.exp(model.predict(test[BASE_FEATURES]) + slope * test.t)
            err = pred / test.posted_rate - 1
            totals[name].append(100 * np.mean(np.abs(err)))
            out.append(f"{name} {100 * np.mean(np.abs(err)):.2f} ({100 * np.median(err):+.2f})"
                       + (f" slope/day {slope:.5f}" if name == "detrend" else ""))
        label = f"month {k + 1}" if k < 8 else "Sep-Oct"
        print(f"  months 1-{k} -> {label:8s}: " + " | ".join(out))
    print("  mean over the 5 folds: " + ", ".join(f"{k} {np.mean(v):.2f}%" for k, v in totals.items()))


# --------------------------------------------------------------------------
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    train, val, dec = load()

    unseen, city_table = section_integrity(train, val, dec)
    train = section_outliers(train)          # adds exp_rpm, log_ratio, is_outlier
    section_distance(train, val)
    section_weight(train, val)
    section_missing(train, val)
    section_quote_signal(train, val)
    section_time(train, val)
    section_geography(train, val, unseen, city_table)
    section_december(train, val, dec, city_table)
    section_validation(train, val)
    print("\nDone. Figures saved to reports/figures/.")


if __name__ == "__main__":
    main()
