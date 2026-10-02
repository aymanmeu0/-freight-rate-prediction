"""Final fit and the two submission files.

``run_final(spec)``:

1. fits the Cleaner on all of train (C3, C5) and the model on all clean Jan-Oct rows;
2. predicts the 12,000 validation rows and writes ``validation_predictions.csv``
   with the pipeline writer, plus a copy in ``outputs/``;
3. predicts the 31 December chart rows and writes them with
   ``data.write_december_predictions`` (``data/december_chart_inputs.csv`` and
   ``outputs/december_predictions.csv``);
4. runs sanity checks (hard failures raise, soft ones are reported) and adds a
   ``final`` section to ``reports/metrics.json``.

No work at import time.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg
from src import data
from src import features as feat
from src import train as T
from src.model import ModelSpec, RateModel

VALIDATION_COPY_PATH = cfg.OUTPUTS_DIR / "validation_predictions.csv"

# December lane history (findings section 6): 21 Dry Van loads, $757.93 to $934.37.
DEC_LANE = ("Lexington", "Fort Wayne", "Dry Van")


def fit_final(spec: ModelSpec, ds: data.Dataset) -> tuple[RateModel, data.Cleaner, pd.DataFrame]:
    """Cleaner fit on all of train; model fit on all clean Jan-Oct rows."""
    cleaner = data.Cleaner(ds.market_index_table).fit(ds.train)
    train_clean = cleaner.transform(ds.train, drop_outliers=True, name="train (final fit)")
    return RateModel(spec).fit(train_clean), cleaner, train_clean


def predict_validation(model: RateModel, cleaner: data.Cleaner, ds: data.Dataset):
    val = cleaner.transform(ds.validation, name="validation (final)")
    return val, model.predict(val)


def predict_december(model: RateModel, cleaner: data.Cleaner, ds: data.Dataset):
    dec = cleaner.transform(data.build_december_frame(ds.city_table, ds.validation), name="december")
    return dec, model.predict(dec)


def _summary(label: str, v) -> dict:
    v = np.asarray(v, float)
    return {"label": label, "n": int(len(v)), "min": float(v.min()), "p05": float(np.percentile(v, 5)),
            "median": float(np.median(v)), "mean": float(v.mean()), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def sanity_checks(train_clean: pd.DataFrame, val: pd.DataFrame, val_pred: np.ndarray,
                  dec: pd.DataFrame, dec_pred: np.ndarray, log=print) -> dict:
    """Hard checks raise; soft checks are logged and returned."""
    for label, p, n in (("validation", val_pred, cfg.N_VALIDATION_ROWS), ("December", dec_pred, cfg.N_DECEMBER_ROWS)):
        if len(p) != n or not np.isfinite(p).all() or (p <= 0).any():
            raise AssertionError(f"{label} predictions must be {n} finite positive values")
    out: dict = {}

    # Validation vs train: rate per mile by equipment, Nov and Dec vs Sep-Oct actuals.
    rpm_pred = val_pred / val["distance"].to_numpy()
    sep_oct = train_clean[train_clean["date"] >= cfg.HOLDOUT_TEST_DATES[0]]
    rows = []
    for e in cfg.EQUIPMENT:
        r = {"equipment": e,
             "train_all_median_rpm": float((train_clean[cfg.TARGET] / train_clean["distance"])[train_clean["equipment"] == e].median()),
             "sep_oct_median_rpm": float((sep_oct[cfg.TARGET] / sep_oct["distance"])[sep_oct["equipment"] == e].median())}
        for m, lab in ((11, "nov"), (12, "dec")):
            sel = (val["equipment"].to_numpy() == e) & (val["date"].dt.month.to_numpy() == m)
            r[f"pred_{lab}_median_rpm"] = float(np.median(rpm_pred[sel]))
        rows.append(r)
    out["rpm_by_equipment"] = rows
    log("  median rate per mile by equipment (train Jan-Oct actual | Sep-Oct actual | predicted Nov | predicted Dec):")
    for r in rows:
        log(f"    {r['equipment']:8s} {r['train_all_median_rpm']:.3f} | {r['sep_oct_median_rpm']:.3f} | "
            f"{r['pred_nov_median_rpm']:.3f} | {r['pred_dec_median_rpm']:.3f}")

    # Same comparison at fixed mix: a B1-style ratio (prediction / Sep-Oct cell median x distance).
    b1 = T.B1Baseline().fit(sep_oct)
    ratio = val_pred / b1.predict(val)
    month = val["date"].dt.month.to_numpy()
    out["ratio_to_sep_oct_cell_median"] = {"nov": float(np.median(ratio[month == 11])),
                                           "dec": float(np.median(ratio[month == 12]))}
    log(f"  median prediction / Sep-Oct (equipment x band) median: Nov {out['ratio_to_sep_oct_cell_median']['nov']:.4f}, "
        f"Dec {out['ratio_to_sep_oct_cell_median']['dec']:.4f}")

    # Rows touching a city never seen in training.
    seen = set(train_clean["pickup"]) | set(train_clean["delivery"])
    new_city = ~(val["pickup"].isin(seen) & val["delivery"].isin(seen)).to_numpy()
    out["unseen_city_rows"] = int(new_city.sum())
    out["unseen_cities"] = sorted((set(val["pickup"]) | set(val["delivery"])) - seen)
    out["ratio_unseen_city_rows"] = float(np.median(ratio[new_city]))
    out["ratio_seen_city_rows"] = float(np.median(ratio[~new_city]))
    log(f"  rows touching an unseen city: {new_city.sum():,} ({', '.join(out['unseen_cities'])}); "
        f"median ratio to cell median {out['ratio_unseen_city_rows']:.4f} vs {out['ratio_seen_city_rows']:.4f} for the rest")

    # December: near the lane history and the expected level, no jumps.
    lane = train_clean[(train_clean["pickup"] == DEC_LANE[0]) & (train_clean["delivery"] == DEC_LANE[1])
                       & (train_clean["equipment"] == DEC_LANE[2])]
    lo, hi = float(lane[cfg.TARGET].min()), float(lane[cfg.TARGET].max())
    jumps = np.abs(np.diff(np.log(dec_pred)))
    out["december"] = {"lane_history_n": int(len(lane)), "lane_history_min": lo, "lane_history_max": hi,
                       "lane_history_median": float(lane[cfg.TARGET].median()),
                       "all_inside_history_range": bool(((dec_pred >= lo) & (dec_pred <= hi)).all()),
                       "largest_day_to_day_change_pct": float(100 * (np.exp(jumps.max()) - 1)),
                       "first_week_mean": float(dec_pred[:7].mean()), "last_week_mean": float(dec_pred[-7:].mean())}
    if not (0.8 * lo <= dec_pred.min() and dec_pred.max() <= 1.2 * hi):
        raise AssertionError(f"December predictions {dec_pred.min():.2f}..{dec_pred.max():.2f} are far "
                             f"outside the lane history {lo:.2f}..{hi:.2f}")
    d = out["december"]
    log(f"  December: lane history {d['lane_history_n']} loads ${lo:.2f}..${hi:.2f} (median ${d['lane_history_median']:.2f}); "
        f"inside range: {d['all_inside_history_range']}; largest day-to-day change {d['largest_day_to_day_change_pct']:.2f}%")
    if d["largest_day_to_day_change_pct"] > 5:
        log("  NOTE: a day-to-day change above 5%; inspect the December chart")
    return out


def december_decomposition(model: RateModel, train_clean: pd.DataFrame, dec: pd.DataFrame,
                           ds: data.Dataset, log=print) -> dict:
    """Where the December level comes from: the same lane row priced on past days, the lane's
    past loads as predicted in-sample, and the drift and lane terms in December."""
    lane = train_clean[(train_clean["pickup"] == DEC_LANE[0]) & (train_clean["delivery"] == DEC_LANE[1])
                       & (train_clean["equipment"] == DEC_LANE[2])]
    in_sample = float(np.median(lane[cfg.TARGET].to_numpy() / model.predict(lane)))
    # The December row (360 mi, 32,000 lb) priced on each Sep-Oct day with that day's mean index.
    mi = ds.market_index_table
    days = mi[(mi.index >= cfg.HOLDOUT_TEST_DATES[0]) & (mi.index <= cfg.HOLDOUT_TEST_DATES[1])]
    rows = pd.concat([dec.iloc[[0]]] * len(days), ignore_index=True)
    rows["date"], rows["market_index"] = days.index, days.to_numpy()
    p = model.predict(rows)
    month = rows["date"].dt.month.to_numpy()
    out = {"lane_actual_over_predicted_median_in_sample": in_sample,
           "same_row_on_sep_days_mean": float(p[month == 9].mean()),
           "same_row_on_oct_days_mean": float(p[month == 10].mean()),
           "lane_actual_sep_oct_weights": lane.loc[lane["date"] >= cfg.HOLDOUT_TEST_DATES[0], "weight"].tolist()}
    if model.drift_ is not None:
        off = model.drift_.offset(dec["date"], damp=model.spec.damp)
        at_end = model.drift_.slope_ * model.drift_.t_end_
        out["drift_pct_oct31"] = float(100 * (np.exp(at_end) - 1))
        out["drift_pct_dec1"] = float(100 * (np.exp(off[0]) - 1))
        out["drift_pct_dec31"] = float(100 * (np.exp(off[-1]) - 1))
    if model.lane_table_ is not None:
        key = feat.lane_key(dec.iloc[[0]], directed=model.spec.lane == "directed").iloc[0]
        out["lane_correction_pct"] = float(100 * (np.exp(model.lane_table_.get(key, 0.0)) - 1))
    log(f"  December decomposition: the Dec row priced on Sep-Oct days averages ${out['same_row_on_sep_days_mean']:.2f} "
        f"(Sep) and ${out['same_row_on_oct_days_mean']:.2f} (Oct); lane loads actual/predicted in-sample "
        f"median {in_sample:.4f}; drift level vs Jan 1: Oct 31 {out.get('drift_pct_oct31', 0):+.2f}%, "
        f"Dec 1 {out.get('drift_pct_dec1', 0):+.2f}%, Dec 31 {out.get('drift_pct_dec31', 0):+.2f}%; "
        f"lane correction {out.get('lane_correction_pct', 0):+.2f}%")
    return out


def fig_december(train_clean: pd.DataFrame, dec: pd.DataFrame, dec_pred: np.ndarray, log=print) -> None:
    """December predictions next to the lane's training loads (scaled to 360 miles)."""
    import matplotlib.pyplot as plt
    lane = train_clean[(train_clean["pickup"] == DEC_LANE[0]) & (train_clean["delivery"] == DEC_LANE[1])
                       & (train_clean["equipment"] == DEC_LANE[2])]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.scatter(lane["date"], lane[cfg.TARGET] / lane["distance"] * 360, s=40, color=T.GRAY,
               label=f"Training loads on this lane, Dry Van ({len(lane)}), scaled to 360 mi", zorder=3)
    ax.plot(dec["date"], dec_pred, color=T.BLUE, linewidth=2, marker="o", markersize=4,
            label="Model prediction, December", zorder=4)
    ax.set_ylabel("Rate ($)", color=T.INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    T._axes(ax, "December chart lane: Lexington to Fort Wayne",
            "360 miles, Dry Van, 32,000 lb. Gray: past loads; blue: model, one load per December day")
    T._save(fig, "14_december_vs_history.png", log)


def run_final(spec: ModelSpec, ds: data.Dataset | None = None, write: bool = True, log=print) -> dict:
    """Final fit, predictions, writes and checks. Returns the ``final`` metrics section."""
    ds = data.load_all() if ds is None else ds
    model, cleaner, train_clean = fit_final(spec, ds)
    log(f"Final fit: {len(train_clean):,} clean Jan-Oct rows; {cleaner.describe()}")
    d = model.describe()
    if model.drift_ is not None:
        log(f"  drift: slope {d['drift_slope_per_day']:.6f} per day, quarter-end ramp {d['drift_ramp']:.4f}, "
            f"damp {spec.damp:g}")
    val, val_pred = predict_validation(model, cleaner, ds)
    dec, dec_pred = predict_december(model, cleaner, ds)

    checks = sanity_checks(train_clean, val, val_pred, dec, dec_pred, log=log)
    checks["december_decomposition"] = december_decomposition(model, train_clean, dec, ds, log=log)
    month = val["date"].dt.month.to_numpy()
    summaries = {"validation": _summary("Validation, all 12,000 rows", val_pred),
                 "validation_nov": _summary("Validation, November", val_pred[month == 11]),
                 "validation_dec": _summary("Validation, December", val_pred[month == 12]),
                 "december_chart": _summary("December chart, 31 days", dec_pred),
                 "train_clean_actual": _summary("Train clean Jan-Oct, actual", train_clean[cfg.TARGET])}
    for s in summaries.values():
        log(f"  {s['label']:32s} min ${s['min']:8.2f} median ${s['median']:8.2f} mean ${s['mean']:8.2f} "
            f"max ${s['max']:8.2f}")

    dec_rows = [{"date": str(r.date.date()), "weekday": r.date.day_name()[:3], "market_index": float(r.market_index),
                 "drift_offset_pct": float(100 * (np.exp(o) - 1)), "predicted_rate": float(p)}
                for r, o, p in zip(dec.itertuples(), model.drift_.offset(dec["date"], damp=spec.damp)
                                   if model.drift_ is not None else np.zeros(len(dec)), dec_pred)]
    log("  December by day: " + ", ".join(f"{r['date'][-2:]} {r['weekday']} ${r['predicted_rate']:.0f}"
                                          for r in dec_rows))

    if write:
        p1 = data.write_validation_predictions(val["load_id"], val_pred)
        cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p1, VALIDATION_COPY_PATH)
        p2 = data.write_december_predictions(dec_pred)
        log(f"  wrote {p1.relative_to(cfg.ROOT)}, {VALIDATION_COPY_PATH.relative_to(cfg.ROOT)}, "
            f"{p2.relative_to(cfg.ROOT)}, {cfg.DECEMBER_PREDICTIONS_COPY_PATH.relative_to(cfg.ROOT)}")
        fig_december(train_clean, dec, dec_pred, log=log)

    final = {"spec": spec.as_dict(), "fit_rows": len(train_clean), "model": d,
             "summaries": summaries, "checks": checks, "december_rows": dec_rows}
    if write:
        m = T.load_metrics()
        m["final"] = final
        T.save_metrics(m)
        T.write_results_md(m)
    return final
