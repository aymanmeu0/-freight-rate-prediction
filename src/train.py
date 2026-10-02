"""Experiment harness: CV selection, main holdout, unseen-city check, reports.

Split spec (docs/eda_findings.md section 5):

* CV, for selection only: expanding window inside Jan-Aug,
  months 1-4 -> May, 1-5 -> Jun, 1-6 -> Jul, 1-7 -> Aug. The Cleaner is re-fit
  on each fold's fit rows only. Selection metric: mean clean MAPE over the four
  folds, clean MAE as the tie-break.
* Main holdout, scored once after selection: fit on clean Jan-Aug, test Sep-Oct.
* Unseen-city check: drop the 8 named cities' Jan-Aug rows from training and
  test on the Sep-Oct rows that touch them.

Entry points: :func:`select`, :func:`run_holdout`, :func:`run_unseen_city` and
:func:`run_evaluation` (all of them plus ``reports/metrics.json``,
``reports/model_results.md`` and the figures). No work at import time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg
from src import data
from src import features as feat
from src import model as mdl
from src.model import B1Baseline, ModelSpec, RateModel, make_model

REPORTS_DIR = cfg.REPORTS_DIR
FIG_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
RESULTS_MD_PATH = REPORTS_DIR / "model_results.md"
FOLD_LABELS = {1: "May", 2: "Jun", 3: "Jul", 4: "Aug"}


# ==========================================================================
# Metrics
# ==========================================================================
def metrics(y_true, y_pred) -> dict:
    """MAE ($), MAPE (%), RMSE ($), median signed % error (bias) and row count."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {"MAE": float(np.mean(np.abs(err))),
            "MAPE": float(100 * np.mean(np.abs(err) / y_true)),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "median_signed_pct": float(100 * np.median(err / y_true)),
            "n": int(len(y_true))}


def score_rows(test_rows: pd.DataFrame, pred) -> dict:
    """Metrics on clean rows (``is_outlier == False``) and on all rows."""
    pred = np.asarray(pred, float)
    clean = ~test_rows["is_outlier"].to_numpy()
    y = test_rows[cfg.TARGET].to_numpy(float)
    return {"clean": metrics(y[clean], pred[clean]), "all": metrics(y, pred)}


# ==========================================================================
# Splits
# ==========================================================================
class Split(NamedTuple):
    name: str
    fit: pd.DataFrame     # cleaned, outliers dropped
    test: pd.DataFrame    # cleaned, all rows, with is_outlier


def make_split(name: str, fit_raw: pd.DataFrame, test_raw: pd.DataFrame,
               mi_table: pd.Series) -> Split:
    """Fit the Cleaner on ``fit_raw`` only, then clean both sides."""
    cleaner = data.Cleaner(mi_table).fit(fit_raw)
    return Split(name, cleaner.transform(fit_raw, drop_outliers=True, name=f"{name} fit"),
                 cleaner.transform(test_raw, name=f"{name} test"))


def cv_splits(ds: data.Dataset) -> list[Split]:
    """The four expanding-window folds inside Jan-Aug."""
    month = ds.train["date"].dt.month
    return [make_split(f"fold{i + 1} 1-{last}->{test}", ds.train[month <= last], ds.train[month == test],
                       ds.market_index_table)
            for i, (last, test) in enumerate(cfg.CV_FOLDS)]


def holdout_split(ds: data.Dataset) -> Split:
    d = ds.train["date"]
    return make_split("holdout Jan-Aug->Sep-Oct", ds.train[d <= cfg.HOLDOUT_FIT_DATES[1]],
                      ds.train[d >= cfg.HOLDOUT_TEST_DATES[0]], ds.market_index_table)


def touches(df: pd.DataFrame, cities) -> pd.Series:
    return df["pickup"].isin(cities) | df["delivery"].isin(cities)


def unseen_city_split(ds: data.Dataset) -> Split:
    """Fit on Jan-Aug rows with neither end in the 8 held-out cities; test on Sep-Oct rows touching them.

    The Cleaner is fit on the reduced Jan-Aug rows, so the held-out cities leave no trace in training.
    """
    d = ds.train["date"]
    jan_aug, sep_oct = ds.train[d <= cfg.HOLDOUT_FIT_DATES[1]], ds.train[d >= cfg.HOLDOUT_TEST_DATES[0]]
    cities = cfg.UNSEEN_CITY_HOLDOUT
    return make_split("unseen-city", jan_aug[~touches(jan_aug, cities)], sep_oct[touches(sep_oct, cities)],
                      ds.market_index_table)


# ==========================================================================
# Candidates and staged selection
# ==========================================================================
# Selection runs in stages. Each stage varies one thing around the winner of
# the stage before. The winner is the XGBoost setup with the lowest mean clean
# MAPE over the four CV folds (rounded to 0.001 points; clean MAE breaks ties).
# B1 and the ridge models are scored for comparison only. The Sep-Oct holdout
# is never used here.
PLAIN = ModelSpec("", learner="xgb", features="geo", target="log_rate")


def named(spec):
    return spec if spec == "B1" else spec.with_(name=spec.label())


def stage_drift() -> list:
    ramp = PLAIN.with_(drift="linear_ramp")
    return ["B1",
            PLAIN.with_(learner="ridge"),
            ramp.with_(learner="ridge", damp=0.5),
            PLAIN,
            PLAIN.with_(half_life=30.0),
            *[PLAIN.with_(drift="linear", damp=d) for d in (1.0, 0.5, 0.0)],
            *[ramp.with_(damp=d) for d in (1.0, 0.5, 0.0)],
            ramp.with_(damp=1.0, half_life=60.0)]


def stage_target(best: ModelSpec) -> list:
    return [best.with_(target=t) for t in ("log_rate", "log_rpm")]


def stage_features(best: ModelSpec) -> list:
    return [best.with_(features=f) for f in feat.FEATURE_SETS]


def stage_damp(best: ModelSpec) -> list:
    return [best.with_(damp=d) for d in (1.0, 0.5, 0.0)] if best.drift != "none" else [best]


def stage_grid(best: ModelSpec) -> list:
    """Small XGBoost grid: depth 4/5/6 with 2,000 trees at learning rate 0.03, one shorter run
    (1,000 trees at 0.05) and one min-child-weight check, next to the default (depth 6, 600 trees)."""
    grid = [best]
    for depth in (4, 5, 6):
        grid.append(best.with_(xgb=(("max_depth", depth), ("n_estimators", 2000), ("learning_rate", 0.03))))
    grid.append(best.with_(xgb=(("max_depth", 5), ("n_estimators", 1000), ("learning_rate", 0.05))))
    grid.append(best.with_(xgb=(("max_depth", 5), ("n_estimators", 2000), ("learning_rate", 0.03),
                                ("min_child_weight", 20.0))))
    return grid


def stage_lane(best: ModelSpec) -> list:
    return [best, best.with_(lane="directed"), best.with_(lane="undirected")]


STAGES = [("1 drift handling", stage_drift), ("2 target", stage_target), ("3 features", stage_features),
          ("4 damping re-check", stage_damp), ("5 XGBoost grid", stage_grid), ("6 lane correction", stage_lane)]


# ==========================================================================
# Running
# ==========================================================================
def spec_name(spec) -> str:
    return B1Baseline.name if spec == "B1" else spec.name


def evaluate(spec, split: Split) -> tuple[dict, np.ndarray, object]:
    """Fit on ``split.fit``, predict ``split.test``; return (scores, predictions, model)."""
    model = make_model(spec).fit(split.fit)
    pred = model.predict(split.test)
    if not np.isfinite(pred).all() or (pred <= 0).any():
        raise AssertionError(f"{spec_name(spec)}: non-finite or non-positive predictions on {split.name}")
    return score_rows(split.test, pred), pred, model


def run_cv(specs, splits: list[Split], log=print) -> pd.DataFrame:
    """Clean and all-rows metrics per candidate and fold, long format."""
    rows = []
    for spec in specs:
        t0 = time.perf_counter()
        for i, split in enumerate(splits):
            scores, _, model = evaluate(spec, split)
            extra = {}
            if isinstance(model, RateModel) and model.drift_ is not None:
                extra = {"slope": model.drift_.slope_, "ramp": model.drift_.ramp_}
            for kind in ("clean", "all"):
                rows.append({"candidate": spec_name(spec), "fold": FOLD_LABELS.get(i + 1, split.name),
                             "rows": kind, **scores[kind], **extra})
        cv = pd.DataFrame([r for r in rows if r["candidate"] == spec_name(spec) and r["rows"] == "clean"])
        log(f"  {spec_name(spec)[:76]:76s} " + " ".join(f"{v:5.2f}" for v in cv["MAPE"])
            + f" | mean {cv['MAPE'].mean():.3f}% ${cv['MAE'].mean():6.2f} | {time.perf_counter() - t0:3.0f}s")
    return pd.DataFrame(rows)


def cv_summary(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean clean metrics over folds per candidate, ranked by MAPE then MAE."""
    clean = cv[cv["rows"] == "clean"]
    s = clean.groupby("candidate", sort=False).agg(MAPE=("MAPE", "mean"), MAE=("MAE", "mean"),
                                                   RMSE=("RMSE", "mean"), worst_fold_MAPE=("MAPE", "max"))
    return s.sort_values(["MAPE", "MAE"])


def _pick(cv: pd.DataFrame, specs: list) -> ModelSpec:
    """Best XGBoost spec among ``specs``: lowest mean clean MAPE (rounded to 0.001), then MAE."""
    summ = cv_summary(cv)
    pool = [s for s in specs if s != "B1" and s.learner == "xgb"]
    return min(pool, key=lambda s: (round(summ.loc[s.name, "MAPE"], 3), summ.loc[s.name, "MAE"]))


def select(splits: list[Split], log=print) -> tuple[ModelSpec, pd.DataFrame, list[dict]]:
    """Run the selection stages on the CV folds. Returns (chosen spec, CV table, stage log)."""
    done: set = set()        # spec keys already scored, so a setup is never fitted twice
    frames, stages = [], []
    best = None
    for title, make in STAGES:
        specs = [named(s) for s in (make() if best is None else make(best))]
        log(f"Stage {title}:   (clean MAPE May Jun Jul Aug | mean MAPE, mean MAE | time)")
        new = []
        for s in specs:
            key = s if s == "B1" else s.key()
            if key in done:
                log(f"  {spec_name(s)[:76]:76s} (scored in an earlier stage)")
            else:
                new.append(s)
                done.add(key)
        if new:
            frames.append(run_cv(new, splits, log=log))
        cv = pd.concat(frames, ignore_index=True)
        best = _pick(cv, specs)
        summ = cv_summary(cv)
        stages.append({"stage": title, "candidates": [spec_name(s) for s in specs], "winner": best.name,
                       "winner_mean_clean_MAPE": float(summ.loc[best.name, "MAPE"]),
                       "winner_mean_clean_MAE": float(summ.loc[best.name, "MAE"])})
        log(f"  -> winner: {best.name} ({summ.loc[best.name, 'MAPE']:.3f}%)")
    cv = pd.concat(frames, ignore_index=True)
    cv["stage"] = cv["candidate"].map({c: st["stage"] for st in reversed(stages) for c in st["candidates"]})
    return best, cv, stages


# ==========================================================================
# Holdout and unseen-city check
# ==========================================================================
def holdout_references(chosen: ModelSpec) -> list:
    """Setups scored on the holdout next to the chosen one, for context only (never for selection)."""
    return ["B1",
            named(PLAIN),
            named(PLAIN.with_(learner="ridge", drift="linear_ramp", damp=0.5)),
            chosen.with_(drift="linear", name="chosen without the ramp (trend only)")]


def run_holdout(chosen: ModelSpec, split: Split, log=print) -> tuple[dict, dict, RateModel]:
    """Score the chosen setup and the references on Sep-Oct. Returns (results, predictions, chosen model)."""
    results, preds = {}, {}
    scores, pred, model = evaluate(chosen, split)
    results[chosen.name], preds[chosen.name] = scores, pred
    # Damping sensitivity: the same fitted model, only the trend extrapolation changes (no refit).
    for d in (0.0, 0.5, 1.0):
        p = np.exp(model.predict_log(split.test, damp=d))
        results[f"chosen fit, damp {d:g} at prediction"] = score_rows(split.test, p)
    for ref in holdout_references(chosen):
        s, p, _ = evaluate(ref, split)
        results[spec_name(ref)], preds[spec_name(ref)] = s, p
    month = split.test["date"].dt.month.to_numpy()
    by_month = {}
    for name in (chosen.name, B1Baseline.name):
        for m, label in ((9, "Sep"), (10, "Oct")):
            sel = month == m
            by_month[f"{name}, {label}"] = score_rows(split.test[sel], preds[name][sel])
    for name, r in results.items():
        log(f"  {name[:58]:58s} clean MAE ${r['clean']['MAE']:6.2f} MAPE {r['clean']['MAPE']:5.2f}% "
            f"RMSE ${r['clean']['RMSE']:6.2f} bias {r['clean']['median_signed_pct']:+.2f}% | all MAE "
            f"${r['all']['MAE']:6.2f} MAPE {r['all']['MAPE']:5.2f}% RMSE ${r['all']['RMSE']:6.2f}")
    for name, r in by_month.items():
        log(f"  {name[:58]:58s} clean MAPE {r['clean']['MAPE']:5.2f}% bias {r['clean']['median_signed_pct']:+.2f}%")
    return {"fit_rows": len(split.fit), "test_rows": len(split.test),
            "test_flagged": int(split.test["is_outlier"].sum()),
            "results": results, "by_month": by_month, "chosen_model": model.describe()}, preds, model


def run_unseen_city(chosen: ModelSpec, split: Split, main_model: RateModel, main_test: pd.DataFrame,
                    log=print) -> dict:
    """Penalty of never having seen a city: chosen setup fit without the 8 cities vs the main holdout model."""
    scores, _, _ = evaluate(chosen, split)
    same_rows = main_test.loc[split.test.index]
    main_scores = score_rows(same_rows, main_model.predict(same_rows))
    b1_scores, _, _ = evaluate("B1", split)
    out = {"cities": list(cfg.UNSEEN_CITY_HOLDOUT), "fit_rows": len(split.fit),
           "test_rows": len(split.test), "test_flagged": int(split.test["is_outlier"].sum()),
           "results": {"chosen, cities unseen": scores, "chosen main holdout model (cities seen)": main_scores,
                       "B1, cities unseen": b1_scores},
           "penalty_clean_MAPE_pp": scores["clean"]["MAPE"] - main_scores["clean"]["MAPE"],
           "penalty_clean_MAE": scores["clean"]["MAE"] - main_scores["clean"]["MAE"]}
    for name, r in out["results"].items():
        log(f"  {name:42s} clean MAE ${r['clean']['MAE']:6.2f} MAPE {r['clean']['MAPE']:5.2f}% "
            f"bias {r['clean']['median_signed_pct']:+.2f}% | all MAE ${r['all']['MAE']:7.2f} "
            f"MAPE {r['all']['MAPE']:5.2f}%")
    log(f"  unseen-city penalty: {out['penalty_clean_MAPE_pp']:+.2f} points clean MAPE, "
        f"{out['penalty_clean_MAE']:+.2f} $ clean MAE")
    return out


# ==========================================================================
# Figures
# ==========================================================================
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _axes(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b4ae")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name: str, log=print) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    import matplotlib.pyplot as plt
    plt.close(fig)
    log(f"  [figure] reports/figures/{name}")


def fig_cv_by_fold(cv: pd.DataFrame, series: dict, log=print) -> None:
    """Clean MAPE per CV fold for a few key setups."""
    import matplotlib.pyplot as plt
    clean = cv[cv["rows"] == "clean"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    folds = list(FOLD_LABELS.values())
    for (label, cand), color in zip(series.items(), (GRAY, ORANGE, AQUA, BLUE)):
        y = clean[clean["candidate"] == cand].set_index("fold").loc[folds, "MAPE"]
        ax.plot(folds, y, color=color, linewidth=2, marker="o", markersize=7, label=label)
        ax.annotate(f"{y.iloc[-1]:.2f}%", (3, y.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_ylabel("Clean MAPE (%)", color=INK2)
    ax.set_ylim(0, None)
    ax.set_xlim(-0.2, 3.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    _axes(ax, "CV error by test month", "Expanding window inside Jan-Aug; lower is better")
    _save(fig, "10_cv_mape_by_fold.png", log)


def fig_pred_vs_actual(test: pd.DataFrame, pred: np.ndarray, log=print) -> None:
    import matplotlib.pyplot as plt
    y = test[cfg.TARGET].to_numpy(float)
    out = test["is_outlier"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(y[~out], pred[~out], s=5, color=BLUE, alpha=0.25, linewidths=0, label="clean rows")
    ax.scatter(y[out], pred[out], s=14, color=ORANGE, alpha=0.8, marker="x", linewidths=1,
               label=f"corrupted labels ({out.sum()})")
    lo, hi = min(y.min(), pred.min()) * 0.9, max(y.max(), pred.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1, linestyle="--")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual posted rate ($, log scale)", color=INK2)
    ax.set_ylabel("Predicted rate ($, log scale)", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper left", markerscale=2)
    m = metrics(y[~out], pred[~out])
    _axes(ax, "Holdout: predicted vs actual (Sep-Oct)",
          f"Chosen model fit on Jan-Aug. Clean rows: MAPE {m['MAPE']:.2f}%, MAE ${m['MAE']:.2f}")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _save(fig, "11_holdout_pred_vs_actual.png", log)


def fig_residual_by_date(test: pd.DataFrame, preds: dict, series: dict, log=print) -> None:
    """Daily median signed % error on clean holdout rows: does drift remain?"""
    import matplotlib.pyplot as plt
    clean = ~test["is_outlier"].to_numpy()
    y = test[cfg.TARGET].to_numpy(float)[clean]
    dates = test["date"].to_numpy()[clean]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axhline(0, color=INK2, linewidth=1)
    for (label, key), color in zip(series.items(), (GRAY, ORANGE, BLUE)):
        err = pd.Series(100 * (preds[key][clean] / y - 1)).groupby(dates).median()
        ax.plot(err.index, err.to_numpy(), color=color, linewidth=2, label=label)
    ax.set_ylabel("Median signed error (%)", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.tick_params(axis="x", rotation=30)
    _axes(ax, "Holdout error by day (Sep-Oct)", "Below 0 = predictions too low. Clean rows, median per day")
    _save(fig, "12_holdout_residual_by_date.png", log)


def fig_importance(model: RateModel, log=print) -> None:
    import matplotlib.pyplot as plt
    imp = model.feature_importance().iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, 100 * imp.to_numpy(), color=BLUE, height=0.7)
    for i, v in enumerate(imp.to_numpy()):
        ax.text(100 * v + 0.4, i, f"{100 * v:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("Share of total gain (%)", color=INK2)
    _axes(ax, "Feature importance (chosen model, fit on Jan-Aug)",
          "XGBoost total gain on the de-drifted log rate-per-mile target")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _save(fig, "13_feature_importance.png", log)


# ==========================================================================
# Level diagnostics (evidence for the drift term)
# ==========================================================================
def level_diagnostics(ds: data.Dataset, log=print) -> dict:
    """How the daily rate level moves beyond the inputs.

    Row-level OLS of log(rate) on the controls without the index (distance,
    equipment, weight, location) gives a residual; its daily mean is the daily
    level. The daily level is then regressed on log(daily mean index), plus the
    day number, plus the quarter-end ramp. Run on clean Jan-Oct rows (Cleaner fit
    on all of train) for the description, and on each CV fit window and on
    Jan-Aug to show the ramp size learned before any test month.
    """
    cleaner = data.Cleaner(ds.market_index_table).fit(ds.train)
    rows = cleaner.transform(ds.train, drop_outliers=True, name="diagnostics")

    def daily(frame):
        X = mdl._control_design(frame, include_index=False)
        y = np.log(frame[cfg.TARGET].to_numpy(float))
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        d = pd.DataFrame({"date": frame["date"].to_numpy(), "r": y - X @ beta,
                          "lmi": np.log(frame["market_index"].to_numpy(float))})
        d = d.groupby("date").mean().reset_index()
        d["t"] = feat.time_index(d["date"])
        d["ramp"] = feat.quarter_end_ramp(d["date"])
        return d

    def ols(d, cols):
        A = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
        beta, *_ = np.linalg.lstsq(A, d["r"].to_numpy(), rcond=None)
        res = d["r"].to_numpy() - A @ beta
        return beta, res, float(1 - res.var() / d["r"].var())

    d = daily(rows)
    out = {"daily_fits": {}}
    for label, cols in (("index only", ["lmi"]), ("index + day", ["lmi", "t"]),
                        ("index + day + quarter-end ramp", ["lmi", "t", "ramp"])):
        beta, res, r2 = ols(d, cols)
        out["daily_fits"][label] = {"coef": dict(zip(["const"] + cols, map(float, beta))), "R2": r2,
                                    "resid_sd": float(res.std())}
        log(f"  daily level ~ {label:32s} R2 {r2:.3f} resid sd {res.std():.4f} | "
            + ", ".join(f"{c} {b:.5f}" for c, b in zip(["const"] + cols, beta)))

    # Residual after index + day: the ramp shows up here.
    beta, res, _ = ols(d, ["lmi", "t"])
    d["resid_index_day"] = res
    dates = pd.DatetimeIndex(d["date"])
    d["month"], d["day"], d["dim"] = dates.month, dates.day, dates.days_in_month
    q = d[d["month"] % 3 == 0]
    out["quarter_end_months"] = {}
    for m, g in q.groupby("month"):
        first5 = g.loc[g["day"] <= 5, "resid_index_day"].mean()
        last5 = g.loc[g["day"] > g["dim"] - 5, "resid_index_day"].mean()
        after = d.loc[(d["month"] == m + 1) & (d["day"] <= 5), "resid_index_day"].mean()
        out["quarter_end_months"][int(m)] = {"first_5_days_pct": 100 * float(first5),
                                             "last_5_days_pct": 100 * float(last5),
                                             "first_5_days_next_month_pct": 100 * float(after)}
        log(f"  month {m}: level after index + day, first 5 days {100 * first5:+.2f}%, last 5 days "
            f"{100 * last5:+.2f}%, first 5 days of the next month {100 * after:+.2f}%")

    # Ramp and slope learned on each fit window.
    out["by_fit_window"] = {}
    month = rows["date"].dt.month
    for last in (4, 5, 6, 7, 8, 10):
        dd = daily(rows[month <= last])
        b_r, _, _ = ols(dd, ["lmi", "t", "ramp"])
        b_n, _, _ = ols(dd, ["lmi", "t"])
        out["by_fit_window"][f"months 1-{last}"] = {"slope_with_ramp": float(b_r[2]), "ramp": float(b_r[3]),
                                                     "slope_without_ramp": float(b_n[2])}
        log(f"  fit months 1-{last:<2d}: ramp {b_r[3]:.4f}, day slope {b_r[2]:.6f} with the ramp vs "
            f"{b_n[2]:.6f} without")
    out["ramp_coef_jan_oct"] = out["daily_fits"]["index + day + quarter-end ramp"]["coef"]["ramp"]
    daily_out = d[["date", "resid_index_day", "ramp"]].copy()
    daily_out["date"] = daily_out["date"].astype(str)
    out["daily"] = daily_out.to_dict(orient="list")
    return out


def fig_quarter_end_ramp(diag: dict, log=print) -> None:
    import matplotlib.pyplot as plt
    d = pd.DataFrame(diag["daily"])
    d["date"] = pd.to_datetime(d["date"])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for m in (3, 6, 9):
        ax.axvspan(pd.Timestamp(2025, m, 1), pd.Timestamp(2025, m + 1, 1), color="#eef3fb", zorder=0)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.plot(d["date"], 100 * d["resid_index_day"], color=GRAY, linewidth=1.4,
            label="Daily rate level after the index and a straight-line trend")
    c = diag["ramp_coef_jan_oct"]
    ramp = c * d["ramp"].to_numpy()
    ax.plot(d["date"], 100 * (ramp - ramp.mean()), color=BLUE, linewidth=2,
            label=f"Quarter-end ramp fitted with them ({100 * c:.1f} log points by the last day)")
    ax.set_ylabel("Level (%)", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    _axes(ax, "Rates climb through each quarter-end month and drop back on the 1st",
          "Clean Jan-Oct rows, daily means; shaded: Mar, Jun, Sep. December is also a quarter-end month")
    _save(fig, "15_quarter_end_ramp.png", log)

# ==========================================================================
# Reports
# ==========================================================================
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_metrics(m: dict, path: Path = METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(m), indent=2), encoding="utf-8")


def load_metrics(path: Path = METRICS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _row(name: str, r: dict) -> str:
    c, a = r["clean"], r["all"]
    return (f"| {name} | ${c['MAE']:.2f} | {c['MAPE']:.2f}% | ${c['RMSE']:.2f} | {c['median_signed_pct']:+.2f}% "
            f"| ${a['MAE']:.2f} | {a['MAPE']:.2f}% | ${a['RMSE']:.2f} |")


_HEAD = ("| Setup | Clean MAE | Clean MAPE | Clean RMSE | Clean bias | All MAE | All MAPE | All RMSE |\n"
         "|---|---|---|---|---|---|---|---|")


def write_results_md(m: dict, path: Path = RESULTS_MD_PATH) -> None:
    """Readable tables from the metrics dict (CV per fold, holdout, unseen-city, final)."""
    L = ["# Model results", "",
         "Every number here is written by `python run_pipeline.py` (see `docs/modeling.md` for the reasoning).",
         "MAPE and MAE on clean rows decide; bias is the median signed error (below 0 = too low).", ""]
    if "cv" in m:
        L += ["## CV folds (selection only)", "",
              "Expanding window inside Jan-Aug. Clean-row MAPE per test month, then the mean over the four folds.",
              "Candidates are listed in the order the stages scored them.", "",
              "| Stage | Setup | May | Jun | Jul | Aug | Mean MAPE | Mean MAE |", "|---|---|---|---|---|---|---|---|"]
        for c in m["cv"]["candidates"]:
            f = c["clean_MAPE_by_fold"]
            mark = " **(chosen)**" if c["name"] == m["chosen"]["name_in_cv"] else ""
            L.append(f"| {c['stage']} | {c['name']}{mark} | " + " | ".join(f"{f[k]:.2f}%" for k in FOLD_LABELS.values())
                     + f" | {c['mean_clean_MAPE']:.3f}% | ${c['mean_clean_MAE']:.2f} |")
        L += ["", "Stage winners:", "", "| Stage | Winner | Mean clean MAPE |", "|---|---|---|"]
        L += [f"| {s['stage']} | {s['winner']} | {s['winner_mean_clean_MAPE']:.3f}% |" for s in m["selection"]]
        L.append("")
    if "holdout" in m:
        h = m["holdout"]
        L += ["## Main holdout (fit Jan-Aug, test Sep-Oct, scored once after selection)", "",
              f"Fit rows {h['fit_rows']:,}; test rows {h['test_rows']:,} ({h['test_flagged']} with corrupted labels).",
              "", _HEAD]
        L += [_row(k, v) for k, v in h["results"].items()]
        L += ["", "By month:", "", _HEAD] + [_row(k, v) for k, v in h["by_month"].items()]
        d = h["chosen_model"]
        L += ["", f"Drift learned on Jan-Aug: slope {d['drift_slope_per_day']:.6f} log units per day "
                  f"({100 * 30 * d['drift_slope_per_day']:.2f}% per 30 days), quarter-end ramp "
                  f"{d['drift_ramp']:.4f} ({100 * (np.exp(d['drift_ramp']) - 1):.2f}% by the last day of the month).", ""]
    if "unseen_city" in m:
        u = m["unseen_city"]
        L += ["## Unseen-city check", "",
              f"Cities held out: {', '.join(u['cities'])}. Fit rows {u['fit_rows']:,}; test rows "
              f"{u['test_rows']:,} Sep-Oct rows touching them ({u['test_flagged']} with corrupted labels).", "", _HEAD]
        L += [_row(k, v) for k, v in u["results"].items()]
        L += ["", f"Penalty for an unseen city: {u['penalty_clean_MAPE_pp']:+.2f} points of clean MAPE, "
                  f"{u['penalty_clean_MAE']:+.2f} $ of clean MAE.", ""]
    if "final" in m:
        f = m["final"]
        L += ["## Final model (fit on all clean Jan-Oct rows)", "",
              f"Fit rows {f['fit_rows']:,}. Drift: slope {f['model']['drift_slope_per_day']:.6f} per day, "
              f"ramp {f['model']['drift_ramp']:.4f}.", "",
              "| Output | Min | Median | Mean | Max |", "|---|---|---|---|---|"]
        for k in ("validation", "validation_nov", "validation_dec", "december_chart"):
            s = f["summaries"][k]
            L.append(f"| {s['label']} | ${s['min']:,.2f} | ${s['median']:,.2f} | ${s['mean']:,.2f} | ${s['max']:,.2f} |")
        L += ["", "December chart (Lexington to Fort Wayne, 360 mi, Dry Van, 32,000 lb):", "",
              "| Date | Weekday | market_index | Predicted rate |", "|---|---|---|---|"]
        for r in f["december_rows"]:
            L.append(f"| {r['date']} | {r['weekday']} | {r['market_index']:.4f} | ${r['predicted_rate']:.2f} |")
        L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def cv_to_dict(cv: pd.DataFrame, stages: list[dict]) -> dict:
    clean = cv[cv["rows"] == "clean"]
    out = []
    for name in clean["candidate"].drop_duplicates():
        c = clean[clean["candidate"] == name].set_index("fold")
        a = cv[(cv["rows"] == "all") & (cv["candidate"] == name)].set_index("fold")
        rec = {"name": name, "stage": c["stage"].iloc[0],
               "clean_MAPE_by_fold": c["MAPE"].to_dict(), "clean_MAE_by_fold": c["MAE"].to_dict(),
               "clean_bias_by_fold": c["median_signed_pct"].to_dict(),
               "all_MAPE_by_fold": a["MAPE"].to_dict(),
               "mean_clean_MAPE": float(c["MAPE"].mean()), "mean_clean_MAE": float(c["MAE"].mean()),
               "mean_clean_RMSE": float(c["RMSE"].mean())}
        if "slope" in c and c["slope"].notna().all():
            rec["drift_slope_by_fold"] = c["slope"].to_dict()
            rec["drift_ramp_by_fold"] = c["ramp"].to_dict()
        out.append(rec)
    return {"folds": [f"months 1-{a} -> month {b}" for a, b in cfg.CV_FOLDS], "candidates": out}


# ==========================================================================
# Everything
# ==========================================================================
def run_evaluation(ds: data.Dataset | None = None, log=print) -> ModelSpec:
    """CV selection, holdout, unseen-city check, figures and reports. Returns the chosen spec."""
    t0 = time.perf_counter()
    ds = data.load_all() if ds is None else ds
    splits = cv_splits(ds)
    log("CV folds (clean fit rows / test rows / flagged): "
        + ", ".join(f"{len(s.fit):,}/{len(s.test):,}/{int(s.test['is_outlier'].sum())}" for s in splits))
    best, cv, stages = select(splits, log=log)
    chosen = best.with_(name="chosen")
    if chosen.key() != mdl.FINAL_SPEC.key():
        log(f"WARNING: CV picked {best.name}, which differs from model.FINAL_SPEC "
            f"({mdl.FINAL_SPEC.label()}). The CV pick is used; update FINAL_SPEC.")
    log(f"Chosen setup: {best.name}")

    log("\nMain holdout (fit Jan-Aug, test Sep-Oct), scored once:")
    hsplit = holdout_split(ds)
    holdout, preds, main_model = run_holdout(chosen, hsplit, log=log)

    log("\nUnseen-city check:")
    unseen = run_unseen_city(chosen, unseen_city_split(ds), main_model, hsplit.test, log=log)

    log("\nLevel diagnostics (daily rate level beyond the inputs):")
    diag = level_diagnostics(ds, log=log)

    log("\nFigures:")
    plain, trend = named(PLAIN).name, named(PLAIN.with_(drift="linear", damp=0.5)).name
    fig_cv_by_fold(cv, {"B1 baseline": B1Baseline.name, "XGBoost, no drift handling": plain,
                        "XGBoost, linear trend (damp 0.5)": trend, "Chosen model": best.name}, log=log)
    fig_pred_vs_actual(hsplit.test, preds["chosen"], log=log)
    fig_residual_by_date(hsplit.test, preds, {"B1 baseline": B1Baseline.name,
                                              "XGBoost, no drift handling": plain, "Chosen model": "chosen"},
                         log=log)
    fig_importance(main_model, log=log)
    fig_quarter_end_ramp(diag, log=log)

    m = {"seed": cfg.SEED, "n_jobs": mdl.N_JOBS,
         "chosen": {"name_in_cv": best.name, "spec": chosen.as_dict()},
         "selection": stages, "cv": cv_to_dict(cv, stages),
         "holdout": holdout, "unseen_city": unseen,
         "level_diagnostics": {k: v for k, v in diag.items() if k != "daily"},
         "evaluation_seconds": round(time.perf_counter() - t0, 1)}
    save_metrics(m)
    write_results_md(m)
    log(f"\nWrote {METRICS_PATH.relative_to(cfg.ROOT)} and {RESULTS_MD_PATH.relative_to(cfg.ROOT)} "
        f"({m['evaluation_seconds']:.0f}s)")
    return chosen


if __name__ == "__main__":
    run_evaluation()
