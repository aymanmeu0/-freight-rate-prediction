"""Models: the B1 baseline and the configurable rate model.

``RateModel(spec).fit(rows).predict(rows)`` takes cleaned rows (from
``data.Cleaner.transform``) and returns dollar predictions. The spec chooses:

* the learner: XGBoost or ridge regression;
* the feature set (``features.FEATURE_SETS``) and the target, log(rate) or
  log(rate per mile);
* the drift handling:
    - ``none``: nothing;
    - recency weights: sample weight ``0.5 ** (days before the last fit day / half_life)``;
    - ``linear``: a straight-line level trend in time;
    - ``linear_ramp``: the same trend plus a quarter-end ramp (the rate level
      climbs through Mar, Jun and Sep and drops back on the 1st).
  The trend and ramp sizes come from an OLS fit on the fit rows only
  (:class:`LevelDrift`). The learner models the target minus that level; the
  level is added back at prediction time. Past the last fit day the trend is
  extrapolated with slope ``damp * b``: ``damp = 1`` extends the line,
  ``damp = 0`` holds the level of the last fit day.
* an optional lane correction: an out-of-fold, shrunk lane mean of the
  learner's residual, added to the prediction; unseen lanes get 0.

Everything is deterministic: fixed seed, fixed thread count. No work at import.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg
from src import data
from src import features as feat

N_JOBS = 4  # fixed thread count, for identical results across runs

XGB_DEFAULTS = dict(n_estimators=600, max_depth=6, learning_rate=0.05, min_child_weight=1.0,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
# The lane correction needs out-of-fold residuals (5 extra fits). They use this
# lighter XGBoost so the extra fits stay cheap whatever the main model is.
LANE_OOF_XGB = dict(XGB_DEFAULTS)
# Memo of out-of-fold residuals (pure function of rows, target and settings).
_OOF_CACHE: dict = {}


def clear_cache() -> None:
    """Forget memoized out-of-fold residuals."""
    _OOF_CACHE.clear()


# ==========================================================================
# B1 baseline
# ==========================================================================
class B1Baseline:
    """Median rate per mile by (equipment, distance band), times distance (findings 5.5)."""

    name = "B1 baseline"

    def fit(self, rows: pd.DataFrame) -> "B1Baseline":
        rpm = rows[cfg.TARGET] / rows["distance"]
        band = data.distance_band(rows["distance"])
        self.table_ = rpm.groupby([rows["equipment"].to_numpy(), band]).median()
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        band = data.distance_band(rows["distance"])
        keys = list(zip(rows["equipment"].to_numpy(), band))
        rpm = self.table_.reindex(pd.MultiIndex.from_tuples(keys)).to_numpy(dtype=float)
        if np.isnan(rpm).any():  # empty cell: nearest band of the same equipment (never on the given folds)
            for i in np.flatnonzero(np.isnan(rpm)):
                eq, b = keys[i]
                cells = self.table_.loc[eq]
                rpm[i] = cells.iloc[int(np.argmin(np.abs(cells.index.to_numpy() - b)))]
        return rpm * rows["distance"].to_numpy(float)


# ==========================================================================
# Level drift
# ==========================================================================
def _control_design(rows: pd.DataFrame, include_index: bool = True) -> np.ndarray:
    """Controls for the drift OLS: distance curve, equipment, weight curve, location, log index."""
    ld = np.log(rows["distance"].to_numpy(float))
    w = rows["weight"].to_numpy(float) / 1e4
    cols = [np.ones(len(rows)), ld, ld ** 2, ld ** 3,
            (rows["equipment"] == "Reefer").to_numpy(float), (rows["equipment"] == "Flatbed").to_numpy(float),
            w, w ** 2, w ** 3]
    if include_index:
        cols.append(np.log(rows["market_index"].to_numpy(float)))
    for c in cfg.COORD_COLUMNS:
        v = rows[c].to_numpy(float)
        cols += [v, v ** 2]
    return np.column_stack(cols)


class LevelDrift:
    """Level of log(rate) over time beyond what the inputs explain.

    ``fit`` runs OLS of log(rate) on the controls plus the time index
    (and the quarter-end ramp when ``ramp=True``), on the fit rows only, and
    keeps the time slope ``slope_`` (log units per day) and the ramp size ``ramp_``.
    """

    def __init__(self, ramp: bool):
        self.ramp = ramp

    def fit(self, rows: pd.DataFrame) -> "LevelDrift":
        t = feat.time_index(rows["date"])
        extra = [t] + ([feat.quarter_end_ramp(rows["date"])] if self.ramp else [])
        X = np.column_stack([_control_design(rows), *extra])
        beta, *_ = np.linalg.lstsq(X, np.log(rows[cfg.TARGET].to_numpy(float)), rcond=None)
        self.slope_ = float(beta[-2] if self.ramp else beta[-1])
        self.ramp_ = float(beta[-1]) if self.ramp else 0.0
        self.t_end_ = float(t.max())
        return self

    def offset(self, dates, damp: float = 1.0) -> np.ndarray:
        """Level in log units: trend (damped past the last fit day) plus ramp."""
        t = feat.time_index(dates)
        trend_t = np.where(t <= self.t_end_, t, self.t_end_ + damp * (t - self.t_end_))
        out = self.slope_ * trend_t
        if self.ramp:
            out = out + self.ramp_ * feat.quarter_end_ramp(dates)
        return out


# ==========================================================================
# Rate model
# ==========================================================================
@dataclass(frozen=True)
class ModelSpec:
    """Everything that defines one candidate. ``xgb`` overrides ``XGB_DEFAULTS``."""

    name: str
    learner: str = "xgb"                # "xgb" | "ridge"
    features: str = "geo"               # key of features.FEATURE_SETS
    target: str = "log_rate"            # "log_rate" | "log_rpm"
    drift: str = "none"                 # "none" | "linear" | "linear_ramp"
    damp: float = 1.0                   # trend slope multiplier past the last fit day
    half_life: float | None = None      # recency weights (days); None = equal weights
    lane: str | None = None             # None | "directed" | "undirected"
    lane_prior: float = 10.0            # shrinkage (pseudo-loads) for the lane correction
    ridge_alpha: float = 1.0
    xgb: tuple = field(default_factory=tuple)  # ((param, value), ...)

    def xgb_params(self) -> dict:
        return {**XGB_DEFAULTS, **dict(self.xgb)}

    def with_(self, **changes) -> "ModelSpec":
        return replace(self, **changes)

    def key(self) -> "ModelSpec":
        """Canonical form without the name: two specs with equal keys are the same model."""
        changed = sorted((k, v) for k, v in self.xgb_params().items() if XGB_DEFAULTS.get(k) != v)
        return replace(self, name="", xgb=tuple(changed) if self.learner == "xgb" else ())

    def label(self) -> str:
        """Readable description built from the settings."""
        drift = {"none": "no drift", "linear": "trend", "linear_ramp": "trend+ramp"}[self.drift]
        if self.drift != "none":
            drift += f" damp {self.damp:g}"
        parts = [self.learner, self.target.replace("_", " "), self.features, drift]
        if self.half_life:
            parts.append(f"recency hl{self.half_life:g}")
        if self.learner == "xgb":
            p = self.xgb_params()
            parts.append(f"d{p['max_depth']} n{p['n_estimators']} lr{p['learning_rate']:g} mcw{p['min_child_weight']:g}")
        if self.lane:
            parts.append(f"lane {self.lane} p{self.lane_prior:g}")
        return ", ".join(parts)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        d = dict(d)
        xgb = d.pop("xgb", None)
        changed = {k: v for k, v in (xgb or {}).items() if XGB_DEFAULTS.get(k) != v}
        return cls(**d, xgb=tuple(changed.items()))

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("name", "learner", "features", "target", "drift", "damp",
                                           "half_life", "lane", "lane_prior")}
        if self.learner == "xgb":
            d["xgb"] = self.xgb_params()
        else:
            d["ridge_alpha"] = self.ridge_alpha
        return d


def _ridge_design(X: pd.DataFrame) -> np.ndarray:
    """Expanded inputs for the ridge model: smooth curves a straight line cannot fit."""
    ld = X["log_distance"].to_numpy()
    w = X["weight"].to_numpy() / 1e4
    cols = [ld, ld ** 2, ld ** 3, w, w ** 2, w ** 3, np.log(X["market_index"].to_numpy())]
    cols += [X[c].to_numpy() for c in feat.EQUIPMENT_DUMMIES[1:]]
    for c in cfg.COORD_COLUMNS:
        v = X[c].to_numpy()
        cols += [v, v ** 2]
    cols += [X["pickup_lat"].to_numpy() * X["pickup_lon"].to_numpy(),
             X["delivery_lat"].to_numpy() * X["delivery_lon"].to_numpy()]
    for c in ("delta_lat", "delta_lon", "bearing_sin", "bearing_cos", "circuity"):
        if c in X:
            cols.append(X[c].to_numpy())
    if "day_of_week" in X:
        dow = X["day_of_week"].to_numpy()
        cols += [(dow == k).astype(float) for k in range(1, 7)]
    return np.column_stack(cols)


def _make_learner(spec: ModelSpec):
    if spec.learner == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(**spec.xgb_params(), tree_method="hist", random_state=cfg.SEED,
                            n_jobs=N_JOBS, verbosity=0)
    if spec.learner == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), Ridge(alpha=spec.ridge_alpha))
    raise ValueError(f"unknown learner {spec.learner!r}")


class RateModel:
    """One candidate model. ``fit`` on cleaned training rows, ``predict`` returns dollars."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    # ---- helpers -----------------------------------------------------------
    def _matrix(self, rows: pd.DataFrame) -> np.ndarray:
        X = feat.build_features(rows, self.spec.features)
        return _ridge_design(X) if self.spec.learner == "ridge" else X.to_numpy()

    def _target_shift(self, rows: pd.DataFrame) -> np.ndarray:
        """log(distance) when the target is log rate per mile, else 0."""
        return np.log(rows["distance"].to_numpy(float)) if self.spec.target == "log_rpm" else 0.0

    def _fit_learner(self, X, y, w, spec: ModelSpec | None = None):
        learner = _make_learner(spec or self.spec)
        if self.spec.learner == "ridge":
            learner.fit(X, y, ridge__sample_weight=w)
        else:
            learner.fit(X, y, sample_weight=w)
        return learner

    # ---- fit / predict -----------------------------------------------------
    def fit(self, rows: pd.DataFrame) -> "RateModel":
        spec = self.spec
        if "is_outlier" in rows.columns and rows["is_outlier"].any():
            raise ValueError("RateModel.fit: drop the C5 outliers first (Cleaner.transform(drop_outliers=True))")
        self.feature_names_ = feat.FEATURE_SETS[spec.features]
        feat.assert_allowed(self.feature_names_)
        X = self._matrix(rows)
        y = np.log(rows[cfg.TARGET].to_numpy(float))
        t = feat.time_index(rows["date"])
        self.t_end_ = float(t.max())

        self.drift_ = None
        if spec.drift != "none":
            self.drift_ = LevelDrift(ramp=spec.drift == "linear_ramp").fit(rows)
            y = y - self.drift_.offset(rows["date"])
        z = y - self._target_shift(rows)

        w = None
        if spec.half_life:
            w = 0.5 ** ((self.t_end_ - t) / spec.half_life)
        self.learner_ = self._fit_learner(X, z, w)

        self.lane_table_ = None
        if spec.lane:
            self.lane_table_ = self._fit_lane_correction(rows, X, z, w)
        return self

    def _fit_lane_correction(self, rows, X, z, w, k: int = 5) -> pd.Series:
        """Out-of-fold residual per lane, shrunk: sum(resid) / (n + prior).

        Residuals come from a 5-fold split of the fit rows, each fold predicted
        by a ``LANE_OOF_XGB`` model trained on the other four.
        """
        from sklearn.model_selection import KFold
        oof_spec = self.spec.with_(xgb=tuple(LANE_OOF_XGB.items())) if self.spec.learner == "xgb" else self.spec
        # The residuals do not depend on the lane key or prior, so the directed and
        # undirected variants share them. The cache key covers everything they do depend on.
        cache_key = (oof_spec.with_(lane=None, lane_prior=0.0, damp=0.0).key(), k, cfg.SEED, len(z),
                     int(pd.util.hash_pandas_object(rows["load_id"], index=False).sum()),
                     float(np.sum(z)), None if w is None else float(np.sum(w)))
        resid = _OOF_CACHE.get(cache_key)
        if resid is None:
            resid = np.empty(len(z))
            for tr, te in KFold(n_splits=k, shuffle=True, random_state=cfg.SEED).split(X):
                m = self._fit_learner(X[tr], z[tr], None if w is None else w[tr], spec=oof_spec)
                resid[te] = z[te] - m.predict(X[te])
            if len(_OOF_CACHE) >= 16:
                _OOF_CACHE.clear()
            _OOF_CACHE[cache_key] = resid
        keys = feat.lane_key(rows, directed=self.spec.lane == "directed").to_numpy()
        g = pd.Series(resid).groupby(keys)
        return g.sum() / (g.size() + self.spec.lane_prior)

    def predict_log(self, rows: pd.DataFrame, damp: float | None = None) -> np.ndarray:
        """Predicted log(rate). ``damp`` overrides the spec's trend damping (for sensitivity checks)."""
        out = self.learner_.predict(self._matrix(rows)) + self._target_shift(rows)
        if self.drift_ is not None:
            out = out + self.drift_.offset(rows["date"], damp=self.spec.damp if damp is None else damp)
        if self.lane_table_ is not None:
            keys = feat.lane_key(rows, directed=self.spec.lane == "directed")
            out = out + keys.map(self.lane_table_).fillna(0.0).to_numpy(float)
        return np.asarray(out, dtype=float)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(rows))

    def feature_importance(self) -> pd.Series:
        """XGBoost gain importance by feature name (XGBoost only)."""
        if self.spec.learner != "xgb":
            raise ValueError("feature importance is only defined for the XGBoost learner")
        gain = self.learner_.get_booster().get_score(importance_type="total_gain")
        imp = pd.Series({name: gain.get(f"f{i}", 0.0) for i, name in enumerate(self.feature_names_)})
        return (imp / imp.sum()).sort_values(ascending=False)

    def describe(self) -> dict:
        d = {"spec": self.spec.as_dict(), "t_end": self.t_end_}
        if self.drift_ is not None:
            d["drift_slope_per_day"] = self.drift_.slope_
            d["drift_ramp"] = self.drift_.ramp_
        if self.lane_table_ is not None:
            d["n_lanes"] = int(len(self.lane_table_))
        return d


# The setup the CV selection in src/train.py picked (docs/modeling.md). Used by
# ``run_pipeline.py --skip-eval`` when reports/metrics.json is absent. A full
# run re-selects on the CV folds and warns if the result differs from this.
FINAL_SPEC = ModelSpec(
    name="chosen", learner="xgb", features="geo_dow", target="log_rpm", drift="linear_ramp", damp=0.5,
    lane="directed", lane_prior=10.0,
    xgb=(("max_depth", 5), ("n_estimators", 2000), ("learning_rate", 0.03)),
)


def make_model(spec: ModelSpec | str):
    """B1 for the string "B1", else a :class:`RateModel`."""
    return B1Baseline() if spec == "B1" else RateModel(spec)
