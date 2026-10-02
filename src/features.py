"""Model features built from cleaned rows.

Input: a frame returned by ``data.Cleaner.transform`` (train, holdout,
validation or the December rows). Output: a float matrix with a fixed column
order. Nothing is learned here, so the same rows always give the same matrix.

Two kinds of columns come out of this module, and they are kept apart:

* Model features (``FEATURE_SETS``). Only these reach a learner.
  :func:`assert_allowed` checks that none of them is in
  ``config.NEVER_FEATURES`` or in ``DRIFT_ONLY``.
* Drift inputs (:func:`time_index`, :func:`quarter_end_ramp`). They feed the
  level-drift term in ``src/model.py`` and never reach a learner. A tree cannot
  extrapolate a day number, so time is handled outside the trees.

No work is done at import time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg

# Day 0 of the time index.
TIME_ORIGIN = pd.Timestamp("2025-01-01")

# Columns used only by the drift term. They must never reach a learner.
DRIFT_ONLY = ("time_index", "quarter_end_ramp")

EQUIPMENT_DUMMIES = [f"equip_{e.lower().replace(' ', '_')}" for e in cfg.EQUIPMENT]

_BASE = ["distance", "log_distance", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
         *EQUIPMENT_DUMMIES, "weight", "market_index"]
_GEOMETRY = ["delta_lat", "delta_lon", "bearing_sin", "bearing_cos", "haversine", "circuity"]

# Named feature sets compared in CV.
FEATURE_SETS: dict[str, list[str]] = {
    "base": _BASE,
    "geo": _BASE + _GEOMETRY,
    "geo_dow": _BASE + _GEOMETRY + ["day_of_week"],
}


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in miles."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def assert_allowed(columns) -> None:
    """Raise if any banned column would reach a learner.

    Banned: ``config.NEVER_FEATURES`` (load_id, quote_signal, posted_rate,
    is_outlier, date, month) and the drift-only inputs (time index, ramp).
    """
    banned = sorted(set(columns) & (set(cfg.NEVER_FEATURES) | set(DRIFT_ONLY)))
    if banned:
        raise AssertionError(f"banned columns in the feature matrix: {banned}")


def time_index(dates) -> np.ndarray:
    """Days since 2025-01-01 (drift input only)."""
    return (pd.DatetimeIndex(pd.to_datetime(dates)) - TIME_ORIGIN).days.to_numpy(dtype=float)


def quarter_end_ramp(dates) -> np.ndarray:
    """0 outside the last month of a quarter; inside it, a straight line from 0
    on day 1 to 1 on the last day of the month (Mar, Jun, Sep, Dec).

    Drift input only. In training the rate level climbs about 4% through
    each quarter-end month and drops back on the 1st (see docs/modeling.md).
    """
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    ramp = (d.day.to_numpy() - 1) / (d.days_in_month.to_numpy() - 1)
    return np.where(d.month.to_numpy() % 3 == 0, ramp, 0.0).astype(float)


def build_features(df: pd.DataFrame, feature_set: str = "geo") -> pd.DataFrame:
    """Feature matrix for ``df`` (cleaned rows), columns in ``FEATURE_SETS[feature_set]`` order.

    Raises if a needed input is missing or not finite, or if a banned column
    would come out. The row index of ``df`` is kept.
    """
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}")
    columns = FEATURE_SETS[feature_set]
    assert_allowed(columns)
    needed = ["distance", "equipment", "weight", "market_index", "date", *cfg.COORD_COLUMNS]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise KeyError(f"build_features: input frame lacks {absent}")

    plat, plon = df["pickup_lat"].to_numpy(float), df["pickup_lon"].to_numpy(float)
    dlat, dlon = df["delivery_lat"].to_numpy(float), df["delivery_lon"].to_numpy(float)
    dist = df["distance"].to_numpy(float)
    hav = haversine_miles(plat, plon, dlat, dlon)
    # Bearing from pickup to delivery on a flat local map (lon scaled by cos(lat)).
    dx = (dlon - plon) * np.cos(np.radians((plat + dlat) / 2))
    dy = dlat - plat
    bearing = np.arctan2(dx, dy)

    out = {
        "distance": dist,
        "log_distance": np.log(dist),
        "pickup_lat": plat, "pickup_lon": plon, "delivery_lat": dlat, "delivery_lon": dlon,
        "weight": df["weight"].to_numpy(float),
        "market_index": df["market_index"].to_numpy(float),
        "delta_lat": dy, "delta_lon": dlon - plon,
        "bearing_sin": np.sin(bearing), "bearing_cos": np.cos(bearing),
        "haversine": hav,
        # Listed distance is about 1.18x the straight line; guard the (absent) zero case.
        "circuity": dist / np.maximum(hav, 1.0),
        "day_of_week": pd.DatetimeIndex(df["date"]).dayofweek.to_numpy(dtype=float),
    }
    for e, col in zip(cfg.EQUIPMENT, EQUIPMENT_DUMMIES):
        out[col] = (df["equipment"].to_numpy() == e).astype(float)

    X = pd.DataFrame({c: out[c] for c in columns}, index=df.index)
    if not np.isfinite(X.to_numpy()).all():
        bad = X.columns[~np.isfinite(X.to_numpy()).all(axis=0)].tolist()
        raise ValueError(f"build_features: non-finite values in {bad}")
    assert_allowed(X.columns)
    return X


def lane_key(df: pd.DataFrame, directed: bool = False) -> pd.Series:
    """Lane label per row. Undirected keys sort the two cities, so A->B and B->A share one key."""
    p, d = df["pickup"].astype(str), df["delivery"].astype(str)
    if directed:
        return (p + " > " + d).set_axis(df.index)
    lo, hi = np.where(p < d, p, d), np.where(p < d, d, p)
    return pd.Series(lo, index=df.index) + " | " + pd.Series(hi, index=df.index)
