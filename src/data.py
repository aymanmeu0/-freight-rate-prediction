"""Data pipeline: load, check, clean, enrich and write.

This module implements the cleaning rules C1 to C8 from ``docs/eda_findings.md``.
It builds no model features; that is the next step.

Typical use (for example inside one CV fold)::

    from src import data

    train = data.load_train()                    # C1, C2, C8 contract checks
    val = data.load_validation()
    mi_table = data.build_market_index_table(train, val)   # C4, inputs only

    cleaner = data.Cleaner(mi_table).fit(fit_rows)          # C3, C5 learned here only
    fit_clean = cleaner.transform(fit_rows, drop_outliers=True, name="fit")
    test_clean = cleaner.transform(test_rows, name="test")  # keeps is_outlier for scoring
    val_clean = cleaner.transform(val, name="validation")   # never drops a row
    print(cleaner.log_table())

    december = data.build_december_frame()                  # recipe in section 6
    data.write_december_predictions(preds_for_31_days)
    data.write_validation_predictions(val_clean["load_id"], preds)

Run ``python src/data.py`` (from any folder) for the self-check: it runs the
whole pipeline on the full data and prints the cleaning log.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    # Allow `python src/data.py` from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg


# ==========================================================================
# Data contract (C1, C2, C8)
# ==========================================================================
class DataContractError(ValueError):
    """Raised when a file or frame breaks the data contract."""


def _fail_if(problems: list[str], label: str) -> None:
    if problems:
        bullet = "\n  - "
        raise DataContractError(f"{label} breaks the data contract:{bullet}{bullet.join(problems)}")


def _read_csv(path: Path, label: str, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DataContractError(f"{label}: file not found: {path}")
    return pd.read_csv(path, **kwargs)


def _check_frame(df: pd.DataFrame, *, label: str, columns: list[str],
                 date_range: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    """Check one load-level frame and return a typed copy.

    Checks: exact columns in order, text columns present and stripped (C2),
    equipment in the known set, numeric columns numeric and finite (weight and
    market_index may be missing, but never infinite), positive distance,
    market_index and posted_rate, valid coordinates, dates parsed with
    ``%Y-%m-%d`` and inside ``date_range`` (C1), unique non-empty ``load_id`` (C8).
    All problems are collected and raised together.
    """
    if list(df.columns) != columns:
        missing = [c for c in columns if c not in df.columns]
        extra = [c for c in df.columns if c not in columns]
        raise DataContractError(
            f"{label}: columns must be exactly {columns} in this order; "
            f"got {list(df.columns)} (missing {missing}, extra {extra})")
    out = df.copy()
    problems: list[str] = []

    # Text columns: no missing values, whitespace stripped (C2).
    for col in [c for c in cfg.TEXT_COLUMNS if c in out.columns]:
        if out[col].isna().any():
            problems.append(f"{col} has {int(out[col].isna().sum())} missing values")
            continue
        if not (pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col])):
            problems.append(f"{col} must be text, got dtype {out[col].dtype}")
            continue
        stripped = out[col].astype(str).str.strip()
        out.attrs.setdefault("whitespace_stripped", {})[col] = int((stripped != out[col].astype(str)).sum())
        out[col] = stripped
        if (stripped == "").any():
            problems.append(f"{col} has {int((stripped == '').sum())} empty values")
    if "equipment" in out.columns:
        bad = sorted(set(out["equipment"]) - set(cfg.EQUIPMENT))
        if bad:
            problems.append(f"equipment has unknown values {bad}; allowed {list(cfg.EQUIPMENT)}")

    # Numeric columns.
    numeric = [c for c in columns if c not in cfg.TEXT_COLUMNS and c not in ("date", "predicted_rate")]
    for col in numeric:
        if not pd.api.types.is_numeric_dtype(out[col]):
            problems.append(f"{col} must be numeric, got dtype {out[col].dtype} "
                            f"(a non-numeric value in the file?)")
            continue
        values = out[col].astype(float)
        out[col] = values
        present = values.dropna()
        if col not in cfg.NULLABLE_NUMERIC and values.isna().any():
            problems.append(f"{col} has {int(values.isna().sum())} missing values")
        if not np.isfinite(present).all():
            problems.append(f"{col} has {int((~np.isfinite(present)).sum())} infinite values")
    if not problems:
        for col in ("distance", "market_index", cfg.TARGET):
            if col in out.columns and (out[col].dropna() <= 0).any():
                problems.append(f"{col} has {int((out[col].dropna() <= 0).sum())} non-positive values")
        for col in [c for c in cfg.COORD_COLUMNS if c in out.columns]:
            limit = 90 if col.endswith("lat") else 180
            if (out[col].abs() > limit).any():
                problems.append(f"{col} has values outside +/-{limit}")

    # Dates (C1).
    parsed = pd.to_datetime(out["date"], format=cfg.DATE_FORMAT, errors="coerce")
    n_bad = int(parsed.isna().sum())
    if n_bad:
        examples = out.loc[parsed.isna(), "date"].astype(str).head(3).tolist()
        problems.append(f"date has {n_bad} values not in {cfg.DATE_FORMAT} format, e.g. {examples}")
    else:
        lo, hi = date_range
        outside = (parsed < lo) | (parsed > hi)
        if outside.any():
            problems.append(f"date has {int(outside.sum())} values outside {lo.date()}..{hi.date()} "
                            f"(found {parsed.min().date()}..{parsed.max().date()})")
        out["date"] = parsed

    # Keys (C8).
    if "load_id" in out.columns and out["load_id"].duplicated().any():
        problems.append(f"load_id has {int(out['load_id'].duplicated().sum())} duplicates")

    _fail_if(problems, label)
    return out


def load_train(path: Path = cfg.TRAIN_PATH) -> pd.DataFrame:
    """Load ``train_test.csv`` (48,000 labeled loads, Jan-Oct 2025) and check its contract.

    Returns the raw values with ``date`` parsed to datetime and numeric columns
    as float. Nothing is cleaned yet; use :class:`Cleaner` for that.
    """
    df = _check_frame(_read_csv(path, "train_test.csv"), label="train_test.csv",
                      columns=cfg.TRAIN_COLUMNS, date_range=cfg.TRAIN_DATES)
    return df


def load_validation(path: Path = cfg.VALIDATION_PATH) -> pd.DataFrame:
    """Load ``validation.csv`` (12,000 unlabeled loads, Nov-Dec 2025) and check its contract.

    Row order is the file order, which is also the template order. Never drop a row.
    """
    return _check_frame(_read_csv(path, "validation.csv"), label="validation.csv",
                        columns=cfg.VALIDATION_COLUMNS, date_range=cfg.VALIDATION_DATES)


def load_template(path: Path = cfg.TEMPLATE_PATH) -> pd.DataFrame:
    """Load the submission template and check it: two columns, 12,000 unique load_ids."""
    df = _read_csv(path, "validation_predictions_template.csv")
    problems = []
    if list(df.columns) != cfg.TEMPLATE_COLUMNS:
        problems.append(f"columns must be {cfg.TEMPLATE_COLUMNS}, got {list(df.columns)}")
    else:
        if len(df) != cfg.N_VALIDATION_ROWS:
            problems.append(f"expected {cfg.N_VALIDATION_ROWS:,} rows, got {len(df):,}")
        if df["load_id"].isna().any() or df["load_id"].duplicated().any():
            problems.append("load_id has missing or duplicate values")
    _fail_if(problems, "validation_predictions_template.csv")
    df["load_id"] = df["load_id"].astype(str).str.strip()
    return df


def load_december_inputs(path: Path = cfg.DECEMBER_PATH) -> pd.DataFrame:
    """Load ``december_chart_inputs.csv`` (31 rows) and check its contract.

    ``predicted_rate`` may be empty (the original file) or filled (after the ML step).
    """
    df = _read_csv(path, "december_chart_inputs.csv")
    if list(df.columns) != cfg.DECEMBER_COLUMNS:
        raise DataContractError(f"december_chart_inputs.csv: columns must be exactly "
                                f"{cfg.DECEMBER_COLUMNS}, got {list(df.columns)}")
    body = _check_frame(df.drop(columns="predicted_rate"), label="december_chart_inputs.csv",
                        columns=cfg.DECEMBER_COLUMNS[:-1], date_range=cfg.DECEMBER_DATES)
    problems = []
    if len(body) != cfg.N_DECEMBER_ROWS or body["date"].duplicated().any():
        problems.append(f"expected {cfg.N_DECEMBER_ROWS} rows with one unique date each, "
                        f"got {len(body)} rows and {body['date'].nunique()} dates")
    if body["weight"].isna().any():
        problems.append("weight has missing values")
    _fail_if(problems, "december_chart_inputs.csv")
    body["predicted_rate"] = df["predicted_rate"]
    return body


# ==========================================================================
# City table (C7)
# ==========================================================================
def build_city_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) from the pickup and delivery columns of the given frames.

    Raises :class:`DataContractError` if any city has more than one coordinate pair.
    Returns a frame indexed by ``city`` with columns ``lat`` and ``lon``, sorted by city.
    """
    parts = []
    for df in frames:
        for end in ("pickup", "delivery"):
            parts.append(df[[end, f"{end}_lat", f"{end}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    coords = pd.concat(parts, ignore_index=True).drop_duplicates()
    pairs = coords.groupby("city").size()
    conflicts = pairs[pairs > 1]
    if len(conflicts):
        raise DataContractError(f"city table: {len(conflicts)} cities have more than one coordinate pair: "
                                f"{conflicts.index.tolist()[:10]}")
    return coords.set_index("city").sort_index()


def load_city_table() -> pd.DataFrame:
    """City table from train + validation (both files), checked to hold 72 cities."""
    table = build_city_table(load_train(), load_validation())
    if len(table) != cfg.N_CITIES:
        raise DataContractError(f"city table: expected {cfg.N_CITIES} cities, got {len(table)}")
    return table


# ==========================================================================
# market_index by date (C4)
# ==========================================================================
def build_market_index_table(*frames: pd.DataFrame, column: str = "market_index") -> pd.Series:
    """Mean of the non-missing ``column`` per date, pooled over the given frames.

    ``market_index`` is an input, not the target, and each date sits in only one
    file, so pooling train and validation is safe (C4). The result is a Series
    indexed by date (sorted) and is used by :func:`values_for_dates`.
    """
    pooled = pd.concat([f[["date", column]] for f in frames], ignore_index=True)
    table = pooled.dropna().groupby("date")[column].mean().sort_index()
    if table.empty:
        raise DataContractError(f"no observed {column} values to build a date table")
    return table


def values_for_dates(table: pd.Series, dates: Iterable) -> tuple[np.ndarray, int]:
    """Look up each date in a date -> value table, with a fallback for unseen dates.

    Exact date first. For a date with no observations, the fallback is the mean
    of the same weekday in the nearest weeks: +/-7 days, then +/-14, up to
    ``MARKET_INDEX_FALLBACK_MAX_WEEKS`` weeks; if even that finds nothing, the
    nearest observed date. Same weekday is used because the index has a strong
    weekly cycle: leave-one-date-out over the 365 observed days gives RMSE 0.024
    for +/-7 days, against 0.063 for the nearest date and 0.084 for the mean of
    the 3 nearest dates on each side.

    Returns (values aligned with ``dates``, number of dates that used the fallback).
    """
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates), dtype="object")))
    values = table.reindex(dates).to_numpy(dtype=float)
    missing = np.flatnonzero(np.isnan(values))
    for i in missing:
        d = dates[i]
        for weeks in range(1, cfg.MARKET_INDEX_FALLBACK_MAX_WEEKS + 1):
            near = table.reindex([d - pd.Timedelta(days=7 * weeks), d + pd.Timedelta(days=7 * weeks)]).dropna()
            if len(near):
                values[i] = near.mean()
                break
        else:
            gap = np.abs((table.index - d).days)
            values[i] = table.iloc[int(np.argmin(gap))]  # earlier date wins a tie
    return values, len(missing)


# ==========================================================================
# Cleaning (C3, C4, C5, C6, C7), fit on the training fold only
# ==========================================================================
def distance_band(distance: pd.Series | np.ndarray) -> np.ndarray:
    """Band number 0..17 of ``DIST_BANDS`` (right-closed, as ``pd.cut``).

    Distances above the last edge (4,000 mi) go to the last band, so a new long
    lane never breaks the outlier rule. No such distance exists today (max 3,440).
    """
    edges = np.asarray(cfg.DIST_BANDS, dtype=float)
    codes = np.searchsorted(edges, np.asarray(distance, dtype=float), side="left") - 1
    return np.clip(codes, 0, len(edges) - 2)


class Cleaner:
    """Cleaning rules learned on a training fold, then applied to any frame.

    ``fit(train_rows)`` learns, from labeled rows only:
      * C3: median ``abs(weight)`` per equipment (zeros count as missing).
      * C5: median rate per mile per (equipment, distance band), 54 cells.

    ``transform(df)`` applies, without dropping or reordering rows:
      * C2: strips whitespace on pickup, delivery and equipment.
      * C3: ``weight = abs(weight)``, 0 -> missing, missing -> equipment median.
      * C4: missing ``market_index`` -> same-date mean from ``market_index_table``.
      * C5 (labeled frames only): adds ``is_outlier`` = |ln(rpm / expected rpm)| > 0.5.
        With ``drop_outliers=True`` the flagged rows are removed: use this on
        training rows only, never on a holdout or validation frame.
      * C6, C7: ``quote_signal``, distance and coordinates pass through unchanged.
        ``quote_signal`` stays in the frame for QA; it must never be a feature
        (see ``config.NEVER_FEATURES``).

    No missing-value flags are added (the findings doc found the gaps random).
    Every call records its counts; see :meth:`log_table`.
    """

    def __init__(self, market_index_table: pd.Series):
        self.market_index_table = market_index_table
        self.log_: dict[str, dict] = {}

    # ---- fit ---------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "Cleaner":
        """Learn the C3 weight medians and the C5 expected rate per mile from labeled rows."""
        if cfg.TARGET not in train_df.columns or train_df[cfg.TARGET].isna().any():
            raise ValueError("Cleaner.fit needs labeled rows with no missing posted_rate")
        weight = train_df["weight"].abs().replace(0, np.nan)
        medians = weight.groupby(train_df["equipment"]).median()
        missing = [e for e in cfg.EQUIPMENT if pd.isna(medians.get(e))]
        if missing:
            raise ValueError(f"Cleaner.fit: no observed weight for equipment {missing}")
        self.weight_medians_ = {e: float(medians[e]) for e in cfg.EQUIPMENT}

        # Expected rate per mile per (equipment, band). A cell with no fit rows
        # (never the case on the given folds) borrows the nearest band of the
        # same equipment, so every row can always be scored.
        rpm = train_df[cfg.TARGET] / train_df["distance"]
        band = distance_band(train_df["distance"])
        cells = rpm.groupby([train_df["equipment"].to_numpy(), band]).median()
        n_bands = len(cfg.DIST_BANDS) - 1
        table = pd.DataFrame(np.nan, index=list(cfg.EQUIPMENT), columns=range(n_bands))
        for (equip, b), value in cells.items():
            table.loc[equip, b] = value
        self.n_empty_cells_ = int(table.isna().sum().sum())
        for equip in cfg.EQUIPMENT:
            row = table.loc[equip]
            known = np.flatnonzero(row.notna().to_numpy())
            for b in np.flatnonzero(row.isna().to_numpy()):
                table.loc[equip, b] = row.iloc[known[np.argmin(np.abs(known - b))]]
        self.expected_rpm_ = table
        self.n_fit_rows_ = len(train_df)
        return self

    # ---- transform ---------------------------------------------------------
    def transform(self, df: pd.DataFrame, *, drop_outliers: bool = False,
                  name: str | None = None) -> pd.DataFrame:
        """Apply the fitted rules to ``df`` and return a cleaned copy (input is not changed).

        Parameters
        ----------
        df : frame with the train or validation columns (as returned by the loaders).
        drop_outliers : remove C5 outlier rows. Training rows only; needs ``posted_rate``.
        name : label for this call in the cleaning log (default ``frame_<n>``).
        """
        if not hasattr(self, "weight_medians_"):
            raise RuntimeError("Cleaner is not fitted; call fit(train_rows) first")
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            raise ValueError("date must be datetime; load frames with the loaders in this module")
        labeled = cfg.TARGET in df.columns
        if drop_outliers and not labeled:
            raise ValueError("drop_outliers=True needs posted_rate; never drop validation rows")
        out = df.copy()
        log: dict[str, float] = {"rows_in": len(out)}

        # C2: text columns stripped; equipment must be known.
        for col in ("pickup", "delivery", "equipment"):
            out[col] = out[col].astype(str).str.strip()
        unknown = sorted(set(out["equipment"]) - set(cfg.EQUIPMENT))
        if unknown:
            raise DataContractError(f"equipment has unknown values {unknown}")

        # C3: weight sign errors, zeros, missing values.
        weight = out["weight"].astype(float)
        log["weight_negative_flipped"] = int((weight < 0).sum())
        weight = weight.abs()
        log["weight_zero_set_missing"] = int((weight == 0).sum())
        weight = weight.mask(weight == 0)
        missing_w = weight.isna()
        log["weight_filled"] = int(missing_w.sum())
        weight[missing_w] = out.loc[missing_w, "equipment"].map(self.weight_medians_)
        if not weight.between(cfg.WEIGHT_MIN, cfg.WEIGHT_MAX).all():
            bad = weight[~weight.between(cfg.WEIGHT_MIN, cfg.WEIGHT_MAX)]
            raise DataContractError(f"weight after C3 outside {cfg.WEIGHT_MIN:,.0f}..{cfg.WEIGHT_MAX:,.0f} "
                                    f"in {len(bad)} rows, e.g. {bad.head(3).tolist()}")
        out["weight"] = weight

        # C4: market_index from the same-date mean (pooled train + validation).
        missing_mi = out["market_index"].isna()
        log["market_index_filled"] = int(missing_mi.sum())
        if missing_mi.any():
            fill, n_fallback = values_for_dates(self.market_index_table, out.loc[missing_mi, "date"])
            out.loc[missing_mi, "market_index"] = fill
        else:
            n_fallback = 0
        log["market_index_filled_by_fallback"] = n_fallback
        if out["market_index"].isna().any() or (out["market_index"] <= 0).any():
            raise DataContractError("market_index still missing or non-positive after C4")

        # C5: rate outliers, labeled frames only.
        if labeled:
            if out[cfg.TARGET].isna().any():
                raise DataContractError("posted_rate has missing values in a labeled frame")
            rpm = out[cfg.TARGET] / out["distance"]
            band = distance_band(out["distance"])
            expected = self.expected_rpm_.to_numpy()[
                out["equipment"].map({e: i for i, e in enumerate(cfg.EQUIPMENT)}).to_numpy(), band]
            log_ratio = np.log(rpm.to_numpy() / expected)
            out["is_outlier"] = np.abs(log_ratio) > cfg.OUTLIER_LOG_THRESHOLD
            log["outliers_flagged"] = int(out["is_outlier"].sum())
            log["outliers_high"] = int((log_ratio > cfg.OUTLIER_LOG_THRESHOLD).sum())
            log["outliers_low"] = int((log_ratio < -cfg.OUTLIER_LOG_THRESHOLD).sum())
        if drop_outliers:
            out = out[~out["is_outlier"]]
        log["outliers_dropped"] = int(log["rows_in"] - len(out))
        log["rows_out"] = len(out)

        if not drop_outliers and len(out) != len(df):  # C8: never lose a row by accident
            raise AssertionError("transform changed the row count without drop_outliers")
        self.log_[name or f"frame_{len(self.log_) + 1}"] = log
        return out

    # ---- log ---------------------------------------------------------------
    def log_table(self) -> pd.DataFrame:
        """Counts of every fix, one column per ``transform`` call (in call order).

        Rows: rows_in, weight_negative_flipped, weight_zero_set_missing,
        weight_filled, market_index_filled, market_index_filled_by_fallback,
        outliers_flagged, outliers_high, outliers_low, outliers_dropped, rows_out.
        Outlier counts are blank for unlabeled frames.
        """
        order = ["rows_in", "weight_negative_flipped", "weight_zero_set_missing", "weight_filled",
                 "market_index_filled", "market_index_filled_by_fallback", "outliers_flagged",
                 "outliers_high", "outliers_low", "outliers_dropped", "rows_out"]
        return pd.DataFrame(self.log_).reindex(order).astype("Int64")

    def describe(self) -> str:
        """Plain-text summary of what was learned in fit (for logs and the report)."""
        medians = ", ".join(f"{e} {v:,.1f}" for e, v in self.weight_medians_.items())
        return (f"fit rows {self.n_fit_rows_:,} | weight medians: {medians} | "
                f"expected rpm cells {self.expected_rpm_.size} (empty, filled from nearest band: "
                f"{self.n_empty_cells_})")


# ==========================================================================
# Convenience: everything loaded at once
# ==========================================================================
class Dataset(NamedTuple):
    train: pd.DataFrame
    validation: pd.DataFrame
    template: pd.DataFrame
    city_table: pd.DataFrame
    market_index_table: pd.Series


def load_all() -> Dataset:
    """Load and check every input, and build the city and market_index tables.

    Also checks that the template ids equal the validation ids in file order.
    """
    train, validation, template = load_train(), load_validation(), load_template()
    if not np.array_equal(template["load_id"].to_numpy(), validation["load_id"].to_numpy()):
        raise DataContractError("template load_ids do not match validation.csv in file order")
    city_table = build_city_table(train, validation)
    if len(city_table) != cfg.N_CITIES:
        raise DataContractError(f"city table: expected {cfg.N_CITIES} cities, got {len(city_table)}")
    return Dataset(train, validation, template, city_table,
                   build_market_index_table(train, validation))


# ==========================================================================
# December chart inputs (section 6 recipe)
# ==========================================================================
def build_december_frame(city_table: pd.DataFrame | None = None,
                         validation: pd.DataFrame | None = None,
                         path: Path = cfg.DECEMBER_PATH) -> pd.DataFrame:
    """The 31 December rows in the validation schema, ready for ``Cleaner.transform``.

    * lat/lon from the city table (C7).
    * ``market_index``: mean of non-missing values in ``validation.csv`` on the
      same date (recipe). A date with none would use the C4 fallback; the
      count is in ``frame.attrs["market_index_fallback"]`` (0 today).
    * ``quote_signal``: same-date validation mean, only so the columns match the
      validation frame. It is not a model input and must not change predictions.
    * ``load_id``: ``DEC-YYYYMMDD``, so the frame passes the same checks as validation.
    * distance, equipment, weight, date: file values (weight already passes C3).

    Rows stay in file order, so predictions for this frame can go straight to
    :func:`write_december_predictions`. No values are missing.
    """
    dec = load_december_inputs(path)
    city_table = load_city_table() if city_table is None else city_table
    validation = load_validation() if validation is None else validation

    unknown = sorted((set(dec["pickup"]) | set(dec["delivery"])) - set(city_table.index))
    if unknown:
        raise DataContractError(f"December cities not in the city table: {unknown}")

    frame = pd.DataFrame({"load_id": "DEC-" + dec["date"].dt.strftime("%Y%m%d"),
                          "pickup": dec["pickup"], "delivery": dec["delivery"]})
    for end in ("pickup", "delivery"):
        frame[f"{end}_lat"] = dec[end].map(city_table["lat"]).to_numpy()
        frame[f"{end}_lon"] = dec[end].map(city_table["lon"]).to_numpy()
    frame["distance"] = dec["distance"]
    frame["equipment"] = dec["equipment"]
    frame["weight"] = dec["weight"]
    frame["date"] = dec["date"]
    mi, n_mi_fallback = values_for_dates(build_market_index_table(validation), dec["date"])
    qs, _ = values_for_dates(build_market_index_table(validation, column="quote_signal"), dec["date"])
    frame["market_index"] = mi
    frame["quote_signal"] = qs

    frame = _check_frame(frame, label="December model frame", columns=cfg.VALIDATION_COLUMNS,
                         date_range=cfg.DECEMBER_DATES)
    if frame.isna().any().any():
        raise DataContractError("December model frame has missing values")
    frame.attrs["market_index_fallback"] = n_mi_fallback
    return frame.reset_index(drop=True)


# ==========================================================================
# Writers
# ==========================================================================
def _check_predictions(preds, n: int, label: str) -> np.ndarray:
    values = np.asarray(preds, dtype=float).ravel()
    if len(values) != n:
        raise ValueError(f"{label}: expected {n} predictions, got {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: {int((~np.isfinite(values)).sum())} predictions are not finite")
    if (np.round(values, 2) <= 0).any():
        raise ValueError(f"{label}: {int((np.round(values, 2) <= 0).sum())} predictions are not positive")
    return values


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write with LF line endings (as the original files) via a temp file, then replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def write_december_predictions(preds, path: Path = cfg.DECEMBER_PATH,
                               copy_path: Path | None = cfg.DECEMBER_PREDICTIONS_COPY_PATH,
                               source_path: Path = cfg.DECEMBER_PATH) -> Path:
    """Fill ``predicted_rate`` in the December file and write it.

    ``preds``: 31 values in the row order of :func:`build_december_frame` (the file order).
    The six input columns are copied as the original text, so values and formats
    stay exactly as given (360, 32000, 2025-12-01); only ``predicted_rate`` is
    filled (rounded to cents). Columns stay the original 7 in their order.
    Writes ``path`` (default ``data/december_chart_inputs.csv``, which
    ``score.py`` reads) and a copy at ``copy_path`` (``outputs/december_predictions.csv``;
    ``None`` skips it). Returns ``path``.
    """
    raw = _read_csv(source_path, "December source", dtype=str, keep_default_na=False)
    if list(raw.columns) != cfg.DECEMBER_COLUMNS or len(raw) != cfg.N_DECEMBER_ROWS:
        raise DataContractError(f"December source must have columns {cfg.DECEMBER_COLUMNS} "
                                f"and {cfg.N_DECEMBER_ROWS} rows")
    values = _check_predictions(preds, cfg.N_DECEMBER_ROWS, "December predictions")
    out = raw.copy()
    out["predicted_rate"] = [f"{v:.2f}" for v in values]

    for target in [path] + ([copy_path] if copy_path is not None else []):
        _write_csv(out, target)
        back = pd.read_csv(target, dtype=str, keep_default_na=False)
        if not back[cfg.DECEMBER_COLUMNS[:-1]].equals(raw[cfg.DECEMBER_COLUMNS[:-1]]):
            raise AssertionError(f"{target}: input columns changed while writing")
    return Path(path)


def write_validation_predictions(load_ids, preds, path: Path = cfg.VALIDATION_PREDICTIONS_PATH,
                                 template_path: Path = cfg.TEMPLATE_PATH) -> Path:
    """Write ``validation_predictions.csv`` (columns ``load_id,predicted_rate``) in template order.

    ``load_ids`` and ``preds`` are aligned with each other in any order. They
    must cover the 12,000 template ids exactly once; every prediction must be
    finite and positive. Predictions are rounded to cents. Returns ``path``.
    """
    template = load_template(template_path)
    ids = pd.Series(np.asarray(load_ids).ravel(), dtype=str).str.strip()
    values = _check_predictions(preds, len(ids), "validation predictions")
    if ids.duplicated().any():
        raise ValueError(f"validation predictions: {int(ids.duplicated().sum())} duplicate load_ids")
    expected = set(template["load_id"])
    missing, extra = expected - set(ids), set(ids) - expected
    if missing or extra or len(ids) != cfg.N_VALIDATION_ROWS:
        raise ValueError(f"validation predictions: ids do not match the template "
                         f"({len(ids):,} given, {len(missing)} missing, {len(extra)} extra)")
    by_id = pd.Series(values, index=ids.to_numpy())
    out = pd.DataFrame({"load_id": template["load_id"],
                        "predicted_rate": [f"{v:.2f}" for v in by_id.reindex(template["load_id"]).to_numpy()]})
    _write_csv(out, path)

    back = pd.read_csv(path)
    if (list(back.columns) != cfg.TEMPLATE_COLUMNS or len(back) != cfg.N_VALIDATION_ROWS
            or not back["load_id"].equals(template["load_id"])
            or not np.isfinite(back["predicted_rate"]).all() or (back["predicted_rate"] <= 0).any()):
        raise AssertionError(f"{path}: written file failed its own check")
    return Path(path)


# ==========================================================================
# Self-check: python src/data.py
# ==========================================================================
def _expect(label: str, got, want) -> bool:
    ok = got == want
    print(f"  [{'ok' if ok else 'MISMATCH'}] {label}: {got}" + ("" if ok else f" (expected {want})"))
    return ok


def main() -> None:
    import tempfile

    pd.set_option("display.width", 200)
    checks: list[bool] = []
    ds = load_all()
    train, val = ds.train, ds.validation
    print("Contract checks passed (C1, C2, C8).")
    print(f"  train {len(train):,} rows {train['date'].min().date()}..{train['date'].max().date()}; "
          f"validation {len(val):,} rows {val['date'].min().date()}..{val['date'].max().date()}; "
          f"template ids match validation order")
    print(f"  whitespace stripped (C2): train {train.attrs.get('whitespace_stripped')}, "
          f"validation {val.attrs.get('whitespace_stripped')}")
    checks.append(_expect("cities in table (C7)", len(ds.city_table), cfg.N_CITIES))
    checks.append(_expect("dates in market_index table", len(ds.market_index_table), 365))

    # 1. Final-fit setup: cleaning learned on all of train.
    full = Cleaner(ds.market_index_table).fit(train)
    train_clean = full.transform(train, drop_outliers=True, name="train (fit: all train)")
    val_clean = full.transform(val, name="validation (fit: all train)")
    print("\nFit on all of train: " + full.describe())
    log = full.log_table()
    for col, want in (("train (fit: all train)", dict(weight_negative_flipped=292, weight_filled=300,
                                                      market_index_filled=374, outliers_flagged=677,
                                                      rows_out=47_323)),
                      ("validation (fit: all train)", dict(weight_negative_flipped=145, weight_filled=165,
                                                           market_index_filled=249, rows_out=12_000))):
        for key, value in want.items():
            checks.append(_expect(f"{col} {key}", int(log.loc[key, col]), value))
    checks.append(_expect("weight medians (doc: 31,444 / 31,577 / 31,532.5)", full.weight_medians_,
                          {"Dry Van": 31444.0, "Reefer": 31577.0, "Flatbed": 31532.5}))

    # 2. Holdout setup: cleaning learned on Jan-Aug only.
    jan_aug = train[train["date"] <= cfg.HOLDOUT_FIT_DATES[1]]
    sep_oct = train[train["date"] >= cfg.HOLDOUT_TEST_DATES[0]]
    early = Cleaner(ds.market_index_table).fit(jan_aug)
    fit_rows = early.transform(jan_aug, drop_outliers=True, name="Jan-Aug (fit: Jan-Aug)")
    test_rows = early.transform(sep_oct, name="Sep-Oct (fit: Jan-Aug)")
    early.transform(val, name="validation (fit: Jan-Aug)")
    print("\nFit on Jan-Aug only: " + early.describe())
    checks.append(_expect("Jan-Aug clean fit rows", len(fit_rows), 37_944))
    checks.append(_expect("Jan-Aug flagged", int(early.log_["Jan-Aug (fit: Jan-Aug)"]["outliers_flagged"]), 533))
    checks.append(_expect("Sep-Oct flagged (kept, scoring only)", int(test_rows["is_outlier"].sum()), 144))
    same = np.array_equal(early.transform(train, name="_tmp")["is_outlier"].to_numpy(),
                          full.transform(train, name="_tmp")["is_outlier"].to_numpy())
    del early.log_["_tmp"], full.log_["_tmp"]
    checks.append(_expect("Jan-Aug medians flag the identical set as full-train medians", same, True))

    print("\nCleaning log, fit on all of train:")
    print(full.log_table().to_string())
    print("\nCleaning log, fit on Jan-Aug only:")
    print(early.log_table().to_string())

    # 3. Invariants.
    print("\nInvariants:")
    for label, frame, raw in (("train", train_clean, train.loc[train_clean.index]), ("validation", val_clean, val)):
        checks.append(_expect(f"{label}: no missing weight or market_index",
                              int(frame[["weight", "market_index"]].isna().sum().sum()), 0))
        unchanged = all(frame[c].equals(raw[c]) for c in ["load_id", "quote_signal", "distance", *cfg.COORD_COLUMNS])
        checks.append(_expect(f"{label}: load_id, quote_signal, distance, coordinates unchanged (C6, C7)",
                              unchanged, True))
    checks.append(_expect("validation keeps all rows in file order (C8)",
                          bool(val_clean["load_id"].equals(val["load_id"])), True))

    # 4. December frame and writers (on temp copies, never on data/).
    dec = build_december_frame(ds.city_table, val)
    print(f"\nDecember frame: {dec.shape}, market_index {dec['market_index'].min():.4f}.."
          f"{dec['market_index'].max():.4f} (mean {dec['market_index'].mean():.4f}), "
          f"fallback dates {dec.attrs['market_index_fallback']}")
    print(f"  {dec.loc[0, 'pickup']} ({dec.loc[0, 'pickup_lat']}, {dec.loc[0, 'pickup_lon']}), "
          f"{dec.loc[0, 'delivery']} ({dec.loc[0, 'delivery_lat']}, {dec.loc[0, 'delivery_lon']})")
    dec_clean = full.transform(dec, name="december")
    checks.append(_expect("December frame unchanged by Cleaner.transform",
                          bool(dec_clean[cfg.VALIDATION_COLUMNS].equals(dec)), True))
    del full.log_["december"]

    sys.path.insert(0, str(cfg.ROOT))
    import score  # the official checker, used read-only

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dec_out = write_december_predictions(2.2 * dec["distance"], path=tmp / "december_chart_inputs.csv",
                                             copy_path=tmp / "outputs" / "december_predictions.csv")
        score.validate_december(pd.read_csv(dec_out))
        original = cfg.DECEMBER_PATH.read_text().splitlines()
        written = dec_out.read_text().splitlines()
        checks.append(_expect("December copy: header and input values byte-identical",
                              [ln.rsplit(",", 1)[0] for ln in written] == [ln.rsplit(",", 1)[0] for ln in original],
                              True))
        val_out = write_validation_predictions(val_clean["load_id"], 2.2 * val_clean["distance"],
                                               path=tmp / "validation_predictions.csv")
        score.validate_predictions(pd.read_csv(val_out))
        print("  score.py checks pass on test writes of both output files (temp dir)")

    print(f"\n{sum(checks)} of {len(checks)} checks match.")
    if not all(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
