"""Cleaning: the documented fixes and counts, no lost or reordered validation rows, fold isolation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic, needs_data, write_csv
from src import config as cfg
from src import data
from src import train as T


# --------------------------------------------------------------------------
# Unit checks on a synthetic frame
# --------------------------------------------------------------------------
@pytest.fixture
def synth(tmp_path):
    raw = make_synthetic(n=90)
    # Keep every load in one distance band so each equipment cell has 30 rows.
    raw["distance"] = np.linspace(410, 490, len(raw)).round(1)
    premium = raw["equipment"].map({"Dry Van": 1.0, "Reefer": 1.13, "Flatbed": 1.08})
    raw["posted_rate"] = (raw["distance"] * 2.2 * premium).round(2)
    raw.loc[0, "weight"] = -20_000.0          # sign error
    raw.loc[1, "weight"] = 0.0                # zero -> missing -> median
    raw.loc[2, "weight"] = np.nan             # missing -> median
    raw.loc[3, "date"] = raw.loc[5, "date"]  # share a date with an observed index
    raw.loc[3, "market_index"] = np.nan       # missing -> same-date mean
    raw.loc[4, "posted_rate"] = raw.loc[4, "posted_rate"] * 3.5   # corrupted label
    df = data.load_train(write_csv(raw, tmp_path / "t.csv"))
    table = data.build_market_index_table(df)
    return df, data.Cleaner(table).fit(df), table


def test_weight_sign_zero_and_missing(synth):
    df, cleaner, _ = synth
    out = cleaner.transform(df)
    assert out.loc[0, "weight"] == 20_000.0
    medians = df["weight"].abs().replace(0, np.nan).groupby(df["equipment"]).median()
    for i in (1, 2):
        assert out.loc[i, "weight"] == medians[df.loc[i, "equipment"]]
    assert out["weight"].notna().all()
    log = cleaner.log_[next(iter(cleaner.log_))]
    assert (log["weight_negative_flipped"], log["weight_zero_set_missing"], log["weight_filled"]) == (1, 1, 2)


def test_market_index_filled_from_same_date(synth):
    df, cleaner, table = synth
    out = cleaner.transform(df)
    assert out.loc[3, "market_index"] == pytest.approx(table[df.loc[3, "date"]])
    assert out["market_index"].notna().all()


def test_corrupted_label_flagged_and_dropped_only_on_request(synth):
    df, cleaner, _ = synth
    kept = cleaner.transform(df)
    assert kept["is_outlier"].sum() == 1 and bool(kept.loc[4, "is_outlier"])
    assert len(kept) == len(df)
    dropped = cleaner.transform(df, drop_outliers=True)
    assert len(dropped) == len(df) - 1 and 4 not in dropped.index


def test_transform_never_changes_its_input_or_the_fitted_params(synth):
    df, cleaner, _ = synth
    before = df.copy()
    medians = dict(cleaner.weight_medians_)
    rpm = cleaner.expected_rpm_.copy()
    wild = df.copy()
    wild["posted_rate"] *= 9
    wild["weight"] = -wild["weight"]
    cleaner.transform(wild)
    cleaner.transform(df, drop_outliers=True)
    pd.testing.assert_frame_equal(df, before)
    assert cleaner.weight_medians_ == medians
    pd.testing.assert_frame_equal(cleaner.expected_rpm_, rpm)


def test_unlabeled_rows_can_never_be_dropped(synth):
    df, cleaner, _ = synth
    with pytest.raises(ValueError):
        cleaner.transform(df.drop(columns=cfg.TARGET), drop_outliers=True)


def test_unfitted_cleaner_refuses_to_transform(synth):
    df, _, table = synth
    with pytest.raises(RuntimeError):
        data.Cleaner(table).transform(df)


# --------------------------------------------------------------------------
# Real data: documented counts
# --------------------------------------------------------------------------
@needs_data
def test_full_train_counts_match_the_docs(full_clean):
    log = full_clean.cleaner.log_
    tr, va = log["train"], log["validation"]
    assert (tr["weight_negative_flipped"], tr["weight_filled"], tr["market_index_filled"]) == (292, 300, 374)
    assert (tr["outliers_flagged"], tr["outliers_high"], tr["outliers_low"]) == (677, 340, 337)
    assert tr["rows_out"] == 47_323 and tr["weight_zero_set_missing"] == 0
    assert (va["weight_negative_flipped"], va["weight_filled"], va["market_index_filled"]) == (145, 165, 249)
    assert va["rows_out"] == 12_000 and va["outliers_dropped"] == 0
    assert full_clean.cleaner.weight_medians_ == {"Dry Van": 31444.0, "Reefer": 31577.0, "Flatbed": 31532.5}


@needs_data
def test_jan_aug_counts_match_the_docs(ds):
    d = ds.train["date"]
    jan_aug, sep_oct = ds.train[d <= cfg.HOLDOUT_FIT_DATES[1]], ds.train[d >= cfg.HOLDOUT_TEST_DATES[0]]
    c = data.Cleaner(ds.market_index_table).fit(jan_aug)
    fit = c.transform(jan_aug, drop_outliers=True, name="fit")
    test = c.transform(sep_oct, name="test")
    f, t = c.log_["fit"], c.log_["test"]
    assert (f["weight_negative_flipped"], f["weight_filled"], f["market_index_filled"], f["outliers_flagged"]) == (233, 235, 301, 533)
    assert (t["weight_negative_flipped"], t["weight_filled"], t["market_index_filled"], t["outliers_flagged"]) == (59, 65, 73, 144)
    assert len(fit) == 37_944 and len(test) == 9_523


@needs_data
def test_validation_rows_never_dropped_or_reordered(ds, full_clean):
    val = full_clean.val
    assert len(val) == 12_000
    assert val["load_id"].tolist() == ds.validation["load_id"].tolist() == ds.template["load_id"].tolist()
    assert val.index.equals(ds.validation.index)


@needs_data
def test_clean_values_are_complete_and_pass_through_columns_unchanged(ds, full_clean):
    for clean, raw in ((full_clean.train, ds.train.loc[full_clean.train.index]), (full_clean.val, ds.validation)):
        assert clean[["weight", "market_index"]].notna().all().all()
        assert clean["weight"].between(cfg.WEIGHT_MIN, cfg.WEIGHT_MAX).all()
        present = raw["weight"].notna() & (raw["weight"] != 0)
        assert np.array_equal(clean.loc[present, "weight"], raw.loc[present, "weight"].abs())
        given = raw["market_index"].notna()
        assert np.array_equal(clean.loc[given, "market_index"], raw.loc[given, "market_index"])
        for col in ["load_id", "quote_signal", "distance", "date", *cfg.COORD_COLUMNS]:
            assert clean[col].equals(raw[col]), col


# --------------------------------------------------------------------------
# Real data: a Cleaner fit on one fold never reads another
# --------------------------------------------------------------------------
def _scramble_after(train: pd.DataFrame, first_day: pd.Timestamp, seed: int = 1) -> pd.DataFrame:
    """Copy of train with every row on or after ``first_day`` badly altered."""
    rng = np.random.default_rng(seed)
    out = train.copy()
    late = out["date"] >= first_day
    n = int(late.sum())
    out.loc[late, cfg.TARGET] = out.loc[late, cfg.TARGET] * rng.uniform(0.2, 6.0, n)
    out.loc[late, "weight"] = -rng.uniform(5_000, 47_500, n)
    out.loc[late, "quote_signal"] = rng.normal(size=n)
    return out


@needs_data
def test_cleaner_fit_on_jan_aug_ignores_sep_oct(ds):
    altered = _scramble_after(ds.train, cfg.HOLDOUT_TEST_DATES[0])
    assert not altered.equals(ds.train)
    jan_aug = lambda t: t[t["date"] <= cfg.HOLDOUT_FIT_DATES[1]]  # noqa: E731
    a = data.Cleaner(ds.market_index_table).fit(jan_aug(ds.train))
    b = data.Cleaner(ds.market_index_table).fit(jan_aug(altered))
    assert a.weight_medians_ == b.weight_medians_
    pd.testing.assert_frame_equal(a.expected_rpm_, b.expected_rpm_)
    # The same through the project's split function.
    pd.testing.assert_frame_equal(T.holdout_split(ds).fit, T.holdout_split(ds._replace(train=altered)).fit)


@needs_data
def test_cv_fold_fit_rows_ignore_later_months(ds):
    altered = _scramble_after(ds.train, pd.Timestamp("2025-05-01"))
    first_fold, first_fold_alt = T.cv_splits(ds)[0], T.cv_splits(ds._replace(train=altered))[0]
    pd.testing.assert_frame_equal(first_fold.fit, first_fold_alt.fit)
