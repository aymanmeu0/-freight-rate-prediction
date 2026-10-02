"""Data contract: the loaders accept the real files and reject broken ones; the writers reject bad predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import score
from conftest import make_synthetic, needs_data, synthetic_december, synthetic_template, write_csv
from src import config as cfg
from src import data


# --------------------------------------------------------------------------
# Real files
# --------------------------------------------------------------------------
@needs_data
def test_real_files_pass_the_contract(ds):
    assert len(ds.train) == cfg.N_TRAIN_ROWS == 48_000
    assert len(ds.validation) == cfg.N_VALIDATION_ROWS == 12_000
    assert list(ds.train.columns) == cfg.TRAIN_COLUMNS
    assert list(ds.validation.columns) == cfg.VALIDATION_COLUMNS
    assert ds.train["date"].min() == pd.Timestamp("2025-01-01")
    assert ds.train["date"].max() == pd.Timestamp("2025-10-31")
    assert ds.validation["date"].min() == pd.Timestamp("2025-11-01")
    assert ds.validation["date"].max() == pd.Timestamp("2025-12-31")
    assert ds.template["load_id"].tolist() == ds.validation["load_id"].tolist()
    assert len(ds.city_table) == 72
    assert len(ds.market_index_table) == 365


@needs_data
def test_december_inputs_load():
    dec = data.load_december_inputs()
    assert len(dec) == 31 and dec["date"].is_unique
    assert set(dec["pickup"]) == {"Lexington"} and set(dec["delivery"]) == {"Fort Wayne"}


# --------------------------------------------------------------------------
# Synthetic files: valid ones load, broken ones are rejected
# --------------------------------------------------------------------------
def test_synthetic_valid_train_loads(tmp_path):
    raw = make_synthetic()
    df = data.load_train(write_csv(raw, tmp_path / "t.csv"))
    assert len(df) == len(raw)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_missing_weight_and_index_are_allowed(tmp_path):
    raw = make_synthetic()
    raw.loc[0, "weight"] = np.nan
    raw.loc[1, "market_index"] = np.nan
    df = data.load_train(write_csv(raw, tmp_path / "t.csv"))
    assert df["weight"].isna().sum() == 1 and df["market_index"].isna().sum() == 1


def _swap_columns(df):
    cols = list(df.columns)
    cols[1], cols[2] = cols[2], cols[1]
    return df[cols]


def _set(col, value, row=0):
    def f(df):
        df = df.copy()
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            df[col] = df[col].astype(object)
        df.loc[row, col] = value
        return df
    return f


BROKEN_TRAIN = {
    "columns reordered": _swap_columns,
    "column missing": lambda df: df.drop(columns="quote_signal"),
    "extra column": lambda df: df.assign(month=1),
    "unknown equipment": _set("equipment", "Box Truck"),
    "bad date format": _set("date", "03/05/2025"),
    "date outside Jan-Oct": _set("date", "2025-11-05"),
    "duplicate load_id": _set("load_id", "TR-000002"),
    "text in a number column": _set("distance", "abc"),
    "missing distance": _set("distance", np.nan),
    "infinite market_index": _set("market_index", np.inf),
    "zero posted_rate": _set("posted_rate", 0.0),
    "missing posted_rate": _set("posted_rate", np.nan),
    "negative distance": _set("distance", -10.0),
    "latitude out of range": _set("pickup_lat", 95.0),
    "missing pickup": _set("pickup", np.nan),
}


@pytest.mark.parametrize("breakage", list(BROKEN_TRAIN))
def test_broken_train_file_is_rejected(tmp_path, breakage):
    broken = BROKEN_TRAIN[breakage](make_synthetic())
    with pytest.raises(data.DataContractError):
        data.load_train(write_csv(broken, tmp_path / "t.csv"))


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(data.DataContractError, match="not found"):
        data.load_train(tmp_path / "nope.csv")


def test_validation_loader_rejects_labels_and_train_dates(tmp_path):
    good = make_synthetic(labeled=False, start="2025-11-01", prefix="TE")
    assert len(data.load_validation(write_csv(good, tmp_path / "v.csv"))) == len(good)
    with pytest.raises(data.DataContractError):  # a posted_rate column breaks the column contract
        data.load_validation(write_csv(make_synthetic(start="2025-11-01", prefix="TE"), tmp_path / "v2.csv"))
    with pytest.raises(data.DataContractError):  # October date in the validation file
        data.load_validation(write_csv(_set("date", "2025-10-31")(good), tmp_path / "v3.csv"))


def test_template_loader_rejects_wrong_rows(tmp_path):
    path = synthetic_template(tmp_path / "tpl.csv")
    assert len(data.load_template(path)) == 12_000
    tpl = pd.read_csv(path)
    with pytest.raises(data.DataContractError):
        data.load_template(write_csv(tpl.iloc[:-1], tmp_path / "short.csv"))
    dup = tpl.copy()
    dup.loc[1, "load_id"] = dup.loc[0, "load_id"]
    with pytest.raises(data.DataContractError):
        data.load_template(write_csv(dup, tmp_path / "dup.csv"))


def test_december_loader_rejects_wrong_shape(tmp_path):
    path = synthetic_december(tmp_path / "dec.csv")
    assert len(data.load_december_inputs(path)) == 31
    dec = pd.read_csv(path)
    with pytest.raises(data.DataContractError):
        data.load_december_inputs(write_csv(dec.iloc[:30], tmp_path / "d30.csv"))
    with pytest.raises(data.DataContractError):
        data.load_december_inputs(write_csv(dec[dec.columns[::-1]], tmp_path / "rev.csv"))


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------
def test_validation_writer_puts_rows_in_template_order(tmp_path):
    tpl = synthetic_template(tmp_path / "tpl.csv")
    ids = pd.read_csv(tpl)["load_id"].to_numpy()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(ids))
    preds = np.arange(1, len(ids) + 1, dtype=float) * 1.0037
    out = data.write_validation_predictions(ids[order], preds[order], path=tmp_path / "vp.csv", template_path=tpl)
    back = pd.read_csv(out)
    score.validate_predictions(back)
    assert back["load_id"].tolist() == ids.tolist()
    assert np.allclose(back["predicted_rate"], preds, rtol=0, atol=0.0051)


@pytest.mark.parametrize("breakage", ["short", "nan", "inf", "zero", "negative", "duplicate id", "foreign id"])
def test_validation_writer_rejects_bad_predictions(tmp_path, breakage):
    tpl = synthetic_template(tmp_path / "tpl.csv")
    ids = pd.read_csv(tpl)["load_id"].to_numpy().astype(object)
    preds = np.full(len(ids), 1000.0)
    if breakage == "short":
        ids, preds = ids[:-1], preds[:-1]
    elif breakage in ("nan", "inf", "zero", "negative"):
        preds[5] = {"nan": np.nan, "inf": np.inf, "zero": 0.0, "negative": -5.0}[breakage]
    elif breakage == "duplicate id":
        ids[1] = ids[0]
    else:
        ids[3] = "TE-999999"
    with pytest.raises(ValueError):
        data.write_validation_predictions(ids, preds, path=tmp_path / "vp.csv", template_path=tpl)
    assert not (tmp_path / "vp.csv").exists()


def test_december_writer_keeps_inputs_and_passes_score(tmp_path):
    src = synthetic_december(tmp_path / "dec.csv")
    out = data.write_december_predictions(np.linspace(800, 880, 31), path=tmp_path / "dec_out.csv",
                                          copy_path=tmp_path / "copy.csv", source_path=src)
    score.validate_december(pd.read_csv(out))
    original = [ln.rsplit(",", 1)[0] for ln in src.read_text().splitlines()]
    written = [ln.rsplit(",", 1)[0] for ln in out.read_text().splitlines()]
    assert written == original
    assert out.read_bytes() == (tmp_path / "copy.csv").read_bytes()


@pytest.mark.parametrize("preds", [np.full(30, 850.0), np.r_[np.full(30, 850.0), np.inf], np.r_[np.full(30, 850.0), -1.0]])
def test_december_writer_rejects_bad_predictions(tmp_path, preds):
    src = synthetic_december(tmp_path / "dec.csv")
    with pytest.raises(ValueError):
        data.write_december_predictions(preds, path=tmp_path / "o.csv", copy_path=None, source_path=src)
