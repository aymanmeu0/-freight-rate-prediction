"""Submission files: shape, ids, values, the original December inputs, and score.py's own validators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import score
from conftest import ROOT_ORIGINALS, needs_outputs, needs_root_originals
from src import config as cfg

pytestmark = needs_outputs

VALIDATION_COPY = cfg.OUTPUTS_DIR / "validation_predictions.csv"


@pytest.fixture(scope="module")
def vp():
    return pd.read_csv(cfg.VALIDATION_PREDICTIONS_PATH, dtype={"load_id": str})


@pytest.fixture(scope="module")
def dec():
    return pd.read_csv(cfg.DECEMBER_PATH)


def test_validation_file_shape_ids_and_values(vp):
    lines = cfg.VALIDATION_PREDICTIONS_PATH.read_text().splitlines()
    assert lines[0] == "load_id,predicted_rate"
    assert list(vp.columns) == ["load_id", "predicted_rate"]
    assert len(vp) == 12_000 and vp["load_id"].is_unique
    template = pd.read_csv(cfg.TEMPLATE_PATH, dtype={"load_id": str})
    assert vp["load_id"].tolist() == template["load_id"].tolist()     # same ids, same order
    rate = vp["predicted_rate"].to_numpy(float)
    assert np.isfinite(rate).all() and (rate > 0).all()
    assert all(len(line.rsplit(",", 1)[1].split(".")[1]) == 2 for line in lines[1:])  # cents


def test_validation_file_passes_score_py(vp):
    score.validate_predictions(pd.read_csv(cfg.VALIDATION_PREDICTIONS_PATH))


def test_outputs_copies_are_byte_identical():
    assert VALIDATION_COPY.read_bytes() == cfg.VALIDATION_PREDICTIONS_PATH.read_bytes()
    assert cfg.DECEMBER_PREDICTIONS_COPY_PATH.read_bytes() == cfg.DECEMBER_PATH.read_bytes()


def test_december_file_keeps_the_original_columns_and_31_dates(dec):
    assert list(dec.columns) == cfg.DECEMBER_COLUMNS
    assert len(dec) == 31
    assert pd.to_datetime(dec["date"]).tolist() == list(pd.date_range("2025-12-01", "2025-12-31"))
    assert (dec["pickup"] == "Lexington").all() and (dec["delivery"] == "Fort Wayne").all()
    assert (dec["distance"] == 360).all() and (dec["weight"] == 32000).all() and (dec["equipment"] == "Dry Van").all()
    assert np.isfinite(dec["predicted_rate"]).all() and (dec["predicted_rate"] > 0).all()


@needs_root_originals
def test_december_input_columns_are_byte_identical_to_the_original():
    original = ROOT_ORIGINALS[cfg.DECEMBER_PATH].read_text().splitlines()
    filled = cfg.DECEMBER_PATH.read_text().splitlines()
    assert len(filled) == len(original) == 32
    assert [ln.rsplit(",", 1)[0] for ln in filled] == [ln.rsplit(",", 1)[0] for ln in original]
    assert all(ln.endswith(",") for ln in original[1:])          # the original was empty
    assert all(not ln.endswith(",") for ln in filled[1:])         # and now every row is filled


def test_december_file_passes_score_py(dec):
    checked = score.validate_december(dec)
    assert len(checked) == 31


@needs_root_originals
def test_input_files_in_data_are_untouched_copies_of_the_originals():
    for copy, original in ROOT_ORIGINALS.items():
        if copy != cfg.DECEMBER_PATH:   # the December file is filled by the pipeline writer
            assert copy.read_bytes() == original.read_bytes(), copy.name


@pytest.mark.parametrize("breakage", ["drop a row", "negative rate", "duplicate id", "extra column"])
def test_score_py_validators_do_reject_broken_files(vp, dec, breakage):
    """The validators are real gates: a broken copy of our own file fails them."""
    bad_vp, bad_dec = vp.copy(), dec.copy()
    if breakage == "drop a row":
        bad_vp, bad_dec = bad_vp.iloc[1:], bad_dec.iloc[1:]
    elif breakage == "negative rate":
        bad_vp.loc[0, "predicted_rate"] = -1.0
        bad_dec.loc[0, "predicted_rate"] = -1.0
    elif breakage == "duplicate id":
        bad_vp.loc[1, "load_id"] = bad_vp.loc[0, "load_id"]
        bad_dec.loc[1, "date"] = bad_dec.loc[0, "date"]
    else:
        bad_vp["x"] = 1
        bad_dec["x"] = 1
    with pytest.raises(SystemExit):
        score.validate_predictions(bad_vp)
    with pytest.raises(SystemExit):
        score.validate_december(bad_dec)
