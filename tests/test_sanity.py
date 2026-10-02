"""Sanity of the submitted predictions against the training data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import needs_outputs
from src import config as cfg
from src import data

pytestmark = needs_outputs


@pytest.fixture(scope="module")
def dec_pred():
    return pd.read_csv(cfg.DECEMBER_PATH)["predicted_rate"].to_numpy(float)


@pytest.fixture(scope="module")
def val_pred():
    return pd.read_csv(cfg.VALIDATION_PREDICTIONS_PATH)["predicted_rate"].to_numpy(float)


def _lane(train):
    return train[(train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")
                 & (train["equipment"] == "Dry Van")]


def test_december_inside_the_lanes_history(full_clean, dec_pred):
    lane = _lane(full_clean.train)
    assert len(lane) == 21
    lo, hi = lane[cfg.TARGET].min(), lane[cfg.TARGET].max()
    assert (lo, hi) == (757.93, 934.37)
    assert ((dec_pred >= lo) & (dec_pred <= hi)).all()


def test_december_line_is_smooth_and_follows_the_index(full_clean, ds, dec_pred):
    jumps = np.abs(np.diff(dec_pred) / dec_pred[:-1])
    assert jumps.max() < 0.05                                 # no day-to-day jump above 5%
    assert dec_pred.max() / dec_pred.min() - 1 > 0.01          # not a flat line
    frame = data.build_december_frame(ds.city_table, ds.validation)
    # Weekly saw-tooth: after removing the month's straight-line climb, the line moves with the index.
    t = np.arange(31)
    detrended = np.log(dec_pred) - np.polyval(np.polyfit(t, np.log(dec_pred), 1), t)
    assert np.corrcoef(detrended, frame["market_index"])[0, 1] > 0.8


def test_validation_per_equipment_and_band_near_training_medians(full_clean, val_pred):
    """Median predicted rate per mile within 10% of the clean training median, in every cell."""
    val, train = full_clean.val, full_clean.train
    pred = pd.DataFrame({"equipment": val["equipment"].to_numpy(), "band": data.distance_band(val["distance"]),
                         "rpm": val_pred / val["distance"].to_numpy()})
    actual = pd.DataFrame({"equipment": train["equipment"].to_numpy(), "band": data.distance_band(train["distance"]),
                           "rpm": (train[cfg.TARGET] / train["distance"]).to_numpy()})
    ratio = pred.groupby(["equipment", "band"])["rpm"].median() / actual.groupby(["equipment", "band"])["rpm"].median()
    assert len(ratio) == 54 and ratio.notna().all()
    assert ratio.between(0.90, 1.10).all(), ratio[~ratio.between(0.90, 1.10)]


def test_validation_range_and_unseen_city_rows(full_clean, val_pred):
    actual = full_clean.train[cfg.TARGET]
    assert 0.5 * actual.min() < val_pred.min() and val_pred.max() < 1.5 * actual.max()
    val = full_clean.val
    seen = set(full_clean.train["pickup"]) | set(full_clean.train["delivery"])
    new = ~(val["pickup"].isin(seen) & val["delivery"].isin(seen)).to_numpy()
    assert new.sum() == 1_447
    rpm = val_pred / val["distance"].to_numpy()
    # Rows touching a new city are priced like other rows of the same equipment and distance band.
    band = data.distance_band(val["distance"])
    key = pd.Series(list(zip(val["equipment"], band)))
    cell_median = pd.Series(rpm).groupby(key).transform("median").to_numpy()
    assert 0.95 < np.median(rpm[new] / cell_median[new]) < 1.05
