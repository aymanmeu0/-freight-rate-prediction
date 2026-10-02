"""Reproducibility: refitting the final model gives the committed files to the cent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import needs_outputs
from src import config as cfg
from src import model as mdl

pytestmark = needs_outputs


def test_saved_setup_is_the_documented_one(metrics_json, final_spec):
    assert final_spec.key() == mdl.FINAL_SPEC.key()
    assert metrics_json["seed"] == cfg.SEED == 42
    assert metrics_json["n_jobs"] == mdl.N_JOBS == 4
    assert final_spec.label() == ("xgb, log rpm, geo_dow, trend+ramp damp 0.5, d5 n2000 lr0.03 mcw1, "
                                  "lane directed p10")


@pytest.mark.slow
def test_final_fit_reproduces_the_validation_file_to_the_cent(final_fit):
    written = pd.read_csv(cfg.VALIDATION_PREDICTIONS_PATH, dtype=str)
    assert written["load_id"].tolist() == final_fit.val["load_id"].tolist()
    assert written["predicted_rate"].tolist() == [f"{v:.2f}" for v in final_fit.val_pred]


@pytest.mark.slow
def test_final_fit_reproduces_the_december_file_to_the_cent(final_fit):
    written = pd.read_csv(cfg.DECEMBER_PATH, dtype=str)
    assert written["predicted_rate"].tolist() == [f"{v:.2f}" for v in final_fit.dec_pred]


@pytest.mark.slow
def test_final_fit_matches_the_saved_metrics(final_fit, metrics_json):
    saved = metrics_json["final"]
    assert len(final_fit.train) == saved["fit_rows"] == 47_323
    d = final_fit.model.describe()
    assert d["drift_slope_per_day"] == pytest.approx(saved["model"]["drift_slope_per_day"], rel=1e-12)
    assert d["drift_ramp"] == pytest.approx(saved["model"]["drift_ramp"], rel=1e-12)
    assert np.median(final_fit.val_pred) == pytest.approx(saved["summaries"]["validation"]["median"], rel=1e-12)
