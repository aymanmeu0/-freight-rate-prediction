"""Leakage: banned columns never reach a learner, and the predictions prove it.

Behavioural checks use a cheap model with the same structure as the chosen one
(drift + ramp, log rate per mile, geo_dow features, directed lane correction).
The slow checks repeat the key ones on the full final model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from conftest import FAST_SPEC, needs_data
from src import config as cfg
from src import features as feat
from src import model as mdl
from src import train as T


# --------------------------------------------------------------------------
# Static checks
# --------------------------------------------------------------------------
def test_never_features_list_is_complete():
    assert set(cfg.NEVER_FEATURES) == {"load_id", "quote_signal", "posted_rate", "is_outlier", "date", "month"}


@pytest.mark.parametrize("name", list(feat.FEATURE_SETS))
def test_no_feature_set_contains_a_banned_column(name):
    cols = set(feat.FEATURE_SETS[name])
    assert not cols & set(cfg.NEVER_FEATURES)
    assert not cols & set(feat.DRIFT_ONLY)


@pytest.mark.parametrize("banned", list(cfg.NEVER_FEATURES) + list(feat.DRIFT_ONLY))
def test_assert_allowed_raises_on_each_banned_column(banned):
    with pytest.raises(AssertionError):
        feat.assert_allowed(["distance", banned])


@needs_data
def test_feature_matrix_has_only_allowed_columns_and_ignores_banned_inputs(full_clean):
    val = full_clean.val
    X = feat.build_features(val, mdl.FINAL_SPEC.features)
    assert list(X.columns) == feat.FEATURE_SETS[mdl.FINAL_SPEC.features]
    assert not set(X.columns) & set(cfg.NEVER_FEATURES)
    rng = np.random.default_rng(0)
    other = val.copy()
    other["quote_signal"] = rng.permutation(other["quote_signal"].to_numpy())
    other["load_id"] = other["load_id"].iloc[::-1].to_numpy()
    other["posted_rate"] = rng.uniform(100, 9000, len(other))
    other["is_outlier"] = True
    pd.testing.assert_frame_equal(X, feat.build_features(other, mdl.FINAL_SPEC.features))


# --------------------------------------------------------------------------
# Behavioural checks on a cheap model with the chosen structure
# --------------------------------------------------------------------------
@pytest.fixture
def fit_rows(full_clean):
    """6,000 clean Jan-Oct rows (every month, so the drift and ramp are both fitted)."""
    return full_clean.train.sample(6_000, random_state=0).sort_index()


@needs_data
def test_quote_signal_is_not_used_when_fitting(fit_rows, full_clean, light_oof):
    val = full_clean.val.iloc[:3_000]
    m1 = mdl.RateModel(FAST_SPEC).fit(fit_rows)
    mdl.clear_cache()   # so the lane residuals are recomputed, not read from the memo
    noisy = fit_rows.copy()
    noisy["quote_signal"] = np.random.default_rng(1).normal(50, 30, len(noisy))
    m2 = mdl.RateModel(FAST_SPEC).fit(noisy)
    assert (m1.drift_.slope_, m1.drift_.ramp_) == (m2.drift_.slope_, m2.drift_.ramp_)
    pd.testing.assert_series_equal(m1.lane_table_, m2.lane_table_)
    assert np.array_equal(m1.predict(val), m2.predict(val))


@needs_data
def test_inputs_of_predicted_rows_that_must_not_matter(fit_rows, full_clean, light_oof):
    m = mdl.RateModel(FAST_SPEC).fit(fit_rows)
    rng = np.random.default_rng(2)
    # Validation rows: quote_signal permuted, or replaced with nonsense.
    val = full_clean.val
    base = m.predict(val)
    for qs in (rng.permutation(val["quote_signal"].to_numpy()), np.full(len(val), 1e6)):
        assert np.array_equal(base, m.predict(val.assign(quote_signal=qs)))
    # Labeled rows: shuffling, dropping or flagging the label of the predicted rows changes nothing.
    rows = full_clean.train.iloc[::7]
    base = m.predict(rows)
    assert np.array_equal(base, m.predict(rows.assign(posted_rate=rng.permutation(rows["posted_rate"].to_numpy()))))
    assert np.array_equal(base, m.predict(rows.drop(columns="posted_rate")))
    assert np.array_equal(base, m.predict(rows.assign(is_outlier=True)))
    assert np.array_equal(base, m.predict(rows.assign(load_id=rows["load_id"].iloc[::-1].to_numpy())))


@needs_data
def test_learner_sees_exactly_the_feature_list(fit_rows, light_oof):
    m = mdl.RateModel(FAST_SPEC).fit(fit_rows)
    assert m.learner_.n_features_in_ == len(feat.FEATURE_SETS[FAST_SPEC.features])
    assert m.feature_names_ == feat.FEATURE_SETS[FAST_SPEC.features]


# --------------------------------------------------------------------------
# Out-of-fold lane correction
# --------------------------------------------------------------------------
def _lane_rows(n=600, seed=0):
    rng = np.random.default_rng(seed)
    cities = np.array(["A", "B", "C", "D"])
    pick = rng.integers(0, 4, n)
    deliv = (pick + rng.integers(1, 4, n)) % 4
    rows = pd.DataFrame({"load_id": [f"X-{i}" for i in range(n)], "pickup": cities[pick], "delivery": cities[deliv]})
    X = rng.normal(size=(n, 5))
    z = X[:, 0] * 0.3 + rng.normal(scale=0.05, size=n)
    return rows, X, z


def _oof_residuals(model, rows, X, z):
    mdl.clear_cache()
    table = model._fit_lane_correction(rows, X, z, None)
    (resid,) = mdl._OOF_CACHE.values()
    return table, resid.copy()


def test_oof_lane_correction_never_uses_a_rows_own_label(light_oof):
    rows, X, z = _lane_rows()
    model = mdl.RateModel(FAST_SPEC)
    _, r1 = _oof_residuals(model, rows, X, z)
    i, delta = 17, 0.8
    z2 = z.copy()
    z2[i] += delta
    _, r2 = _oof_residuals(model, rows, X, z2)
    # The prediction for row i is unchanged, so its residual moves by exactly delta.
    assert r2[i] - r1[i] == pytest.approx(delta, abs=1e-9)
    folds = KFold(n_splits=5, shuffle=True, random_state=cfg.SEED).split(X)
    same_fold = next(te for _, te in folds if i in te)
    others = np.setdiff1d(same_fold, [i])
    assert np.array_equal(r1[others], r2[others])        # same-fold rows: model never saw row i
    changed = np.flatnonzero(r1 != r2)
    assert len(np.setdiff1d(changed, same_fold)) > 0       # other folds did use it (the check is not vacuous)


def test_lane_table_is_the_shrunk_mean_of_oof_residuals(light_oof):
    rows, X, z = _lane_rows(seed=3)
    model = mdl.RateModel(FAST_SPEC)
    table, resid = _oof_residuals(model, rows, X, z)
    keys = feat.lane_key(rows, directed=True)
    g = pd.Series(resid).groupby(keys.to_numpy())
    expected = g.sum() / (g.size() + FAST_SPEC.lane_prior)
    pd.testing.assert_series_equal(table.sort_index(), expected.sort_index(), check_names=False)


def test_oof_memo_is_keyed_on_the_features_too(light_oof):
    """Same ids and labels but different feature values must not reuse stale residuals."""
    rows, X, z = _lane_rows(seed=4)
    model = mdl.RateModel(FAST_SPEC)
    mdl.clear_cache()
    model._fit_lane_correction(rows, X, z, None)
    X2 = X.copy()
    X2[:, 0] = -X2[:, 0]
    stale_or_fresh = model._fit_lane_correction(rows, X2, z, None)
    _, fresh = _oof_residuals(model, rows, X2, z)
    g = pd.Series(fresh).groupby(feat.lane_key(rows, directed=True).to_numpy())
    pd.testing.assert_series_equal(stale_or_fresh.sort_index(), (g.sum() / (g.size() + 10.0)).sort_index(),
                                   check_names=False)


@needs_data
def test_unseen_lanes_get_no_correction(fit_rows, full_clean, light_oof):
    m = mdl.RateModel(FAST_SPEC).fit(fit_rows)
    val = full_clean.val
    unseen = ~feat.lane_key(val, directed=True).isin(m.lane_table_.index).to_numpy()
    assert unseen.sum() > 0
    with_lane = m.predict_log(val)
    table, m.lane_table_ = m.lane_table_, None
    without = m.predict_log(val)
    m.lane_table_ = table
    assert np.array_equal(with_lane[unseen], without[unseen])
    assert not np.array_equal(with_lane[~unseen], without[~unseen])


# --------------------------------------------------------------------------
# Drift and ramp: fit on the fit rows of each split only
# --------------------------------------------------------------------------
@needs_data
def test_cv_drift_is_fit_on_the_fold_fit_rows_only(ds, light_oof):
    split = T.cv_splits(ds)[0]                             # Jan-Apr -> May
    scores, pred, model = T.evaluate(FAST_SPEC, split)
    ref = mdl.LevelDrift(ramp=True).fit(split.fit)
    assert (model.drift_.slope_, model.drift_.ramp_) == (ref.slope_, ref.ramp_)
    assert model.drift_.t_end_ == feat.time_index([pd.Timestamp("2025-04-30")])[0]
    # Wreck the test-month labels: the fitted drift and the predictions stay identical.
    rng = np.random.default_rng(5)
    wrecked = split._replace(test=split.test.assign(posted_rate=split.test["posted_rate"] * rng.uniform(0.2, 5, len(split.test))))
    mdl.clear_cache()
    _, pred2, model2 = T.evaluate(FAST_SPEC, wrecked)
    assert (model2.drift_.slope_, model2.drift_.ramp_) == (model.drift_.slope_, model.drift_.ramp_)
    assert np.array_equal(pred, pred2)


@needs_data
def test_final_drift_uses_training_rows_only(full_clean):
    """The final drift is a function of clean Jan-Oct rows; validation rows have no label to leak."""
    assert cfg.TARGET not in full_clean.val.columns
    d = mdl.LevelDrift(ramp=True).fit(full_clean.train)
    assert d.t_end_ == feat.time_index([pd.Timestamp("2025-10-31")])[0]


# --------------------------------------------------------------------------
# The same on the full final model (slow)
# --------------------------------------------------------------------------
@pytest.mark.slow
@needs_data
def test_final_model_ignores_quote_signal_and_labels_of_predicted_rows(final_fit):
    rng = np.random.default_rng(7)
    val = final_fit.val
    assert np.array_equal(final_fit.val_pred,
                          final_fit.model.predict(val.assign(quote_signal=rng.permutation(val["quote_signal"].to_numpy()))))
    dec = final_fit.dec
    assert np.array_equal(final_fit.dec_pred, final_fit.model.predict(dec.assign(quote_signal=-123.0)))
    rows = final_fit.train.iloc[::11]
    base = final_fit.model.predict(rows)
    shuffled = rows.assign(posted_rate=rng.permutation(rows["posted_rate"].to_numpy()),
                           quote_signal=rng.permutation(rows["quote_signal"].to_numpy()))
    assert np.array_equal(base, final_fit.model.predict(shuffled))
