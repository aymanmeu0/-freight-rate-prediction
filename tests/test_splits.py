"""Split integrity: CV folds, the Sep-Oct holdout and the unseen-city holdout."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import needs_data
from src import config as cfg
from src import train as T

pytestmark = needs_data


@pytest.fixture(scope="module")
def cv(ds):
    return T.cv_splits(ds)


def test_cv_folds_are_expanding_time_ordered_and_disjoint(cv):
    assert [s.fit["date"].dt.month.max() for s in cv] == [4, 5, 6, 7]
    for (last, test_month), split in zip(cfg.CV_FOLDS, cv):
        assert set(split.fit["date"].dt.month) == set(range(1, last + 1))
        assert set(split.test["date"].dt.month) == {test_month}
        assert split.fit["date"].max() < split.test["date"].min()
        assert not set(split.fit["load_id"]) & set(split.test["load_id"])
        assert split.test["date"].max() <= cfg.HOLDOUT_FIT_DATES[1]   # CV never touches Sep-Oct
        assert not split.fit["is_outlier"].any()


def test_cv_fold_sizes_match_the_findings(cv):
    assert [len(s.fit) for s in cv] == [18_835, 23_686, 28_406, 33_253]
    assert [len(s.test) for s in cv] == [4_913, 4_783, 4_912, 4_759]
    assert [int(s.test["is_outlier"].sum()) for s in cv] == [62, 63, 65, 68]


def test_holdout_is_jan_aug_to_sep_oct(ds):
    h = T.holdout_split(ds)
    assert h.fit["date"].min() == pd.Timestamp("2025-01-01")
    assert h.fit["date"].max() == pd.Timestamp("2025-08-31")
    assert h.test["date"].min() == pd.Timestamp("2025-09-01")
    assert h.test["date"].max() == pd.Timestamp("2025-10-31")
    assert set(h.test["date"].dt.month) == {9, 10}
    assert not set(h.fit["load_id"]) & set(h.test["load_id"])
    assert (len(h.fit), len(h.test), int(h.test["is_outlier"].sum())) == (37_944, 9_523, 144)
    # Test rows are never dropped: every Sep-Oct row is there.
    assert len(h.test) == int((ds.train["date"] >= cfg.HOLDOUT_TEST_DATES[0]).sum())


def test_unseen_city_holdout_really_hides_the_cities(ds):
    cities = set(cfg.UNSEEN_CITY_HOLDOUT)
    assert len(cities) == 8
    seen_in_train = set(ds.train["pickup"]) | set(ds.train["delivery"])
    assert cities <= seen_in_train          # the check is not vacuous
    u = T.unseen_city_split(ds)
    assert not (u.fit["pickup"].isin(cities) | u.fit["delivery"].isin(cities)).any()
    assert (u.test["pickup"].isin(cities) | u.test["delivery"].isin(cities)).all()
    assert u.fit["date"].max() <= cfg.HOLDOUT_FIT_DATES[1]
    assert u.test["date"].min() >= cfg.HOLDOUT_TEST_DATES[0]
    assert (len(u.fit), len(u.test)) == (33_819, 1_003)
