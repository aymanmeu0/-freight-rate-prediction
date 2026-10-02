"""Shared fixtures and helpers for the QA suite.

The data files are not in the repo. Tests that need them are marked with
``needs_data`` (input files in ``data/``) or ``needs_outputs`` (the prediction
files written by ``python run_pipeline.py``) and skip with a clear reason when
the files are absent. Synthetic-data tests always run.

Slow tests (marked ``slow``) fit the full final model once per session.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import data  # noqa: E402
from src import model as mdl  # noqa: E402

# Original files from the assessment pack, kept at the project root (not in git).
ROOT_ORIGINALS = {
    cfg.TRAIN_PATH: ROOT / "train-test.csv",
    cfg.VALIDATION_PATH: ROOT / "validation.csv",
    cfg.TEMPLATE_PATH: ROOT / "validation-predictions-template.csv",
    cfg.DECEMBER_PATH: ROOT / "december-chart-inputs.csv",
}

_INPUTS = [cfg.TRAIN_PATH, cfg.VALIDATION_PATH, cfg.TEMPLATE_PATH, cfg.DECEMBER_PATH]
_MISSING = [p.name for p in _INPUTS if not p.is_file()]

needs_data = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"input data not found in data/ ({', '.join(_MISSING)}); the data is not in the repo, see README")


def _outputs_ready() -> bool:
    if _MISSING or not cfg.VALIDATION_PREDICTIONS_PATH.is_file():
        return False
    return bool(pd.read_csv(cfg.DECEMBER_PATH)["predicted_rate"].notna().all())


needs_outputs = pytest.mark.skipif(
    not _outputs_ready(),
    reason="prediction files not written yet (or data missing); run python run_pipeline.py first")

needs_root_originals = pytest.mark.skipif(
    not all(p.is_file() for p in ROOT_ORIGINALS.values()),
    reason="original assessment files not found at the project root")

# A cheap model with the same structure as the chosen one (drift + ramp, log
# rate per mile, geo_dow features, directed lane correction), so behavioural
# leakage checks run in seconds.
FAST_SPEC = mdl.FINAL_SPEC.with_(name="fast", xgb=(("max_depth", 4), ("n_estimators", 40),
                                                    ("learning_rate", 0.3)))


@contextlib.contextmanager
def light_lane_oof():
    """Use a 30-tree XGBoost for the lane-correction out-of-fold fits, then restore."""
    saved = mdl.LANE_OOF_XGB
    mdl.LANE_OOF_XGB = dict(mdl.XGB_DEFAULTS, n_estimators=30, learning_rate=0.3)
    mdl.clear_cache()
    try:
        yield
    finally:
        mdl.LANE_OOF_XGB = saved
        mdl.clear_cache()


@pytest.fixture
def light_oof():
    with light_lane_oof():
        yield


# --------------------------------------------------------------------------
# Synthetic frames (no data needed)
# --------------------------------------------------------------------------
CITIES = {"Alpha": (40.0, -80.0), "Beta": (41.0, -85.0), "Gamma": (35.0, -90.0)}


def make_synthetic(n: int = 60, labeled: bool = True, seed: int = 0, start: str = "2025-03-01",
                   prefix: str = "TR") -> pd.DataFrame:
    """A small frame in the raw file schema (dates as text), valid under the data contract."""
    rng = np.random.default_rng(seed)
    names = list(CITIES)
    pick = rng.integers(0, 3, n)
    deliv = (pick + 1 + rng.integers(0, 2, n)) % 3
    dist = rng.uniform(100, 900, n).round(1)
    equip = np.array(cfg.EQUIPMENT)[np.arange(n) % 3]
    dates = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, 50, n), unit="D")
    premium = np.where(equip == "Reefer", 1.13, np.where(equip == "Flatbed", 1.08, 1.0))
    df = pd.DataFrame({
        "load_id": [f"{prefix}-{i:06d}" for i in range(1, n + 1)],
        "pickup": [names[i] for i in pick], "delivery": [names[i] for i in deliv],
        "pickup_lat": [CITIES[names[i]][0] for i in pick], "pickup_lon": [CITIES[names[i]][1] for i in pick],
        "delivery_lat": [CITIES[names[i]][0] for i in deliv], "delivery_lon": [CITIES[names[i]][1] for i in deliv],
        "distance": dist, "equipment": equip, "weight": rng.uniform(10_000, 45_000, n).round(0),
        "date": dates.strftime("%Y-%m-%d"), "market_index": rng.uniform(0.8, 1.2, n).round(5),
        "quote_signal": rng.uniform(1.5, 2.5, n).round(5),
    })
    if labeled:
        df["posted_rate"] = (dist * 2.2 * premium * rng.uniform(0.95, 1.05, n)).round(2)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def synthetic_template(path: Path) -> Path:
    ids = [f"TE-{i:06d}" for i in range(1, cfg.N_VALIDATION_ROWS + 1)]
    return write_csv(pd.DataFrame({"load_id": ids, "predicted_rate": np.nan}), path)


def synthetic_december(path: Path) -> Path:
    """31 rows in the original December file format, predicted_rate empty."""
    lines = ["pickup,delivery,distance,equipment,weight,date,predicted_rate"]
    lines += [f"Lexington,Fort Wayne,360,Dry Van,32000,2025-12-{d:02d}," for d in range(1, 32)]
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# Real data (session scoped)
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ds():
    if _MISSING:
        pytest.skip(f"input data not found in data/ ({', '.join(_MISSING)})")
    return data.load_all()


@pytest.fixture(scope="session")
def full_clean(ds):
    """Cleaner fit on all of train; clean train (outliers dropped) and clean validation."""
    cleaner = data.Cleaner(ds.market_index_table).fit(ds.train)
    train_clean = cleaner.transform(ds.train, drop_outliers=True, name="train")
    val_clean = cleaner.transform(ds.validation, name="validation")
    return SimpleNamespace(cleaner=cleaner, train=train_clean, val=val_clean)


@pytest.fixture(scope="session")
def metrics_json():
    path = cfg.REPORTS_DIR / "metrics.json"
    if not path.is_file():
        pytest.skip("reports/metrics.json not found; run python run_pipeline.py first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def final_spec(metrics_json):
    return mdl.ModelSpec.from_dict(metrics_json["chosen"]["spec"])


@pytest.fixture(scope="session")
def final_fit(ds, final_spec):
    """The final model exactly as the pipeline fits it (all clean Jan-Oct rows), with its predictions."""
    from src import predict
    mdl.clear_cache()
    model, cleaner, train_clean = predict.fit_final(final_spec, ds)
    val, val_pred = predict.predict_validation(model, cleaner, ds)
    dec, dec_pred = predict.predict_december(model, cleaner, ds)
    return SimpleNamespace(model=model, cleaner=cleaner, train=train_clean, val=val, val_pred=val_pred,
                           dec=dec, dec_pred=dec_pred)
