"""Paths and constants shared by the whole pipeline.

Every path is built from this file's location, so the code works from any
working directory. The numbers come from ``docs/eda_findings.md`` (the
cleaning rules C1 to C8 and the split spec); the section is named next to each one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

# Submission files. score.py reads the December file from data/, so the filled
# file goes back there; outputs/ keeps a copy.
VALIDATION_PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
DECEMBER_PREDICTIONS_COPY_PATH = OUTPUTS_DIR / "december_predictions.csv"

SEED = 42

# --------------------------------------------------------------------------
# Data contract: columns in file order (C1, C2, C8)
# --------------------------------------------------------------------------
VALIDATION_COLUMNS = [
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "distance", "equipment", "weight",
    "date", "market_index", "quote_signal",
]
TARGET = "posted_rate"
TRAIN_COLUMNS = VALIDATION_COLUMNS + [TARGET]
TEMPLATE_COLUMNS = ["load_id", "predicted_rate"]
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]

TEXT_COLUMNS = ["load_id", "pickup", "delivery", "equipment"]
COORD_COLUMNS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
# Numeric columns that may hold missing values in the raw files (Q4, Q5).
NULLABLE_NUMERIC = ["weight", "market_index"]

EQUIPMENT = ("Dry Van", "Reefer", "Flatbed")
DATE_FORMAT = "%Y-%m-%d"

TRAIN_DATES = (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-10-31"))
VALIDATION_DATES = (pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"))
DECEMBER_DATES = (pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-31"))

N_TRAIN_ROWS = 48_000
N_VALIDATION_ROWS = 12_000
N_DECEMBER_ROWS = 31
N_CITIES = 72  # train + validation, C7

# --------------------------------------------------------------------------
# Cleaning rules
# --------------------------------------------------------------------------
# C3: weight range after abs() and the fill. 47,500 is a hard cap in the data (Q7).
WEIGHT_MIN = 5_000.0
WEIGHT_MAX = 47_500.0

# C4 fallback for a date with no observed market_index (never triggered on
# the given files). The index has a strong weekly cycle, so the fallback uses
# the same weekday in the nearest weeks: +/-7 days first, then +/-14, and so on.
MARKET_INDEX_FALLBACK_MAX_WEEKS = 4

# C5: distance bands (miles, right-closed as in pd.cut) for the expected rate
# per mile, and the outlier threshold on |ln(rpm / expected rpm)|.
DIST_BANDS = [0, 100, 150, 200, 250, 300, 400, 500, 600, 750, 900,
              1100, 1300, 1550, 1800, 2100, 2500, 2900, 4000]
OUTLIER_LOG_THRESHOLD = 0.5

# C6 and section 4: columns that must never reach the model as features.
NEVER_FEATURES = ("load_id", "quote_signal", TARGET, "is_outlier", "date", "month")

# --------------------------------------------------------------------------
# Split spec (eda_findings.md section 5), for the ML step
# --------------------------------------------------------------------------
HOLDOUT_FIT_DATES = (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-08-31"))
HOLDOUT_TEST_DATES = (pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-31"))
# Expanding-window CV inside Jan-Aug: (last fit month, test month).
CV_FOLDS = [(4, 5), (5, 6), (6, 7), (7, 8)]
UNSEEN_CITY_HOLDOUT = ("Birmingham", "Dallas", "Detroit", "Las Vegas",
                       "Louisville", "New Orleans", "St. Louis", "Washington")
