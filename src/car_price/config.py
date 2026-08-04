"""Column names and constants shared across the pipeline."""

import os
from pathlib import Path

# Repository root (src/car_price/config.py -> repo root is two parents up from src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / "data" / "csv" / "adverts.csv"
MODELS_DIR = PROJECT_ROOT / "models"
# Env overrides let a container point at artifacts outside the source tree.
MODEL_PATH = Path(os.environ.get("CAR_PRICE_MODEL_PATH", MODELS_DIR / "model.joblib"))
METRICS_PATH = Path(os.environ.get("CAR_PRICE_METRICS_PATH", MODELS_DIR / "metrics.json"))

# Raw dataset columns
COL_PUBLIC_REF = "public_reference"
COL_MILEAGE = "mileage"
COL_REG_CODE = "reg_code"
COL_STD_COLR = "standard_colour"
COL_STD_MK = "standard_make"
COL_STD_MDL = "standard_model"
COL_VHL_COND = "vehicle_condition"
COL_REG_YEAR = "year_of_registration"
COL_PRICE = "price"
COL_BD_TYPE = "body_type"
COL_VHL_TYPE = "crossover_car_and_van"
COL_FL_TYPE = "fuel_type"

# Engineered columns
COL_AGE = "age"

VAL_USED = "USED"
VAL_NEW = "NEW"

TARGET = COL_PRICE

# Feature sets used by the sklearn pipeline (post-cleaning)
NUMERIC_FEATURES = [COL_MILEAGE, COL_AGE, COL_VHL_TYPE]
CATEGORICAL_FEATURES = [
    COL_STD_COLR,
    COL_STD_MK,
    COL_STD_MDL,
    COL_VHL_COND,
    COL_BD_TYPE,
    COL_FL_TYPE,
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.25
