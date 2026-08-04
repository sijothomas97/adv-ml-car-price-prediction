import json

import joblib
import numpy as np
import pandas as pd
import pytest

from car_price import config as c
from car_price.train import train


@pytest.fixture
def csv_path(raw_df, tmp_path):
    path = tmp_path / "adverts.csv"
    raw_df.to_csv(path, index=False)
    return path


def test_train_saves_model_and_metrics_and_roundtrips(csv_path, tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    metrics = train(
        data_path=csv_path,
        model="gbr",
        sample=None,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    # metrics written and sane
    assert model_path.exists() and metrics_path.exists()
    on_disk = json.loads(metrics_path.read_text())
    assert on_disk["mae"] == metrics["mae"]
    for key in ("mae", "rmse", "r2", "n_train", "n_test"):
        assert key in on_disk
    assert on_disk["mae"] > 0
    assert on_disk["rmse"] >= on_disk["mae"]

    # round-trip: loaded pipeline predicts on raw-style feature rows
    pipe = joblib.load(model_path)
    row = pd.DataFrame(
        [
            {
                c.COL_MILEAGE: 30_000.0,
                c.COL_AGE: 5.0,
                c.COL_VHL_TYPE: 0,
                c.COL_STD_COLR: "Blue",
                c.COL_STD_MK: "Ford",
                c.COL_STD_MDL: "Fiesta",
                c.COL_VHL_COND: "USED",
                c.COL_BD_TYPE: "Hatchback",
                c.COL_FL_TYPE: "Petrol",
            }
        ]
    )
    pred = pipe.predict(row[c.FEATURES])
    assert pred.shape == (1,)
    assert np.isfinite(pred[0])


@pytest.mark.skipif(not c.MODEL_PATH.exists(), reason="no trained model artifact yet")
def test_saved_champion_model_roundtrip():
    pipe = joblib.load(c.MODEL_PATH)
    row = pd.DataFrame(
        [
            {
                c.COL_MILEAGE: 42_000.0,
                c.COL_AGE: 6.0,
                c.COL_VHL_TYPE: 0,
                c.COL_STD_COLR: "Black",
                c.COL_STD_MK: "BMW",
                c.COL_STD_MDL: "3 Series",
                c.COL_VHL_COND: "USED",
                c.COL_BD_TYPE: "Saloon",
                c.COL_FL_TYPE: "Diesel",
            }
        ]
    )
    pred = pipe.predict(row[c.FEATURES])
    assert np.isfinite(pred[0])
    assert 100 < pred[0] < 500_000  # a plausible GBP price
