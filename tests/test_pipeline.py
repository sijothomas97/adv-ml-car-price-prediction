import numpy as np
import pytest

from car_price import config as c
from car_price import preprocessing as pp
from car_price.pipeline import build_pipeline, build_preprocessor


@pytest.fixture
def clean_df(raw_df):
    return pp.clean(raw_df, current_year=2026)


def test_preprocessor_encodes_all_features(clean_df):
    X = clean_df[c.FEATURES]
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] >= len(c.NUMERIC_FEATURES)  # numeric + one-hot columns
    assert np.isfinite(np.asarray(Xt, dtype=float)).all()


def test_pipeline_fit_predict(clean_df):
    X, y = clean_df[c.FEATURES], clean_df[c.TARGET]
    pipe = build_pipeline("gbr")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()


def test_pipeline_handles_unseen_categories(clean_df):
    X, y = clean_df[c.FEATURES], clean_df[c.TARGET]
    pipe = build_pipeline("gbr")
    pipe.fit(X, y)
    X_new = X.head(3).copy()
    X_new[c.COL_STD_MK] = "Lamborghini"  # never seen in training
    X_new[c.COL_STD_MDL] = "Huracan"
    preds = pipe.predict(X_new)
    assert np.isfinite(preds).all()


def test_build_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("xgboost")
