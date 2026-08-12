"""Sklearn Pipeline construction (encoders + regressor), per notebooks 05/08/09."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from car_price import config as c

MODEL_CHOICES = ("gbr", "hgbr", "stacking")


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median impute + scale. Categorical: mode impute + one-hot.

    ``handle_unknown="ignore"`` keeps inference safe for unseen makes/models;
    ``min_frequency`` caps the ~1200-level ``standard_model`` cardinality.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",
                    min_frequency=20,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, c.NUMERIC_FEATURES),
            ("cat", categorical_transformer, c.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _make_regressor(model: str, random_state: int):
    if model == "gbr":
        return GradientBoostingRegressor(random_state=random_state)
    if model == "hgbr":
        return HistGradientBoostingRegressor(random_state=random_state)
    if model == "stacking":
        estimators = [
            ("ridge", Ridge(alpha=1.0)),
            (
                "rfr",
                RandomForestRegressor(
                    n_estimators=50,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
            ("gbr", HistGradientBoostingRegressor(random_state=random_state)),
        ]
        return StackingRegressor(
            estimators=estimators, final_estimator=RidgeCV(), cv=3, n_jobs=-1
        )
    raise ValueError(f"Unknown model {model!r}; choose from {MODEL_CHOICES}")


def build_pipeline(model: str = "gbr", random_state: int = c.RANDOM_STATE) -> Pipeline:
    """Full pipeline: preprocessing + regressor, serializable as one artifact."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", _make_regressor(model, random_state)),
        ]
    )
