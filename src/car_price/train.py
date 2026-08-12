"""Train the used-car price model and serialize pipeline + metrics.

Usage:
    python train.py [--model gbr|hgbr|stacking] [--sample N] [--data PATH]
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from car_price import config as c
from car_price.data import load_raw
from car_price.pipeline import MODEL_CHOICES, build_pipeline
from car_price.preprocessing import clean


def train(
    data_path: str | Path = c.DATA_CSV,
    model: str = "gbr",
    sample: int | None = 50_000,
    test_size: float = c.TEST_SIZE,
    random_state: int = c.RANDOM_STATE,
    model_path: str | Path = c.MODEL_PATH,
    metrics_path: str | Path = c.METRICS_PATH,
) -> dict:
    """Fit the pipeline, evaluate on a holdout, save model + metrics.

    Returns the metrics dict.
    """
    t0 = time.time()
    df = clean(load_raw(data_path))
    if sample and 0 < sample < len(df):
        df = df.sample(n=sample, random_state=random_state).reset_index(drop=True)

    X = df[c.FEATURES]
    y = df[c.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline(model=model, random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "model": model,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_test, y_pred)), 2),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "features": c.FEATURES,
        "target": c.TARGET,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_seconds": round(time.time() - t0, 1),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="gbr")
    parser.add_argument("--data", default=str(c.DATA_CSV))
    parser.add_argument(
        "--sample",
        type=int,
        default=50_000,
        help="Row cap for training (0 = full dataset)",
    )
    parser.add_argument("--test-size", type=float, default=c.TEST_SIZE)
    parser.add_argument("--model-out", default=str(c.MODEL_PATH))
    parser.add_argument("--metrics-out", default=str(c.METRICS_PATH))
    args = parser.parse_args(argv)

    metrics = train(
        data_path=args.data,
        model=args.model,
        sample=args.sample or None,
        test_size=args.test_size,
        model_path=args.model_out,
        metrics_path=args.metrics_out,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
