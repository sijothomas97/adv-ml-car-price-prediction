"""FastAPI inference service for the used-car price model.

Run locally:
    uvicorn car_price.api:app --reload

Endpoints:
    GET  /         — minimal HTML form that calls /predict
    GET  /health   — liveness + model status
    POST /predict  — price prediction for one car
"""

import json
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from car_price import config as c

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact once at startup (missing artifact -> 503s)."""
    app.state.model = joblib.load(c.MODEL_PATH) if c.MODEL_PATH.exists() else None
    app.state.metrics = (
        json.loads(c.METRICS_PATH.read_text()) if c.METRICS_PATH.exists() else {}
    )
    yield
    app.state.model = None


app = FastAPI(
    title="Used-car price API",
    version="1.0.0",
    description="Predicts UK used-car prices with a trained sklearn pipeline.",
    lifespan=lifespan,
)


class CarFeatures(BaseModel):
    """One car, described with the raw features the model was trained on.

    ``age`` is derived server-side from ``year_of_registration``.
    """

    mileage: float = Field(ge=0, le=500_000, description="Odometer reading in miles")
    year_of_registration: int = Field(ge=1950, le=date.today().year)
    standard_make: str = Field(min_length=1, max_length=64, examples=["BMW"])
    standard_model: str = Field(min_length=1, max_length=64, examples=["3 Series"])
    standard_colour: str = Field(min_length=1, max_length=32, examples=["Black"])
    body_type: str = Field(min_length=1, max_length=32, examples=["Saloon"])
    fuel_type: str = Field(min_length=1, max_length=32, examples=["Diesel"])
    vehicle_condition: Literal["USED", "NEW"] = "USED"
    crossover_car_and_van: bool = False

    def to_frame(self) -> pd.DataFrame:
        """Single-row frame with exactly the pipeline's FEATURES columns."""
        row = {
            c.COL_MILEAGE: self.mileage,
            c.COL_AGE: float(max(0, date.today().year - self.year_of_registration)),
            c.COL_VHL_TYPE: int(self.crossover_car_and_van),
            c.COL_STD_COLR: self.standard_colour,
            c.COL_STD_MK: self.standard_make,
            c.COL_STD_MDL: self.standard_model,
            c.COL_VHL_COND: self.vehicle_condition,
            c.COL_BD_TYPE: self.body_type,
            c.COL_FL_TYPE: self.fuel_type,
        }
        return pd.DataFrame([row])[c.FEATURES]


class Prediction(BaseModel):
    predicted_price: float = Field(description="Estimated price in GBP")
    currency: Literal["GBP"] = "GBP"
    model_name: str
    mae: float | None = Field(default=None, description="Holdout MAE of the model, GBP")


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    model_name: str | None = None
    sklearn_version: str | None = None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=Health)
def health() -> Health:
    metrics = app.state.metrics
    return Health(
        model_loaded=app.state.model is not None,
        model_name=metrics.get("model"),
        sklearn_version=metrics.get("sklearn_version"),
    )


@app.post("/predict", response_model=Prediction)
def predict(car: CarFeatures) -> Prediction:
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact not found at {c.MODEL_PATH}; train first.",
        )
    price = float(app.state.model.predict(car.to_frame())[0])
    return Prediction(
        predicted_price=round(max(0.0, price), 2),
        model_name=app.state.metrics.get("model", "unknown"),
        mae=app.state.metrics.get("mae"),
    )
