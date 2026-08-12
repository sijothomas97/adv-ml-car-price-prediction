import pytest
from fastapi.testclient import TestClient

from car_price import config as c
from car_price.api import app

VALID_PAYLOAD = {
    "mileage": 42_000,
    "year_of_registration": 2019,
    "standard_make": "BMW",
    "standard_model": "3 Series",
    "standard_colour": "Black",
    "body_type": "Saloon",
    "fuel_type": "Diesel",
    "vehicle_condition": "USED",
    "crossover_car_and_van": False,
}

needs_model = pytest.mark.skipif(
    not c.MODEL_PATH.exists(), reason="no trained model artifact yet"
)


@pytest.fixture
def client():
    with TestClient(app) as client:  # context manager runs the lifespan
        yield client


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] == c.MODEL_PATH.exists()


def test_index_serves_form(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "/predict" in res.text


@needs_model
def test_predict_valid_payload(client):
    res = client.post("/predict", json=VALID_PAYLOAD)
    assert res.status_code == 200
    body = res.json()
    assert body["currency"] == "GBP"
    assert 100 < body["predicted_price"] < 500_000
    assert body["model_name"]


@needs_model
def test_predict_defaults_condition_and_crossover(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items()}
    payload.pop("vehicle_condition")
    payload.pop("crossover_car_and_van")
    res = client.post("/predict", json=payload)
    assert res.status_code == 200


@needs_model
def test_predict_handles_unseen_categories(client):
    payload = {**VALID_PAYLOAD, "standard_make": "Notacar", "standard_model": "Zzz9"}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200  # OneHotEncoder(handle_unknown="ignore")
    assert res.json()["predicted_price"] > 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"mileage": -1},
        {"mileage": 600_000},
        {"year_of_registration": 1900},
        {"year_of_registration": 3000},
        {"vehicle_condition": "SCRAP"},
        {"standard_make": ""},
        {"mileage": "not-a-number"},
    ],
)
def test_predict_rejects_invalid_values(client, overrides):
    res = client.post("/predict", json={**VALID_PAYLOAD, **overrides})
    assert res.status_code == 422


@pytest.mark.parametrize("missing", ["mileage", "year_of_registration", "standard_make"])
def test_predict_rejects_missing_required_fields(client, missing):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing}
    res = client.post("/predict", json=payload)
    assert res.status_code == 422


def test_predict_without_model_returns_503(client, monkeypatch):
    monkeypatch.setattr(app.state, "model", None)
    res = client.post("/predict", json=VALID_PAYLOAD)
    assert res.status_code == 503
