# adv-ml-car-price-prediction
An advanced ML project to predict price of used cars from historic data.

## Quickstart (training pipeline)

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -e ".[dev]"

# Train (default: GradientBoostingRegressor on a 50k-row sample)
.venv/bin/python train.py

# Champion model: stacking regressor (ridge + random forest + hist-GBR)
.venv/bin/python train.py --model stacking --sample 50000

# Full dataset (slower): --sample 0
```

Outputs: `models/model.joblib` (full sklearn Pipeline: encoders + regressor) and
`models/metrics.json` (MAE / RMSE / R2 on a 25% holdout).

## Quickstart (API + frontend)

```bash
# after training (models/model.joblib must exist)
.venv/bin/uvicorn car_price.api:app --reload --app-dir src
```

- `GET  /` — minimal HTML form (calls `/predict` from the browser)
- `GET  /health` — liveness + model status
- `POST /predict` — JSON in, JSON out

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"mileage":42000,"year_of_registration":2019,"standard_make":"BMW",
       "standard_model":"3 Series","standard_colour":"Black",
       "body_type":"Saloon","fuel_type":"Diesel"}'
# {"predicted_price":21114.56,"currency":"GBP","model_name":"stacking","mae":3175.25}
```

Interactive docs: http://127.0.0.1:8000/docs

## Docker

```bash
docker build -t car-price .
docker run --rm -p 8000:8000 car-price
curl -s http://localhost:8000/health
```

The image is a multi-stage build (build venv → slim runtime); it expects
`models/model.joblib` and `models/metrics.json` to already exist (run
`train.py` first — they are not trained inside the image).

## CI

`.github/workflows/ci.yml` runs on push/PR to `main`: `ruff check`, a smoke
training run (`train.py --sample 3000`), `pytest`, a `docker build`, and a
container smoke test against `/health` and `/predict`.

## Layout

- `src/car_price/` — importable package: `data`, `preprocessing`, `pipeline`, `train`, `config`, `api` (FastAPI service), `static/index.html` (frontend form)
- `train.py` — CLI entry point (also installed as `car-price-train`)
- `data/csv/adverts.csv` — raw dataset (~400k UK used-car adverts)
- `notebooks/` — original EDA / modelling notebooks the pipeline was consolidated from
- `tests/` — pytest suite (preprocessing, encoding, train round-trip, API via `TestClient`)
- `Dockerfile`, `.dockerignore` — multi-stage image serving the API
- `.github/workflows/ci.yml` — lint + smoke-train + test + docker build/run CI

## Develop

```bash
.venv/bin/python -m pytest   # tests (29 incl. API)
.venv/bin/ruff check .       # lint
```
