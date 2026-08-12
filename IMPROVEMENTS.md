# Improvements — adv-ml-car-price-prediction

**Goal:** A deployed used-car price estimator: a web app where a user enters car details and gets an instant price prediction, served by a versioned model behind a REST API, with reproducible training and CI/CD.

## TL;DR — Path to production
- [x] Fix `src/` into a runnable training pipeline that saves a model + metrics
- [x] FastAPI `/predict` service + simple web UI
- [x] Tests, pinned deps, lint
- [x] Docker + GitHub Actions CI/CD
- [x] Git LFS for `data/csv/adverts.csv` (kept it out of raw git storage without needing a cloud DVC remote)
- [ ] MLflow experiment tracking + DVC data/model versioning
- [ ] Public live demo (scaffolded — `fly.toml` + CI `deploy` job, needs `FLY_API_TOKEN` + `FLY_DEPLOY_ENABLED`)

## Current state
- `src/car_price/` is a clean, tested package: `data`, `preprocessing`, `pipeline`, `train` produce `models/model.joblib` (full sklearn `Pipeline`) + `models/metrics.json`.
- `src/car_price/api.py` is a FastAPI service (`/`, `/health`, `/predict`) backed by that artifact, with a static HTML+JS form (`src/car_price/static/index.html`) as the frontend and Pydantic v2 request/response validation.
- 29 pytest tests cover preprocessing, the pipeline, train round-trip, and the API (`TestClient`); `ruff check .` is clean.
- `Dockerfile` (multi-stage: build venv → slim runtime) + `.dockerignore`; `.github/workflows/ci.yml` runs ruff, a smoke-train, pytest, `docker build`, and a container smoke test hitting `/health` + `/predict`.
- Not yet done: MLflow tracking, DVC data versioning, public live deployment.
- 11 Jupyter notebooks remain as the historical record of the original EDA / modelling work the pipeline was consolidated from.

## Key improvements
- ML: finish `src/` into a runnable `train.py` that fits the stacking model and serializes a full sklearn `Pipeline` (encoders + model) to disk; pin a champion model and log metrics (MAE/RMSE/R²).
- API: FastAPI service with a `/predict` endpoint, Pydantic request/response schemas, and input validation matching the feature set.
- Frontend: lightweight Streamlit or React form for car attributes returning a live price estimate + confidence range.
- Quality: pytest suite for preprocessing/encoding/prediction, populated `requirements.txt` / lockfile, and pre-commit (ruff + black).
- Ops: Dockerize API + model, GitHub Actions for lint/test/build, and a public live demo.

## Latest tech to showcase
- FastAPI + Pydantic v2 for a typed inference service.
- MLflow (or Weights & Biases) for experiment tracking and model registry.
- Git LFS for the 31MB CSV (chosen over DVC: LFS authenticates via the same
  token `actions/checkout` already uses, so CI needs no extra cloud-storage
  secret; DVC remains an option later for experiment/model-version tracking).
- uv for fast, reproducible dependency management; ruff for linting.
- Docker + GitHub Actions CI, deployed to Hugging Face Spaces / Render / Fly.io.

## Roadmap
1. Consolidate notebooks into a working, tested training pipeline that outputs a saved model + metrics.
2. Wrap the model in a FastAPI service and a simple UI; containerize both.
3. Add MLflow tracking, DVC data versioning, CI/CD, and ship a public live demo.
