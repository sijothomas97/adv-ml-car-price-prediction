# syntax=docker/dockerfile:1
# ---- Stage 1: build the virtualenv ----
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --upgrade pip \
    && /venv/bin/pip install --no-cache-dir .

# ---- Stage 2: slim runtime ----
FROM python:3.12-slim

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CAR_PRICE_MODEL_PATH=/app/models/model.joblib \
    CAR_PRICE_METRICS_PATH=/app/models/metrics.json

WORKDIR /app
COPY --from=builder /venv /venv
# Model artifact must exist before building -- use scripts/build.sh, which
# trains one automatically if models/model.joblib is missing.
COPY models/model.joblib models/metrics.json /app/models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "car_price.api:app", "--host", "0.0.0.0", "--port", "8000"]
