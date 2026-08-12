#!/usr/bin/env bash
# Builds the car-price Docker image, training a model artifact first if one
# isn't present. Run this instead of a bare `docker build` on a fresh clone --
# the Dockerfile's COPY of models/model.joblib fails with an opaque error
# otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-car-price:local}"

if [[ ! -f models/model.joblib || ! -f models/metrics.json ]]; then
  echo "No model artifact at models/model.joblib -- training one now (python train.py --model gbr)..."
  python train.py --model gbr
fi

docker build -t "$TAG" .
echo "Built $TAG"
