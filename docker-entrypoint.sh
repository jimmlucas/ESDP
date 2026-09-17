#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo "ESDP v2 - Sequential Adaptive Polishing Controller"
echo "Starting API service"
echo "=================================================="

MODEL_PATH="${MODEL_PATH:-/app/outputs/frozen_sequential_model_v2/esdp_sequential_rf_v2.joblib}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Frozen ESDP v2 model not found:"
    echo "  $MODEL_PATH"
    exit 1
fi

echo "Model found: $MODEL_PATH"
echo "Python: $(python --version)"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo "Log level: $LOG_LEVEL"

echo ""
echo "Running ESDP v2 startup validation..."

python - <<'PY'
from esdp_decide import model_info

info = model_info()

assert info["model_version"] == "esdp-sequential-rf-v2"
assert info["decision_threshold"] == 0.45
assert info["feature_count"] == 10

print("OK: frozen ESDP v2 model validated")
print("Model SHA256:", info["model_sha256"])
print("Feature schema SHA256:", info["feature_schema_sha256"])
PY

echo ""
echo "Starting FastAPI..."
echo "=================================================="

exec uvicorn api_service:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL"