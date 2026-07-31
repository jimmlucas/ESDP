#!/bin/bash
set -euo pipefail

if [ "${1:-}" = "esdp" ] || [ "${1:-}" = "/app/esdp_cli.py" ]; then
  exec "$@"
fi

echo "=================================================="
echo "ESDP - Early Stop Decision Polishing"
echo "Starting API Service..."
echo "=================================================="

MODEL_PATH="${MODEL_PATH:-/app/models/best_model_pipeline.pkl}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Ensure logs directory exists (as non-root user)
mkdir -p /app/logs

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: Model file not found at: $MODEL_PATH"
  echo "Expected: /app/models/best_model_pipeline.pkl"
  echo "If running with docker-compose, ensure ./models is mounted and contains the file."
  exit 1
fi

echo "Model found: $MODEL_PATH"

# Check optional feature names
FEATURE_NAMES_PATH="$(dirname "$MODEL_PATH")/feature_names.txt"
if [ ! -f "$FEATURE_NAMES_PATH" ]; then
  echo "WARNING: Feature names file not found at $FEATURE_NAMES_PATH"
  echo "Model may still work if feature names are embedded in the pipeline."
fi

echo ""
echo "Environment Configuration:"
echo "  - Python: $(python --version)"
echo "  - Model Path: $MODEL_PATH"
echo "  - Port: $PORT"
echo "  - Workers: $WORKERS"
echo "  - Log Level: $LOG_LEVEL"
echo ""

# Startup import check
echo "Running startup import check..."
python - <<'PY'
import sys
try:
    import esdp_decide
    import api_service
    print("OK: Modules imported successfully")
except Exception as e:
    print(f"ERROR: Module import failed: {e}")
    sys.exit(1)
PY

echo ""
echo "Startup checks passed"
echo "Starting server..."
echo "=================================================="
echo ""

exec "$@"
