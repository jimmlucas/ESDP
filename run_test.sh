#!/bin/bash
# Test runner for ESDP project
# Usage: bash run_test.sh [unit|integration|api|all]

set -euo pipefail

TEST_TYPE="${1:-all}"

echo "=========================================="
echo "ESDP Test Suite"
echo "=========================================="

# Activate environment if needed
# conda activate ont-polishing

# Check that Python is available
if ! command -v python >/dev/null 2>&1; then
  echo "Error: python is not available in the current environment."
  exit 1
fi

# Check that pytest is installed in the active Python environment
if ! python -m pytest --version >/dev/null 2>&1; then
  echo "Error: pytest is not installed in the current Python environment."
  echo "Install it with:"
  echo "  pip install pytest pytest-cov"
  exit 1
fi

case "$TEST_TYPE" in
  unit)
    echo "Running unit tests..."
    python -m pytest test/test_esdp_decide.py -v \
      --cov=esdp_decide \
      --cov-report=term-missing
    ;;
  
  integration)
    echo "Running integration tests..."
    python -m pytest test/test_pipeline_integration.py -v
    ;;
  
  api)
    echo "Running API tests..."
    python -m pytest test/test_api_service.py -v
    ;;
  
  all)
    echo "Running all tests..."
    python -m pytest test/ -v \
      --cov=esdp_decide \
      --cov=api_service \
      --cov-report=term-missing \
      --cov-report=html
    echo ""
    echo "Coverage report generated in htmlcov/index.html"
    ;;
  
  *)
    echo "Unknown test type: $TEST_TYPE"
    echo "Usage: bash run_test.sh [unit|integration|api|all]"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "Tests completed successfully!"
echo "=========================================="