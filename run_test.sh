#!/bin/bash
# Test runner for ESDP project
# Usage: bash run_test.sh [unit|integration|api|all]

set -e

TEST_TYPE=${1:-all}

echo "=========================================="
echo "ESDP Test Suite"
echo "=========================================="

# Activate environment if needed
# source activate polishin_ml

case $TEST_TYPE in
  unit)
    echo "Running unit tests..."
    pytest test/test_esdp_decide.py -v --cov=esdp_decide --cov-report=term-missing
    ;;
  
  integration)
    echo "Running integration tests..."
    pytest test/test_pipeline_integration.py -v
    ;;
  
  api)
    echo "Running API tests..."
    pytest test/test_api_service.py -v
    ;;
  
  all)
    echo "Running all tests..."
    pytest test/ -v --cov=esdp_decide --cov=api_service --cov-report=term-missing --cov-report=html
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