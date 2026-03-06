#!/bin/bash
# Complete pipeline execution script

set -euo pipefail

echo "========================================"
echo "Oxford Nanopore Polishing ML Pipeline"
echo "========================================"

# Create directories
mkdir -p data models outputs/plots outputs/results logs

# Check input data
if [ ! -f "data/all_samples_polishing_metrics.csv" ]; then
    echo "ERROR: data/all_samples_polishing_metrics.csv not found!"
    echo "Please copy your dataset to data/ directory"
    exit 1
fi

# Check required scripts
required_scripts=(
    "2_exploratory_analysis.py"
    "3_feature_engineering.py"
    "4_label_optimal_round.py"
    "5_train_models.py"
    "8_evaluate_models.py"
    "9_benchmark_resources.py"
    "10_sensitivity_analysis.py"
)

for script in "${required_scripts[@]}"; do
    if [ ! -f "$script" ]; then
        echo "ERROR: Required script not found: $script"
        exit 1
    fi
done

echo ""
echo "Step 1: Exploratory Data Analysis"
echo "========================================"
python 2_exploratory_analysis.py

echo ""
echo "Step 2: Feature Engineering"
echo "========================================"
python 3_feature_engineering.py

echo ""
echo "Step 3: Label Optimal Rounds (3-Class System)"
echo "========================================"
python 4_label_optimal_round.py

echo ""
echo "Step 4: Train Models (XGBoost, RF, Ordinal, Ensemble)"
echo "========================================"
python 5_train_models.py

echo ""
echo "Step 5: Final Evaluation vs Baselines"
echo "========================================"
python 8_evaluate_models.py

echo ""
echo "Step 6: Resource Benchmarking"
echo "========================================"
python 9_benchmark_resources.py

echo ""
echo "Step 7: Sensitivity Analysis"
echo "========================================"
python 10_sensitivity_analysis.py

echo ""
echo "========================================"
echo "Pipeline completed successfully!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  - outputs/plots/: Visualizations"
echo "  - outputs/results/: Metrics and reports"
echo "  - models/: Trained models"
echo ""
echo "To make predictions on new data:"
echo "  python 7_inference_pipeline.py --input data/new_samples.csv --output predictions.csv"