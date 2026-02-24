#!/bin/bash
# Complete pipeline execution script

set -e  # Exit on error

echo "========================================"
echo "Oxford Nanopore Polishing ML Pipeline"
echo "========================================"

# Create directories
mkdir -p data models outputs/{plots,results} logs

# Check if data exists
if [ ! -f "data/all_samples_polishing_metrics.csv" ]; then
    echo "ERROR: data/all_samples_polishing_metrics.csv not found!"
    echo "Please copy your dataset to data/ directory"
    exit 1
fi

# Step 2: Exploratory Data Analysis
echo ""
echo "Step 1: Exploratory Data Analysis"
echo "========================================"
python 2_exploratory_analysis.py

# Step 3: Feature Engineering
echo ""
echo "Step 2: Feature Engineering"
echo "========================================"
python 3_feature_engineering.py

# Step 4: Label Optimal Rounds
echo ""
echo "Step 3: Label Optimal Rounds (3-Class System)"
echo "========================================"
python 4_label_optimal_round.py

# Step 5: Train Models
echo ""
echo "Step 4: Train Models (XGBoost, RF, Ordinal, Ensemble)"
echo "========================================"
python 5_train_models.py

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  - outputs/plots/: Visualizations"
echo "  - outputs/results/: Metrics and reports"
echo "  - models/: Trained models"
echo ""
echo "To make predictions on new data:"
echo "  python 7_inference_pipeline.py --input data/new_samples.csv --output predictions.csv"
echo ""
echo ""
echo "Step 5: Final Evaluation vs Baselines"
echo "========================================"
python 8_evaluate_models.py

echo ""
echo "========================================"
echo "Step 6: Resource Benchmarking"
echo "========================================"
python 9_benchmark_resources.py

echo ""
echo "========================================"
echo "Step 7: Sensitivity Analysis"
echo "========================================"
python 10_sensitivity_analysis.py
echo "========================================"
echo "All steps completed successfully!"
echo "========================================"