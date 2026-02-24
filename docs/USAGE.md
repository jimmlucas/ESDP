# ESDP manual

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Input Data](#input-data-format)
- [Quick Start - Minimal Example](#quick-start)
- [Usage](#usage)
  - [Complete Pipeline](#complete-pipeline)
  - [Individual Steps](#individual-steps)
  - [Making Predictions](#making-predictions)
- [Output](#-output)
- [Config](#config)

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Polishing Metrics                     │
│              (QUAST + BUSCO results for rounds 1-5)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Exploratory Data Analysis (2_exploratory_analysis.py)  │
│         • Validate data integrity                               │
│         • Analyze distributions                                 │
│         • Check for outliers                                    │
│         • Generate summary statistics                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Feature Engineering (3_feature_engineering.py)         │
│         • Generate 40+ derived features                         │
│         • Delta features (Δ between rounds)                     │
│         • Ratio features (efficiency metrics)                   │
│         • R1-normalized features                                │
│         • Domain-specific scores                                │
│         • Plateau detection indicators                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Label Optimal Rounds (4_label_optimal_round.py)        │
│         • Identify optimal stopping round per group             │
│         • Map 5 rounds → 3 classes (Early/Medium/Late)          │
│         • Validate label                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Train Models (5_train_models.py)                       │
│         • XGBoost with class weights                            │
│         • Random Forest (800 trees)                             │
│         • Ordinal Regression (LogisticAT)                       │
│         • Ensemble (Voting Classifier)                          │
│         • 5-fold stratified group CV                            │
│         • SMOTE for class balancing                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Evaluate Models (8_evaluate_models.py)                 │
│         • Classification metrics (Balanced Acc, Macro F1)       │
│         • Ordinal metrics (MAE, QWK, Acc±1)                     │
│         • Confusion matrices                                    │
│         • Feature importance analysis                           │
│         • Practical impact assessment                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT: Trained Models + Reports              │
│         • models/*.pkl (trained classifiers)                    │
│         • outputs/plots/ (visualizations)                       │
│         • outputs/results/ (metrics, reports)                   │
└─────────────────────────────────────────────────────────────────┘
```


### Pipeline Scripts Description

| Script | Purpose |
|--------|---------|
| `1_csv_merge.py` | Merge polishing metrics from multiple sources |
| `2_exploratory_analysis.py` | Data validation and visualization |
| `3_feature_engineering.py` | Generate 40+ derived features |
| `4_label_optimal_round.py` | Assign 3-class labels |
| `5_train_models.py` | Train XGBoost, RF, Ordinal, Ensemble |
| `7_inference_pipeline.py` | Make predictions on new data |
| `8_evaluate_models.py` | Comprehensive evaluation |
| `run_pipeline.sh` | Execute complete pipeline |

---

## Input Data Format

The tool expects a CSV file with polishing metrics from QUAST and BUSCO for each round.

### Required Columns

```csv
Sample,Genus,Coverage,round,n50,qv,error_rate,busco_complete,
busco_fragmented,busco_missing,assembly_frac,num_contigs,total_length
```

### Example Data

```csv
Sample,Genus,Coverage,round,n50,qv,error_rate,busco_complete,busco_fragmented,busco_missing,assembly_frac,num_contigs,total_length
sample_001,Escherichia,40,1,234567,35.2,0.082,95.3,2.1,2.6,0.987,45,4856234
sample_001,Escherichia,40,2,345678,38.1,0.045,97.8,1.2,1.0,0.993,32,4862341
sample_001,Escherichia,40,3,456789,39.5,0.032,98.5,0.8,0.7,0.995,28,4865123
sample_001,Escherichia,40,4,467890,39.8,0.029,98.7,0.6,0.7,0.996,26,4866234
sample_001,Escherichia,40,5,468901,40.0,0.028,98.8,0.5,0.7,0.996,25,4866789
```

### Column Descriptions

| Column | Description | Source | Example |
|--------|-------------|--------|---------|
| `sample_id` | Unique sample identifier | User-defined | sample_001 |
| `genus` | Bacterial genus | User-defined | Escherichia |
| `coverage` | Sequencing coverage | User-defined | 40 |
| `round` | Polishing round (1-5) | Sequential | 3 |
| `n50` | N50 contig length | QUAST | 456789 |
| `qv` | Consensus quality value | QUAST | 39.5 |
| `error_rate` | Per-base error rate | QUAST | 0.032 |
| `busco_complete` | Complete BUSCO genes (%) | BUSCO | 98.5 |
| `busco_fragmented` | Fragmented BUSCO genes (%) | BUSCO | 0.8 |
| `busco_missing` | Missing BUSCO genes (%) | BUSCO | 0.7 |
| `assembly_frac` | Fraction of reference covered | QUAST | 0.995 |
| `num_contigs` | Number of contigs | QUAST | 28 |
| `total_length` | Total assembly length | QUAST | 4865123 |

### Data Requirements

- **Minimum 5 rounds per sample**: Each sample must have metrics for rounds 1-5
- **Unique groups**: Each (sample_id, coverage) combination forms a group
- **No missing values**: All required columns must be populated
- **Consistent coverage labels**: Use 10, 20, 40 (or your actual coverages)

### Generating Input Data

To prepare the Input Data go to [BUILD_DATASET](../docs/BUILD_DATASET.md)

---

### Quick Start

Minimal example if you already have the input data ready.  
By default, `run_pipeline.sh` starts at `2_exploratory_analysis.py` assuming that
`data/all_samples_polishing_metrics.csv` already exists.


```bash
# 1. Clone and install
git clone https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing.git
cd ESDP-Early-Stop-Decision-Polishing
pip install -r requirements.txt

# 2. Prepare your data
# use your own CSV (must follow "Input Data Format")
cp your_polishing_metrics.csv data/all_samples_polishing_metrics.csv

# 3. Run the complete pipeline (starts at Step 1: EDA)
bash run_pipeline.sh

# 4. Check results
ls outputs/plots/      # Visualizations
ls outputs/results/    # Metrics and reports
ls models/             # Trained models + scaler + feature list
```

## Usage

### Complete Pipeline

The easiest way to run the entire pipeline:

```bash
bash run_pipeline.sh
```

This executes all steps sequentially:
1. Exploratory data analysis
2. Feature engineering
3. Label assignment (3-class system)
4. Model training (XGBoost, RF, Ordinal, Ensemble)
5. Comprehensive evaluation

### Individual Steps

You can also run each step independently:

#### Optional Step 0 – Build merged polishing metrics CSV

If you start from raw polishing outputs (QUAST/BUSCO/Flye per round), first build
the consolidated CSV that all subsequent steps expect:

```bash
python 1_csv_merge.py
```
**Output**
- create `data/all_samples_polishing_metrics.csv`

#### Step 1: Exploratory Data Analysis

```bash
python 2_exploratory_analysis.py
```

**Outputs:**
- `outputs/plots/correlation_matrix.png` – Feature correlation heatmap
- `outputs/plots/coverage_distribution.png` – Coverage distribution
- `outputs/plots/genus_distribution.png` – Genus distribution
- `outputs/plots/metric_distributions.png` – Distributions of base polishing metrics
- `outputs/plots/improvement_distributions.png` – Distributions of metric improvements
- `outputs/plots/metrics_by_round.png` – Metrics evolution across rounds
- `outputs/results/eda_summary.txt` – data quality / EDA summary report

outputs/results/eda_summary.txt – Data quality / EDA summary report
**What it does:**
- Validates data integrity
- Checks for missing values
- Summarizes distributions of key metrics (QV, BUSCO, error rate, N50, contigs)
- Analyzes class distribution (genus, coverage, classes)
- Identifies outliers
- Generates summary statistics and EDA core EDA plots

#### Step 2: Feature Engineering

```bash
python 3_feature_engineering.py
```

**Outputs:**
- `data/training_dataset_engineered.csv` - Dataset with base metrics + engineered features


**Generated features:**
- Delta features (Δ between rounds:delta_qv, delta_busco_complete, delta_error_rate, etc.)
- Ratio features (efficiency metrics: qv_improvement_rate, busco_per_contig, n50_fraction, cost_benefit_ratio)
- R1-normalized features (qv_from_r1, busco_complete_from_r1, error_rate_from_r1, etc.)
- Domain-specific scores (completeness_score, assembly_quality, polishing_effectiveness)
- Plateau indicators (is_plateau, plateau_streak)

#### Step 3: Label Optimal Rounds

```bash
python 4_label_optimal_round.py
```

**Outputs:**
- `data/training_dataset_with_target.csv` - Labeled dataset

**Labeling strategy:**
- Validates label consistency
- Applies conservative early-stop rules when R1 is already stable
- Identifies optimal stopping round per group
- Computes optimal round per Sample+Coverage group and maps 5→3 classes
- Detects plateus using score improvement and a relative threshold from config.yaml


#### Step 4: Train Models

```bash
python 5_train_models.py
```

**Outputs:**
Models and artifacts:
- `models/best_model.pkl` – best-performing classifier (XGBoost, RF, ordinal, or ensemble)
- `models/scaler.pkl` – fitted StandardScaler used for features
- `models/feature_names.txt` – exact list of features used for training

Metrics:
- outputs/results/model_comparison.csv – metrics for all trained models
- outputs/results/training_metrics.json – same metrics in JSON format

Plots:
- outputs/plots/cm_xgboost.png – confusion matrix (XGBoost)
- outputs/plots/cm_random_forest.png – confusion matrix (Random Forest)
- outputs/plots/cm_ordinal.png – confusion matrix (Ordinal regression, if available)
- outputs/plots/cm_ensemble.png – confusion matrix (Ensemble, if available)
- outputs/plots/fi_xgboost.png – top feature importances (XGBoost)
- outputs/plots/fi_random_forest.png – top feature importances (Random Forest)

Training configuration:
- Group-aware split: train/test split stratified by class at group level (Sample+Coverage)
- SMOTE for class balancing (configurable in config.yaml)
- Class weights for imbalanced classes
- Multiple models trained:
  - XGBoost (multi-class)
  - Random Forest (multi-class)
  - Ordinal regression
  - Soft-voting ensemble (excluding ordinal model)
- Best model selected by highest balanced accuracy on the test set


#### Step 5: Evaluate Models

```bash
python 8_evaluate_models.py
```

**Outputs:**
- outputs/results/baseline_comparison.csv – comparison between the trained model and simple baselines

What it compares:
- Best_Model – the best_model.pkl from Step 4
- Baseline_Always_Late – heuristic that always predicts class 3 (Late)
- Baseline_QV_Threshold_30 – heuristic: Early if QV > 30, else Late.
- Baseline_R1_Only_RF – Random Forest using only R1-level features.

### Making Predictions

Once models are trained, use them for inference:

```bash
# Predict on new data
python 7_inference_pipeline.py \
  --input data/new_samples.csv \
  --output predictions.csv \
  --model models/best_model.pkl
```

**Prediction output format:**

```csv
sample_id,coverage,predicted_class,predicted_strategy,confidence,recommended_rounds
sample_001,40,1,Early,0.87,2
sample_002,20,2,Medium,0.65,3
sample_003,10,3,Late,0.91,5
```

### Config

Edit `config.yaml` to customize:

<details>
<summary>Click to see key configuration options</summary>

```yaml
# Class Configuration (CRITICAL IMPROVEMENT: 3-class system)
classes:
  n_classes: 3
  class_mapping:
    1: "Early (R1-R2)"  # Stop early: rounds 1-2
    2: "Medium (R3-R4)"  # Stop mid-way: rounds 3-4
    3: "Late (R5)"      # Continue to end: round 5
  
# Model Training Configuration
models:
  random_state: 42
  test_size: 0.20
  cv_folds: 5
  
  # XGBoost Configuration
  xgboost:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 3
    gamma: 0.1
    reg_alpha: 0.1
    reg_lambda: 1.0
  
  # Random Forest Configuration
  random_forest:
    n_estimators: 800
    max_depth: 12
    min_samples_leaf: 2
    max_features: "sqrt"
  
  # Ordinal Regression Configuration
  ordinal_regression:
    alpha: 1.0

# Imbalance Handling
imbalance:
  use_smote: true
  smote_k_neighbors: 3
  smote_sampling_strategy: "auto"
  
  # Class weights
  class_weights:
    1: 2.0   # Early - boost minority
    2: 1.5   # Medium
    3: 1.0   # Late - most common

# Evaluation Metrics
evaluation:
  primary_metric: "balanced_accuracy"
  target_metrics:
    balanced_accuracy: 0.65
    macro_f1: 0.60
    min_class_recall: 0.40
    mae: 0.5
    qwk: 0.50

# Hierarchical Policy (3-class version)
hierarchical:
  stage_a:
    # Binary: Early (1) vs Continue (2-3)
    threshold: 0.5
    model: "logistic_regression"
  
  stage_b:
    # Binary: Medium (2) vs Late (3)
    model: "random_forest"

```

</details>

---
## Output

### Directory Structure

```
ont-polishing-optimizer/
├── data/
│   ├── all_samples_polishing_metrics.csv          # Input data
│   ├── training_dataset_with_target.csv           # Labeled data
│   └── training_dataset_engineered.csv            # Engineered features
├── models/
│   ├── best_model.pkl        # Best-performing model (XGBoost / RF / Ensemble / Ordinal)
│   ├── scaler.pkl            # Fitted StandardScaler used for features
│   └── feature_names.txt     # One feature name per line (order used for training)
├── outputs/
    ├── plots/                                # All figures (EDA + model evaluation)
    │   ├── cm_ensemble.png                   # Confusion matrix - Ensemble
    │   ├── cm_ordinal.png                    # Confusion matrix - Ordinal regression
    │   ├── cm_random_forest.png              # Confusion matrix - Random Forest
    │   ├── cm_xgboost.png                    # Confusion matrix - XGBoost
    │   ├── correlation_matrix.png            # Feature correlation heatmap
    │   ├── coverage_distribution.png         # Coverage distribution
    │   ├── genus_distribution.png            # Genus distribution
    │   ├── fi_random_forest.png              # Feature importance - Random Forest
    │   ├── fi_xgboost.png                    # Feature importance - XGBoost
    │   ├── improvement_distributions.png     # Distributions of metric improvements
    │   ├── metric_distributions.png          # Distributions of base polishing metrics
    │   └── metrics_by_round.png              # Metrics evolution across rounds
    ├── results/
    │   ├── eda_summary.txt                   # Data quality / EDA summary report
    │   ├── baseline_comparison.csv           # Baseline vs model performance comparison
    │   ├── model_comparison.csv              # Per-model metric table (XGB, RF, Ordinal, Ensemble)
    │   └── training_metrics.json             # Same metrics as JSON (one record per model)
    └── logs/
        └── pipeline.log                      # Training pipeline log (INFO/WARN/ERROR)

```