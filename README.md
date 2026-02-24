# ONT-Polishing-Optimizer

### Machine Learning-Based Early Stopping for Oxford Nanopore Bacterial Genome Polishing 


[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)]()
[![Paper](https://img.shields.io/badge/paper-BMC%20Bioinformatics-orange)]()

> **Stop polishing when it matters, not when it's scheduled**

---

**Note**: This tool is for research purposes. Always validate predictions with field expertise before making decisions.

---

**Version: 1.0**
---

## Table of Contents

- [Installation](docs/INSTALL.md)
- [Build Data-Set](docs/BUILD_DATASET.md)
- [Usage](docs/USAGE.md)
- [Overview](#overview)
  - [Problem Statement](#problem-statement)
  - [Our Solution](#our-solution)
  - [Key Features](#key-features)
  - [Performance Highlights](#performance-highlights)
  - [Model Details](#model-details)
  - [Citation](#citation)
  - [Acknowledgments](#acknowledgments)
  - [Contact](#contact)
  - [License](#license)
  - [Future Improvements](#future-improvements)
  - [References](#references)

---
## Overview

Oxford Nanopore sequencing produces long reads that enable high-quality genome assemblies. However, standard polishing pipelines typically run a fixed number of Racon rounds (e.g. 4–5) regardless of whether early rounds already yield near-optimal assemblies.

**This leads to:**

- Unnecessary compute time and cost (redundant polishing rounds)
- No curate way to decide when to stop
- Poor support for "early-stop" scenarios in current tooling

**This tool uses machine learning** to predict the optimal stopping strategy (early / medium / late) based on assembly quality metrics from early polishing rounds, **reducing computational cost while maintaining assembly quality**.

---
### Problem Statement
---
**Original challenge:**

- 5-class optimal-round problem (rounds 1–5) with strong class imbalance
- Highly skewed label distribution:
  - **Class 5** (late, round 5) ≈ **89 groups**
  - **Class 1** (early, round 1) ≈ **7 groups**
- Baseline performance with direct 5-class prediction:
  - Balanced Accuracy: ~0.41–0.54
  - Macro F1: ~0.40–0.50
- **Critical issue**: at least one intermediate class had ~0% recall in early models

---
### Our Solution
---
- Reformulate the problem as a **3-class ordinal decision**:
  - **Early** (rounds 1–2)
  - **Medium** (rounds 3–4)
  - **Late** (round 5)
- Advanced feature engineering with **40+ derived features** capturing dynamics across rounds
- Multiple model families:
  - XGBoost, Random Forest, Ordinal Regression, Ensembles
- Imbalance handling:
  - SMOTE, class weights
- Stratified Group K-Fold splits at the **polishing group** level (`Sample` × `Coverage`)

---

## Key Features

### 3-Class System (Core Design)

Instead of predicting exact rounds (1–5), the tool predicts **stopping strategies**:

| Class | Strategy | Original Rounds | Groups | Description                                  |
|:-----:|----------|:---------------:|:------:|----------------------------------------------|
| **1** | Early    | 1–2             | 31     | Stop early – assembly already high quality   |
| **2** | Medium   | 3–4             | 41     | Stop mid-way – quality/cost trade-off        |
| **3** | Late     | 5               | 89     | Continue to end – assembly needs more work   |

**Why this works:**
- Reduces confusion between adjacent rounds (R3 vs R4)
- More balanced class distribution than original 5-class formulation
- Ordinal relationship preserved (Early → Medium → Late)

### Advanced Feature Engineering

The pipeline derives [**40+ features**](#feature-groups) from per-round metrics:

#### Delta Features (Round-to-Round Changes)
- Captures improvement velocity between rounds

#### Ratio Features
- QV improvement rate
- BUSCO per contig
- Cost–benefit ratios
- Efficiency metrics

#### R1-Normalized Features
- All metrics relative to polishing round 1
- Enables comparison across samples with different baselines

#### Domain-Specific Features
- **Completeness Score**: Combines BUSCO, error rate, contiguity
- **Assembly Quality Score**: Weighted combination of N50, assembly fraction, contig count
- **R1 Quality Indicators**: Early stopping feasibility flags

#### Cumulative Features
- Running sums of improvements across rounds
- Cumulative gains tracking

#### Plateau Detection Features
- Indicators of diminishing returns in QV, BUSCO, and error rate
- Automatically detects when polishing is no longer beneficial

### Multiple ML Models

The training pipeline includes:

1. **XGBoost** 
   - Gradient boosting with class weights
   - Excellent with heterogeneous and imbalanced data
   - Tuned for the 3-class ordinal decision

2. **Random Forest**
   - Robust baseline with interpretable feature importance
   - 800 trees with controlled depth
   - Less prone to overfitting

3. **Ordinal Regression (LogisticAT)** 
   - Respects the natural order between Early/Medium/Late
   - Threshold-based approach
   - Optimized for ordinal outcomes

4. **Ensemble** 
   - Voting classifier combining multiple models
   - Reduces variance and improves robustness
   - “Best of both worlds” between XGBoost and RF

### Comprehensive Evaluation

#### Classification Metrics
- **Accuracy**
- **Balanced Accuracy** (class-imbalance aware)
- **Macro F1-Score** (equal weight for all classes)
- Per-class precision, recall, F1

#### Ordinal Metrics
- **Mean Absolute Error (MAE)** in class index
- **Accuracy within ±1 class**
- **Quadratic Weighted Kappa (QWK)**

#### Visual Outputs
- Confusion matrices
- Feature importance plots
- Practical impact plots (rounds saved vs quality retained)
- Performance stratified by genus and coverage

---

## Performance Highlights

### Model Performance (3-Class System)

The dataset used in this repository was 805 rows, 161 groups, the **best model (XGBoost)** achieves on the held-out test set:


| Metric            | Value  | Description                           |
|-------------------|--------|---------------------------------------|
| **Accuracy**      | 0.661  | Overall accuracy                      |
| **Balanced Accuracy** | **0.635** | Class-imbalance aware accuracy |
| **Macro F1-Score**| 0.627  | Average F1 across all classes         |
| **MAE**           | 0.388  | Average class distance error          |
| **Accuracy ±1**   | 0.952  | Predictions within 1 class            |
| **QWK**           | 0.600  | Quadratic Weighted Kappa              |
| **Early Recall**  | 0.543  | Correctly identified early stops      |

### Practical Impact

| Metric | Value | Impact |
|--------|-------|--------|
| **Computational Time Saved** | **46.7%** | 4.0h vs 7.5h per sample |
| **Quality Retained** | **93.9%** | Only 6.1% quality loss |
| **Avg Rounds Recommended** | **2.67** | vs 5 rounds maximum |
| **Savings for 100 Genomes** | **350 hours** | CPU time saved |

### Dataset Statistics

- **805 observations** (valid polishing rounds)
- **161 unique groups** (samples × coverage combinations)
- **9 bacterial genera**: *Acinetobacter*, *Corynebacterium*, *Enterococcus*, *Escherichia*, *Klebsiella*, *Pseudomonas*, *Salmonella*, *Staphylococcus*, *Streptococcus*
- **3 coverage levels**: 10×, 20×, 40×
- **5 polishing rounds** evaluated per sample


**Summary statistics:**
- Average time saved: **46.7%** (4.0h vs 7.5h per sample)
- Average quality retained: **93.9%**
- Recommended rounds: **2.67** (vs 5 maximum)

---

## Model Details

### Architecture

The tool uses an **assemble approach** combining multiple model families:

#### 1. XGBoost

```python
XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight={1: 2.0, 2: 1.5, 3: 1.0},
    random_state=42
)
```

**Why XGBoost:**
- Handles class imbalance well with `scale_pos_weight`
- Robust to feature scaling differences
- Built-in feature importance
- Fast training with GPU support

#### 2. Random Forest

```python
RandomForestClassifier(
    n_estimators=800,
    max_depth=12,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight={1: 2.0, 2: 1.5, 3: 1.0},
    random_state=42,
    n_jobs=-1
)
```

**Why Random Forest:**
- Less prone to overfitting than single trees
- Good baseline performance
- Interpretable feature importance
- Handles non-linear relationships

#### 3. Ordinal Regression (LogisticAT)

```python
LogisticAT(alpha=1.0)
```

**Why Ordinal:**
- Respects the natural order (Early < Medium < Late)
- More sample-efficient for ordinal outcomes
- Reduces confusion between adjacent classes
- Theoretically motivated for this problem

#### 4. Ensemble (Voting Classifier)

```python
VotingClassifier(
    estimators=[
        ('xgboost', xgb_model),
        ('random_forest', rf_model)
    ],
    voting='soft',         # Use predicted probabilities
    weights=[2, 1]         # XGBoost weighted higher
)
```

**Why Ensemble:**
- Combines strengths of multiple models
- Reduces variance
- More robust predictions
- Higher confidence in unanimous votes

### Feature Groups

The 40+ features are organized into categories:

<details><summary>Click to see complete feature list</summary>

#### Base Features (13)
- `n50`, `qv`, `error_rate`
- `busco_complete`, `busco_fragmented`, `busco_missing`
- `assembly_frac`, `assembly_error`
- `num_contigs`, `total_length`
- `coverage`, `genus_encoded`
- `round`

#### Delta Features (15)
- `delta_qv_r1_r2`, `delta_busco_r1_r2`, `delta_error_r1_r2`
- `delta_qv_r2_r3`, `delta_busco_r2_r3`, `delta_error_r2_r3`
- ... (for each consecutive round pair)

#### R1-Normalized Features (5)
- `qv_from_r1` - (QV_current - QV_r1)
- `busco_complete_from_r1`
- `error_rate_from_r1`
- `assembly_frac_from_r1`
- `n50_from_r1`

#### Ratio Features (6)
- `qv_per_round` - QV improvement per round
- `busco_per_round`
- `improvement_rate` - Overall quality gain velocity
- `cost_benefit_ratio` - Quality gain per computational unit
- `busco_per_contig` - BUSCO completeness per contig
- `error_reduction_rate`

#### Domain-Specific Scores (3)
- `completeness_score` - Weighted: 40% BUSCO + 30% error_rate + 30% N50
- `assembly_quality_score` - Weighted: 40% N50 + 30% assembly_frac + 30% (1/contigs)
- `r1_quality_flag` - Binary: Is R1 good enough for early stop?

#### Plateau Indicators (3)
- `plateau_qv` - Boolean: Has QV improvement stalled?
- `plateau_busco` - Boolean: Has BUSCO improvement stalled?
- `plateau_error` - Boolean: Has error rate improvement stalled?

#### Cumulative Features (2)
- `cumulative_qv_gain` - Sum of QV improvements up to current round
- `cumulative_busco_gain` - Sum of BUSCO improvements

</details>

### Training Configuration

```yaml
# Cross-Validation (conceptual setup)
cv_strategy: StratifiedGroupKFold
cv_folds: 5
grouping: (Sample, Coverage)

# Train/Test Split
test_size: 0.20
stratify: optimal_rounds_3class
random_state: 42

# Class Balancing
method: SMOTE + class_weights
smote_k_neighbors: 3
smote_sampling_strategy: "auto"
class_weights: {1: 2.0, 2: 1.5, 3: 1.0}

# Hyperparameter Tuning
search_method: manual / grid search
validation: 5-fold CV
metric: balanced_accuracy
```

### Model Selection Criteria

Best model selected based on:
1. **Balanced Accuracy** (primary metric)
2. **Macro F1-Score** (secondary)
3. **Minimum class recall** (especially Early) as practical constraint
4. **MAE** and **QWK** for ordinal consistency

---

### Metrics
On the reference dataset shipped with this project (805 rows, 161 groups), the default configuration yields:

| Metric            | Value |
| ----------------- | ----- |
| Balanced Accuracy | 0.635 |
| Macro F1          | 0.627 |
| MAE               | 0.388 |
| Accuracy ±1       | 0.952 |
| QWK               | 0.600 |

Small deviations are expected if library versions or random seeds differ.

---
### Tools & Dependencies

This project builds upon:
- [Flye](https://github.com/mikolmogorov/Flye) -  assembler for single-molecule sequencing reads
- [Racon](https://github.com/isovic/racon) - Genome polishing
- [Medaka]() - ONT polish assemble
- [QUAST](https://github.com/ablab/quast) - Assembly quality assessment
- [BUSCO](https://busco.ezlab.org/) - Genome completeness
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [imbalanced-learn](https://imbalanced-learn.org/) - SMOTE implementation

---
## Citation

If you use this tool in your research, please cite:

```bibtex
@software{ESDP-Early-Stop-Decision-Polishing,
  author={Jimmy Lucas},
  year={2025},
  url={https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing}
}
```
---

## Acknowledgments

We thank:

- **Oxford Nanopore Technologies** for long-read sequencing technology
- **All reasearching groups** for publish data in **NCBI**
- **The Racon team** for the excellent polishing tool
- **QUAST and BUSCO developers** for assembly quality assessment tools
- **scikit-learn and XGBoost communities** for ML frameworks
- **All contributors** who helped improve this tool

---

## Contact

### Maintainer
- GitHub: [@jimmlycas](https://github.com/jimmlucas)
- Website: [LinkedIn](https://www.linkedin.com/in/jimmlucas)

### Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing/discussions)
- **Questions**: [GitHub Discussions Q&A](https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing/discussions/categories/q-a)

### Community

- **GitHub**: [@AMRmicrobiology ](https://github.com/AMRmicrobiology)
---
## License

This project is licensed under the MIT License - see LICENSE file for details.

---
## Limitations

1. **Small dataset**: Only 161 groups total
2. **Genus imbalance**: Some genera underrepresented
3. **Coverage dependency**: Performance varies by coverage
4. **Feature availability**: Requires complete metrics
---
## Future Improvements

1. **More data collection**: Improve model robustness
2. **Transfer learning**: Pre-train on related tasks
3. **Deep learning**: Try neural networks with more data
4. **Active learning**: Prioritize labeling uncertain samples
5. **Real-time prediction**: Deploy as web service

---

## References

### Related Tools

- **[Racon](https://github.com/isovic/racon)** - Ultrafast consensus module for raw de novo genome assembly
- **[Medaka](https://github.com/nanoporetech/medaka)** - Neural network-based polishing for ONT
- **[Pilon](https://github.com/broadinstitute/pilon)** - Automated genome assembly improvement tool
- **[QUAST](http://quast.sourceforge.net/)** - Quality assessment tool for genome assemblies
- **[BUSCO](https://busco.ezlab.org/)** - Assessing genome assembly completeness
- **[Flye](https://github.com/mikolmogorov/Flye)** -Assembler for single-molecule sequencing reads

---

### Publications
---

1. **Racon**: Vaser, R., Sović, I., Nagarajan, N., & Šikić, M. (2017). Fast and accurate de novo genome assembly from long uncorrected reads. *Genome Research*, 27(5), 737-746.

2. **Medaka**: Oxford Nanopore Technologies. (2020). Medaka: Sequence correction provided by ONT Research.

3. **BUSCO**: Simão, F. A., et al. (2015). BUSCO: assessing genome assembly and annotation completeness with single-copy orthologs. *Bioinformatics*, 31(19), 3210-3212.

4. **Class Imbalance**: Chawla, N. V., et al. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

5. **Ordinal Classification**: Frank, E., & Hall, M. (2001). A simple approach to ordinal classification. *EMCL* 2001, 145-156.
