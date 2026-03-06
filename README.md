---
title: "README"
output: github_document
---

# ESDP

### A Decision Framework for Resource-Efficient Bacterial Genome Polishing

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)]()
[![Paper](https://img.shields.io/badge/paper-under%20preparation-orange)]()

> **Stop polishing when it matters, not when it is scheduled**

---

**Note:** This tool is intended for research use and decision support. Predictions should be interpreted alongside assembly-quality metrics and domain expertise.

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

ESDP is a machine learning-based decision framework for Oxford Nanopore bacterial genome polishing. Instead of always running a fixed number of polishing rounds, ESDP uses early polishing and assembly-quality signals to recommend whether a sample can stop early, continue to an intermediate stage, or proceed to the full polishing schedule.

The repository includes:

- a full data-processing and model-training pipeline
- feature engineering and target relabeling for a three-class stopping task
- formal evaluation against naive baselines
- resource benchmarking and sensitivity analysis
- a command-line interface for workflow integration
- a FastAPI service for online inference
- Docker and Docker Compose support for reproducible deployment

### Problem Statement

Iterative polishing is a standard step in long-read bacterial genome assembly, but most practical workflows still apply a **fixed number of rounds**, commonly five, regardless of whether quality has already stabilized. This creates three problems:

- **Computational inefficiency:** assemblies that converge early continue consuming CPU time unnecessarily.
- **Limited decision support:** standard polishing workflows do not provide a principled recommendation of when to stop.
- **Workflow rigidity:** fixed-round execution prevents adaptive resource allocation across samples with different quality profiles.

### Our Solution

ESDP addresses this by learning **data-driven stopping decisions** from polishing trajectories. The current system predicts a **three-class stopping recommendation** derived from the original five-round formulation:

| Class | Recommendation | Original rounds | Interpretation |
|:---:|---|:---:|---|
| 1 | Early | 1-2 | assembly reaches acceptable quality quickly |
| 2 | Medium | 3-4 | additional polishing is useful but full execution may be unnecessary |
| 3 | Late | 5 | full polishing schedule is recommended |

This formulation reduces sparsity in the original five-round problem and provides a more stable and operationally useful decision target.

---

## Key Features

### 1. Three-class stopping system

Instead of predicting an exact round from 1 to 5, ESDP predicts an **Early / Medium / Late** stopping recommendation. This design:

- reduces confusion between adjacent rounds
- improves class balance relative to the original five-class task
- preserves the ordinal structure of the polishing process
- maps naturally to practical workflow decisions

### 2. Feature engineering from polishing trajectories

The pipeline derives features from assembly-quality metrics across polishing rounds, including:

- base features such as `n50`, `qv`, `error_rate`, `busco_complete`, `assembly_frac`, `num_contigs`, and `total_length`
- delta and improvement features such as `delta_qv`, `delta_busco_complete`, `delta_n50`, and `delta_error_rate`
- normalized features relative to round 1 such as `qv_from_r1`, `n50_from_r1`, `error_rate_from_r1`, and `busco_complete_from_r1`
- cumulative and trend features such as `delta_qv_cumsum`, `delta_busco_complete_cumsum`, `delta_qv_trend`, and `score_improvement_trend`
- plateau-oriented indicators such as `is_plateau` and `plateau_streak`
- domain-specific summary features such as `completeness_score`, `assembly_quality`, and `polishing_effectiveness`
- Flye-derived coverage features such as `coverage_est`, `mean_edge_coverage`, and `align_err_consensus`

### 3. Multiple model families

The training pipeline evaluates several model families:

1. **XGBoost**
2. **Random Forest**
3. **Ordinal Regression**
4. **Soft-voting Ensemble**

The reference run included in this repository selected **Random Forest** as the best-performing model on the held-out sample-level test split.

### 4. Conservative decision logic for online use

Offline predictions are complemented by a decision layer that supports operational deployment, including:

- confidence-aware conservative bias
- optional force-conservative behavior
- domain-specific rule overrides
- transparent probabilities and reasoning in the final response

### 5. Deployable interfaces

ESDP can be used in different ways:

- **CLI** for local execution and workflow integration
- **FastAPI REST service** for online inference
- **Docker / Docker Compose** for reproducible deployment
- **tests and structured logging** for more reliable operational use

---

## Performance Highlights

### Dataset Statistics

The reference dataset included in this repository contains:

- **805 rows** in the final training table
- **41 bacterial samples**
- **161 sample-coverage groups**
- **5 original polishing rounds**
- **3 stopping classes** after relabeling

### Model Performance

In the reference run, the **best model was Random Forest**.

| Metric | Value |
|---|---:|
| Accuracy | 0.629 |
| Balanced Accuracy | 0.592 |
| Macro F1 | 0.568 |
| MAE | 0.482 |
| Accuracy within one class | 0.888 |
| Quadratic Weighted Kappa | 0.561 |

Formal baseline comparison:

| Model | Balanced accuracy | Macro F1 | MAE | QWK |
|---|---:|---:|---:|---:|
| Best model | 0.592 | 0.568 | 0.482 | 0.561 |
| Always late baseline | 0.333 | 0.231 | 0.735 | 0.000 |
| QV threshold baseline | 0.333 | 0.231 | 0.735 | 0.000 |
| R1-only Random Forest | 0.566 | 0.571 | 0.465 | 0.538 |

### Practical Impact

Compared with a fixed five-round baseline, the reference ESDP benchmark reported:

| Metric | Value |
|---|---:|
| CPU reduction | 44.71% |
| Mean CPU saved per trajectory | 0.60 h |
| Mean QV loss | -0.0038 |
| Mean BUSCO loss | -0.51% |
| Efficiency gain | 200.17% |
| Zero QV loss trajectories | 33 / 34 |
| Acceptable-loss trajectories | 34 / 34 |

These results indicate that ESDP can reduce polishing cost substantially while largely preserving assembly quality.

---

## Model Details

### Training pipeline

The repository includes a complete end-to-end workflow:

- `1_csv_merge.py`: merge and harmonize polishing metrics
- `2_exploratory_analysis.py`: exploratory analysis and plots
- `3_feature_engineering.py`: derived feature construction
- `4_label_optimal_round.py`: optimal-round labeling and three-class conversion
- `5_train_models.py`: multi-model training and model selection
- `8_evaluate_models.py`: formal evaluation against baselines
- `9_benchmark_resources.py`: quality-versus-cost benchmark
- `10_sensitivity_analysis.py`: confidence-threshold analysis

### Inference and serving

- `7_inference_pipeline.py`: inference workflow
- `esdp_decide.py`: final decision logic and conservative overrides
- `esdp_cli.py`: command-line entry point
- `api_service.py`: FastAPI service for online inference

### Leakage control

To avoid leakage between related observations, the reference split is performed at the **sample level**, not the row level. This prevents different rows from the same biological sample from being distributed across train and test partitions.

### Deployment-oriented components

The project includes:

- `Dockerfile`
- `docker-compose.yml`
- `docker-entrypoint.sh`
- `config.yaml`
- `test/` for API, decision, and integration tests
- `docs/` for installation, usage, and dataset-building guidance

---

## Citation

If you use this repository, please cite the associated software release and manuscript when available.

```bibtex
@software{esdp,
  author = {Jimmy Lucas and collaborators},
  title = {ESDP: A Decision Framework for Resource-Efficient Bacterial Genome Polishing},
  year = {2026},
  url = {https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing}
}
```

---

## Acknowledgments

We acknowledge:

- the developers of **Racon**, **Medaka**, **Flye**, **QUAST**, and **BUSCO**
- public data contributors and repositories that enabled dataset construction
- the open-source communities behind **scikit-learn**, **XGBoost**, **FastAPI**, and related tools

---

## Contact

**Jimmy Lucas**  
ISGlobal, Barcelona Institute for Global Health, Barcelona, Spain  
Email: `jimmy.lucas@isglobal.org`

Project issues and feature requests should be reported through the repository issue tracker.

---

## License

This project is distributed under the MIT License. See `LICENSE` for details.

---

## Future Improvements

Planned directions include:

1. expanding the dataset across more taxa and coverage regimes
2. improving calibration and threshold selection for online decisions
3. extending benchmarking across additional polishing workflows
4. refining deployment and monitoring components for production settings
5. improving model robustness under limited or noisy input metrics

---

## References

### Related tools

- **Racon**: ultrafast consensus for raw de novo genome assembly
- **Medaka**: Oxford Nanopore consensus polishing
- **Flye**: assembler for long and noisy reads
- **QUAST**: quality assessment for genome assemblies
- **BUSCO**: completeness assessment using conserved orthologs

### Selected publications

1. Vaser R, Sović I, Nagarajan N, Šikić M. Fast and accurate de novo genome assembly from long uncorrected reads. *Genome Research*. 2017.
2. Simão FA, Waterhouse RM, Ioannidis P, Kriventseva EV, Zdobnov EM. BUSCO: assessing genome assembly and annotation completeness with single-copy orthologs. *Bioinformatics*. 2015.
3. Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*. 2002.
4. Frank E, Hall M. A simple approach to ordinal classification. *ECML*. 2001.
