# ESDP

### Early Stop Decision Polishing for Long-read Bacterial Genome Polishing

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Docker Hub](https://img.shields.io/badge/docker-jimmlucas%2Fesdp-blue)](https://hub.docker.com/r/jimmlucas/esdp)
[![Version](https://img.shields.io/badge/version-v1.0.2-success.svg)](https://github.com/jimmlucas/ESDP/releases/tag/v1.0.2)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18910597.svg)](https://doi.org/10.5281/zenodo.18910597)

> **Stop polishing when it matters, not when it is scheduled**

---

**Research-use software.** Predictions should be interpreted together with assembly-quality metrics and domain expertise.

**Version:** 1.0.2

---

## Table of Contents

- [Installation](docs/INSTALL.md)
- [Build Dataset](docs/BUILD_DATASET.md)
- [Usage](docs/USAGE.md)
- [Production CLI](docs/CLI.md)
- [ESDP-light scientific design](docs/ESDP_LIGHT_DESIGN.md)
- [ESDP v2 trajectory contract](docs/TRAJECTORY_INPUT_SCHEMA.md)
- [Model manifest and artifact verification](docs/MODEL_MANIFEST.md)
- [Overview](#overview)
- [Problem Formulation](#problem-formulation)
- [Key Features](#key-features)
- [Performance Highlights](#performance-highlights)
- [Model Details](#model-details)
- [Tools & Dependencies](#tools--dependencies)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [License](#license)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## Overview

Oxford Nanopore sequencing enables high-contiguity bacterial genome assemblies, but iterative polishing pipelines are commonly executed with a fixed number of rounds. In practice, this can lead to unnecessary computation after assembly quality has already stabilized.

ESDP is a machine learning-based decision-support framework designed to recommend whether polishing can stop early or should continue. Rather than predicting exact polishing rounds directly for deployment, ESDP combines statistical prediction with explicit operational safeguards to produce conservative and transparent stopping recommendations.

The framework was developed and evaluated on 805 polishing records derived from 41 bacterial samples spanning 9 genera, five polishing rounds, and four coverage groups (10X, 20X, 40X, and FULL). To avoid information leakage, evaluation was performed with sample-level separation, using 32 training samples and 9 held-out test samples.

ESDP supports both command-line and service-based use. The software includes a REST API implemented with FastAPI, containerized deployment, reproducible model artifacts, and dedicated verification scripts for decision logic, API behavior, benchmarking, and sensitivity analysis.

In the current benchmark, Random Forest was selected as the final model for downstream deployment and benchmarking based on its overall balance across exact classification, ordinal agreement, and error minimization on the held-out test set.

### Why ESDP?

Fixed-round polishing workflows are simple to run, but they often continue processing after assembly quality has already stabilized. This can lead to unnecessary computational cost, limited visibility into convergence behavior, and little practical support for adaptive stopping decisions.

ESDP addresses this problem by using machine learning to recommend a conservative polishing strategy based on early-round assembly-quality signals. Instead of relying only on a raw classifier output, the framework combines statistical prediction with explicit decision rules to support transparent and reproducible early-stop recommendations.

---

## Problem Formulation

During development, several challenges had to be addressed to make adaptive polishing decisions reliable and deployable.

### 1. From round-level prediction to operational stopping classes

A direct five-class formulation based on exact polishing rounds proved difficult to model robustly and was less suitable for deployment. To make the task more stable and operationally meaningful, ESDP reformulates the problem into three stopping categories:

- **Early**: stop after 1–2 rounds
- **Medium**: stop after 3–4 rounds
- **Late**: continue to round 5

This representation improves decision stability while preserving the ordinal structure of the polishing process.

### 2. Capturing polishing dynamics, not only absolute quality

Single-round metrics alone do not fully describe whether polishing is still improving meaningfully or has already plateaued. To address this, ESDP uses engineered features that summarize both assembly quality and convergence behavior, including:

- round-to-round delta features
- first-round normalized metrics
- ratio-based and cumulative descriptors
- plateau indicators
- domain-specific summary features

These features help the model distinguish between early convergence, transitional cases, and assemblies that still benefit from continued polishing.

### 3. Preventing leakage and preserving realistic evaluation

Because multiple records can be derived from the same biological sample across rounds and coverage groups, row-level splitting would produce information leakage. ESDP therefore uses sample-level grouping during training and evaluation so that held-out predictions reflect genuinely unseen isolates rather than repeated observations from the same sample.

### 4. Separating prediction from decision logic

A key design principle of ESDP is that model prediction and deployment-time decision support are treated separately. The final recommendation layer applies explicit safeguards to make outputs more conservative and interpretable, including:

- first-round quality overrides
- confidence-aware escalation to more conservative recommendations
- optional forced-conservative operation

This design makes ESDP more suitable for practical workflow integration than a classifier alone.

---

## Key Features

### 3-class stopping framework

Instead of predicting exact polishing rounds directly for deployment, ESDP predicts **stopping categories** that are more stable and operationally meaningful:

| Class | Strategy | Recommended rounds | Description |
|:-----:|----------|:------------------:|-------------|
| **1** | Early    | 1–2                | Assembly quality stabilizes early |
| **2** | Medium   | 3–4                | Transitional regime with moderate additional benefit |
| **3** | Late     | 5                  | Continued polishing is recommended |

This formulation preserves the ordinal structure of the polishing process while reducing unnecessary sensitivity to adjacent round-level differences. The held-out test results also showed that most residual errors occurred between adjacent classes rather than as extreme misclassifications. 

### Feature engineering for polishing dynamics

ESDP uses a structured feature space built from per-round polishing and assembly-quality measurements. The feature set includes:

- **round-to-round delta features**
- **ratio-based descriptors**
- **cumulative improvement variables**
- **first-round normalized metrics**
- **plateau indicators**
- **domain-specific summary features**

These features were designed to capture not only absolute assembly quality, but also convergence behavior and diminishing returns across polishing rounds. 

### Decision-support layer

A key design element of ESDP is the separation between **statistical prediction** and **operational decision logic**. Model outputs are processed by a dedicated decision module that applies explicit safeguards to support conservative and transparent recommendations. These include:

- **first-round quality overrides**
- **confidence-aware escalation to more conservative recommendations**
- **optional forced-conservative operation**

Returned outputs include the predicted class, recommended number of polishing rounds, class probabilities, confidence score, decision rationale, warning flags, and metadata on applied overrides. 

### Interfaces and deployment

ESDP supports both **command-line execution** and **service-based inference**. The software includes:

- a **REST API** implemented with **FastAPI**
- containerized deployment
- bundled model artifacts and feature metadata
- verification scripts for decision logic, API behavior, and pipeline integration
- reproducible benchmarking and sensitivity-analysis workflows

This design allows the framework to function both as software and as a reproducible decision system for adaptive genome polishing. 

### Evaluation outputs

The evaluation workflow generates standardized artifacts for inspection and reuse, including:

- trained models
- preprocessing objects
- selected feature lists
- confusion matrices
- feature-importance visualizations
- comparative performance summaries

These outputs support reproducibility, model inspection, and downstream benchmarking.

---

## Performance Highlights

### Dataset statistics

- **41 bacterial samples**
- **805 polishing records**
- **9 bacterial genera**
- **5 polishing rounds**
- **4 coverage groups:** 10X, 20X, 40X, and FULL
- **Sample-level split:** 32 training samples / 9 held-out test samples

The current benchmark dataset was generated from Oxford Nanopore bacterial read sets assembled with Flye and processed through an iterative polishing workflow under a fixed experimental configuration. 

### Final model performance

On the held-out sample-level test set, the selected **Random Forest** model achieved:

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.629 |
| **Balanced Accuracy** | **0.592** |
| **Macro-F1** | 0.568 |
| **MAE** | 0.482 |
| **Accuracy ±1 class** | 0.888 |
| **QWK** | 0.561 |

Class-wise performance was strongest for the early and late stopping categories, while the intermediate class remained the most difficult to resolve. For Random Forest, recall was **0.822** for class 1, **0.286** for class 2, and **0.667** for class 3. 

### Practical benchmark

Across 34 valid test trajectories derived from 9 held-out samples, ESDP achieved:

- **0.60 CPU-hours saved per trajectory** on average (95% CI: 0.44–0.76)
- **44.71% mean CPU reduction** (95% CI: 32.94–56.50)
- **33/34 trajectories with zero QV loss** (97.1%)
- **34/34 trajectories within the predefined acceptable loss range** (100.0%)
- **200.17% mean efficiency gain** relative to fixed five-round polishing

These results indicate that ESDP can provide practical computational savings while preserving assembly quality in the evaluated benchmark setting.

---

## Model Details

### Final model

During development, ESDP compared multiple supervised learning strategies, including **XGBoost**, **Random Forest**, **ordinal regression**, and **ensemble configurations**. On the held-out sample-level test set, **Random Forest** achieved the best overall balance across exact classification, ordinal agreement, and error minimization, and was therefore selected as the final model for downstream benchmarking and deployment. 

### Decision-support design

A key design principle of ESDP is the separation between **statistical prediction** and **deployment-time decision support**. Model outputs are processed through a dedicated decision layer that applies explicit safeguards to support conservative and interpretable recommendations. These safeguards include:

- **first-round quality overrides**
- **confidence-aware escalation** toward more conservative recommendations
- **optional forced-conservative operation**

Returned outputs include the predicted stopping class, recommended number of polishing rounds, class probabilities, confidence score, rationale, warning flags, and metadata on applied overrides. 

### Feature groups

ESDP uses engineered features designed to capture both absolute assembly quality and polishing dynamics. These include:

- **base assembly and polishing metrics**
- **round-to-round delta features**
- **first-round normalized metrics**
- **ratio-based descriptors**
- **cumulative improvement variables**
- **plateau indicators**
- **domain-specific summary features**

To reduce training-serving skew, preprocessing is embedded within the serialized model pipeline. Infinite values are converted to missing values, missing features are preserved until inference-time preprocessing, and imputation and scaling are applied consistently during both training and deployment. 

### Training and evaluation

The current benchmark dataset contains **805 polishing records** derived from **41 bacterial samples**, spanning **9 genera**, **5 polishing rounds**, and **4 coverage groups** (**10X, 20X, 40X, and FULL**). Evaluation was performed with **sample-level separation** to prevent leakage between coverage-derived observations from the same isolate, resulting in **32 training samples** and **9 held-out test samples**. 

On the held-out test set, the final **Random Forest** model achieved:

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.629 |
| **Balanced Accuracy** | **0.592** |
| **Macro-F1** | 0.568 |
| **MAE** | 0.482 |
| **Accuracy ±1 class** | 0.888 |
| **QWK** | 0.561 |

Most residual errors occurred between adjacent stopping categories rather than as extreme misclassifications. Class-wise recall for the final Random Forest model was **0.822** for class 1, **0.286** for class 2, and **0.667** for class 3.

### Baseline comparison

To assess practical value, ESDP was compared against three baseline decision strategies on the held-out test set:

- **Always Late** (fixed five-round polishing)
- **QV-threshold rule**
- **R1-only Random Forest**

| Strategy | Balanced Accuracy | Macro-F1 | MAE | QWK |
|----------|------------------:|---------:|----:|----:|
| **ESDP** | **0.592** | 0.568 | 0.482 | **0.561** |
| Always Late | 0.333 | 0.231 | 0.735 | 0.000 |
| QV Threshold | 0.333 | 0.231 | 0.735 | 0.000 |
| R1-only RF | 0.566 | **0.571** | **0.465** | 0.538 |

These comparisons show that ESDP clearly outperforms naive fixed or heuristic strategies, while also indicating that first-round signals already contain substantial predictive information. The full ESDP framework retained the best overall balanced accuracy and quadratic weighted kappa.

---

## Tools & Dependencies

ESDP builds on the following core tools and libraries:

- [Flye](https://github.com/mikolmogorov/Flye) - long-read assembler for bacterial genome reconstruction
- [Racon](https://github.com/isovic/racon) - consensus polishing for long-read assemblies
- [Medaka](https://github.com/nanoporetech/medaka) - neural network-based polishing for Oxford Nanopore data
- [QUAST](https://github.com/ablab/quast) - assembly quality assessment
- [BUSCO](https://busco.ezlab.org/) - genome completeness assessment
- [scikit-learn](https://scikit-learn.org/) - machine learning framework
- [XGBoost](https://xgboost.readthedocs.io/) - gradient boosting models evaluated during development
- [imbalanced-learn](https://imbalanced-learn.org/) - resampling utilities used during model development
- [FastAPI](https://fastapi.tiangolo.com/) - REST API layer
- [Docker](https://www.docker.com/) - containerized deployment

---
## Citation

If you use ESDP in your research, please cite the archived Zenodo release:

```bibtex
@software{lucas_esdp_2026,
  author       = {Jimmy Lucas and Roger de Pedro},
  title        = {ESDP: Early Stop Decision Polishing},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.0.2},
  doi          = {10.5281/zenodo.18910597},
  url          = {https://doi.org/10.5281/zenodo.18910597}
}
```
---

## Acknowledgments

We thank the bioinformatics community for the development and maintenance of the open-source software that
supports this project, including Flye, Racon, Medaka, QUAST, BUSCO, scikit-learn, XGBoost, FastAPI, and Docker.
We also acknowledge the NCBI Sequence Read Archive for providing public access to the sequencing datasets used to
construct the benchmark dataset.

---

## Contact

### Maintainer

- GitHub: [@jimmlucas](https://github.com/jimmlucas)

### Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing/discussions)
- **Questions**: [GitHub Discussions Q&A](https://github.com/jimmlucas/ESDP-Early-Stop-Decision-Polishing/discussions/categories/q-a)

### Community

- **GitHub**: [@AMRmicrobiology](https://github.com/AMRmicrobiology)

---

## License

ESDP is released under the MIT License. See the LICENSE file for details.

---

## Limitations

Current limitations of ESDP include:

1. The benchmark dataset comprises 805 polishing records derived from 41 bacterial samples, so the effective biological diversity represented during development remains limited.
2. The current evaluation was performed under a specific Oxford Nanopore + Flye polishing configuration, and performance may differ across other assemblers, polishers, sequencing conditions, or species distributions.
3. The intermediate stopping category remains the most difficult to resolve, indicating that transitional polishing states are less separable than clear early or late stopping scenarios.
4. ESDP requires the availability of the expected assembly-quality metrics and feature inputs used by the trained pipeline.

---

## Future Improvements

Planned directions for future development include:

1. Broader external validation across additional bacterial genera, coverage regimes, and polishing workflows.
2. Recalibration of confidence thresholds for different operational settings.
3. Incorporation of richer early-round features to improve transitional-state detection.
4. Exploration of alternative decision architectures for more robust medium-class recommendations.
5. Prospective evaluation in routine deployment settings.

---

## References

1. **Flye**  
   Kolmogorov M, Yuan J, Lin Y, Pevzner PA. Assembly of long, error-prone reads using repeat graphs. *Nat Biotechnol.* 2019;37(5):540-546.

2. **Racon**  
   Vaser R, Sović I, Nagarajan N, Šikić M. Fast and accurate de novo genome assembly from long uncorrected reads. *Genome Res.* 2017;27(5):737-746.

3. **Medaka**  
   Oxford Nanopore Technologies. Medaka. Available from: https://github.com/nanoporetech/medaka

4. **QUAST**  
   Gurevich A, Saveliev V, Vyahhi N, Tesler G. QUAST: quality assessment tool for genome assemblies. *Bioinformatics.* 2013;29(8):1072-1075.

5. **BUSCO**  
   Simão FA, Waterhouse RM, Ioannidis P, Kriventseva EV, Zdobnov EM. BUSCO: assessing genome assembly and annotation completeness with single-copy orthologs. *Bioinformatics.* 2015;31(19):3210-3212.

6. **XGBoost**  
   Chen T, Guestrin C. XGBoost: a scalable tree boosting system. In: *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*; 2016. p. 785-794.

7. **scikit-learn**  
   Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. *J Mach Learn Res.* 2011;12:2825-2830.

8. **SMOTE**  
   Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: synthetic minority over-sampling technique. *J Artif Intell Res.* 2002;16:321-357.

9. **Ordinal classification**  
   Frank E, Hall M. A simple approach to ordinal classification. In: *European Conference on Machine Learning*; 2001. p. 145-156.
