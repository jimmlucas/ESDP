---
title: "ESDP"
output:
  github_document:
    toc: false
---

# ESDP

### Early Stop Decision Polishing for resource-efficient long-read bacterial genome polishing

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Docker Hub](https://img.shields.io/badge/docker-jimmlucas%2Fesdp-blue)](https://hub.docker.com/r/jimmlucas/esdp)
[![Version](https://img.shields.io/badge/version-v2.0.0--rc1-orange.svg)](https://hub.docker.com/r/jimmlucas/esdp/tags)
[![Status](https://img.shields.io/badge/status-release%20candidate-orange.svg)](#release-status)

> **Replace a fixed Racon polishing budget with a sample-adaptive stopping policy.**

---

## Release status

This branch contains the ESDP v2 sequential controller used for the revised software release.

Current container release candidate:

```text
jimmlucas/esdp:v2.0.0-rc1
```

The final `v2.0.0` GitHub release, Docker tag, and Zenodo DOI will be created after the repository-wide v2 audit and full test suite are completed.

---

## Table of contents

- [Overview](#overview)
- [What changed in ESDP v2](#what-changed-in-esdp-v2)
- [Sequential decision formulation](#sequential-decision-formulation)
- [Operational feature contract](#operational-feature-contract)
- [Model and preprocessing](#model-and-preprocessing)
- [Decision policy](#decision-policy)
- [Installation](#installation)
- [Docker](#docker)
- [REST API](#rest-api)
- [Python / CLI use](#python--cli-use)
- [Development and evaluation data](#development-and-evaluation-data)
- [Performance summary](#performance-summary)
- [Post-Medaka evaluation](#post-medaka-evaluation)
- [Resource benchmark](#resource-benchmark)
- [Reproducibility and frozen artifacts](#reproducibility-and-frozen-artifacts)
- [Repository structure](#repository-structure)
- [Scientific scope](#scientific-scope)
- [Limitations](#limitations)
- [Citation](#citation)
- [Availability](#availability)
- [License](#license)
- [Contact](#contact)

---

## Overview

Oxford Nanopore sequencing can produce highly contiguous bacterial genome assemblies, but polishing workflows are frequently executed with a fixed number of Racon iterations. A fixed budget is simple to automate, yet it can continue computation after additional polishing has little material value for a given assembly.

**ESDP** is a sequential decision-support framework for adapting the number of Racon polishing rounds to the observed state of each assembly.

Instead of predicting a single final round before polishing begins, ESDP v2 re-evaluates the assembly **after each Racon iteration** and asks:

> **Is another Racon round likely to provide sufficient material benefit to justify continuing?**

At each decision point, ESDP returns either:

- `STOP` → proceed directly to **Medaka**, or
- `CONTINUE` → execute the next Racon iteration.

The maximum Racon budget remains five rounds. **Medaka is mandatory after the selected stopping point**, including after R5.

The deployed controller uses only operational features available at the current polishing stage. It does **not** require BUSCO, a reference genome, mapping-derived QV/error metrics, genus, or future polishing states at inference time.

---

## What changed in ESDP v2

ESDP v2 replaces the earlier three-class deployment formulation with a **stage-aware sequential STOP/CONTINUE controller**.

The previous descriptive labels:

- Early: R1-R2
- Medium: R3-R4
- Late: R5

may still be used for retrospective reporting, but they are **not deployment commands** in v2.

The deployed architecture is:

```text
Flye assembly
    |
    v
Racon R1
    |
    v
Decision at R1 ---- STOP ----> Medaka
    |
 CONTINUE
    v
Racon R2
    |
    v
Decision at R2 ---- STOP ----> Medaka
    |
 CONTINUE
    v
Racon R3
    |
    v
Decision at R3 ---- STOP ----> Medaka
    |
 CONTINUE
    v
Racon R4
    |
    v
Decision at R4 ---- STOP ----> Medaka
    |
 CONTINUE
    v
Racon R5
    |
    v
Medaka
```

This converts a fixed five-round Racon budget into a **sample-adaptive sequential budget**.

---

## Sequential decision formulation

For trajectory `i` at current Racon round `k`, the retrospective development target is:

\[
Y_{i,k} =
\mathbf{1}
\left[
\max_{j>k} B_{i,j} - B_{i,k} \geq 2
\right]
\]

where:

- `B` is the number of complete BUSCO markers,
- `k` is the current round (`R1`-`R4`),
- `Y = 1` means `CONTINUE`,
- `Y = 0` means `STOP`,
- the material-improvement threshold is fixed at **2 complete BUSCO markers**.

This target is a **retrospective material-improvement oracle used for supervised model development**. It should not be interpreted as a universal biological optimum.

The use of `max` over later rounds accommodates non-monotonic polishing trajectories during target construction.

### Important distinction

BUSCO is used to define the retrospective development outcome and to evaluate quality.

**BUSCO is not an operational predictor of the deployed ESDP v2 controller.**

Consequently, operational inference does not require:

- BUSCO execution,
- a reference genome,
- read-to-reference quality assessment,
- genus labels,
- access to future polishing states.

---

## Operational feature contract

The frozen ESDP v2 controller uses exactly **10 predictors**:

| Feature | Description |
|---|---|
| `round` | Current Racon round, restricted to R1-R4 |
| `coverage_est` | Estimated sequencing coverage |
| `expected_genome_size` | Expected genome size |
| `raw_read_n50` | N50 of the raw long reads |
| `ai_cov_cv` | Coefficient of variation of assembly-info coverage |
| `current_n50` | N50 of the current polished assembly |
| `current_num_contigs` | Number of contigs in the current assembly |
| `current_assembly_frac` | Current assembly length / expected genome size |
| `delta_n50_last` | Current N50 minus previous-round N50 |
| `n50_from_R1` | Current N50 minus R1 N50 |

Feature definitions:

\[
ai\_cov\_cv = \frac{\sigma_{\mathrm{coverage}}}{\mu_{\mathrm{coverage}}}
\]

\[
current\_assembly\_frac_k =
\frac{\mathrm{assembly\ length}_k}
{\mathrm{expected\ genome\ size}}
\]

\[
delta\_n50\_last_k =
N50_k - N50_{k-1}
\]

\[
n50\_from\_R1_k =
N50_k - N50_{R1}
\]

At R1, unavailable history-derived values are represented as missing values and handled by the preprocessing pipeline.

### Frozen feature schema

```text
SHA256:
2f50cc9c6169c325da21e189b309623d872cbfdf4a025588c786c6e9f4576ced
```

The operational controller does **not** use:

- `busco_complete`
- QV
- mapping-derived error rate
- genus
- Sample ID
- future-round values
- plateau labels
- retrospective oracle variables

---

## Model and preprocessing

The frozen v2 model is a pooled, stage-aware `RandomForestClassifier`.

Model configuration:

```text
n_estimators = 500
max_depth = 3
min_samples_leaf = 8
max_features = "sqrt"
class_weight = "balanced"
random_state = 42
n_jobs = -1
```

Preprocessing is embedded in the serialized pipeline:

1. non-finite values are converted to missing values,
2. median imputation is applied,
3. missing-value indicators are added,
4. no feature scaling is applied.

The production environment is pinned to:

```text
Python          3.10
scikit-learn    1.7.2
joblib          1.5.3
pandas          2.3.3
numpy           2.2.6
```

---

## Decision policy

At each legal decision stage (`R1`-`R4`), the model returns a `p_continue` score.

The frozen policy is:

```text
if p_continue >= 0.45:
    CONTINUE
else:
    STOP
```

The threshold is therefore **inclusive**:

```text
p_continue >= 0.45  -> CONTINUE
p_continue <  0.45  -> STOP
```

Possible actions are:

| Current round | Decision | `next_action` |
|---|---|---|
| R1 | STOP | `MEDAKA` |
| R1 | CONTINUE | `RACON_R2` |
| R2 | STOP | `MEDAKA` |
| R2 | CONTINUE | `RACON_R3` |
| R3 | STOP | `MEDAKA` |
| R3 | CONTINUE | `RACON_R4` |
| R4 | STOP | `MEDAKA` |
| R4 | CONTINUE | `RACON_R5_THEN_MEDAKA` |

R5 is the maximum Racon round and is therefore **not** a decision state.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jimmlucas/ESDP.git
cd ESDP
```

For the v2 development branch:

```bash
git checkout ESDP-V2.0
```

Create a Python 3.10 environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Docker

### Release candidate

The current public v2 release candidate is:

```bash
docker pull jimmlucas/esdp:v2.0.0-rc1
```

Run the API:

```bash
docker run --rm -p 8000:8000 jimmlucas/esdp:v2.0.0-rc1
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Model metadata:

```bash
curl http://127.0.0.1:8000/model/info
```

The frozen model must report:

```text
model_version:
esdp-sequential-rf-v2

decision_threshold:
0.45

feature_count:
10

model_sha256:
9f9f08428242e98381546b9c79cd3d9b013c412b3edcbfe01ca84bb6f4c10dcf

feature_schema_sha256:
2f50cc9c6169c325da21e189b309623d872cbfdf4a025588c786c6e9f4576ced
```

### Docker Compose

```bash
docker compose up --build -d
docker compose ps
```

The service should reach:

```text
healthy
```

Stop it with:

```bash
docker compose down
```

### Final release

After the repository-wide v2 audit is complete, the release candidate will be replaced by the immutable final tag:

```text
jimmlucas/esdp:v2.0.0
```

---

## REST API

ESDP v2 exposes:

```text
GET  /
GET  /health
GET  /model/info
POST /predict
```

### Prediction example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sample_id": "example_sample",
    "round": 1,
    "coverage_est": 40,
    "expected_genome_size": 5000000,
    "raw_read_n50": 15000,
    "ai_cov_cv": 0.20,
    "current_n50": 4800000,
    "current_num_contigs": 2,
    "current_assembly_frac": 0.99,
    "delta_n50_last": null,
    "n50_from_R1": null
  }'
```

Reference output for this frozen test case:

```json
{
  "sample_id": "example_sample",
  "round": 1,
  "p_continue": 0.5939402938271254,
  "threshold": 0.45,
  "decision": "CONTINUE",
  "next_action": "RACON_R2",
  "model_version": "esdp-sequential-rf-v2",
  "model_sha256": "9f9f08428242e98381546b9c79cd3d9b013c412b3edcbfe01ca84bb6f4c10dcf",
  "feature_schema_sha256": "2f50cc9c6169c325da21e189b309623d872cbfdf4a025588c786c6e9f4576ced",
  "missing_features": [
    "delta_n50_last",
    "n50_from_R1"
  ]
}
```

The API rejects legacy deployment inputs such as:

```text
busco_complete
qv
genus
force_conservative
confidence_threshold
```

---

## Python / CLI use

The v2 decision logic is implemented in:

```text
esdp_decide.py
```

The model artifact is loaded relative to the repository by default from:

```text
outputs/frozen_sequential_model_v2/esdp_sequential_rf_v2.joblib
```

A direct smoke test can be run with:

```bash
python esdp_decide.py
```

The operational command-line interface is being aligned with this sequential contract for the final v2.0.0 release. Until that repository-wide audit is complete, the REST API and `esdp_decide.py` are the canonical v2 deployment interfaces.

---

## Development and evaluation data

The complete historical dataset contains:

- **41 biological samples**
- **161 Sample x Coverage trajectories**
- **805 Racon round-level observations**
- **9 bacterial genera**
- Racon rounds **R1-R5**
- coverage regimes including **10x, 20x, and 40x**

The v2 development/evaluation split is performed at the **biological-sample level**.

### Development set

- **32 biological samples**
- **127 trajectories**
- **508 sequential decision states**
- decision states from **R1-R4**

These states were used for model development under grouped sample-level resampling.

### Held-out evaluation set

- **9 biological samples**
- **34 trajectories**
- not used for:
  - target definition,
  - feature selection,
  - model fitting,
  - hyperparameter selection,
  - decision-threshold selection.

This set is described as a **held-out evaluation set**, not as an external validation cohort.

---

## Performance summary

### Development out-of-fold model discrimination

For the frozen 10-feature Random Forest:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.698 |
| Average Precision | 0.438 |
| Brier score | 0.208 |
| Balanced Accuracy | 0.659 |

Stage-specific ROC-AUC:

| Stage | ROC-AUC |
|---|---:|
| R1 | 0.722 |
| R2 | 0.692 |
| R3 | 0.707 |
| R4 | 0.627 |

### Nested sample-level policy evaluation

Nested threshold selection was used to estimate development policy performance without reusing the same held-out folds for threshold tuning and policy evaluation.

Observed results:

| Metric | Value |
|---|---:|
| Unsafe trajectories | 8 / 127 (6.30%) |
| Wilson 95% upper bound | 11.94% |
| Mean Racon rounds saved | 1.835 |
| Mean stopping round | 3.165 |
| Mean regret | 0.252 |

Observed stopping rounds:

```text
R1: 54
R2:  0
R3:  6
R4:  5
R5: 62
```

### Frozen development threshold

After development-only threshold selection, the frozen deployment threshold was set to:

```text
p_continue >= 0.45 -> CONTINUE
```

This threshold was selected on development data and was **not re-tuned on the held-out evaluation set**.

### Held-out evaluation

State-level predictive performance:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.665 |
| Average Precision | 0.264 |
| Brier score | 0.234 |

Sequential policy performance across 34 trajectories:

| Metric | Value |
|---|---:|
| Unsafe trajectories | 3 / 34 (8.82%) |
| Wilson 95% CI | 3.05%-22.96% |
| Mean Racon rounds saved | 1.647 |
| Mean stopping round | 3.353 |
| Mean regret | 0.265 |

At the biological-sample level, unsafe decisions occurred in 2 of 9 held-out samples. The small held-out sample size should be considered when interpreting these estimates.

---

## Post-Medaka evaluation

Because Medaka is mandatory after the ESDP-selected Racon stopping point, final quality was evaluated by comparing:

```text
ESDP-selected Racon stop + Medaka
```

against:

```text
fixed R5 + Medaka
```

Two samples were excluded because the available metadata were insufficient for a defensible Medaka chemistry/basecaller compatibility assignment.

The resulting comparison included:

- **7 biological samples**
- **26 unique trajectories**
- **14 unique early-stop trajectories**

Among the 14 unique early-stop trajectories:

- 3/14 showed any decrease in complete BUSCO markers,
- 2/14 showed a decrease of at least 2 markers,
- 12/14 showed no material loss under the predefined 2-marker criterion,
- maximum observed loss was 2 complete markers.

A sample-clustered bootstrap yielded:

```text
mean ΔComplete = +0.262 markers
95% CI = [-0.738, +1.357]
```

These results do not support a systematic average loss in the evaluated subset, but the limited sample size and the presence of individual trajectory losses do **not** establish equivalence to fixed R5 polishing.

---

## Resource benchmark

A separate computational benchmark was performed on:

- **8 biological samples**
- **12 trajectories**

This benchmark evaluates computational cost and is not an external validation cohort.

Using the frozen v2 policy:

```text
R1 stops: 4
R2 stops: 0
R3 stops: 1
R4 stops: 1
R5 stops: 6
```

Across the 12 trajectories:

```text
Fixed policy:  60 Racon rounds
ESDP policy:   41 Racon rounds
Avoided:       19 / 60 rounds = 31.67%
```

Core runtime benchmark:

| Metric | Value |
|---|---:|
| Mean trajectory wall-time saving | 31.56% |
| Median trajectory wall-time saving | 9.89% |
| Cost-weighted wall-time saving | 3.95% |
| Total wall seconds saved | 106.12 s |

Controller overhead across 35 reachable decisions:

| Metric | Value |
|---|---:|
| Total controller overhead | 4.303 s |
| Median decision latency | 0.102 s |
| 95th percentile latency | 0.162 s |
| Cold model load | 0.066 s |

After accounting for controller overhead, the net cost-weighted core wall-time saving was approximately **3.79%** in this benchmark.

The proportion of polishing rounds avoided should not be interpreted as equivalent to wall-time reduction because Racon rounds do not have identical runtime cost.

---

## Reproducibility and frozen artifacts

### Frozen model

```text
outputs/frozen_sequential_model_v2/esdp_sequential_rf_v2.joblib
```

SHA256:

```text
9f9f08428242e98381546b9c79cd3d9b013c412b3edcbfe01ca84bb6f4c10dcf
```

### Frozen feature contract

SHA256:

```text
2f50cc9c6169c325da21e189b309623d872cbfdf4a025588c786c6e9f4576ced
```

### Frozen development protocol

SHA256:

```text
74b5c35e572911530da47e46ed3d87c615dfa20fd251fd87f92b4b0fffae82c2
```

### Verification tests

The current v2 API regression suite includes checks for:

- API health,
- model metadata,
- R1 inference,
- exact frozen reference prediction,
- rejection of R5 as a decision state,
- rejection of BUSCO as an inference input,
- rejection of QV as an inference input,
- rejection of genus as an inference input,
- valid R4 continuation behavior.

Run:

```bash
python -m pytest -v tests/test_api_v2.py
```

The frozen reference case is expected to return:

```text
p_continue = 0.5939402938271254
decision   = CONTINUE
next_action = RACON_R2
```

---

## Repository structure

Key v2 deployment files:

```text
ESDP/
├── api_service.py
├── esdp_decide.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── outputs/
│   └── frozen_sequential_model_v2/
│       ├── esdp_sequential_rf_v2.joblib
│       ├── esdp_sequential_rf_v2.metadata.json
│       └── SHA256.txt
└── tests/
    └── test_api_v2.py
```

The repository also contains historical development scripts, datasets, figures, and model artifacts from earlier experimental formulations.

Those historical materials may contain BUSCO, QV, error-rate, multi-class, and other retrospective variables. Their presence does **not** imply that these variables are required by the frozen v2 operational controller.

The canonical production interfaces for v2 are:

```text
esdp_decide.py
api_service.py
outputs/frozen_sequential_model_v2/
```

---

## Scientific scope

ESDP v2 was developed for **Oxford Nanopore bacterial genome polishing** using trajectories generated under the evaluated assembly/polishing workflow.

It should be understood as a decision-support framework for this defined setting rather than as a universal stopping rule for all long-read polishing pipelines.

The target is an operational proxy for material future improvement under the historical trajectories. It is not a direct biological definition of genome perfection.

---

## Limitations

1. The number of independent biological samples remains modest despite the larger number of trajectory-level observations.

2. The held-out evaluation contains 9 biological samples and 34 trajectories; confidence intervals are therefore wide for uncommon policy failures.

3. The evaluated trajectories arise from a defined Oxford Nanopore / Flye / Racon / Medaka context. Performance should not be assumed to transfer unchanged to other assemblers, polishers, sequencing chemistries, organisms, or coverage distributions.

4. The retrospective target is BUSCO-based and uses a fixed material-improvement threshold of 2 complete markers. This definition was frozen for the reported model but is not necessarily optimal for every scientific objective.

5. BUSCO is used for target construction and retrospective quality assessment, but not for operational inference.

6. The final post-Medaka paired analysis is limited in size and does not establish statistical equivalence between early stopping and fixed R5 polishing.

7. The current threshold was selected under a development safety constraint and should not be interpreted as a universally optimal operating point.

---

## Citation

### Current manuscript

The revised manuscript describing ESDP v2 is under peer review.

Suggested final BibTeX template:

```bibtex
@software{lucas_esdp_v2_2026,
  author    = {Lucas, Jimmy and de Pedro-Jové, Roger},
  title     = {ESDP: A Decision Framework for Resource-Efficient Long-Read Bacterial Genome Polishing},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v2.0.0},
  doi       = {ZENODO_V2_DOI},
  url       = {https://doi.org/ZENODO_V2_DOI}
}
```

---

## Availability

**Project name:** ESDP

**Project home page:**  
https://github.com/jimmlucas/ESDP

**Current v2 container release candidate:**  
https://hub.docker.com/r/jimmlucas/esdp  
Tag: `v2.0.0-rc1`

**Final archived version:**  
Zenodo v2.0.0 DOI: pending final release

**Operating system(s):**  
Platform independent

**Programming language:**  
Python 3.10

**Other requirements:**  
Docker 20.10+ for containerized deployment, or the Python dependencies listed in `requirements.txt`

**License:**  
MIT License

**Restrictions for use by non-academics:**  
None

---

## License

ESDP is released under the [MIT License](LICENSE).

There are no additional restrictions on use by non-academic users beyond the terms of the MIT License.

---

## Contact

### Maintainer

- GitHub: [@jimmlucas](https://github.com/jimmlucas)

### Repository support

- Issues: https://github.com/jimmlucas/ESDP/issues
- Repository: https://github.com/jimmlucas/ESDP

---

## Acknowledgments

We acknowledge the developers and maintainers of the open-source software used in the ESDP workflow and analysis, including Flye, Racon, Medaka, BUSCO, scikit-learn, FastAPI, Docker, NumPy, pandas, SciPy, and related scientific Python tools.

We also acknowledge the NCBI Sequence Read Archive for public access to the sequencing datasets used to construct the study trajectories.

---

## References

1. Kolmogorov M, Yuan J, Lin Y, Pevzner PA. Assembly of long, error-prone reads using repeat graphs. *Nature Biotechnology*. 2019;37:540-546.

2. Vaser R, Sović I, Nagarajan N, Šikić M. Fast and accurate de novo genome assembly from long uncorrected reads. *Genome Research*. 2017;27:737-746.

3. Oxford Nanopore Technologies. Medaka. https://github.com/nanoporetech/medaka

4. Simão FA, Waterhouse RM, Ioannidis P, Kriventseva EV, Zdobnov EM. BUSCO: assessing genome assembly and annotation completeness with single-copy orthologs. *Bioinformatics*. 2015;31:3210-3212.

5. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*. 2011;12:2825-2830.
