# ESDP-light development-only feasibility experiment

## Scientific status

This is a development-only feasibility study. It does not constitute an
independent validation or a released ESDP-light model. The frozen nine-sample
test set was not used for feature selection, model fitting, threshold tuning,
candidate comparison, or reporting.

The experiment answers one bounded question: do the currently deployable
low-cost features retain enough cross-sample signal to justify building an
adaptive Nextflow decision module?

## Protocol

The study uses the 32 frozen development samples:

- 635 round observations;
- 127 `Sample+Coverage_effective` trajectories;
- complete R1-R5 histories;
- class distribution of 23 early, 33 medium, and 71 late trajectories.

All reported model predictions are out of fold. Five-fold
`StratifiedGroupKFold` keeps every biological sample entirely inside one
validation fold. The same folds, Random Forest configuration, class weights,
and random seeds are used for every feature family. No hyperparameters are
selected from outer-fold performance.

The primary sequential policy uses a predeclared confidence threshold of
0.60. At each observed round, an adaptive candidate may stop only when its
predicted class is eligible at that round and its confidence reaches the
threshold. Otherwise polishing continues, and R5 always terminates the
trajectory.

Four model candidates are compared:

- `legacy_full_retrospective`: an explicitly optimistic benchmark containing
  expensive and leakage-prone legacy features; it cannot become a light model;
- `light_core_prospective`: 23 deployment-approved raw metrics and 12 causally
  rebuilt temporal features;
- `light_fasta_prospective`: the four metrics currently produced by
  `light-observe` and their 12 causal temporal features;
- `r1_core_only`: the 23 approved raw metrics restricted to R1, producing a
  fixed planned final round rather than adaptive updates.

`always_r1` and `always_r5` are operational controls. The exploratory
threshold sweep from 0.40 to 0.90 describes a safety-savings frontier; it is
not a threshold-selection procedure.

## Primary results

| Candidate | Balanced accuracy | Unsafe early stop | Quality failure | Mean rounds saved | Compute reduction |
|---|---:|---:|---:|---:|---:|
| Always R5 | 0.333 | 0.0% | 0.0% | 0.000 | 0.0% |
| Legacy retrospective | 0.334 | 3.9% | 0.0% | 0.181 | 3.6% |
| Light core prospective | 0.345 | 7.9% | 3.1% | 0.291 | 5.8% |
| Light FASTA prospective | 0.330 | 10.2% | 3.1% | 0.425 | 8.5% |
| R1 core only | 0.329 | 2.4% | 0.0% | 0.094 | 1.9% |
| Always R1 | 0.333 | 97.6% | 20.5% | 4.000 | 80.0% |

For light core, sample-level bootstrap intervals were:

- balanced accuracy: 0.345, 95% CI 0.314-0.386;
- unsafe early-stop rate: 7.9%, 95% CI 3.1-13.3%;
- mean rounds saved: 0.291, 95% CI 0.078-0.531;
- absolute round error: 1.071, 95% CI 0.810-1.326.

The balanced-accuracy interval includes the 1/3 constant-class reference. The
candidate also falls below the repository's pre-existing targets of 0.65
balanced accuracy, 0.60 macro F1, 0.40 minimum class recall, and 0.50 quadratic
weighted kappa.

## Signal before the stopping policy

Out-of-fold per-round classification separates model discrimination from the
confidence policy. The best balanced accuracy observed for each family was:

- legacy retrospective: 0.551 at R2;
- light core prospective: 0.388 at R4/R5;
- light FASTA prospective: 0.425 at R3;
- R1 core only: 0.362 at R1.

Therefore, low-cost features contain weak cross-sample signal, but not enough
to support the current three-class decision objective reliably. The poor
operational result is not explained only by the 0.60 confidence threshold.

## Safety-savings frontier

For light core:

| Confidence threshold | Unsafe stop | Quality failure | Mean rounds saved |
|---:|---:|---:|---:|
| 0.40 | 23.6% | 3.9% | 0.969 |
| 0.50 | 18.1% | 3.9% | 0.772 |
| 0.60 | 7.9% | 3.1% | 0.291 |
| 0.70 | 1.6% | 0.0% | 0.063 |
| 0.80 | 0.0% | 0.0% | 0.000 |

At safer thresholds the compute benefit collapses. At thresholds that produce
material savings, premature-stop risk becomes too large. No threshold on this
development curve is approved for deployment.

Compared with R1 core only, light core saved 0.197 additional rounds per
trajectory, with a sample-bootstrap 95% CI of 0.063-0.360. It also increased
unsafe stopping by 5.5 percentage points, with a 95% CI of 1.6-10.2 points.
The added temporal behavior therefore produces measurable activity, but the
current gain is coupled to measurable risk.

## Endpoint audit

The exact optimal-round label and the partial QV/BUSCO resource endpoint are
not interchangeable:

- 79.5% of all development trajectories met the QV/BUSCO loss limits at R1;
- 83.1% of trajectories labeled as requiring R5 also met those two limits at
  R1.

This does not invalidate the labels. Their construction also uses error rate,
assembly fraction/error, R1 vetoes, stability, and plateau behavior. It does
show that QV and BUSCO loss alone cannot validate decision safety and that the
endpoint must be audited before changing model families or adding metrics.

## Interpretation and decision

The experiment demonstrates engineering validity and a modest biological
signal, but it does not demonstrate sufficient decision utility for an
ESDP-light release. The scientifically appropriate decision is:

- **no-go** for training a release artifact or implementing a production
  `light-decide` Nextflow module now;
- **go** for a bounded endpoint and formulation study inside the 32 development
  samples;
- keep the nine test samples sealed until the feature contract, formulation,
  calibration method, and stopping policy are frozen.

The next study should first audit which label components drive late decisions
and whether the three-class target is the correct objective for a sequential
policy. Only then should it compare ordinal, cost-sensitive, or direct
continue/stop formulations through nested grouped validation. Additional
metrics, including post-polish alignment error, should be admitted only when a
predeclared ablation demonstrates that they improve the safety-savings
frontier.

## Reproducibility

Run from the repository root:

```bash
python -m experiments.esdp_light_feasibility
```

Machine-readable outputs are stored in
`outputs/esdp_light_feasibility/`, including:

- complete protocol, checksums, bootstrap intervals, and comparisons;
- out-of-fold trajectory and round predictions;
- sample-fold assignments;
- per-fold and per-round metrics;
- feature importances;
- exploratory threshold sensitivity.
