# Direct binary stopping experiment for ESDP-light

## Scientific status

This is a development-only formulation study. The frozen nine-sample test set
remained sealed. Its purpose is to determine whether a direct sequential
target, `CONTINUE` versus `STOP_ELIGIBLE`, is more appropriate than predicting
an early/medium/late final-round class. It does not authorize a production
model or a `light-decide` Nextflow module.

## Predeclared formulation

For each development trajectory and each decision round R1-R4:

```text
STOP_ELIGIBLE = current_round >= optimal_rounds_5class
```

At inference, the simulated policy stops at the first round whose out-of-fold
STOP probability is at least 0.70; otherwise it continues to R5. Thresholds
from 0.30 to 0.90 are exploratory only. The experiment reuses the frozen
sample-grouped folds, Random Forest configuration, and 32 development samples
from the three-class feasibility study.

The target is strongly imbalanced at early rounds: 3/127 trajectories are
eligible at R1, 23/127 at R2, 45/127 at R3, and 56/127 at R4. R5 is the
mandatory fallback and is excluded from model fitting.

## Controls for temporal leakage

Five candidates separate formulation, feature cost, and leakage effects:

- `legacy_historical_leaky_binary` uses the historical feature table before
  correction of retrospective plateau variables. It is an invalid diagnostic
  control and can never be selected for deployment;
- `legacy_rebuilt_binary` uses the same broad legacy feature family after
  rebuilding temporal variables causally;
- `quality_causal_binary` uses costly quality metrics rebuilt causally but
  excludes plateau and policy variables;
- `light_core_binary` uses the approved low-cost core feature contract;
- `light_fasta_binary` uses only the currently operational FASTA family.

## Primary results

| Candidate | Balanced accuracy | Macro F1 | Unsafe stop | Quality failure | Absolute round error | Rounds saved |
|---|---:|---:|---:|---:|---:|---:|
| Historical leaky control | 0.844 | 0.879 | 0.0% | 7.9% | 0.244 | 0.756 |
| Legacy rebuilt causally | 0.540 | 0.539 | 2.4% | 0.8% | 0.717 | 0.346 |
| Causal quality control | 0.433 | 0.430 | 2.4% | 2.4% | 0.882 | 0.213 |
| Light core binary | 0.406 | 0.371 | 7.1% | 4.7% | 0.913 | 0.291 |
| Light FASTA binary | 0.371 | 0.329 | 7.9% | 3.9% | 0.961 | 0.276 |

For light core, the sample-bootstrap 95% intervals were 0.361-0.455 for
balanced accuracy, 3.1-12.5% for unsafe stopping, and 0.133-0.469 rounds for
mean savings. These intervals do not support release-level performance.

## What the experiment demonstrates

The direct binary formulation is a real improvement over three-class
prediction for the two light feature families. Light core reduced mean
absolute round error by 0.157 rounds (paired sample-bootstrap 95% CI
0.039-0.281), and FASTA reduced it by 0.197 rounds (95% CI 0.031-0.391).
Neither comparison showed a resolved improvement in savings, unsafe stopping,
or quality failure. The architectural direction is therefore better, but the
available causal signal remains insufficient.

The leakage control is equally important. Balanced accuracy falls from 0.844
with historical variables to 0.540 when the same legacy family is rebuilt
causally. Historical importance is led by `plateau_streak` and `is_plateau`,
which were derived with retrospective information. The high historical score
must not be interpreted as prospective scientific utility.

Costly causal quality metrics reach only 0.433 balanced accuracy, compared
with 0.406 for light core. The small difference does not justify adding their
runtime cost. Causally rebuilt plateau features improve the broad legacy
control to 0.540, suggesting genuine trajectory information, but still remain
below the existing 0.65 balanced-accuracy release gate and have very low
minimum class recall.

## Decision

- **Go** for the direct binary decision architecture as the next research
  formulation.
- **No-go** for training a release artifact, exposing `light-decide`, or
  integrating adaptive stopping into nf-core/bacass.
- **No-go** for selecting the 0.70 threshold from this development result or
  opening the held-out test set.
- **Go** for one bounded endpoint study: define a prospective stopping utility
  based on the material future benefit of additional polishing, then evaluate
  it with nested sample-grouped validation before adding new metrics.

The next study should freeze the endpoint, asymmetric cost of premature versus
late stopping, calibration method, and release gates before fitting models.
Only after those choices pass internal grouped validation should the single
held-out evaluation and Nextflow module be considered.

The resulting bounded endpoint study is documented in
`ESDP_LIGHT_MATERIAL_ENDPOINT.md`.

## Reproducibility

Run from the repository root:

```bash
python -m experiments.esdp_light_binary_decision
```

Machine-readable results, complete out-of-fold predictions, fold assignments,
bootstrap intervals, threshold sensitivity, and feature importances are stored
in `outputs/esdp_light_binary_feasibility/`.
