# Future-material-benefit endpoint study

## Scientific status

This is a development-only endpoint study. The frozen nine-sample test set
remained sealed, no new predictor metrics were introduced, and no threshold
was selected for deployment. The study asks whether ESDP-light should predict
the absence of meaningful future polishing benefit instead of reproducing the
legacy plateau-derived optimal-round label.

## Endpoint definition

At each observed round, every later round is compared with the current
assembly using four directional quality metrics and the pre-existing stability
tolerances:

| Metric | Better direction | Material tolerance |
|---|---:|---:|
| QV | higher | 0.05 |
| BUSCO complete | higher | 1.0 percentage point |
| Error rate | lower | 0.0005 |
| Assembly error | lower | 0.01 |

A future material benefit exists when at least one later round improves any
metric by more than its tolerance. The material optimal round is the first
round after which no such improvement exists. Once reached, STOP eligibility
is treated as absorbing:

```text
MATERIAL_STOP_ELIGIBLE = current_round >= material_optimal_round
```

This formulation is deliberately conservative about missed improvements: it
does not cancel a gain in one metric with deterioration in another through an
arbitrary weighted score. Safety and QV/BUSCO loss remain separately reported.

A stricter Pareto formulation was audited before fitting and rejected. It
created non-monotonic decisions because small cross-metric trade-offs made many
rounds incomparable.

## Endpoint composition

The primary endpoint assigns the 127 development trajectories as follows:

| Material optimal round | R1 | R2 | R3 | R4 | R5 |
|---|---:|---:|---:|---:|---:|
| Number of trajectories | 88 | 17 | 10 | 7 | 5 |

This differs profoundly from the legacy distribution of 3/20/22/11/71.
Exact agreement is 3.9%, and the mean absolute difference is 2.78 rounds.
The result shows that the legacy plateau label and material future benefit are
different scientific objectives.

The endpoint also exposes a major limitation in the current measurements:

- no development trajectory improves QV by more than 0.05 after any observed
  decision round;
- none improves error rate by more than 0.0005;
- one R1 trajectory has a future assembly-error improvement above 0.01;
- BUSCO is therefore responsible for almost all positive future-benefit
  labels.

Halving the tolerances changes the R1-R5 distribution to 62/27/9/13/16;
doubling them changes it to 109/8/5/3/2. The endpoint is scientifically
sensitive to the definition of materiality and cannot yet be considered a
universally validated biological endpoint.

## Grouped out-of-fold results

The experiment reuses the frozen five sample-grouped folds and the direct
binary model configuration. The primary simulated stopping threshold remains
0.70.

| Candidate | Balanced accuracy | Macro F1 | Unsafe stop | Quality failure | Rounds saved |
|---|---:|---:|---:|---:|---:|
| Light core | 0.502 | 0.460 | 14.2% | 12.6% | 3.024 |
| Light FASTA | 0.369 | 0.337 | 17.3% | 13.4% | 2.874 |
| Causal quality | 0.329 | 0.329 | 18.9% | 16.5% | 3.110 |
| Legacy rebuilt | 0.332 | 0.328 | 16.5% | 18.9% | 3.157 |
| Always R1 | 0.333 | 0.302 | 30.7% | 20.5% | 4.000 |
| Always R5 | 0.333 | 0.025 | 0.0% | 0.0% | 0.000 |

Light core is the only candidate showing meaningful discrimination. Its
sample-bootstrap 95% interval for balanced accuracy is 0.336-0.664, however,
and its unsafe-stop interval is 7.8-20.3%. Only five trajectories belong to
the late class, so class-level estimates remain unstable.

Per-round light-core ROC AUC increases from 0.685 at R1 to 0.761 at R4. This is
evidence that low-cost Flye and assembly features contain cross-sample signal
about future material BUSCO improvement. It is not evidence of adequate safety
for autonomous stopping.

The leading light-core importances are raw sequencing yield, initial overlap
divergence, edge/overlap coverage, N50, and assembly coverage. This supports a
causal low-cost signal; it does not imply that individual impurity-based
importance values have a biological causal interpretation.

## Safety-savings frontier

For light core:

| STOP threshold | Unsafe stop | Quality failure | Rounds saved |
|---:|---:|---:|---:|
| 0.60 | 19.7% | 15.0% | 3.299 |
| 0.70 | 14.2% | 12.6% | 3.024 |
| 0.80 | 9.4% | 13.4% | 2.496 |
| 0.90 | 7.1% | 9.4% | 1.669 |

No explored threshold provides both low premature-stop risk and a validated
quality guarantee. Compared under the new endpoint, retraining light core on
material-benefit labels saves 2.73 more rounds than the previous-label model,
but increases unsafe stopping by 11.0 percentage points and quality failure by
7.9 points. These paired differences demonstrate activity, not acceptable
clinical or production utility.

## Decision

- **Go** for the future-material-benefit concept as a clearer scientific
  objective than reproducing the historical plateau label.
- **No-go** for treating the current four-metric definition as the final
  endpoint because it is dominated by BUSCO and sensitive to tolerances.
- **No-go** for `light-decide`, held-out testing, or adaptive nf-core/bacass
  integration.
- **Go** for a measurement-validity stage that establishes biologically and
  technically justified minimal important differences, including assay noise,
  before any further model selection.

The next gate is scientific rather than computational: freeze defensible
metric tolerances and an asymmetric maximum acceptable premature-stop rate.
Only then should calibration and nested grouped evaluation resume.

## Reproducibility

Run from the repository root:

```bash
python -m experiments.esdp_light_material_benefit
```

Machine-readable endpoint audits, complete out-of-fold predictions, bootstrap
intervals, threshold sensitivity, and feature importances are stored in
`outputs/esdp_light_material_benefit/`.
