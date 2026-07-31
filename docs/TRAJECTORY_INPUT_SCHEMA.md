# ESDP v2 trajectory contract

ESDP v2 represents an inference request as the complete polishing history
available at the decision point. The v1 single-row interface remains
unchanged.

The canonical Python models are defined in `esdp_trajectory.py`:

- `RoundMetrics`: raw observations produced after one polishing round.
- `PolishingTrajectory`: one ordered history for a sample and coverage
  condition.
- `DecisionV2`: a workflow-facing decision with an explicit action.

## JSON input

```json
{
  "schema_version": "2.0.0",
  "sample_id": "sample_001",
  "genus": "Escherichia",
  "coverage": 40.0,
  "coverage_effective": 38.5,
  "max_rounds": 5,
  "incomplete_history_policy": "CONSERVATIVE_CONTINUE",
  "rounds": [
    {
      "round": 1,
      "qv": 31.0,
      "busco_complete": 92.0,
      "n50": 1010000,
      "num_contigs": 6,
      "error_rate": 0.009,
      "total_length": 4900000,
      "assembly_frac": 0.96
    },
    {
      "round": 2,
      "qv": 32.0,
      "busco_complete": 93.0,
      "n50": 1020000,
      "num_contigs": 5,
      "error_rate": 0.008,
      "total_length": 4900000,
      "assembly_frac": 0.97
    }
  ]
}
```

The same example is available as `examples/trajectory.v2.json`.

## Structural validation

A trajectory is rejected when:

- its schema version is unsupported;
- rounds are out of order, duplicated, or non-contiguous;
- the history does not start at R1;
- a round or `max_rounds` is outside R1–R5;
- a metric is outside its biological range;
- a boolean or numeric string is supplied in place of a numeric metric;
- `sample_id` is empty or contains only whitespace;
- a round contains no observed metric;
- an unknown field is supplied.

These are structural errors and must not be silently imputed.

## Metric completeness

By default, prediction readiness requires these metrics in every round:

- `qv`
- `busco_complete`
- `n50`
- `num_contigs`
- `error_rate`
- `total_length`

`build_trajectory_features()` returns both the prospective feature table and
the missing-metric map. Its `can_predict` property is true only when all
required metrics are present.

Two policies are supported:

- `ERROR`: raise `IncompleteTrajectoryError`.
- `CONSERVATIVE_CONTINUE`: do not treat missing values as evidence of a
  plateau; request another polishing round when one remains.

The required metric set will later be read from the v2 model manifest. It is
explicit here so the schema can be developed without altering the published
v1 artifact.

## Decision actions

- `STOP`: select the current assembly; `next_round` must be absent.
- `CONTINUE`: run exactly the next sequential round.
- `CONSERVATIVE_CONTINUE`: continue because metrics or model state are not
  sufficient for a safe early stop.

Every decision enforces:

```text
current_round <= recommended_final_round <= max_rounds
```

A continuation must set `next_round = current_round + 1` and cannot exceed
`max_rounds`.

## Feature construction

```python
import json

from esdp_trajectory import PolishingTrajectory, build_trajectory_features

with open("examples/trajectory.v2.json", encoding="utf-8") as input_file:
    trajectory = PolishingTrajectory.model_validate(json.load(input_file))

result = build_trajectory_features(trajectory)
if result.can_predict:
    latest_features = result.latest
else:
    missing = result.missing_metrics
```

The builder computes every row prospectively. Adding a future polishing round
cannot change the feature vector of an earlier round.

## Nextflow mapping

The future Nextflow module will maintain one JSON trajectory per sample:

```text
R1 metrics
  -> append R1 to trajectory
  -> ESDP action
  -> STOP or run R2
  -> append R2 to the same trajectory
  -> ESDP action
```

Nextflow should branch only on the `action` field. On validation, metric
collection, model-loading, or low-confidence failures, the safe workflow
default is `CONSERVATIVE_CONTINUE` until `max_rounds` is reached.

## JSON Schema generation

Pydantic can expose a machine-readable schema without maintaining a duplicate
definition:

```python
from esdp_trajectory import PolishingTrajectory

schema = PolishingTrajectory.model_json_schema()
```

Static schema export and CLI validation will be added with
`feat/production-cli`.
