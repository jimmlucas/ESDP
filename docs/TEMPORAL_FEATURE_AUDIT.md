# Temporal feature audit

## Objective

ESDP decisions at polishing round `r` must be based only on information
available at or before `r`. This audit classifies every training feature by its
temporal dependency and records the correction made for ESDP v2.

## Predictor audit

| Feature group | Temporal dependency | Status |
|---|---|---|
| Base assembly metrics | Current round only | Prospective |
| Round deltas | Current and immediately previous rounds | Prospective |
| Ratio features | Current round and prospective deltas | Prospective |
| Cumulative improvements | Rounds up to the current round | Prospective |
| R1-normalized features | R1 and current round | Prospective |
| Trend features | Current and previous deltas | Prospective |
| Plateau features, before this change | Maximum gain from the full trajectory | Future leakage |
| Plateau features, after this change | Running maximum gain through the current round | Prospective |
| Domain-specific features | Current prospective features | Prospective |
| `r1_ok_group` | R1 measurements only | Prospective |
| Flye and alignment features | Current assembly/run measurements | Prospective |

## Labels and non-predictor fields

The optimal-round labels intentionally use the complete trajectory because
they represent the retrospective outcome that the model is trained to predict.
This is valid only for target construction.

`stable_all_group` also evaluates the complete trajectory. It is retained as a
labeling and reporting field but is not included in the predictor feature list.
It must not be introduced into a prospective model.

## Plateau correction

The previous implementation defined the plateau threshold as a fraction of the
maximum `score_improvement` across all five rounds:

```text
threshold(r) = relative_threshold * max(score_improvement[1..5])
```

Consequently, a large gain in a later round could change `is_plateau` and
`plateau_streak` for earlier rounds.

The v2 implementation uses the maximum positive gain observed so far:

```text
threshold(r) =
    relative_threshold * max(0, score_improvement[1..r])
```

This makes the feature causal and reproducible from a truncated trajectory.

## Impact on the current dataset

An in-memory comparison against `data/training_dataset_engineered.csv` found:

- 805 records compared.
- 203 records (25.2%) changed in `is_plateau` or `plateau_streak`.
- Changes by round: 160 in R1, 24 in R2, 15 in R3, and 4 in R4.
- Total positive `is_plateau` flags changed from 290 to 88.

The concentration of changes in R1 is expected. At R1 there is no observed
post-R1 improvement history, so a plateau cannot be established using later
gains.

These results show that the correction is scientifically material. Existing
v1 datasets, model artifacts, and manuscript benchmark outputs remain
unchanged. A candidate v2 model must be trained and evaluated separately.

## Regression protection

The temporal-invariance tests rebuild features from:

1. a trajectory truncated at decision round `r`; and
2. the same trajectory containing all later rounds.

For every cutoff from R1 through R5, all rows up to the cutoff must be
identical. A targeted regression test also verifies that a large R5 gain cannot
alter plateau states from R1 through R4.
