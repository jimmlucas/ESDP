"""Tests for versioned, prospective ESDP-light history aggregation."""

import json

import pytest

from esdp_light_history import (
    LIGHT_HISTORY_FEATURE_NAMES,
    LightFeatureHistory,
    LightHistoryError,
    build_light_feature_history,
)
from esdp_light_metrics import FastaMetrics, LightRoundObservation


def observation(
    round_number: int,
    *,
    sample_id: str = "sample-1",
) -> LightRoundObservation:
    values = {
        1: (100, 5, 1000, 50.0),
        2: (130, 4, 1010, 50.2),
        3: (150, 4, 1015, 50.1),
    }
    n50, num_contigs, total_length, gc = values[round_number]
    return LightRoundObservation(
        sample_id=sample_id,
        round=round_number,
        assembly_sha256=f"{round_number:064x}",
        fasta=FastaMetrics(
            n50=n50,
            num_contigs=num_contigs,
            total_length=total_length,
            gc_percent=gc,
            acgt_bases=total_length,
            ambiguous_bases=0,
        ),
    )


def test_history_sorts_rounds_and_emits_json_safe_features():
    history = build_light_feature_history(
        [observation(3), observation(1), observation(2)],
        coverage_effective=40.0,
    )

    assert history.schema_version == "1.0.0"
    assert history.observation_schema_version == "1.0.0"
    assert history.sample_id == "sample-1"
    assert history.current_round == 3
    assert history.coverage_effective == 40.0
    assert history.feature_names == LIGHT_HISTORY_FEATURE_NAMES
    assert [row.round for row in history.rows] == [1, 2, 3]
    assert history.rows[0].delta_n50 is None
    assert history.rows[0].n50_from_r1 == 0.0
    assert history.rows[1].delta_n50 == 30.0
    assert history.rows[1].delta_n50_trend is None
    assert history.rows[2].delta_n50_trend == -10.0
    assert history.rows[2].gc_from_r1 == pytest.approx(0.1)
    assert "mapping_error_rate" not in history.feature_names
    serialized = json.dumps(history.model_dump(mode="json"), allow_nan=False)
    restored = LightFeatureHistory.model_validate(json.loads(serialized))
    assert restored == history


def test_history_is_invariant_to_future_observations():
    prefix = build_light_feature_history(
        [observation(1), observation(2)],
        coverage_effective=30.0,
    )
    complete = build_light_feature_history(
        [observation(1), observation(2), observation(3)],
        coverage_effective=30.0,
    )

    assert prefix.rows == complete.rows[:2]


def test_history_rejects_mixed_samples():
    with pytest.raises(LightHistoryError, match="same sample_id"):
        build_light_feature_history(
            [observation(1), observation(2, sample_id="sample-2")],
            coverage_effective=30.0,
        )


@pytest.mark.parametrize(
    "observations, message",
    [
        ([], "at least one"),
        ([observation(2)], "start at round 1"),
        ([observation(1), observation(3)], "contiguous"),
        ([observation(1), observation(1)], "contiguous"),
    ],
)
def test_history_rejects_incomplete_round_sequences(observations, message):
    with pytest.raises(LightHistoryError, match=message):
        build_light_feature_history(
            observations,
            coverage_effective=30.0,
        )


@pytest.mark.parametrize("coverage", [0.0, -1.0, float("inf"), float("nan")])
def test_history_rejects_invalid_coverage(coverage):
    with pytest.raises(LightHistoryError, match="finite and positive"):
        build_light_feature_history(
            [observation(1)],
            coverage_effective=coverage,
        )


def test_observation_normalizes_sample_id():
    normalized = observation(1, sample_id="  sample-1  ")

    assert normalized.sample_id == "sample-1"
