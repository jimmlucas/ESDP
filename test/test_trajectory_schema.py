"""Tests for the trajectory-aware ESDP v2 contracts."""

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from esdp_trajectory import (
    DecisionAction,
    DecisionV2,
    IncompleteHistoryPolicy,
    IncompleteTrajectoryError,
    PolishingTrajectory,
    RoundMetrics,
    build_trajectory_features,
    conservative_fallback_decision,
)


def _round(round_number: int, **overrides) -> RoundMetrics:
    values = {
        "round": round_number,
        "qv": 30.0 + round_number,
        "busco_complete": 90.0 + round_number,
        "n50": 1_000_000 + round_number * 10_000,
        "num_contigs": 7 - round_number,
        "error_rate": 0.01 - round_number * 0.001,
        "total_length": 4_900_000,
        "assembly_frac": 0.95 + round_number * 0.01,
    }
    values.update(overrides)
    return RoundMetrics(**values)


def _trajectory(**overrides) -> PolishingTrajectory:
    values = {
        "sample_id": "sample_1",
        "genus": "Escherichia",
        "coverage": 40.0,
        "rounds": [_round(1), _round(2), _round(3)],
    }
    values.update(overrides)
    return PolishingTrajectory(**values)


def test_valid_trajectory_builds_prospective_features():
    result = build_trajectory_features(_trajectory())

    assert result.can_predict is True
    assert result.missing_metrics == {}
    assert list(result.features["round"]) == [1, 2, 3]
    assert result.latest["delta_qv"] == pytest.approx(1.0)
    assert result.latest["qv_from_r1"] == pytest.approx(2.0)
    assert result.latest["Sample"] == "sample_1"


def test_feature_frame_is_deterministic_and_workflow_friendly():
    trajectory = _trajectory(coverage_effective=38.5)

    frame = trajectory.to_feature_frame()

    assert isinstance(frame, pd.DataFrame)
    assert frame["Sample"].eq("sample_1").all()
    assert frame["Coverage"].eq(38.5).all()
    assert list(frame["round"]) == [1, 2, 3]


@pytest.mark.parametrize(
    ("rounds", "message"),
    [
        ([_round(2), _round(1)], "ascending"),
        ([_round(1), _round(1)], "unique"),
        ([_round(1), _round(3)], "contiguous"),
        ([_round(2), _round(3)], "start at round 1"),
    ],
)
def test_invalid_round_sequences_are_rejected(rounds, message):
    with pytest.raises(ValidationError, match=message):
        _trajectory(rounds=rounds)


def test_round_metric_ranges_and_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        RoundMetrics(round=1, busco_complete=101.0)

    with pytest.raises(ValidationError):
        RoundMetrics(round=1, qv=30.0, future_metric=12.0)


def test_round_requires_at_least_one_observed_metric():
    with pytest.raises(ValidationError, match="at least one observed metric"):
        RoundMetrics(round=1)


def test_incomplete_metrics_can_trigger_conservative_fallback():
    incomplete_round = _round(3, qv=None)
    trajectory = _trajectory(
        rounds=[_round(1), _round(2), incomplete_round],
    )

    result = build_trajectory_features(trajectory)
    decision = conservative_fallback_decision(
        trajectory,
        reason="Required metrics are missing",
    )

    assert result.can_predict is False
    assert result.missing_metrics == {3: ["qv"]}
    assert decision.action is DecisionAction.CONSERVATIVE_CONTINUE
    assert decision.current_round == 3
    assert decision.next_round == 4
    assert decision.recommended_final_round == 5


def test_incomplete_metrics_can_be_configured_as_an_error():
    trajectory = _trajectory(
        incomplete_history_policy=IncompleteHistoryPolicy.ERROR,
        rounds=[_round(1, error_rate=None)],
    )

    with pytest.raises(IncompleteTrajectoryError, match="error_rate"):
        build_trajectory_features(trajectory)


def test_conservative_fallback_stops_at_configured_maximum():
    trajectory = _trajectory(max_rounds=3)

    decision = conservative_fallback_decision(
        trajectory,
        reason="Model unavailable",
    )

    assert decision.action is DecisionAction.STOP
    assert decision.current_round == 3
    assert decision.next_round is None
    assert decision.recommended_final_round == 3


def test_decision_never_recommends_an_earlier_round():
    with pytest.raises(ValidationError, match="cannot precede current_round"):
        DecisionV2(
            sample_id="sample_1",
            action=DecisionAction.STOP,
            current_round=3,
            recommended_final_round=2,
            reason="Invalid recommendation",
        )


def test_continue_decision_requires_the_immediate_next_round():
    with pytest.raises(ValidationError, match="next_round=current_round\\+1"):
        DecisionV2(
            sample_id="sample_1",
            action=DecisionAction.CONTINUE,
            current_round=2,
            next_round=4,
            recommended_final_round=5,
            reason="Invalid skipped round",
        )


def test_stop_decision_must_select_the_current_round():
    with pytest.raises(ValidationError, match="must select the current round"):
        DecisionV2(
            sample_id="sample_1",
            action=DecisionAction.STOP,
            current_round=2,
            recommended_final_round=3,
            reason="Invalid future stop",
        )


def test_continue_decision_cannot_exceed_configured_maximum():
    with pytest.raises(ValidationError, match="cannot continue"):
        DecisionV2(
            sample_id="sample_1",
            action=DecisionAction.CONTINUE,
            current_round=3,
            next_round=4,
            recommended_final_round=3,
            max_rounds=3,
            reason="Invalid continuation",
        )


def test_documented_json_example_matches_the_runtime_schema():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "trajectory.v2.json"
    )
    with example_path.open(encoding="utf-8") as example_file:
        trajectory = PolishingTrajectory.model_validate(json.load(example_file))

    assert trajectory.sample_id == "sample_001"
    assert trajectory.current_round == 2
    assert trajectory.model_json_schema()["properties"]["schema_version"][
        "const"
    ] == "2.0.0"
