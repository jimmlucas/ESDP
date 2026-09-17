#!/usr/bin/env python3

import pytest

from esdp_decide import (
    DECISION_THRESHOLD,
    FEATURE_SCHEMA_SHA256,
    FROZEN_FEATURES,
    FROZEN_MODEL_SHA256,
    LEGAL_DECISION_ROUNDS,
    MODEL_VERSION,
    InputValidationError,
    PolishingState,
    decide,
    model_info,
    prepare_features,
)


def reference_state(**overrides):
    values = {
        "sample_id": "core_test",
        "round": 1,
        "coverage_est": 40,
        "expected_genome_size": 5_000_000,
        "raw_read_n50": 15_000,
        "ai_cov_cv": 0.20,
        "current_n50": 4_800_000,
        "current_num_contigs": 2,
        "current_assembly_frac": 0.99,
        "delta_n50_last": None,
        "n50_from_R1": None,
    }

    values.update(overrides)

    return PolishingState(**values)


def test_frozen_constants():
    assert MODEL_VERSION == "esdp-sequential-rf-v2"
    assert DECISION_THRESHOLD == 0.45
    assert LEGAL_DECISION_ROUNDS == (1, 2, 3, 4)

    assert FROZEN_MODEL_SHA256 == (
        "9f9f08428242e98381546b9c79cd3d9b013c412b3edcbfe01ca84bb6f4c10dcf"
    )

    assert FEATURE_SCHEMA_SHA256 == (
        "2f50cc9c6169c325da21e189b309623d872cbfdf4a025588c786c6e9f4576ced"
    )


def test_exact_feature_contract():
    assert FROZEN_FEATURES == (
        "round",
        "coverage_est",
        "expected_genome_size",
        "raw_read_n50",
        "ai_cov_cv",
        "current_n50",
        "current_num_contigs",
        "current_assembly_frac",
        "delta_n50_last",
        "n50_from_R1",
    )


def test_prepare_features_exact_order():
    frame = prepare_features(reference_state())

    assert tuple(frame.columns) == FROZEN_FEATURES
    assert frame.shape == (1, 10)


def test_r1_history_features_are_missing():
    frame = prepare_features(reference_state())

    assert frame["delta_n50_last"].isna().iloc[0]
    assert frame["n50_from_R1"].isna().iloc[0]


def test_frozen_reference_prediction():
    result = decide(reference_state())

    assert result.p_continue == pytest.approx(
        0.5939402938271254,
        abs=1e-12,
    )

    assert result.threshold == 0.45
    assert result.decision == "CONTINUE"
    assert result.next_action == "RACON_R2"

    assert result.model_version == MODEL_VERSION
    assert result.model_sha256 == FROZEN_MODEL_SHA256
    assert result.feature_schema_sha256 == FEATURE_SCHEMA_SHA256

    assert result.missing_features == [
        "delta_n50_last",
        "n50_from_R1",
    ]


@pytest.mark.parametrize("round_number", [0, 5, 6])
def test_illegal_decision_rounds_rejected(round_number):
    state = reference_state(
        round=round_number
    )

    with pytest.raises(InputValidationError):
        decide(state)


def test_negative_coverage_rejected():
    state = reference_state(
        coverage_est=-1
    )

    with pytest.raises(InputValidationError):
        decide(state)


def test_negative_current_n50_rejected():
    state = reference_state(
        current_n50=-1
    )

    with pytest.raises(InputValidationError):
        decide(state)


def test_empty_sample_id_rejected():
    state = reference_state(
        sample_id=""
    )

    with pytest.raises(InputValidationError):
        decide(state)


def test_r4_action_is_valid():
    state = reference_state(
        round=4,
        delta_n50_last=1000,
        n50_from_R1=5000,
    )

    result = decide(state)

    assert result.decision in {
        "STOP",
        "CONTINUE",
    }

    if result.decision == "CONTINUE":
        assert result.next_action == "RACON_R5_THEN_MEDAKA"
    else:
        assert result.next_action == "MEDAKA"


def test_model_info_matches_frozen_contract():
    info = model_info()

    assert info["model_version"] == MODEL_VERSION
    assert info["decision_threshold"] == DECISION_THRESHOLD
    assert info["legal_decision_rounds"] == [1, 2, 3, 4]
    assert info["maximum_racon_round"] == 5
    assert info["feature_count"] == 10
    assert info["features"] == list(FROZEN_FEATURES)

    assert info["model_sha256"] == FROZEN_MODEL_SHA256
    assert info["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256
