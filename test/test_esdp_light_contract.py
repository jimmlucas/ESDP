"""Tests for the cost-aware ESDP-light feature boundary."""

import json
from pathlib import Path

import pandas as pd
import pytest

from esdp_light import (
    FORBIDDEN_LIGHT_FEATURES,
    LIGHT_ALIGNMENT_CANDIDATES,
    LIGHT_CORE_FEATURES,
    METRIC_REGISTRY,
    LightFeatureContractError,
    audit_feature_availability,
    validate_frozen_sample_split,
    validate_light_feature_contract,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def test_core_contract_excludes_expensive_and_label_derived_features():
    validated = validate_light_feature_contract(LIGHT_CORE_FEATURES)

    assert validated == LIGHT_CORE_FEATURES
    assert not set(validated) & FORBIDDEN_LIGHT_FEATURES
    assert "qv" not in validated
    assert "busco_complete" not in validated
    assert "r1_ok_group" not in validated
    assert "raw_n_reads" not in validated
    assert "ai_circular_n" not in validated


def test_metric_registry_has_unique_names():
    names = [spec.name for spec in METRIC_REGISTRY]

    assert len(names) == len(set(names))


def test_alignment_features_remain_provisional_by_default():
    with pytest.raises(
        LightFeatureContractError,
        match="approved deployment provenance",
    ):
        validate_light_feature_contract(LIGHT_ALIGNMENT_CANDIDATES)

    assert validate_light_feature_contract(
        LIGHT_ALIGNMENT_CANDIDATES,
        allow_provisional_alignment=True,
    ) == LIGHT_ALIGNMENT_CANDIDATES


def test_legacy_error_rate_is_not_silently_rebranded_as_light():
    assert "error_rate" in FORBIDDEN_LIGHT_FEATURES
    assert "mapping_error_rate" in LIGHT_ALIGNMENT_CANDIDATES


def test_forbidden_derived_quality_feature_is_rejected():
    with pytest.raises(LightFeatureContractError, match="forbidden"):
        validate_light_feature_contract(["n50", "score_improvement"])


def test_availability_audit_never_imputes_missing_values():
    frame = pd.DataFrame({"n50": [1, None], "raw_mean_read_len": [None, None]})

    audit = audit_feature_availability(
        frame,
        ["n50", "raw_mean_read_len", "absent"],
    ).set_index("feature")

    assert audit.loc["n50", "availability_fraction"] == 0.5
    assert audit.loc["raw_mean_read_len", "availability_fraction"] == 0.0
    assert audit.loc["absent", "availability_fraction"] == 0.0
    assert frame["n50"].isna().sum() == 1


def test_repository_split_is_disjoint_and_exhaustive():
    frame = pd.read_csv(
        REPOSITORY / "data" / "training_dataset_with_target.csv",
        usecols=["Sample"],
    )
    split = json.loads(
        (REPOSITORY / "outputs" / "train_test_split_samples.json").read_text(
            encoding="utf-8"
        )
    )

    validate_frozen_sample_split(
        frame,
        split["train_samples"],
        split["test_samples"],
    )


def test_repository_has_complete_observations_for_ready_core_features():
    frame = pd.read_csv(
        REPOSITORY / "data" / "training_dataset_with_target.csv",
    )

    audit = audit_feature_availability(frame, LIGHT_CORE_FEATURES)

    assert not audit.empty
    assert audit["availability_fraction"].eq(1.0).all()


def test_split_rejects_sample_overlap():
    frame = pd.DataFrame({"Sample": ["A", "B"]})

    with pytest.raises(ValueError, match="overlap"):
        validate_frozen_sample_split(frame, ["A"], ["A", "B"])
