"""Guards for the development-only material-benefit endpoint study."""

import json
from pathlib import Path

import pandas as pd

from experiments.esdp_light_feasibility import prepare_training_frame
from experiments.esdp_light_material_benefit import (
    ENDPOINT_COLUMN,
    TARGET_COLUMN,
    add_material_benefit_endpoint,
    rejected_pareto_audit,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY / "data" / "training_dataset_with_target.csv"
SPLIT = REPOSITORY / "outputs" / "train_test_split_samples.json"
RESULTS = REPOSITORY / "outputs" / "esdp_light_material_benefit"


def _development_frame():
    frame, audit = prepare_training_frame(DATASET, SPLIT)
    return frame, audit


def test_material_endpoint_is_absorbing_and_complete():
    frame, _ = _development_frame()
    endpoint, audit = add_material_benefit_endpoint(frame)

    assert len(endpoint) == 635
    assert len(audit) == 635
    assert endpoint.groupby(["Sample", "Coverage_effective"]).size().eq(5).all()
    assert endpoint.groupby(["Sample", "Coverage_effective"])[TARGET_COLUMN].apply(
        lambda values: values.is_monotonic_increasing
    ).all()
    assert endpoint[endpoint["round"] == 5][TARGET_COLUMN].eq(1).all()


def test_material_endpoint_has_frozen_primary_distribution():
    frame, _ = _development_frame()
    endpoint, _ = add_material_benefit_endpoint(frame)
    distribution = (
        endpoint[endpoint["round"] == 1][ENDPOINT_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    assert distribution == {1: 88, 2: 17, 3: 10, 4: 7, 5: 5}


def test_material_endpoint_does_not_depend_on_legacy_target_columns():
    frame, _ = _development_frame()
    original, _ = add_material_benefit_endpoint(frame)
    relabeled = frame.copy()
    relabeled["optimal_rounds_5class"] = 5
    relabeled["optimal_rounds_3class"] = 3
    rebuilt, _ = add_material_benefit_endpoint(relabeled)

    keys = ["Sample", "Coverage_effective", "round"]
    original = original.sort_values(keys).reset_index(drop=True)
    rebuilt = rebuilt.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_series_equal(
        original[ENDPOINT_COLUMN],
        rebuilt[ENDPOINT_COLUMN],
    )
    pd.testing.assert_series_equal(original[TARGET_COLUMN], rebuilt[TARGET_COLUMN])


def test_rejected_pareto_endpoint_audit_is_reproducible():
    frame, _ = _development_frame()
    audit = rejected_pareto_audit(frame)

    assert audit["status"] == "rejected before model fitting"
    assert audit["raw_non_monotonic_trajectories"] == 42
    assert audit["earliest_stop_distribution"] == {
        "1": 99,
        "2": 14,
        "3": 6,
        "4": 4,
        "5": 4,
    }


def test_committed_material_predictions_are_grouped_and_complete():
    predictions = pd.read_csv(RESULTS / "oof_material_round_predictions.csv")

    assert set(predictions["round"]) == {1, 2, 3, 4}
    assert predictions.groupby(["candidate", "round"]).size().eq(127).all()
    assert predictions.groupby(
        ["candidate", "Sample", "Coverage_effective"]
    ).size().eq(4).all()
    assert predictions["probability_stop"].between(0.0, 1.0).all()


def test_committed_material_report_preserves_test_firewall():
    report = json.loads(
        (RESULTS / "material_experiment_report.json").read_text(encoding="utf-8")
    )

    assert report["audit"]["development_samples"] == 32
    assert report["audit"]["held_out_samples_reserved"] == 9
    assert report["audit"]["held_out_samples_used"] == []
    assert report["protocol"]["training_rounds"] == [1, 2, 3, 4]
    assert report["protocol"]["primary_stop_threshold"] == 0.7
    assert report["scientific_status"].startswith("development-only")
