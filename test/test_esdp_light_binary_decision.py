"""Guards for the development-only binary ESDP-light experiment."""

import json
from pathlib import Path

import pandas as pd

from experiments.esdp_light_binary_decision import (
    CAUSAL_QUALITY_FEATURES,
    HISTORICAL_PREFIX,
    define_binary_candidates,
    rebuild_causal_quality_features,
)
from experiments.esdp_light_feasibility import (
    define_candidates,
    prepare_training_frame,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY / "data" / "training_dataset_with_target.csv"
SPLIT = REPOSITORY / "outputs" / "train_test_split_samples.json"
RESULTS = REPOSITORY / "outputs" / "esdp_light_binary_feasibility"


def _binary_frame_and_candidates():
    frame, audit = prepare_training_frame(DATASET, SPLIT)
    source = {
        candidate.name: candidate
        for candidate in define_candidates(REPOSITORY, frame)
    }
    frame = rebuild_causal_quality_features(
        frame,
        source["legacy_full_retrospective"].feature_names,
    )
    candidates = {
        candidate.name: candidate
        for candidate in define_binary_candidates(REPOSITORY, frame)
    }
    return frame, audit, candidates


def test_binary_candidates_make_historical_leakage_explicit():
    frame, _, candidates = _binary_frame_and_candidates()

    historical = candidates["legacy_historical_leaky_binary"]
    rebuilt = candidates["legacy_rebuilt_binary"]
    assert all(name.startswith(HISTORICAL_PREFIX) for name in historical.feature_names)
    assert "Invalid diagnostic" in historical.description
    assert not any(name.startswith(HISTORICAL_PREFIX) for name in rebuilt.feature_names)
    assert set(historical.feature_names).issubset(frame.columns)
    assert set(rebuilt.feature_names).issubset(frame.columns)


def test_causal_quality_control_excludes_policy_and_plateau_predictors():
    forbidden = {
        "is_plateau",
        "plateau_streak",
        "r1_ok_group",
        "stable_all_group",
        "optimal_rounds_5class",
    }

    assert not set(CAUSAL_QUALITY_FEATURES) & forbidden


def test_binary_predictions_are_complete_and_use_only_decision_rounds():
    predictions = pd.read_csv(RESULTS / "oof_binary_round_predictions.csv")

    assert set(predictions["round"]) == {1, 2, 3, 4}
    assert predictions.groupby(["candidate", "round"]).size().eq(127).all()
    assert predictions.groupby(
        ["candidate", "Sample", "Coverage_effective"]
    ).size().eq(4).all()
    assert predictions["probability_stop"].between(0.0, 1.0).all()


def test_binary_report_preserves_test_firewall_and_fixed_protocol():
    report = json.loads(
        (RESULTS / "binary_experiment_report.json").read_text(encoding="utf-8")
    )

    assert report["audit"]["development_samples"] == 32
    assert report["audit"]["held_out_samples_reserved"] == 9
    assert report["audit"]["held_out_samples_used"] == []
    assert report["protocol"]["training_rounds"] == [1, 2, 3, 4]
    assert report["protocol"]["primary_stop_threshold"] == 0.7
    assert report["protocol"]["target"].startswith("STOP_ELIGIBLE")
    assert {row["n_trajectories"] for row in report["summary"]} == {127}


def test_stop_eligible_target_is_monotonic_within_each_trajectory():
    predictions = pd.read_csv(RESULTS / "oof_binary_round_predictions.csv")
    reference = predictions[
        predictions["candidate"] == "light_core_binary"
    ].sort_values(["Sample", "Coverage_effective", "round"])

    assert reference.groupby(["Sample", "Coverage_effective"])[
        "stop_eligible"
    ].apply(lambda values: values.is_monotonic_increasing).all()
