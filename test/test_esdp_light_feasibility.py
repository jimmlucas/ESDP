"""Guards for the development-only ESDP-light feasibility experiment."""

import json
from pathlib import Path

import pandas as pd

from esdp_light import FORBIDDEN_LIGHT_FEATURES
from experiments.esdp_light_feasibility import (
    define_candidates,
    make_sample_folds,
    prepare_training_frame,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY / "data" / "training_dataset_with_target.csv"
SPLIT = REPOSITORY / "outputs" / "train_test_split_samples.json"
RESULTS = REPOSITORY / "outputs" / "esdp_light_feasibility"


def test_feasibility_frame_uses_only_frozen_development_samples():
    frame, audit = prepare_training_frame(DATASET, SPLIT)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))

    assert len(frame) == 635
    assert frame["Sample"].nunique() == 32
    assert audit["development_trajectories"] == 127
    assert audit["held_out_samples_used"] == []
    assert not set(frame["Sample"]) & set(split["test_samples"])
    assert frame.groupby(["Sample", "Coverage_effective"]).size().eq(5).all()


def test_light_candidates_respect_the_frozen_feature_boundary():
    frame, _ = prepare_training_frame(DATASET, SPLIT)
    candidates = {
        candidate.name: candidate
        for candidate in define_candidates(REPOSITORY, frame)
    }

    assert set(candidates) == {
        "legacy_full_retrospective",
        "light_core_prospective",
        "light_fasta_prospective",
        "r1_core_only",
    }
    for name in (
        "light_core_prospective",
        "light_fasta_prospective",
        "r1_core_only",
    ):
        assert not set(candidates[name].feature_names) & FORBIDDEN_LIGHT_FEATURES
    assert set(candidates["legacy_full_retrospective"].feature_names) & (
        FORBIDDEN_LIGHT_FEATURES
    )


def test_fold_assignment_is_unique_at_biological_sample_level():
    frame, _ = prepare_training_frame(DATASET, SPLIT)
    folds = make_sample_folds(frame)

    assert len(folds) == 32
    assert folds["Sample"].nunique() == 32
    assert set(folds["fold"]) == {1, 2, 3, 4, 5}


def test_committed_report_preserves_the_held_out_firewall():
    report = json.loads(
        (RESULTS / "experiment_report.json").read_text(encoding="utf-8")
    )

    assert report["audit"]["development_samples"] == 32
    assert report["audit"]["held_out_samples_reserved"] == 9
    assert report["audit"]["held_out_samples_used"] == []
    assert report["scientific_status"].startswith("development-only")
    assert {
        row["n_trajectories"] for row in report["summary"]
    } == {127}


def test_round_diagnostics_are_not_conditioned_on_primary_stopping():
    predictions = pd.read_csv(RESULTS / "oof_round_predictions.csv")
    adaptive = predictions[
        predictions["candidate"].isin(
            {
                "legacy_full_retrospective",
                "light_core_prospective",
                "light_fasta_prospective",
            }
        )
    ]

    assert adaptive.groupby(["candidate", "round"]).size().eq(127).all()
