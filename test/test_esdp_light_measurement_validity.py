"""Guards for the ESDP-light measurement-validity gate."""

import json
from pathlib import Path

from experiments.esdp_light_feasibility import prepare_training_frame
from experiments.esdp_light_measurement_validity import (
    busco_audit,
    dependency_audit,
    materiality_contract,
    replicate_audit,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATASET = REPOSITORY / "data" / "training_dataset_with_target.csv"
SPLIT = REPOSITORY / "outputs" / "train_test_split_samples.json"
RESULTS = REPOSITORY / "outputs" / "esdp_light_measurement_validity"


def _development_frame():
    return prepare_training_frame(DATASET, SPLIT)[0]


def test_qv_and_assembly_error_are_exact_derived_representations():
    audit = dependency_audit(_development_frame())

    assert audit["qv_is_exact_error_transform"]
    assert audit["qv_error_max_absolute_residual"] < 1e-12
    assert audit["assembly_error_is_exact_fraction_transform"]
    assert audit["assembly_error_max_absolute_residual"] < 1e-12


def test_busco_resolution_and_transient_gain_audit_are_frozen():
    audit, trajectories = busco_audit(_development_frame())

    assert audit["busco_n_values"] == [116]
    assert audit["one_percent_threshold_requires_at_least_two_markers"]
    assert len(trajectories) == 127
    assert trajectories["any_future_gain_gt_1pp"].sum() == 38
    assert trajectories["terminal_gain_gt_1pp"].sum() == 26
    assert trajectories["transient_gain_only"].sum() == 12


def test_current_dataset_cannot_estimate_technical_repeatability():
    audit = replicate_audit(_development_frame())

    assert audit["rows"] == 635
    assert audit["unique_sample_coverage_rounds"] == 635
    assert audit["technical_replicate_cells"] == 0
    assert not audit["technical_repeatability_estimable"]


def test_materiality_contract_blocks_downstream_release_gates():
    frame = _development_frame()
    dependency = dependency_audit(frame)
    busco, _ = busco_audit(frame)
    replicates = replicate_audit(frame)
    contract = materiality_contract(dependency, busco, replicates)

    assert contract["release_gate"] == "blocked"
    limits = contract["global_limitations"]
    assert not limits["model_retraining_authorized"]
    assert not limits["held_out_test_may_be_opened"]
    assert not limits["nextflow_adaptive_stopping_authorized"]


def test_committed_measurement_report_preserves_test_firewall():
    report = json.loads(
        (RESULTS / "measurement_validity_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["split_audit"]["development_samples"] == 32
    assert report["split_audit"]["held_out_samples_reserved"] == 9
    assert report["split_audit"]["held_out_samples_used"] == []
    assert report["materiality_contract"]["release_gate"] == "blocked"
