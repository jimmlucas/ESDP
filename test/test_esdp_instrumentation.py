"""Tests for immutable prospective ESDP instrumentation projects."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from esdp_instrumentation import (
    InstrumentationError,
    LongReadTechnology,
    OfflineQualityOutcome,
    ToolIdentity,
    init_project,
    record_round,
)
from esdp_manifest import sha256_file


def _technology():
    return LongReadTechnology(
        platform="ont",
        chemistry="R10.4.1",
        basecaller="Dorado",
        basecaller_version="0.9.0",
        basecaller_model="sup@v5.0.0",
    )


def _tool(name):
    return ToolIdentity(name=name, version="1.0.0", parameters=("--frozen",))


def _init(tmp_path: Path) -> Path:
    project = tmp_path / "study"
    init_project(
        project,
        project_id="study-001",
        technology=_technology(),
        assembler=_tool("Flye"),
        polisher=_tool("Racon"),
    )
    return project


def _assembly(path: Path, sequence: str) -> Path:
    path.write_text(f">contig\n{sequence}\n", encoding="utf-8")
    return path


def test_init_creates_frozen_non_decision_contract_and_qc_schema(tmp_path):
    project = _init(tmp_path)

    contract = json.loads(
        (project / "esdp-project.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (project / "schemas" / "offline-qc.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["project_id"] == "study-001"
    assert contract["decision_enabled"] is False
    assert contract["materiality_contract_status"] == (
        "blocked_pending_validation"
    )
    assert contract["technology"]["basecaller_model"] == "sup@v5.0.0"
    assert schema["title"] == "OfflineQualityOutcome"
    assert (project / "trajectories").is_dir()


def test_ont_init_requires_complete_basecalling_provenance():
    with pytest.raises(ValidationError, match="basecalling provenance"):
        LongReadTechnology(platform="ont", chemistry="R10.4.1")


def test_record_round_commits_immutable_cumulative_histories(tmp_path):
    project = _init(tmp_path)
    reads = tmp_path / "reads.fastq"
    reads.write_text("@r1\nACGT\n+\n!!!!\n", encoding="utf-8")
    r1 = _assembly(tmp_path / "r1.fasta", "ACGTACGT")
    r2 = _assembly(tmp_path / "r2.fasta", "ACGTACGTACGT")

    first = record_round(
        project,
        sample_id="sample-A",
        coverage_effective=40,
        round_number=1,
        assembly_path=r1,
        read_paths=(reads,),
    )
    first_history_before = Path(first.history_file).read_bytes()
    second = record_round(
        project,
        sample_id="sample-A",
        coverage_effective=40,
        round_number=2,
        assembly_path=r2,
        read_paths=(reads,),
    )

    first_history = json.loads(Path(first.history_file).read_text(encoding="utf-8"))
    second_history = json.loads(Path(second.history_file).read_text(encoding="utf-8"))
    record = json.loads(Path(second.record_file).read_text(encoding="utf-8"))
    assert Path(first.history_file).read_bytes() == first_history_before
    assert first_history["current_round"] == 1
    assert second_history["current_round"] == 2
    assert [row["round"] for row in second_history["rows"]] == [1, 2]
    assert second_history["rows"][1]["delta_total_length"] == 4.0
    assert record["decision_enabled"] is False
    assert record["project_contract_sha256"] == sha256_file(
        project / "esdp-project.json"
    )
    assert record["read_artifacts"] == [
        {"file_name": "reads.fastq", "sha256": sha256_file(reads)}
    ]


def test_record_round_rejects_skips_overwrites_and_unsafe_ids(tmp_path):
    project = _init(tmp_path)
    assembly = _assembly(tmp_path / "assembly.fasta", "ACGT")

    with pytest.raises(InstrumentationError, match="contiguously"):
        record_round(
            project,
            sample_id="sample-A",
            coverage_effective=40,
            round_number=2,
            assembly_path=assembly,
        )
    with pytest.raises(InstrumentationError, match="sample_id must match"):
        record_round(
            project,
            sample_id="../sample-A",
            coverage_effective=40,
            round_number=1,
            assembly_path=assembly,
        )
    record_round(
        project,
        sample_id="sample-A",
        coverage_effective=40,
        round_number=1,
        assembly_path=assembly,
    )
    with pytest.raises(InstrumentationError, match="immutable"):
        record_round(
            project,
            sample_id="sample-A",
            coverage_effective=40,
            round_number=1,
            assembly_path=assembly,
        )


def test_record_round_rejects_project_contract_drift(tmp_path):
    project = _init(tmp_path)
    r1 = _assembly(tmp_path / "r1.fasta", "ACGT")
    r2 = _assembly(tmp_path / "r2.fasta", "ACGTACGT")
    record_round(
        project,
        sample_id="sample-A",
        coverage_effective=40,
        round_number=1,
        assembly_path=r1,
    )
    contract_path = project / "esdp-project.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["polisher"]["version"] = "changed-after-R1"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(InstrumentationError, match="differs"):
        record_round(
            project,
            sample_id="sample-A",
            coverage_effective=40,
            round_number=2,
            assembly_path=r2,
        )


def test_offline_qc_is_validated_and_bound_to_exact_assembly(tmp_path):
    project = _init(tmp_path)
    assembly = _assembly(tmp_path / "assembly.fasta", "ACGTACGT")
    qc = OfflineQualityOutcome(
        assembly_sha256=sha256_file(assembly),
        busco={
            "complete_count": 110,
            "single_copy_count": 109,
            "duplicated_count": 1,
            "fragmented_count": 2,
            "missing_count": 4,
            "marker_count": 116,
            "version": "6.1.0",
            "lineage_dataset": "bacteria_odb12.2",
            "lineage_creation_date": "2026-05-22",
            "mode": "genome",
            "options": ["--offline"],
        },
    )
    qc_path = tmp_path / "qc.json"
    qc_path.write_text(qc.model_dump_json(indent=2), encoding="utf-8")

    receipt = record_round(
        project,
        sample_id="sample-A",
        coverage_effective=40,
        round_number=1,
        assembly_path=assembly,
        offline_qc_path=qc_path,
    )
    committed = json.loads(
        Path(receipt.offline_qc_file).read_text(encoding="utf-8")
    )
    assert committed["busco"]["marker_count"] == 116
    assert committed["assembly_sha256"] == sha256_file(assembly)


def test_offline_qc_rejects_different_assembly(tmp_path):
    project = _init(tmp_path)
    assembly = _assembly(tmp_path / "assembly.fasta", "ACGT")
    qc_path = tmp_path / "qc.json"
    qc_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "assembly_sha256": "0" * 64,
                "coding_integrity": {
                    "predicted_cds": 1,
                    "frameshifts": 0,
                    "premature_stops": 0,
                    "truncated_cds": 0,
                    "tool": {
                        "name": "QC",
                        "version": "1",
                        "parameters": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InstrumentationError, match="does not match"):
        record_round(
            project,
            sample_id="sample-A",
            coverage_effective=40,
            round_number=1,
            assembly_path=assembly,
            offline_qc_path=qc_path,
        )
