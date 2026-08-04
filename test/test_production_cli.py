"""Process-level tests for the workflow-oriented ESDP CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPOSITORY = Path(__file__).resolve().parents[1]
CLI = REPOSITORY / "esdp_cli.py"
TRAJECTORY_EXAMPLE = (
    REPOSITORY / "examples" / "trajectory.v2.json"
)


def _run_cli(*arguments, cwd, stdin=None):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(CLI), *arguments],
        cwd=cwd,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )


def _valid_metrics():
    return {
        "sample_id": "cli_sample",
        "round": 1,
        "coverage": 40.0,
        "qv": 35.0,
        "busco_complete": 95.0,
        "n50": 4_000_000.0,
        "num_contigs": 2,
        "error_rate": 0.001,
        "total_length": 4_800_000,
    }


def _light_observation(sample_id, round_number, *, n50, num_contigs):
    return {
        "schema_version": "1.0.0",
        "sample_id": sample_id,
        "round": round_number,
        "assembly_sha256": f"{round_number:064x}",
        "fasta": {
            "num_contigs": num_contigs,
            "total_length": 1000 + (round_number - 1) * 10,
            "n50": n50,
            "gc_percent": 50.0 + (round_number - 1) * 0.1,
            "acgt_bases": 1000 + (round_number - 1) * 10,
            "ambiguous_bases": 0,
        },
        "alignment": None,
        "alignment_reference_sha256": None,
        "samtools_stats_sha256": None,
    }


def test_version_command_works_outside_repository(tmp_path):
    result = _run_cli("--version", cwd=tmp_path)

    assert result.returncode == 0
    assert result.stdout == "esdp 2.0.0.dev0\n"
    assert result.stderr == ""


def test_decide_reads_stdin_and_emits_only_deterministic_json(tmp_path):
    serialized_input = json.dumps(_valid_metrics())

    first = _run_cli(
        "decide",
        "--input",
        "-",
        "--output",
        "-",
        cwd=tmp_path,
        stdin=serialized_input,
    )
    second = _run_cli(
        "decide",
        "--input",
        "-",
        "--output",
        "-",
        cwd=tmp_path,
        stdin=serialized_input,
    )

    assert first.returncode == 0
    assert first.stderr == ""
    assert first.stdout == second.stdout
    decision = json.loads(first.stdout)
    assert decision["decision_schema_version"] == "1.0.0"
    assert decision["feature_schema_version"] == "1.0.0"
    assert decision["model_version"] == "v1.1.0"
    assert decision["workflow_action"] == "STOP"
    assert decision["selected_final_round"] == 1


def test_decide_writes_a_complete_json_file(tmp_path):
    input_path = tmp_path / "metrics.json"
    output_path = tmp_path / "decision.json"
    input_path.write_text(json.dumps(_valid_metrics()), encoding="utf-8")

    result = _run_cli(
        "decide",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert json.loads(output_path.read_text(encoding="utf-8"))["sample_id"] == (
        "cli_sample"
    )
    assert not list(tmp_path.glob(".decision.json.*.tmp"))


def test_invalid_input_has_stable_exit_code_and_no_json_output(tmp_path):
    payload = _valid_metrics()
    payload["unexpected"] = True

    result = _run_cli(
        "decide",
        "--input",
        "-",
        cwd=tmp_path,
        stdin=json.dumps(payload),
    )

    assert result.returncode == 3
    assert result.stdout == ""
    assert result.stderr.startswith("esdp: invalid input:")


def test_invalid_confidence_threshold_has_stable_exit_code(tmp_path):
    result = _run_cli(
        "decide",
        "--input",
        "-",
        "--confidence-threshold",
        "1.1",
        cwd=tmp_path,
        stdin=json.dumps(_valid_metrics()),
    )

    assert result.returncode == 3
    assert "confidence-threshold must be between 0 and 1" in result.stderr


def test_validate_auto_detects_v2_trajectory(tmp_path):
    result = _run_cli(
        "validate",
        "--input",
        str(TRAJECTORY_EXAMPLE),
        cwd=tmp_path,
    )

    assert result.returncode == 0
    receipt = json.loads(result.stdout)
    assert receipt == {
        "complete": True,
        "current_round": 2,
        "kind": "trajectory",
        "missing_metrics": {},
        "sample_id": "sample_001",
        "schema_version": "2.0.0",
        "valid": True,
    }


def test_model_info_verifies_bundled_artifact_outside_repository(tmp_path):
    result = _run_cli("model-info", cwd=tmp_path)

    assert result.returncode == 0
    model_info = json.loads(result.stdout)
    assert model_info["artifact_verified"] is True
    assert model_info["feature_names_verified"] is True
    assert model_info["manifest"]["model_version"] == "v1.1.0"
    assert model_info["training_data_verified"] is False


def test_light_observe_emits_deterministic_reference_free_json(tmp_path):
    assembly = tmp_path / "polished.fasta"
    assembly.write_text(
        ">long\nGGCCAAAANN\n>short\nATGC\n",
        encoding="utf-8",
    )
    arguments = (
        "light-observe",
        "--sample-id",
        "sample-light",
        "--round",
        "2",
        "--assembly",
        str(assembly),
    )

    first = _run_cli(*arguments, cwd=tmp_path)
    second = _run_cli(*arguments, cwd=tmp_path)

    assert first.returncode == 0
    assert first.stderr == ""
    assert first.stdout == second.stdout
    observation = json.loads(first.stdout)
    assert observation["schema_version"] == "1.0.0"
    assert observation["sample_id"] == "sample-light"
    assert observation["round"] == 2
    assert len(observation["assembly_sha256"]) == 64
    assert observation["fasta"] == {
        "acgt_bases": 12,
        "ambiguous_bases": 2,
        "gc_percent": 50.0,
        "n50": 10,
        "num_contigs": 2,
        "total_length": 14,
    }
    assert observation["alignment"] is None
    assert observation["alignment_reference_sha256"] is None
    assert observation["samtools_stats_sha256"] is None


def test_light_observe_writes_atomic_alignment_observation(tmp_path):
    assembly = tmp_path / "polished.fasta"
    assembly.write_text(">contig\nACGTACGT\n", encoding="utf-8")
    stats = tmp_path / "alignment.stats"
    stats.write_text(
        "SN\terror rate:\t0.0125\n"
        "SN\tbases mapped (cigar):\t1,000\n"
        "SN\tmismatches:\t12\n",
        encoding="utf-8",
    )
    output = tmp_path / "observation.json"

    result = _run_cli(
        "light-observe",
        "--sample-id",
        "sample-light",
        "--round",
        "3",
        "--assembly",
        str(assembly),
        "--samtools-stats",
        str(stats),
        "--alignment-reference",
        str(assembly),
        "--output",
        str(output),
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    observation = json.loads(output.read_text(encoding="utf-8"))
    assert observation["alignment"]["mapping_error_rate"] == 0.0125
    assert observation["alignment_reference_sha256"] == (
        observation["assembly_sha256"]
    )
    assert not list(tmp_path.glob(".observation.json.*.tmp"))


def test_light_observe_rejects_mismatched_alignment_reference(tmp_path):
    assembly = tmp_path / "polished.fasta"
    assembly.write_text(">contig\nACGT\n", encoding="utf-8")
    reference = tmp_path / "pre-polish.fasta"
    reference.write_text(">contig\nAGGT\n", encoding="utf-8")
    stats = tmp_path / "alignment.stats"
    stats.write_text(
        "SN\terror rate:\t0.01\n"
        "SN\tbases mapped (cigar):\t100\n"
        "SN\tmismatches:\t1\n",
        encoding="utf-8",
    )

    result = _run_cli(
        "light-observe",
        "--sample-id",
        "sample-light",
        "--round",
        "2",
        "--assembly",
        str(assembly),
        "--samtools-stats",
        str(stats),
        "--alignment-reference",
        str(reference),
        cwd=tmp_path,
    )

    assert result.returncode == 3
    assert result.stdout == ""
    assert "not the polished assembly" in result.stderr


def test_light_history_aggregates_observations_in_round_order(tmp_path):
    round_1 = tmp_path / "r1.json"
    round_2 = tmp_path / "r2.json"
    round_1.write_text(
        json.dumps(
            _light_observation(
                "sample-light",
                1,
                n50=100,
                num_contigs=5,
            )
        ),
        encoding="utf-8",
    )
    round_2.write_text(
        json.dumps(
            _light_observation(
                "sample-light",
                2,
                n50=130,
                num_contigs=4,
            )
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "light-history",
        "--observation",
        str(round_2),
        "--observation",
        str(round_1),
        "--coverage-effective",
        "40",
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    history = json.loads(result.stdout)
    assert history["schema_version"] == "1.0.0"
    assert history["current_round"] == 2
    assert history["coverage_effective"] == 40.0
    assert [row["round"] for row in history["rows"]] == [1, 2]
    assert history["rows"][0]["delta_n50"] is None
    assert history["rows"][1]["delta_n50"] == 30.0
    assert history["rows"][1]["num_contigs_from_r1"] == -1.0


def test_light_history_rejects_mixed_sample_observations(tmp_path):
    round_1 = tmp_path / "r1.json"
    round_2 = tmp_path / "r2.json"
    round_1.write_text(
        json.dumps(_light_observation("sample-a", 1, n50=100, num_contigs=5)),
        encoding="utf-8",
    )
    round_2.write_text(
        json.dumps(_light_observation("sample-b", 2, n50=130, num_contigs=4)),
        encoding="utf-8",
    )

    result = _run_cli(
        "light-history",
        "--observation",
        str(round_1),
        "--observation",
        str(round_2),
        "--coverage-effective",
        "40",
        cwd=tmp_path,
    )

    assert result.returncode == 3
    assert result.stdout == ""
    assert "same sample_id" in result.stderr


def test_init_and_record_round_create_prospective_project(tmp_path):
    project = tmp_path / "prospective-study"
    initialized = _run_cli(
        "init",
        "--project-directory",
        str(project),
        "--project-id",
        "study-cli",
        "--platform",
        "ont",
        "--chemistry",
        "R10.4.1",
        "--basecaller",
        "Dorado",
        "--basecaller-version",
        "0.9.0",
        "--basecaller-model",
        "sup-v5",
        "--assembler",
        "Flye",
        "--assembler-version",
        "2.9.6",
        "--polisher",
        "Racon",
        "--polisher-version",
        "1.5.0",
        cwd=tmp_path,
    )

    assert initialized.returncode == 0
    assert initialized.stderr == ""
    assert json.loads(initialized.stdout)["decision_enabled"] is False
    assembly = tmp_path / "r1.fasta"
    assembly.write_text(">contig\nACGTACGT\n", encoding="utf-8")
    recorded = _run_cli(
        "record-round",
        "--project-directory",
        str(project),
        "--sample-id",
        "sample-cli",
        "--coverage-effective",
        "40",
        "--round",
        "1",
        "--assembly",
        str(assembly),
        cwd=tmp_path,
    )

    assert recorded.returncode == 0
    assert recorded.stderr == ""
    receipt = json.loads(recorded.stdout)
    assert receipt["decision_enabled"] is False
    assert Path(receipt["observation_file"]).is_file()
    assert Path(receipt["history_file"]).is_file()
    assert Path(receipt["record_file"]).is_file()


def test_init_rejects_incomplete_ont_provenance(tmp_path):
    result = _run_cli(
        "init",
        "--project-directory",
        str(tmp_path / "study"),
        "--project-id",
        "study-cli",
        "--platform",
        "ont",
        "--chemistry",
        "R10.4.1",
        "--assembler",
        "Flye",
        "--assembler-version",
        "2.9.6",
        "--polisher",
        "Racon",
        "--polisher-version",
        "1.5.0",
        cwd=tmp_path,
    )

    assert result.returncode == 3
    assert "basecalling provenance" in result.stderr


def test_explicit_manifest_does_not_require_repeating_model_path(tmp_path):
    result = _run_cli(
        "decide",
        "--input",
        "-",
        "--manifest",
        str(REPOSITORY / "models" / "model_manifest.v1.json"),
        cwd=tmp_path,
        stdin=json.dumps(_valid_metrics()),
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["feature_schema_version"] == "1.0.0"


def test_decide_rejects_missing_core_metrics(tmp_path):
    payload = _valid_metrics()
    payload.pop("error_rate")

    result = _run_cli(
        "decide",
        "--input",
        "-",
        cwd=tmp_path,
        stdin=json.dumps(payload),
    )

    assert result.returncode == 3
    assert "missing required decision metrics" in result.stderr


def test_validate_respects_incomplete_history_error_policy(tmp_path):
    payload = json.loads(TRAJECTORY_EXAMPLE.read_text(encoding="utf-8"))
    payload["incomplete_history_policy"] = "ERROR"
    payload["rounds"][1].pop("error_rate")

    result = _run_cli(
        "validate",
        "--input",
        "-",
        cwd=tmp_path,
        stdin=json.dumps(payload),
    )

    assert result.returncode == 3
    assert "incomplete trajectory metrics by round" in result.stderr


def test_missing_output_directory_has_stable_exit_code(tmp_path):
    result = _run_cli(
        "validate",
        "--input",
        str(TRAJECTORY_EXAMPLE),
        "--output",
        str(tmp_path / "missing" / "receipt.json"),
        cwd=tmp_path,
    )

    assert result.returncode == 5
    assert result.stdout == ""
    assert result.stderr.startswith("esdp: output error:")


def test_pyproject_separates_core_api_training_and_dev_dependencies():
    with (REPOSITORY / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)["project"]

    assert project["requires-python"] == ">=3.10"
    assert project["version"] == "2.0.0.dev0"
    assert "fastapi>=0.104.0,<1.0.0" not in project["dependencies"]
    assert set(project["optional-dependencies"]) == {"api", "training", "dev"}


def test_docker_runtime_exposes_cli_without_api_banner():
    dockerfile = (REPOSITORY / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (REPOSITORY / ".dockerignore").read_text(encoding="utf-8")
    entrypoint = (REPOSITORY / "docker-entrypoint.sh").read_text(
        encoding="utf-8"
    )

    assert "COPY --chown=esdp:esdp esdp_cli.py ." in dockerfile
    assert "COPY --chown=esdp:esdp esdp_light_metrics.py ." in dockerfile
    assert "COPY --chown=esdp:esdp esdp_light_history.py ." in dockerfile
    assert "COPY --chown=esdp:esdp esdp_instrumentation.py ." in dockerfile
    assert "ln -s /app/esdp_cli.py /usr/local/bin/esdp" in dockerfile
    assert '[ "${1:-}" = "esdp" ]' in entrypoint
    assert "!models/feature_names.txt" in dockerignore
    assert "!models/model_manifest.v1.json" in dockerignore
