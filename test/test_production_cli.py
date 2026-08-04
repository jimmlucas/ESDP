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
    assert "ln -s /app/esdp_cli.py /usr/local/bin/esdp" in dockerfile
    assert '[ "${1:-}" = "esdp" ]' in entrypoint
    assert "!models/feature_names.txt" in dockerignore
    assert "!models/model_manifest.v1.json" in dockerignore
