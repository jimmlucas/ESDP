#!/usr/bin/env python3

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "esdp_cli.py"

MODEL_SHA256 = (
    "9f9f08428242e98381546b9c79cd3d9b013c412b3edcbfe01ca84bb6f4c10dcf"
)

FEATURE_SCHEMA_SHA256 = (
    "2f50cc9c6169c325da21e189b309623d872cbfdf4a025588c786c6e9f4576ced"
)


def reference_payload():
    return {
        "sample_id": "cli_test",
        "round": 1,
        "coverage_est": 40,
        "expected_genome_size": 5000000,
        "raw_read_n50": 15000,
        "ai_cov_cv": 0.20,
        "current_n50": 4800000,
        "current_num_contigs": 2,
        "current_assembly_frac": 0.99,
        "delta_n50_last": None,
        "n50_from_R1": None,
    }


def run_cli(
    args=None,
    stdin_payload=None,
):
    if args is None:
        args = []

    command = [
        sys.executable,
        str(CLI),
        *args,
    ]

    if stdin_payload is None:
        input_text = None
    else:
        input_text = json.dumps(stdin_payload)

    return subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        check=False,
    )


def test_cli_reference_case_from_stdin():
    result = run_cli(
        stdin_payload=reference_payload()
    )

    assert result.returncode == 0, result.stderr

    data = json.loads(result.stdout)

    assert data["sample_id"] == "cli_test"
    assert data["round"] == 1

    assert data["p_continue"] == pytest.approx(
        0.5939402938271254,
        abs=1e-12,
    )

    assert data["threshold"] == 0.45
    assert data["decision"] == "CONTINUE"
    assert data["next_action"] == "RACON_R2"

    assert data["model_version"] == "esdp-sequential-rf-v2"

    assert data["model_sha256"] == MODEL_SHA256
    assert data["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256

    assert data["missing_features"] == [
        "delta_n50_last",
        "n50_from_R1",
    ]


def test_cli_json_file_input(tmp_path):
    input_path = tmp_path / "sample.json"
    output_path = tmp_path / "decision.json"

    input_path.write_text(
        json.dumps(reference_payload()),
        encoding="utf-8",
    )

    result = run_cli(
        args=[
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    data = json.loads(
        output_path.read_text(
            encoding="utf-8",
        )
    )

    assert data["decision"] == "CONTINUE"
    assert data["next_action"] == "RACON_R2"


def test_cli_csv_batch_input(tmp_path):
    input_path = tmp_path / "samples.csv"
    output_path = tmp_path / "decisions.csv"

    rows = [
        reference_payload(),
        {
            **reference_payload(),
            "sample_id": "cli_test_2",
            "round": 2,
            "delta_n50_last": 10000,
            "n50_from_R1": 15000,
        },
    ]

    fieldnames = list(rows[0].keys())

    with input_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    result = run_cli(
        args=[
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        output_rows = list(
            csv.DictReader(handle)
        )

    assert len(output_rows) == 2

    assert output_rows[0]["sample_id"] == "cli_test"
    assert output_rows[1]["sample_id"] == "cli_test_2"


def test_cli_rejects_busco():
    payload = reference_payload()
    payload["busco_complete"] = 98.5

    result = run_cli(
        stdin_payload=payload
    )

    assert result.returncode != 0

    assert "busco_complete" in result.stderr
    assert "not accepted" in result.stderr


def test_cli_rejects_qv():
    payload = reference_payload()
    payload["qv"] = 40.0

    result = run_cli(
        stdin_payload=payload
    )

    assert result.returncode != 0

    assert "qv" in result.stderr
    assert "not accepted" in result.stderr


def test_cli_rejects_genus():
    payload = reference_payload()
    payload["genus"] = "Escherichia"

    result = run_cli(
        stdin_payload=payload
    )

    assert result.returncode != 0

    assert "genus" in result.stderr
    assert "not accepted" in result.stderr


def test_cli_rejects_r5():
    payload = reference_payload()
    payload["round"] = 5

    result = run_cli(
        stdin_payload=payload
    )

    assert result.returncode != 0

    assert "R1-R4" in result.stderr


def test_cli_rejects_unknown_field():
    payload = reference_payload()
    payload["unexpected_feature"] = 123

    result = run_cli(
        stdin_payload=payload
    )

    assert result.returncode != 0

    assert "unexpected_feature" in result.stderr
    assert "Unknown input fields" in result.stderr


def test_cli_assigns_sample_id_when_missing():
    payload = reference_payload()
    payload.pop("sample_id")

    result = run_cli(
        stdin_payload=payload
    )

    assert result.returncode == 0, result.stderr

    data = json.loads(result.stdout)

    assert data["sample_id"] == "sample_1"


def test_cli_rejects_missing_required_feature():
    payload = reference_payload()
    payload.pop("current_n50")

    result = run_cli(
        stdin_payload=payload
    )

    assert result.returncode != 0

    assert "current_n50" in result.stderr
    assert "Missing required" in result.stderr


def test_cli_empty_stdin_is_rejected():
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
        ],
        input="",
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode != 0
    assert "No input received on stdin" in result.stderr


def test_cli_r4_action_is_valid():
    payload = reference_payload()

    payload.update(
        {
            "sample_id": "r4_test",
            "round": 4,
            "delta_n50_last": 1000,
            "n50_from_R1": 5000,
        }
    )

    result = run_cli(
        stdin_payload=payload
    )

    assert result.returncode == 0, result.stderr

    data = json.loads(result.stdout)

    assert data["round"] == 4
    assert data["decision"] in {
        "STOP",
        "CONTINUE",
    }

    if data["decision"] == "CONTINUE":
        assert data["next_action"] == "RACON_R5_THEN_MEDAKA"
    else:
        assert data["next_action"] == "MEDAKA"