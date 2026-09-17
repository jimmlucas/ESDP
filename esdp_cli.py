#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
esdp_cli.py

Command-line interface for the ESDP v2 sequential polishing controller.

ESDP v2 evaluates the current Racon polishing state after rounds R1-R4 and
returns a STOP/CONTINUE decision.

Decision rule
-------------
    p_continue >= 0.45  -> CONTINUE
    p_continue <  0.45  -> STOP

Workflow action
---------------
    STOP at R1-R4:
        proceed to mandatory Medaka.

    CONTINUE at R1-R3:
        execute the next Racon round and re-evaluate.

    CONTINUE at R4:
        execute R5 and then proceed directly to mandatory Medaka.

Operational inference does NOT require:
- BUSCO
- QV
- mapping-derived error metrics
- bacterial genus
- future polishing states

Supported input modes
---------------------
1. JSON file
2. CSV file
3. stdin JSON

Examples
--------
Single JSON file:

    python esdp_cli.py \
        --input sample.json \
        --output decision.json

Batch CSV:

    python esdp_cli.py \
        --input samples.csv \
        --output decisions.csv

stdin:

    echo '{
      "sample_id": "sample_01",
      "round": 1,
      "coverage_est": 40,
      "expected_genome_size": 5000000,
      "raw_read_n50": 15000,
      "ai_cov_cv": 0.20,
      "current_n50": 4800000,
      "current_num_contigs": 2,
      "current_assembly_frac": 0.99,
      "delta_n50_last": null,
      "n50_from_R1": null
    }' | python esdp_cli.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from esdp_decide import (
    DEFAULT_MODEL_PATH,
    Decision,
    ESDPError,
    InputValidationError,
    ModelValidationError,
    PolishingState,
    decide,
)


# =============================================================================
# INPUT CONTRACT
# =============================================================================

ALLOWED_INPUT_FIELDS = {
    "sample_id",
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
}

REQUIRED_INPUT_FIELDS = {
    "round",
    "coverage_est",
    "expected_genome_size",
    "raw_read_n50",
    "ai_cov_cv",
    "current_n50",
    "current_num_contigs",
    "current_assembly_frac",
}

LEGACY_FORBIDDEN_FIELDS = {
    "qv",
    "error_rate",
    "busco_complete",
    "busco_fragmented",
    "busco_missing",
    "genus",
    "predicted_class",
    "predicted_strategy",
    "recommended_rounds",
    "confidence",
    "confidence_threshold",
    "force_conservative",
}


# =============================================================================
# HELPERS
# =============================================================================

def _none_if_missing(value: Any) -> Any:
    """
    Convert pandas/CSV missing values to None.

    JSON null values already arrive as None.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def _validate_input_keys(record: Dict[str, Any]) -> None:
    """
    Fail closed on unsupported or legacy operational fields.
    """

    keys = set(record)

    forbidden = sorted(keys & LEGACY_FORBIDDEN_FIELDS)

    if forbidden:
        raise InputValidationError(
            "Legacy/non-operational fields are not accepted by ESDP v2: "
            + ", ".join(forbidden)
        )

    unknown = sorted(keys - ALLOWED_INPUT_FIELDS)

    if unknown:
        raise InputValidationError(
            "Unknown input fields for ESDP v2: "
            + ", ".join(unknown)
        )

    missing = sorted(REQUIRED_INPUT_FIELDS - keys)

    if missing:
        raise InputValidationError(
            "Missing required ESDP v2 fields: "
            + ", ".join(missing)
        )


def _record_to_state(
    record: Dict[str, Any],
    index: int = 1,
) -> PolishingState:
    """
    Convert one JSON/CSV record into the frozen PolishingState contract.
    """

    _validate_input_keys(record)

    sample_id = record.get("sample_id")

    if sample_id is None or not str(sample_id).strip():
        sample_id = f"sample_{index}"

    return PolishingState(
        sample_id=str(sample_id),
        round=int(record["round"]),
        coverage_est=_none_if_missing(record.get("coverage_est")),
        expected_genome_size=_none_if_missing(
            record.get("expected_genome_size")
        ),
        raw_read_n50=_none_if_missing(record.get("raw_read_n50")),
        ai_cov_cv=_none_if_missing(record.get("ai_cov_cv")),
        current_n50=_none_if_missing(record.get("current_n50")),
        current_num_contigs=_none_if_missing(
            record.get("current_num_contigs")
        ),
        current_assembly_frac=_none_if_missing(
            record.get("current_assembly_frac")
        ),
        delta_n50_last=_none_if_missing(
            record.get("delta_n50_last")
        ),
        n50_from_R1=_none_if_missing(
            record.get("n50_from_R1")
        ),
    )


def _decision_to_dict(decision: Decision) -> Dict[str, Any]:
    """
    Convert Decision to a JSON/CSV serializable dictionary.
    """

    return decision.to_dict()


# =============================================================================
# INPUT
# =============================================================================

def read_input(
    input_path: str | None = None,
) -> Dict[str, Any] | List[Dict[str, Any]]:
    """
    Read JSON or CSV from file, or JSON from stdin.
    """

    if input_path is None:
        raw = sys.stdin.read()

        if not raw.strip():
            raise InputValidationError(
                "No input received on stdin."
            )

        data = json.loads(raw)

        if not isinstance(data, (dict, list)):
            raise InputValidationError(
                "stdin JSON must contain an object or a list of objects."
            )

        return data

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)

        if not isinstance(data, (dict, list)):
            raise InputValidationError(
                "JSON input must contain an object or a list of objects."
            )

        return data

    if suffix == ".csv":
        frame = pd.read_csv(path)
        return frame.to_dict(orient="records")

    raise ValueError(
        f"Unsupported input format: {suffix}. "
        "Use .json or .csv."
    )


# =============================================================================
# INFERENCE
# =============================================================================

def process_single(
    record: Dict[str, Any],
    model_path: str | Path,
    verify_model_sha256: bool = True,
    index: int = 1,
) -> Decision:
    """
    Run one ESDP v2 sequential decision.
    """

    state = _record_to_state(
        record,
        index=index,
    )

    return decide(
        state=state,
        model_path=model_path,
        verify_model_sha256=verify_model_sha256,
    )


def process_batch(
    records: Sequence[Dict[str, Any]],
    model_path: str | Path,
    verify_model_sha256: bool = True,
) -> List[Decision]:
    """
    Run ESDP v2 sequential inference for multiple records.
    """

    decisions: List[Decision] = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        decisions.append(
            process_single(
                record=record,
                model_path=model_path,
                verify_model_sha256=verify_model_sha256,
                index=index,
            )
        )

    return decisions


# =============================================================================
# OUTPUT
# =============================================================================

def write_output(
    decisions: Decision | Sequence[Decision],
    output_path: str | None = None,
    output_format: str = "json",
) -> None:
    """
    Write decisions to JSON, CSV, or stdout.
    """

    if isinstance(decisions, Decision):
        decision_list = [decisions]
    else:
        decision_list = list(decisions)

    rows = [
        _decision_to_dict(decision)
        for decision in decision_list
    ]

    if output_path is not None:
        suffix = Path(output_path).suffix.lower()

        if suffix == ".csv":
            output_format = "csv"
        elif suffix == ".json":
            output_format = "json"

    if output_format == "json":
        payload: Any

        if len(rows) == 1:
            payload = rows[0]
        else:
            payload = rows

        text = json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        )

        if output_path is None:
            print(text)
        else:
            Path(output_path).write_text(
                text + "\n",
                encoding="utf-8",
            )

        return

    if output_format == "csv":
        frame = pd.DataFrame(rows)

        if output_path is None:
            frame.to_csv(
                sys.stdout,
                index=False,
            )
        else:
            frame.to_csv(
                output_path,
                index=False,
            )

        return

    raise ValueError(
        f"Unsupported output format: {output_format}"
    )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "ESDP v2 sequential STOP/CONTINUE controller "
            "for adaptive Racon polishing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------

JSON input:

  python esdp_cli.py \
    --input sample.json \
    --output decision.json

CSV batch input:

  python esdp_cli.py \
    --input samples.csv \
    --output decisions.csv

stdin:

  echo '{
    "sample_id": "sample_01",
    "round": 1,
    "coverage_est": 40,
    "expected_genome_size": 5000000,
    "raw_read_n50": 15000,
    "ai_cov_cv": 0.20,
    "current_n50": 4800000,
    "current_num_contigs": 2,
    "current_assembly_frac": 0.99,
    "delta_n50_last": null,
    "n50_from_R1": null
  }' | python esdp_cli.py
""",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help=(
            "Input JSON or CSV file. "
            "If omitted, JSON is read from stdin."
        ),
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Output JSON or CSV file. "
            "If omitted, output is written to stdout."
        ),
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=("json", "csv"),
        default="json",
        help="Output format when writing to stdout. Default: json.",
    )

    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=(
            "Path to the frozen ESDP v2 model. "
            f"Default: {DEFAULT_MODEL_PATH}"
        ),
    )

    parser.add_argument(
        "--skip-model-sha256-check",
        action="store_true",
        help=(
            "Disable frozen model SHA256 verification. "
            "Not recommended for reproducible use."
        ),
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Write processing information to stderr.",
    )

    return parser


def main() -> int:
    """
    CLI entry point.
    """

    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.verbose:
            print(
                f"Input: {args.input or 'stdin'}",
                file=sys.stderr,
            )
            print(
                f"Model: {args.model_path}",
                file=sys.stderr,
            )

        input_data = read_input(
            args.input
        )

        verify_sha = not args.skip_model_sha256_check

        if isinstance(
            input_data,
            list,
        ):
            if not input_data:
                raise InputValidationError(
                    "Batch input contains no records."
                )

            if not all(
                isinstance(record, dict)
                for record in input_data
            ):
                raise InputValidationError(
                    "Batch JSON input must contain only objects."
                )

            decisions = process_batch(
                records=input_data,
                model_path=args.model_path,
                verify_model_sha256=verify_sha,
            )

            if args.verbose:
                print(
                    f"Processed {len(decisions)} decision states.",
                    file=sys.stderr,
                )

        else:
            if not isinstance(
                input_data,
                dict,
            ):
                raise InputValidationError(
                    "Single input must be a JSON object."
                )

            decisions = process_single(
                record=input_data,
                model_path=args.model_path,
                verify_model_sha256=verify_sha,
            )

            if args.verbose:
                print(
                    "Processed 1 decision state.",
                    file=sys.stderr,
                )

        write_output(
            decisions=decisions,
            output_path=args.output,
            output_format=args.format,
        )

        return 0

    except (
        InputValidationError,
        ModelValidationError,
        ESDPError,
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    except Exception as exc:
        print(
            f"ERROR: Unexpected failure: {exc}",
            file=sys.stderr,
        )

        if args.verbose:
            import traceback

            traceback.print_exc(
                file=sys.stderr
            )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())