#!/usr/bin/env python3
"""Stable, workflow-oriented command-line interface for ESDP."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from esdp_decide import DEFAULT_MANIFEST_PATH, PolishingMetrics, decide
from esdp_light_metrics import LightMetricError, collect_light_round_observation
from esdp_manifest import (
    ManifestError,
    load_verified_model,
    verify_manifest_files,
)
from esdp_trajectory import (
    DEFAULT_REQUIRED_METRICS,
    IncompleteHistoryPolicy,
    PolishingTrajectory,
)


ESDP_CLI_VERSION = "2.0.0.dev0"
DECISION_SCHEMA_VERSION = "1.0.0"

EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_INVALID_INPUT = 3
EXIT_MODEL_ERROR = 4
EXIT_OUTPUT_ERROR = 5


class InputContractError(ValueError):
    """Raised when CLI input cannot satisfy its declared contract."""


class OutputWriteError(OSError):
    """Raised when a deterministic output cannot be written."""


class LegacyMetricsInput(BaseModel):
    """Strict input contract for the bundled legacy single-round model."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        strict=True,
    )

    sample_id: str = Field(min_length=1)
    genus: str | None = None
    round: int = Field(ge=1, le=5)

    coverage: float | None = Field(default=None, ge=0)
    coverage_effective: float | None = Field(default=None, ge=0)
    n50: float | None = Field(default=None, ge=1)
    qv: float | None = Field(default=None, ge=0)
    error_rate: float | None = Field(default=None, ge=0, le=1)
    busco_complete: float | None = Field(default=None, ge=0, le=100)
    num_contigs: int | None = Field(default=None, ge=1)
    total_length: int | None = Field(default=None, ge=1)

    raw_total_bp: float | None = Field(default=None, ge=0)
    raw_read_n50: float | None = Field(default=None, ge=0)
    raw_mean_read_len: float | None = Field(default=None, ge=0)

    ai_num_contigs: int | None = Field(default=None, ge=0)
    ai_total_bp: int | None = Field(default=None, ge=0)
    ai_mean_cov: float | None = Field(default=None, ge=0)
    ai_median_cov: float | None = Field(default=None, ge=0)
    ai_cov_cv: float | None = Field(default=None, ge=0)
    ai_circular_n: int | None = Field(default=None, ge=0)
    ai_circular_bp_frac: float | None = Field(default=None, ge=0, le=1)
    ai_repeat_bp_frac: float | None = Field(default=None, ge=0, le=1)
    ai_longest_len: int | None = Field(default=None, ge=0)
    ai_longest_cov: float | None = Field(default=None, ge=0)

    polish_mean_contig_cov: float | None = Field(default=None, ge=0)
    align_err_consensus: float | None = Field(default=None, ge=0, le=1)
    align_err_polishing: float | None = Field(default=None, ge=0, le=1)

    ovlp_div_initial: float | None = Field(default=None, ge=0)
    ovlp_median_div_first: float | None = Field(default=None, ge=0)
    mean_edge_coverage: float | None = Field(default=None, ge=0)

    delta_qv: float | None = None
    delta_busco_complete: float | None = None
    delta_error_rate: float | None = None
    qv_improvement_rate: float | None = None
    assembly_error: float | None = Field(default=None, ge=0)

    @field_validator("sample_id", mode="before")
    @classmethod
    def normalize_sample_id(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def require_core_decision_metrics(self):
        missing = [
            name
            for name in DEFAULT_REQUIRED_METRICS
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"missing required decision metrics: {missing}"
            )
        return self


def _canonical_json_value(value: Any) -> Any:
    """Normalize floating-point noise before deterministic serialization."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OutputWriteError("output contains a non-finite number")
        return round(value, 10)
    if isinstance(value, dict):
        return {
            key: _canonical_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    return value


def _json_text(payload: Any) -> str:
    try:
        return json.dumps(
            _canonical_json_value(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as error:
        raise OutputWriteError(f"output is not valid JSON: {error}") from error


def read_json(path_text: str) -> Any:
    """Read one JSON document from a file or stdin."""
    try:
        if path_text == "-":
            return json.load(sys.stdin)
        with Path(path_text).expanduser().open(encoding="utf-8") as input_file:
            return json.load(input_file)
    except json.JSONDecodeError as error:
        raise InputContractError(
            f"invalid JSON at line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from error
    except OSError as error:
        raise InputContractError(f"unable to read input {path_text}: {error}") from error


def write_json(payload: Any, path_text: str) -> None:
    """Write deterministic JSON, atomically when targeting a file."""
    output_text = _json_text(payload)
    if path_text == "-":
        sys.stdout.write(output_text)
        return

    output_path = Path(path_text).expanduser().resolve()
    temporary_path: Path | None = None
    try:
        if not output_path.parent.is_dir():
            raise OSError(
                f"parent directory does not exist: {output_path.parent}"
            )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            output_file.write(output_text)
            output_file.flush()
            os.fsync(output_file.fileno())
            temporary_path = Path(output_file.name)
        os.replace(temporary_path, output_path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise OutputWriteError(
            f"unable to write output {output_path}: {error}"
        ) from error


def _validate_metrics(payload: Any) -> LegacyMetricsInput:
    if not isinstance(payload, dict):
        raise InputContractError("metrics input must be one JSON object")
    try:
        return LegacyMetricsInput.model_validate(payload)
    except ValidationError as error:
        raise InputContractError(str(error)) from error


def _validate_trajectory(payload: Any) -> PolishingTrajectory:
    if not isinstance(payload, dict):
        raise InputContractError("trajectory input must be one JSON object")
    try:
        return PolishingTrajectory.model_validate(payload)
    except ValidationError as error:
        raise InputContractError(str(error)) from error


def _decision_payload(decision, current_round: int) -> dict[str, Any]:
    payload = asdict(decision)
    selected_final_round = max(current_round, decision.recommended_rounds)
    can_continue = current_round < selected_final_round and current_round < 5
    payload.update(
        {
            "decision_schema_version": DECISION_SCHEMA_VERSION,
            "workflow_action": "CONTINUE" if can_continue else "STOP",
            "current_round": current_round,
            "next_round": current_round + 1 if can_continue else None,
            "selected_final_round": selected_final_round,
        }
    )
    return payload


def command_decide(args: argparse.Namespace) -> dict[str, Any]:
    metrics_input = _validate_metrics(read_json(args.input))
    metrics = PolishingMetrics(**metrics_input.model_dump())
    decision = decide(
        metrics,
        model_path=args.model,
        manifest_path=args.manifest,
        confidence_threshold=args.confidence_threshold,
        force_conservative=args.force_conservative,
    )
    return _decision_payload(decision, metrics_input.round)


def _infer_validation_kind(payload: Any) -> Literal["metrics", "trajectory"]:
    if not isinstance(payload, dict):
        raise InputContractError("validation input must be one JSON object")
    if payload.get("schema_version") == "2.0.0" or "rounds" in payload:
        return "trajectory"
    return "metrics"


def command_validate(args: argparse.Namespace) -> dict[str, Any]:
    payload = read_json(args.input)
    kind = args.kind if args.kind != "auto" else _infer_validation_kind(payload)
    if kind == "trajectory":
        trajectory = _validate_trajectory(payload)
        missing_metrics = trajectory.missing_metrics()
        if (
            missing_metrics
            and trajectory.incomplete_history_policy
            is IncompleteHistoryPolicy.ERROR
        ):
            raise InputContractError(
                f"incomplete trajectory metrics by round: {missing_metrics}"
            )
        return {
            "complete": not missing_metrics,
            "current_round": trajectory.current_round,
            "kind": "trajectory",
            "missing_metrics": missing_metrics,
            "sample_id": trajectory.sample_id,
            "schema_version": trajectory.schema_version,
            "valid": True,
        }

    metrics = _validate_metrics(payload)
    return {
        "complete": True,
        "current_round": metrics.round,
        "kind": "metrics",
        "missing_metrics": {},
        "sample_id": metrics.sample_id,
        "schema_version": DECISION_SCHEMA_VERSION,
        "valid": True,
    }


def command_model_info(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest is not None
        else DEFAULT_MANIFEST_PATH
    )
    verified = load_verified_model(manifest_path)
    verified_files = verify_manifest_files(
        verified.manifest,
        manifest_path,
        include_training_data=args.verify_training_data,
    )
    return {
        "artifact_verified": True,
        "feature_names_verified": True,
        "manifest": verified.manifest.model_dump(mode="json"),
        "training_data_verified": args.verify_training_data,
        "verified_file_names": {
            key: path.name
            for key, path in verified_files.items()
            if key != "training_data" or args.verify_training_data
        },
    }


def command_light_observe(args: argparse.Namespace) -> dict[str, Any]:
    """Collect one model-independent, provenance-aware light observation."""
    observation = collect_light_round_observation(
        sample_id=args.sample_id,
        round_number=args.round_number,
        assembly_path=args.assembly,
        samtools_stats_path=args.samtools_stats,
        alignment_reference_path=args.alignment_reference,
    )
    return observation.model_dump(mode="json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="esdp",
        description="Workflow-oriented ESDP decision interface",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {ESDP_CLI_VERSION}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="report successful operations to stderr",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    decide_parser = subparsers.add_parser(
        "decide",
        help="make one decision with the bundled legacy model",
    )
    decide_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="metrics JSON path, or - for stdin",
    )
    decide_parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="decision JSON path, or - for stdout",
    )
    decide_parser.add_argument("--model", help="optional custom joblib model")
    decide_parser.add_argument("--manifest", help="optional model manifest")
    decide_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        choices=None,
        help="conservative escalation threshold in [0,1] (default: 0.5)",
    )
    decide_parser.add_argument(
        "--force-conservative",
        action="store_true",
        help="force the maximum-round recommendation",
    )
    decide_parser.set_defaults(handler=command_decide)

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate metrics or a v2 trajectory JSON document",
    )
    validate_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="input JSON path, or - for stdin",
    )
    validate_parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="validation receipt path, or - for stdout",
    )
    validate_parser.add_argument(
        "--kind",
        choices=("auto", "metrics", "trajectory"),
        default="auto",
    )
    validate_parser.set_defaults(handler=command_validate)

    info_parser = subparsers.add_parser(
        "model-info",
        help="verify and report the model manifest",
    )
    info_parser.add_argument("--manifest", help="optional model manifest")
    info_parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="model information path, or - for stdout",
    )
    info_parser.add_argument(
        "--verify-training-data",
        action="store_true",
        help="also verify the recorded training dataset checksum",
    )
    info_parser.set_defaults(handler=command_model_info)

    light_observe_parser = subparsers.add_parser(
        "light-observe",
        help="collect low-cost metrics from one polished assembly",
    )
    light_observe_parser.add_argument(
        "--sample-id",
        required=True,
        help="stable biological sample identifier",
    )
    light_observe_parser.add_argument(
        "--round",
        dest="round_number",
        required=True,
        type=int,
        help="current polishing round (1-5)",
    )
    light_observe_parser.add_argument(
        "--assembly",
        required=True,
        help="current polished assembly FASTA or FASTA.gz",
    )
    light_observe_parser.add_argument(
        "--samtools-stats",
        help="optional samtools stats generated for this round",
    )
    light_observe_parser.add_argument(
        "--alignment-reference",
        help="FASTA used as the optional alignment reference",
    )
    light_observe_parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="observation JSON path, or - for stdout",
    )
    light_observe_parser.set_defaults(handler=command_light_observe)
    return parser


def _validate_cli_options(args: argparse.Namespace) -> None:
    threshold = getattr(args, "confidence_threshold", None)
    if threshold is not None and not 0 <= threshold <= 1:
        raise InputContractError(
            "confidence-threshold must be between 0 and 1"
        )
    if (
        getattr(args, "model", None) is not None
        and getattr(args, "manifest", None) is None
    ):
        print(
            "esdp: warning: custom model is unverified without --manifest",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_cli_options(args)
        result = args.handler(args)
        write_json(result, args.output)
        if args.verbose:
            print(f"esdp: {args.command} completed", file=sys.stderr)
        return EXIT_SUCCESS
    except (InputContractError, LightMetricError) as error:
        print(f"esdp: invalid input: {error}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except ManifestError as error:
        print(f"esdp: model error: {error}", file=sys.stderr)
        return EXIT_MODEL_ERROR
    except OutputWriteError as error:
        print(f"esdp: output error: {error}", file=sys.stderr)
        return EXIT_OUTPUT_ERROR
    except Exception as error:
        print(
            f"esdp: inference error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        if args.verbose:
            import traceback

            traceback.print_exc(file=sys.stderr)
        return EXIT_MODEL_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
