#!/usr/bin/env python3
"""
esdp_decide.py

Sequential inference controller for ESDP
(Early Stop Decision Polishing).

This module implements the frozen sequential ESDP policy described in the
revised manuscript.

At each legal Racon decision state (R1-R4), the frozen model estimates:

    p_continue = P(CONTINUE | current and prior operational information)

Decision rule
-------------
    p_continue >= 0.45  -> CONTINUE
    p_continue <  0.45  -> STOP

Workflow action
---------------
    STOP at R1-R4:
        retain current Racon assembly and proceed to mandatory Medaka.

    CONTINUE at R1-R3:
        execute Racon round k+1 and re-evaluate.

    CONTINUE at R4:
        execute R5 and then proceed directly to mandatory Medaka.

No ESDP decision is made after R5.

Operational inference does NOT require:
- BUSCO execution
- a reference genome
- mapping-derived QV/error metrics
- bacterial genus
- sample identity as a predictor
- information from future polishing rounds

Frozen model
------------
Default artifact:
    outputs/frozen_sequential_model_v2/esdp_sequential_rf_v2.joblib

Frozen decision threshold:
    0.45

Expected artifact SHA256:
    9f9f08428242e98381546b9c79cd3d9b013c412b3edcbfe01ca84bb6f4c10dcf
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# FROZEN SEQUENTIAL POLICY
# =============================================================================

MODEL_VERSION = "esdp-sequential-rf-v2"

DECISION_THRESHOLD = 0.45

LEGAL_DECISION_ROUNDS = (1, 2, 3, 4)

MAX_RACON_ROUND = 5

FROZEN_FEATURES = (
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
)

FEATURE_SCHEMA_SHA256 = (
    "2f50cc9c6169c325da21e189b309623d872cbfdf4a025588c786c6e9f4576ced"
)

FROZEN_MODEL_SHA256 = (
    "9f9f08428242e98381546b9c79cd3d9b013c412b3edcbfe01ca84bb6f4c10dcf"
)

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "frozen_sequential_model_v2"
    / "esdp_sequential_rf_v2.joblib"
)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ESDPError(RuntimeError):
    """Base exception for ESDP inference."""


class ModelValidationError(ESDPError):
    """Raised when the frozen model artifact fails validation."""


class InputValidationError(ESDPError):
    """Raised when an inference request violates the operational contract."""


# =============================================================================
# INPUT CONTRACT
# =============================================================================

@dataclass(frozen=True)
class PolishingState:
    """
    Operational state supplied to ESDP at the current Racon round.

    sample_id is metadata only and is NEVER supplied to the model.

    Predictors
    ----------
    round
        Current Racon polishing round. Legal decision states are R1-R4.

    coverage_est
        Estimated sequencing coverage obtained from the Flye process.

    expected_genome_size
        Expected genome size in base pairs.

    raw_read_n50
        N50 of the raw sequencing reads.

    ai_cov_cv
        Coverage coefficient of variation derived from Flye assembly_info.txt:
            std(coverage) / mean(coverage)

    current_n50
        N50 of the current Racon-polished assembly.

    current_num_contigs
        Number of contigs in the current assembly.

    current_assembly_frac
        Current assembly length / expected genome size.

    delta_n50_last
        N50_k - N50_(k-1).
        At R1 this feature may be missing.

    n50_from_R1
        N50_k - N50_R1.
        At R1 this feature may be missing.
    """

    sample_id: str

    round: int

    coverage_est: Optional[float]
    expected_genome_size: Optional[float]
    raw_read_n50: Optional[float]
    ai_cov_cv: Optional[float]

    current_n50: Optional[float]
    current_num_contigs: Optional[float]
    current_assembly_frac: Optional[float]

    delta_n50_last: Optional[float] = None
    n50_from_R1: Optional[float] = None


# =============================================================================
# OUTPUT CONTRACT
# =============================================================================

@dataclass(frozen=True)
class Decision:
    """
    Structured ESDP sequential decision.

    decision
        STOP or CONTINUE.

    p_continue
        Frozen Random Forest probability assigned to CONTINUE.

    threshold
        Frozen operational threshold (0.45).

    next_action
        Concrete workflow action following the decision.
    """

    sample_id: str
    round: int

    p_continue: float
    threshold: float

    decision: str
    next_action: str

    model_version: str
    model_sha256: str
    feature_schema_sha256: str

    missing_features: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


# =============================================================================
# UTILITIES
# =============================================================================

def _sha256_file(path: Path) -> str:
    """Calculate SHA256 for a file."""

    digest = sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _normalize_numeric(
    value: Optional[float],
) -> float:
    """
    Convert operational values to numeric form.

    None and non-finite values are represented as NaN so that the
    preprocessing embedded in the frozen sklearn pipeline handles them
    consistently with model development.
    """

    if value is None:
        return np.nan

    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            f"Expected numeric predictor, received {value!r}."
        ) from exc

    if not np.isfinite(value):
        return np.nan

    return value


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_state(
    state: PolishingState,
) -> None:
    """Validate legal sequential inference state."""

    if not isinstance(state.sample_id, str):
        raise InputValidationError(
            "sample_id must be a string."
        )

    if not state.sample_id.strip():
        raise InputValidationError(
            "sample_id cannot be empty."
        )

    if state.round not in LEGAL_DECISION_ROUNDS:
        raise InputValidationError(
            "ESDP decisions are only defined after Racon rounds "
            f"R1-R4. Received round={state.round}. "
            "R5 is the maximum Racon budget and proceeds directly "
            "to mandatory Medaka without another ESDP decision."
        )

    # Quantities that should never be negative when present.
    non_negative_fields = (
        "coverage_est",
        "expected_genome_size",
        "raw_read_n50",
        "ai_cov_cv",
        "current_n50",
        "current_num_contigs",
        "current_assembly_frac",
    )

    for field_name in non_negative_fields:

        value = getattr(
            state,
            field_name,
        )

        if value is None:
            continue

        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise InputValidationError(
                f"{field_name} must be numeric."
            ) from exc

        if np.isfinite(numeric) and numeric < 0:
            raise InputValidationError(
                f"{field_name} cannot be negative."
            )


# =============================================================================
# FEATURE CONSTRUCTION
# =============================================================================

def prepare_features(
    state: PolishingState,
) -> pd.DataFrame:
    """
    Construct the exact frozen 10-feature vector.

    No retrospective or outcome-derived metric is created here.

    In particular, this function does NOT calculate or consume:
    BUSCO, QV, reference-derived errors, genus, or future-round information.
    """

    validate_state(state)

    values = {
        "round": float(state.round),
        "coverage_est":
            _normalize_numeric(state.coverage_est),
        "expected_genome_size":
            _normalize_numeric(state.expected_genome_size),
        "raw_read_n50":
            _normalize_numeric(state.raw_read_n50),
        "ai_cov_cv":
            _normalize_numeric(state.ai_cov_cv),
        "current_n50":
            _normalize_numeric(state.current_n50),
        "current_num_contigs":
            _normalize_numeric(state.current_num_contigs),
        "current_assembly_frac":
            _normalize_numeric(state.current_assembly_frac),
        "delta_n50_last":
            _normalize_numeric(state.delta_n50_last),
        "n50_from_R1":
            _normalize_numeric(state.n50_from_R1),
    }

    frame = pd.DataFrame(
        [values],
        columns=list(FROZEN_FEATURES),
    )

    # Fail closed if the operational schema is altered accidentally.
    if tuple(frame.columns) != FROZEN_FEATURES:
        raise InputValidationError(
            "Operational feature order does not match "
            "the frozen ESDP v2 feature contract."
        )

    return frame


# =============================================================================
# MODEL LOADING AND VALIDATION
# =============================================================================

def _extract_estimator(
    artifact: Any,
) -> Any:
    """
    Return the sklearn-compatible estimator from the serialized artifact.

    The canonical v2 artifact is expected to be directly callable through
    predict_proba(). This small compatibility layer also supports a wrapper
    dictionary containing a 'pipeline' or 'model' entry.
    """

    if hasattr(
        artifact,
        "predict_proba",
    ):
        return artifact

    if isinstance(
        artifact,
        dict,
    ):
        for key in (
            "pipeline",
            "model",
            "estimator",
        ):
            candidate = artifact.get(key)

            if (
                candidate is not None
                and hasattr(
                    candidate,
                    "predict_proba",
                )
            ):
                return candidate

    raise ModelValidationError(
        "Frozen artifact does not expose a sklearn-compatible "
        "predict_proba() estimator."
    )


def _validate_model_features(
    estimator: Any,
) -> None:
    """
    Validate feature names when exposed by sklearn.

    Older sklearn composites may not expose feature_names_in_ directly.
    In that case the explicit DataFrame contract remains authoritative.
    """

    names = getattr(
        estimator,
        "feature_names_in_",
        None,
    )

    if names is None:
        return

    observed = tuple(
        str(x)
        for x in names
    )

    if observed != FROZEN_FEATURES:
        raise ModelValidationError(
            "Frozen model feature contract mismatch.\n"
            f"Expected: {FROZEN_FEATURES}\n"
            f"Observed: {observed}"
        )


@lru_cache(maxsize=4)
def load_frozen_model(
    model_path: str,
    verify_sha256: bool = True,
) -> tuple[Any, str]:
    """
    Load and validate the frozen ESDP v2 model.

    The result is cached so API requests do not reload the model from disk.
    """

    path = Path(
        model_path
    ).expanduser().resolve()

    if not path.exists():
        raise ModelValidationError(
            f"Frozen ESDP model not found: {path}"
        )

    observed_sha = _sha256_file(
        path
    )

    if (
        verify_sha256
        and observed_sha != FROZEN_MODEL_SHA256
    ):
        raise ModelValidationError(
            "Frozen model SHA256 mismatch.\n"
            f"Expected: {FROZEN_MODEL_SHA256}\n"
            f"Observed: {observed_sha}\n"
            f"Artifact: {path}"
        )

    artifact = joblib.load(
        path
    )

    estimator = _extract_estimator(
        artifact
    )

    _validate_model_features(
        estimator
    )

    return estimator, observed_sha


# =============================================================================
# PROBABILITY EXTRACTION
# =============================================================================

def _continue_probability(
    estimator: Any,
    X: pd.DataFrame,
) -> float:
    """
    Return probability assigned to the CONTINUE class.

    The sequential target is binary:
        0 = STOP
        1 = CONTINUE
    """

    probabilities = estimator.predict_proba(
        X
    )

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.shape != (1, 2):
        raise ModelValidationError(
            "Sequential ESDP model must return exactly two "
            "class probabilities (STOP, CONTINUE). "
            f"Observed shape: {probabilities.shape}"
        )

    classes = getattr(
        estimator,
        "classes_",
        None,
    )

    # sklearn Pipeline exposes classes_ from the final estimator.
    if classes is None:
        raise ModelValidationError(
            "Frozen classifier does not expose classes_."
        )

    classes = list(classes)

    if 1 not in classes:
        raise ModelValidationError(
            "Frozen model does not contain CONTINUE class label 1. "
            f"Observed classes: {classes}"
        )

    continue_index = classes.index(1)

    p_continue = float(
        probabilities[
            0,
            continue_index,
        ]
    )

    if not np.isfinite(
        p_continue
    ):
        raise ModelValidationError(
            "Model returned a non-finite CONTINUE probability."
        )

    if not 0.0 <= p_continue <= 1.0:
        raise ModelValidationError(
            "Model returned p_continue outside [0, 1]."
        )

    return p_continue


# =============================================================================
# WORKFLOW DECISION
# =============================================================================

def _next_action(
    round_number: int,
    decision: str,
) -> str:
    """Translate STOP/CONTINUE into a concrete polishing action."""

    if decision == "STOP":
        return "MEDAKA"

    if decision != "CONTINUE":
        raise ESDPError(
            f"Unexpected decision: {decision}"
        )

    if round_number in (
        1,
        2,
        3,
    ):
        return (
            f"RACON_R{round_number + 1}"
        )

    if round_number == 4:
        return "RACON_R5_THEN_MEDAKA"

    raise InputValidationError(
        f"No sequential action is defined for R{round_number}."
    )


def decide(
    state: PolishingState,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    verify_model_sha256: bool = True,
) -> Decision:
    """
    Apply the frozen sequential ESDP policy.

    This function is the single source of truth for operational inference.

    No post-hoc rule layer or confidence override is applied.
    """

    X = prepare_features(
        state
    )

    estimator, model_sha = load_frozen_model(
        str(
            Path(model_path)
            .expanduser()
            .resolve()
        ),
        verify_model_sha256,
    )

    p_continue = _continue_probability(
        estimator,
        X,
    )

    # Frozen operational decision rule.
    if p_continue >= DECISION_THRESHOLD:
        decision = "CONTINUE"
    else:
        decision = "STOP"

    next_action = _next_action(
        state.round,
        decision,
    )

    missing_features = [
        feature
        for feature in FROZEN_FEATURES
        if pd.isna(
            X.iloc[0][feature]
        )
    ]

    return Decision(
        sample_id=state.sample_id,
        round=state.round,
        p_continue=p_continue,
        threshold=DECISION_THRESHOLD,
        decision=decision,
        next_action=next_action,
        model_version=MODEL_VERSION,
        model_sha256=model_sha,
        feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        missing_features=missing_features,
    )


# =============================================================================
# MODEL INFORMATION
# =============================================================================

def model_info(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    verify_model_sha256: bool = True,
) -> Dict[str, Any]:
    """
    Return frozen controller metadata.

    Useful for FastAPI /model/info and reproducibility audits.
    """

    estimator, model_sha = load_frozen_model(
        str(
            Path(model_path)
            .expanduser()
            .resolve()
        ),
        verify_model_sha256,
    )

    model_type = type(
        estimator
    ).__name__

    if hasattr(
        estimator,
        "named_steps",
    ):
        final_estimator = list(
            estimator.named_steps.values()
        )[-1]

        model_type = type(
            final_estimator
        ).__name__

    return {
        "model_version":
            MODEL_VERSION,
        "model_type":
            model_type,
        "decision_threshold":
            DECISION_THRESHOLD,
        "legal_decision_rounds":
            list(LEGAL_DECISION_ROUNDS),
        "maximum_racon_round":
            MAX_RACON_ROUND,
        "feature_count":
            len(FROZEN_FEATURES),
        "features":
            list(FROZEN_FEATURES),
        "feature_schema_sha256":
            FEATURE_SCHEMA_SHA256,
        "model_sha256":
            model_sha,
        "model_path":
            str(
                Path(model_path)
                .expanduser()
                .resolve()
            ),
    }


# =============================================================================
# SIMPLE CLI SMOKE TEST
# =============================================================================

if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Run one ESDP sequential STOP/CONTINUE decision."
        )
    )

    parser.add_argument(
        "--sample-id",
        required=True,
    )

    parser.add_argument(
        "--round",
        required=True,
        type=int,
        choices=LEGAL_DECISION_ROUNDS,
    )

    parser.add_argument(
        "--coverage-est",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--expected-genome-size",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--raw-read-n50",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--ai-cov-cv",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--current-n50",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--current-num-contigs",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--current-assembly-frac",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--delta-n50-last",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--n50-from-r1",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--skip-sha-check",
        action="store_true",
        help=(
            "Disable frozen model SHA256 verification. "
            "Intended only for development/testing."
        ),
    )

    args = parser.parse_args()

    state = PolishingState(
        sample_id=args.sample_id,
        round=args.round,
        coverage_est=args.coverage_est,
        expected_genome_size=args.expected_genome_size,
        raw_read_n50=args.raw_read_n50,
        ai_cov_cv=args.ai_cov_cv,
        current_n50=args.current_n50,
        current_num_contigs=args.current_num_contigs,
        current_assembly_frac=args.current_assembly_frac,
        delta_n50_last=args.delta_n50_last,
        n50_from_R1=args.n50_from_r1,
    )

    result = decide(
        state,
        model_path=args.model,
        verify_model_sha256=(
            not args.skip_sha_check
        ),
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )
