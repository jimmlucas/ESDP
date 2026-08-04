"""Versioned aggregation contract for prospective ESDP-light observations."""

from __future__ import annotations

import math
from typing import Literal, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from esdp_light import LIGHT_PROSPECTIVE_DERIVED_FEATURES
from esdp_light_features import LIGHT_DYNAMIC_METRICS, LightFeatureBuilder
from esdp_light_metrics import (
    LIGHT_OBSERVATION_SCHEMA_VERSION,
    LightRoundObservation,
)


LIGHT_HISTORY_SCHEMA_VERSION = "1.0.0"
LIGHT_HISTORY_FEATURE_NAMES = (
    *LIGHT_DYNAMIC_METRICS,
    *LIGHT_PROSPECTIVE_DERIVED_FEATURES,
)


class LightHistoryError(ValueError):
    """Raised when observations cannot form one prospective trajectory."""


class LightFeatureRow(BaseModel):
    """Deployment-ready light features available after one polishing round."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    round: int = Field(ge=1, le=5)
    assembly_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n50: int = Field(ge=1)
    num_contigs: int = Field(ge=1)
    total_length: int = Field(ge=1)
    gc: float = Field(ge=0, le=100)

    delta_n50: float | None = None
    delta_num_contigs: float | None = None
    delta_total_length: float | None = None
    delta_gc: float | None = None
    n50_from_r1: float
    num_contigs_from_r1: float
    total_length_from_r1: float
    gc_from_r1: float
    delta_n50_trend: float | None = None
    delta_num_contigs_trend: float | None = None
    delta_total_length_trend: float | None = None
    delta_gc_trend: float | None = None


class LightFeatureHistory(BaseModel):
    """One sample and coverage trajectory, safe for prospective inference."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )

    schema_version: Literal["1.0.0"] = LIGHT_HISTORY_SCHEMA_VERSION
    observation_schema_version: Literal["1.0.0"] = (
        LIGHT_OBSERVATION_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1)
    coverage_effective: float = Field(gt=0)
    current_round: int = Field(ge=1, le=5)
    feature_names: tuple[str, ...]
    rows: tuple[LightFeatureRow, ...] = Field(min_length=1, max_length=5)

    @field_validator("feature_names", "rows", mode="before")
    @classmethod
    def normalize_json_arrays(cls, value):
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_history_contract(self):
        if self.feature_names != LIGHT_HISTORY_FEATURE_NAMES:
            raise ValueError("feature_names do not match the ESDP-light contract")
        rounds = [row.round for row in self.rows]
        if rounds != list(range(1, self.current_round + 1)):
            raise ValueError("history rows must be contiguous from R1 to current_round")
        return self


def _optional_float(value) -> float | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise LightHistoryError("derived light feature is not finite")
    return numeric


def build_light_feature_history(
    observations: Sequence[LightRoundObservation],
    *,
    coverage_effective: float,
) -> LightFeatureHistory:
    """Aggregate exact round observations without consulting future rounds."""
    if not observations:
        raise LightHistoryError("at least one light observation is required")
    if not math.isfinite(coverage_effective) or coverage_effective <= 0:
        raise LightHistoryError("coverage_effective must be finite and positive")
    if not all(isinstance(item, LightRoundObservation) for item in observations):
        raise LightHistoryError("all inputs must be LightRoundObservation objects")

    ordered = sorted(observations, key=lambda item: item.round)
    sample_ids = {item.sample_id for item in ordered}
    if len(sample_ids) != 1:
        raise LightHistoryError(
            "all light observations must belong to the same sample_id"
        )

    frame = pd.DataFrame(
        [
            {
                "Sample": item.sample_id,
                "Coverage_effective": coverage_effective,
                "round": item.round,
                "assembly_sha256": item.assembly_sha256,
                "n50": item.fasta.n50,
                "num_contigs": item.fasta.num_contigs,
                "total_length": item.fasta.total_length,
                "gc": item.fasta.gc_percent,
            }
            for item in ordered
        ]
    )
    try:
        features = LightFeatureBuilder().transform(frame)
    except ValueError as error:
        raise LightHistoryError(str(error)) from error

    rows = []
    for _, row in features.iterrows():
        rows.append(
            LightFeatureRow(
                round=int(row["round"]),
                assembly_sha256=str(row["assembly_sha256"]),
                n50=int(row["n50"]),
                num_contigs=int(row["num_contigs"]),
                total_length=int(row["total_length"]),
                gc=float(row["gc"]),
                **{
                    name: _optional_float(row[name])
                    for name in LIGHT_PROSPECTIVE_DERIVED_FEATURES
                },
            )
        )

    return LightFeatureHistory(
        sample_id=ordered[0].sample_id,
        coverage_effective=float(coverage_effective),
        current_round=ordered[-1].round,
        feature_names=LIGHT_HISTORY_FEATURE_NAMES,
        rows=tuple(rows),
    )
