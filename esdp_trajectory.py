"""Typed v2 contracts for trajectory-aware ESDP inference.

The v1 API accepts one metrics row and remains unchanged. This module defines
the prospective v2 contract shared by future API, CLI, and workflow adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Literal, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from esdp_features import FeatureBuilder


FEATURE_SCHEMA_VERSION = "2.0.0"
DEFAULT_REQUIRED_METRICS = (
    "qv",
    "busco_complete",
    "n50",
    "num_contigs",
    "error_rate",
    "total_length",
)


class IncompleteHistoryPolicy(str, Enum):
    """How a caller handles missing metrics in a structurally valid history."""

    ERROR = "ERROR"
    CONSERVATIVE_CONTINUE = "CONSERVATIVE_CONTINUE"


class DecisionAction(str, Enum):
    """Workflow action emitted by trajectory-aware inference."""

    STOP = "STOP"
    CONTINUE = "CONTINUE"
    CONSERVATIVE_CONTINUE = "CONSERVATIVE_CONTINUE"


class RoundMetrics(BaseModel):
    """Raw metrics observed after one polishing round."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    round: int = Field(ge=1, le=5)

    n50: int | None = Field(default=None, ge=1)
    qv: float | None = Field(default=None, ge=0)
    error_rate: float | None = Field(default=None, ge=0, le=1)
    busco_complete: float | None = Field(default=None, ge=0, le=100)
    num_contigs: int | None = Field(default=None, ge=1)
    total_length: int | None = Field(default=None, ge=1)
    assembly_frac: float | None = Field(default=None, ge=0)

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

    OBSERVATION_FIELDS: ClassVar[tuple[str, ...]] = (
        "n50",
        "qv",
        "error_rate",
        "busco_complete",
        "num_contigs",
        "total_length",
        "assembly_frac",
        "raw_total_bp",
        "raw_read_n50",
        "raw_mean_read_len",
        "ai_num_contigs",
        "ai_total_bp",
        "ai_mean_cov",
        "ai_median_cov",
        "ai_cov_cv",
        "ai_circular_n",
        "ai_circular_bp_frac",
        "ai_repeat_bp_frac",
        "ai_longest_len",
        "ai_longest_cov",
        "polish_mean_contig_cov",
        "align_err_consensus",
        "align_err_polishing",
        "ovlp_div_initial",
        "ovlp_median_div_first",
        "mean_edge_coverage",
    )

    @model_validator(mode="after")
    def require_observed_metric(self):
        if all(getattr(self, name) is None for name in self.OBSERVATION_FIELDS):
            raise ValueError("a round must contain at least one observed metric")
        return self


class PolishingTrajectory(BaseModel):
    """Ordered metrics history for one sample and coverage condition."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["2.0.0"] = FEATURE_SCHEMA_VERSION
    sample_id: str = Field(min_length=1)
    genus: str | None = None
    coverage: float | None = Field(default=None, ge=0)
    coverage_effective: float | None = Field(default=None, ge=0)
    max_rounds: int = Field(default=5, ge=1, le=5)
    incomplete_history_policy: IncompleteHistoryPolicy = (
        IncompleteHistoryPolicy.CONSERVATIVE_CONTINUE
    )
    rounds: tuple[RoundMetrics, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_history(self):
        round_numbers = [metrics.round for metrics in self.rounds]
        if round_numbers != sorted(round_numbers):
            raise ValueError("rounds must be sorted in ascending order")
        if len(round_numbers) != len(set(round_numbers)):
            raise ValueError("round numbers must be unique")
        if round_numbers[0] != 1:
            raise ValueError("trajectory history must start at round 1")

        expected = list(range(1, round_numbers[-1] + 1))
        if round_numbers != expected:
            raise ValueError("trajectory rounds must be contiguous")
        if round_numbers[-1] > self.max_rounds:
            raise ValueError("current round cannot exceed max_rounds")
        return self

    @property
    def current_round(self) -> int:
        return self.rounds[-1].round

    def missing_metrics(
        self,
        required_metrics: Sequence[str] = DEFAULT_REQUIRED_METRICS,
    ) -> dict[int, list[str]]:
        """Return required metrics absent from each round."""
        unknown = set(required_metrics) - set(RoundMetrics.model_fields)
        if unknown:
            raise ValueError(f"unknown required metrics: {sorted(unknown)}")

        missing = {}
        for metrics in self.rounds:
            absent = [
                name
                for name in required_metrics
                if getattr(metrics, name) is None
            ]
            if absent:
                missing[metrics.round] = absent
        return missing

    def to_feature_frame(self) -> pd.DataFrame:
        """Convert the typed history to the canonical FeatureBuilder table."""
        coverage_group = (
            self.coverage_effective
            if self.coverage_effective is not None
            else self.coverage
            if self.coverage is not None
            else "UNKNOWN"
        )
        records = []
        for metrics in self.rounds:
            record = metrics.model_dump()
            record.update(
                {
                    "Sample": self.sample_id,
                    "Genus": self.genus,
                    "Coverage": coverage_group,
                }
            )
            records.append(record)
        return pd.DataFrame(records)


@dataclass(frozen=True)
class TrajectoryFeatureResult:
    """Feature table plus the completeness state used by decision adapters."""

    features: pd.DataFrame
    missing_metrics: dict[int, list[str]]

    @property
    def can_predict(self) -> bool:
        return not self.missing_metrics

    @property
    def latest(self) -> pd.Series:
        return self.features.iloc[-1]


class IncompleteTrajectoryError(ValueError):
    """Raised when an incomplete trajectory is configured to fail closed."""


def build_trajectory_features(
    trajectory: PolishingTrajectory,
    builder: FeatureBuilder | None = None,
    required_metrics: Sequence[str] = DEFAULT_REQUIRED_METRICS,
) -> TrajectoryFeatureResult:
    """Validate completeness and build prospective features for a trajectory."""
    missing = trajectory.missing_metrics(required_metrics)
    if (
        missing
        and trajectory.incomplete_history_policy is IncompleteHistoryPolicy.ERROR
    ):
        raise IncompleteTrajectoryError(
            f"incomplete trajectory metrics by round: {missing}"
        )

    feature_builder = builder or FeatureBuilder()
    features = feature_builder.transform(trajectory.to_feature_frame())
    return TrajectoryFeatureResult(
        features=features.reset_index(drop=True),
        missing_metrics=missing,
    )


class DecisionV2(BaseModel):
    """Validated workflow-facing decision contract for ESDP v2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0.0"] = FEATURE_SCHEMA_VERSION
    sample_id: str = Field(min_length=1)
    action: DecisionAction
    current_round: int = Field(ge=1, le=5)
    next_round: int | None = Field(default=None, ge=1, le=5)
    recommended_final_round: int = Field(ge=1, le=5)
    max_rounds: int = Field(default=5, ge=1, le=5)
    reason: str = Field(min_length=1)
    warnings: tuple[str, ...] = ()
    feature_schema_version: Literal["2.0.0"] = FEATURE_SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_action(self):
        if self.current_round > self.max_rounds:
            raise ValueError("current_round cannot exceed max_rounds")
        if self.recommended_final_round < self.current_round:
            raise ValueError(
                "recommended_final_round cannot precede current_round"
            )
        if self.recommended_final_round > self.max_rounds:
            raise ValueError(
                "recommended_final_round cannot exceed max_rounds"
            )

        if self.action is DecisionAction.STOP:
            if self.next_round is not None:
                raise ValueError("STOP decisions cannot define next_round")
            if self.recommended_final_round != self.current_round:
                raise ValueError(
                    "STOP decisions must select the current round"
                )
        else:
            if self.current_round >= self.max_rounds:
                raise ValueError("cannot continue at or after max_rounds")
            if self.next_round != self.current_round + 1:
                raise ValueError(
                    "continuation decisions require next_round=current_round+1"
                )
            if self.recommended_final_round < self.next_round:
                raise ValueError(
                    "recommended_final_round cannot precede next_round"
                )
        return self


def conservative_fallback_decision(
    trajectory: PolishingTrajectory,
    reason: str,
) -> DecisionV2:
    """Continue safely when possible, otherwise stop at the configured limit."""
    if trajectory.current_round >= trajectory.max_rounds:
        return DecisionV2(
            sample_id=trajectory.sample_id,
            action=DecisionAction.STOP,
            current_round=trajectory.current_round,
            recommended_final_round=trajectory.current_round,
            max_rounds=trajectory.max_rounds,
            reason=reason,
            warnings=["Conservative continuation unavailable at max_rounds"],
        )

    return DecisionV2(
        sample_id=trajectory.sample_id,
        action=DecisionAction.CONSERVATIVE_CONTINUE,
        current_round=trajectory.current_round,
        next_round=trajectory.current_round + 1,
        recommended_final_round=trajectory.max_rounds,
        max_rounds=trajectory.max_rounds,
        reason=reason,
        warnings=["Incomplete metrics: conservative continuation applied"],
    )
