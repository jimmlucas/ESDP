"""Canonical prospective feature engineering for ESDP.

This module is the single source of truth for feature formulas used by
training and inference. Every feature produced for round ``r`` is restricted
to observations available at or before that round.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def _replace_infinite_with_nan(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities without pandas' deprecated silent downcasting."""
    result = frame.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    if len(numeric_columns):
        numeric = result.loc[:, numeric_columns]
        result.loc[:, numeric_columns] = numeric.mask(np.isinf(numeric))
    return result


@dataclass(frozen=True)
class FeatureBuilderConfig:
    """Configuration that affects canonical feature definitions."""

    plateau_relative_threshold: float = 0.12

    def __post_init__(self):
        if not 0 <= self.plateau_relative_threshold <= 1:
            raise ValueError("plateau_relative_threshold must be between 0 and 1")


class FeatureBuilder:
    """Build prospective ESDP features from polishing-round histories."""

    def __init__(self, config: FeatureBuilderConfig | None = None):
        self.config = config or FeatureBuilderConfig()

    @staticmethod
    def _coverage_column(df: pd.DataFrame) -> str:
        if "Coverage_effective" in df.columns:
            return "Coverage_effective"
        if "Coverage" in df.columns:
            return "Coverage"
        raise ValueError("Feature input requires Coverage or Coverage_effective")

    def add_basic_deltas(self, df: pd.DataFrame) -> pd.DataFrame:
        cov_col = self._coverage_column(df)
        result = df.sort_values(["Sample", cov_col, "round"]).reset_index(drop=True)

        for metric in ["qv", "busco_complete", "n50", "num_contigs", "error_rate"]:
            if metric in result.columns:
                result[f"delta_{metric}"] = result.groupby(
                    ["Sample", cov_col]
                )[metric].diff()

        if "delta_error_rate" in result.columns:
            result["delta_error_improvement"] = -result["delta_error_rate"]

        if "assembly_frac" in result.columns:
            result["assembly_error"] = (result["assembly_frac"] - 1.0).abs()
            result["delta_assembly_error"] = result.groupby(
                ["Sample", cov_col]
            )["assembly_error"].diff()

        return result

    @staticmethod
    def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if "delta_qv" in result.columns:
            result["qv_improvement_rate"] = result["delta_qv"] / result["round"]

        if {"busco_complete", "num_contigs"}.issubset(result.columns):
            result["busco_per_contig"] = (
                result["busco_complete"] / (result["num_contigs"] + 1)
            )

        if {"n50", "total_length"}.issubset(result.columns):
            result["n50_fraction"] = result["n50"] / (result["total_length"] + 1)

        if {"delta_qv", "delta_busco_complete"}.issubset(result.columns):
            improvement = (
                result["delta_qv"].fillna(0) * 0.5
                + result["delta_busco_complete"].fillna(0) * 0.5
            )
            cov_col = FeatureBuilder._coverage_column(result)
            has_round_one = result.groupby(
                ["Sample", cov_col],
            )["round"].transform(lambda rounds: rounds.eq(1).any())
            result["cost_benefit_ratio"] = (
                improvement / result["round"]
            ).where(has_round_one)

        return result

    def add_cumulative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        cov_col = self._coverage_column(result)

        for metric in ["delta_qv", "delta_busco_complete"]:
            if metric in result.columns:
                result[f"{metric}_cumsum"] = result.groupby(
                    ["Sample", cov_col]
                )[metric].cumsum()

        if {"delta_qv", "delta_busco_complete"}.issubset(result.columns):
            has_round_one = result.groupby(
                ["Sample", cov_col],
            )["round"].transform(lambda rounds: rounds.eq(1).any())
            delta_error = result.get(
                "delta_error_improvement",
                pd.Series(0.0, index=result.index),
            )
            delta_assembly = result.get(
                "delta_assembly_error",
                pd.Series(0.0, index=result.index),
            )
            score_improvement = (
                result["delta_qv"].fillna(0) * 0.4
                + result["delta_busco_complete"].fillna(0) * 0.3
                + delta_error.fillna(0) * 0.2
                + delta_assembly.fillna(0) * 0.1
            )
            result["score_improvement"] = score_improvement.where(has_round_one)
            result["gain_cumulative"] = result.groupby(
                ["Sample", cov_col]
            )["score_improvement"].cumsum()

        return result

    def add_normalized_to_r1(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        cov_col = self._coverage_column(result)

        for metric in [
            "qv",
            "n50",
            "error_rate",
            "busco_complete",
            "assembly_frac",
        ]:
            if metric not in result.columns:
                continue

            r1_values = (
                result[result["round"] == 1]
                .drop_duplicates(["Sample", cov_col])
                .set_index(["Sample", cov_col])[metric]
            )

            def difference_from_r1(row):
                current = row[metric]
                baseline = r1_values.get(
                    (row["Sample"], row[cov_col]),
                    np.nan,
                )
                if pd.isna(current) or pd.isna(baseline):
                    return np.nan
                return current - baseline

            result[f"{metric}_from_r1"] = result.apply(
                difference_from_r1,
                axis=1,
            )

        return result

    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        cov_col = self._coverage_column(result)

        for metric in [
            "delta_qv",
            "delta_busco_complete",
            "score_improvement",
        ]:
            if metric in result.columns:
                result[f"{metric}_trend"] = result.groupby(
                    ["Sample", cov_col]
                )[metric].diff()

        return result

    def add_plateau_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "score_improvement" not in df.columns:
            return df.copy()

        cov_col = self._coverage_column(df)

        def detect_plateau(group: pd.DataFrame) -> pd.DataFrame:
            result = group.sort_values("round").copy()
            running_max_gain = result["score_improvement"].clip(lower=0).cummax()
            threshold = (
                self.config.plateau_relative_threshold * running_max_gain
            )
            is_plateau = (
                threshold.gt(0)
                & result["score_improvement"].abs().lt(threshold)
            ).astype(float)
            result["is_plateau"] = is_plateau.where(
                result["score_improvement"].notna()
            )

            streak = []
            run = 0
            for value in result["is_plateau"]:
                if pd.isna(value):
                    run = 0
                    streak.append(np.nan)
                else:
                    run = run + 1 if value == 1 else 0
                    streak.append(float(run))
            result["plateau_streak"] = streak
            return result

        return df.groupby(
            ["Sample", cov_col],
            group_keys=False,
        ).apply(detect_plateau)

    @staticmethod
    def add_domain_specific_features(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if {"busco_complete", "error_rate"}.issubset(result.columns):
            result["completeness_score"] = (
                result["busco_complete"] / 100.0 * (1 - result["error_rate"])
            )

        if {
            "busco_complete",
            "num_contigs",
            "assembly_frac",
        }.issubset(result.columns):
            result["assembly_quality"] = (
                result["busco_complete"]
                / 100.0
                * (1 / (result["num_contigs"] + 1))
                * (1 - (result["assembly_frac"] - 1.0).abs())
            )

        if "score_improvement" in result.columns:
            result["polishing_effectiveness"] = (
                result["score_improvement"] / (result["round"] + 1)
            )

        return result

    @staticmethod
    def add_polynomial_features(
        df: pd.DataFrame,
        degree: int = 2,
    ) -> pd.DataFrame:
        result = df.copy()
        candidates = ["qv", "busco_complete", "error_rate"]
        available = [name for name in candidates if name in result.columns]
        if len(available) < 2:
            return result

        polynomial = PolynomialFeatures(
            degree=degree,
            include_bias=False,
            interaction_only=True,
        )
        transformed = polynomial.fit_transform(result[available].fillna(0))
        names = polynomial.get_feature_names_out(available)
        interaction_mask = polynomial.powers_.sum(axis=1) > 1
        interaction_names = [
            name.replace(" ", "*")
            for name, selected in zip(names, interaction_mask)
            if selected
        ]
        interaction_values = transformed[:, interaction_mask]

        for index, name in enumerate(interaction_names):
            result[f"interaction_{name}"] = interaction_values[:, index]

        return result

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        return _replace_infinite_with_nan(df)

    def transform(self, history: pd.DataFrame) -> pd.DataFrame:
        """Build all canonical features for one or more trajectories."""
        required = {"Sample", "round"}
        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"Missing required feature columns: {sorted(missing)}")

        result = self.add_basic_deltas(history.copy())
        result = self.add_ratio_features(result)
        result = self.add_cumulative_features(result)
        result = self.add_normalized_to_r1(result)
        result = self.add_trend_features(result)
        result = self.add_plateau_features(result)
        result = self.add_domain_specific_features(result)
        return self.validate(result)

    def transform_as_of(
        self,
        history: pd.DataFrame,
        round_number: int,
    ) -> pd.DataFrame:
        """Build features using only records available through a round."""
        if round_number < 1:
            raise ValueError("round_number must be at least 1")
        truncated = history[history["round"] <= round_number].copy()
        return self.transform(truncated)


def align_model_features(
    records: Mapping[str, Any] | Iterable[Mapping[str, Any]] | pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Align raw or engineered records to a model's canonical feature order."""
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, Mapping):
        frame = pd.DataFrame([records])
    else:
        frame = pd.DataFrame(list(records))

    aligned = frame.reindex(columns=list(feature_names))
    aligned = aligned.where(pd.notna(aligned), np.nan)
    return _replace_infinite_with_nan(aligned)
