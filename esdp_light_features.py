"""Prospective history features for ESDP-light candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from esdp_light import LIGHT_PROSPECTIVE_DERIVED_FEATURES


LIGHT_DYNAMIC_METRICS = ("n50", "num_contigs", "total_length", "gc")


@dataclass(frozen=True)
class LightFeatureBuilder:
    """Build temporal features using only rounds observed so far."""

    @staticmethod
    def _coverage_column(frame: pd.DataFrame) -> str:
        if "Coverage_effective" in frame:
            return "Coverage_effective"
        if "Coverage" in frame:
            return "Coverage"
        raise ValueError("light history requires Coverage or Coverage_effective")

    def transform(self, history: pd.DataFrame) -> pd.DataFrame:
        required = {"Sample", "round", *LIGHT_DYNAMIC_METRICS}
        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"light history missing columns: {sorted(missing)}")
        coverage = self._coverage_column(history)
        if history.empty:
            raise ValueError("light history cannot be empty")
        valid_samples = history["Sample"].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if not valid_samples.all():
            raise ValueError("light history requires non-empty string Sample values")
        numeric_columns = [coverage, "round", *LIGHT_DYNAMIC_METRICS]
        invalid_numeric = [
            column
            for column in numeric_columns
            if not pd.api.types.is_numeric_dtype(history[column])
            or history[column].isna().any()
            or not np.isrealobj(history[column].to_numpy())
            or not np.isfinite(history[column].to_numpy(dtype=float)).all()
        ]
        if invalid_numeric:
            raise ValueError(
                "light history requires complete numeric columns: "
                f"{invalid_numeric}"
            )
        if (history[coverage] <= 0).any():
            raise ValueError("light history coverage must be positive")
        for column in ("round", "n50", "num_contigs", "total_length"):
            values = history[column].to_numpy(dtype=float)
            if not np.equal(values, np.floor(values)).all():
                raise ValueError(f"light history {column} must contain integers")
        if (history[["n50", "num_contigs", "total_length"]] <= 0).any().any():
            raise ValueError("light assembly count and length metrics must be positive")
        if not history["gc"].between(0, 100, inclusive="both").all():
            raise ValueError("light history gc must be between 0 and 100")
        result = history.sort_values(["Sample", coverage, "round"]).reset_index(
            drop=True
        )

        group_columns = ["Sample", coverage]
        for _, group in result.groupby(group_columns, sort=False):
            rounds = group["round"].tolist()
            rounds = [int(round_value) for round_value in rounds]
            if rounds[0] != 1:
                raise ValueError("every light trajectory must start at round 1")
            if rounds != list(range(1, rounds[-1] + 1)):
                raise ValueError("light trajectory rounds must be contiguous")

        grouped = result.groupby(group_columns, sort=False)
        for metric in LIGHT_DYNAMIC_METRICS:
            result[f"delta_{metric}"] = grouped[metric].diff()
            baseline = grouped[metric].transform("first")
            result[f"{metric}_from_r1"] = result[metric] - baseline
            result[f"delta_{metric}_trend"] = result.groupby(
                group_columns,
                sort=False,
            )[f"delta_{metric}"].diff()

        produced = set(result.columns)
        absent_contract_features = set(LIGHT_PROSPECTIVE_DERIVED_FEATURES) - produced
        if absent_contract_features:
            raise RuntimeError(
                "light builder did not produce contracted features: "
                f"{sorted(absent_contract_features)}"
            )
        return result

    def transform_as_of(
        self,
        history: pd.DataFrame,
        round_number: int,
    ) -> pd.DataFrame:
        if round_number < 1:
            raise ValueError("round_number must be at least 1")
        return self.transform(history[history["round"] <= round_number].copy())
