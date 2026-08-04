"""Tests for prospective, causal ESDP-light trajectory features."""

import pandas as pd
import pandas.testing as pdt
import pytest

from esdp_light_features import LightFeatureBuilder


def trajectory() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sample": ["A", "A", "A", "B", "B"],
            "Coverage_effective": [30, 30, 30, 50, 50],
            "round": [1, 2, 3, 1, 2],
            "n50": [100, 130, 150, 200, 180],
            "num_contigs": [5, 4, 4, 2, 3],
            "total_length": [1000, 1010, 1015, 2000, 1990],
            "gc": [50.0, 50.2, 50.1, 45.0, 45.0],
        }
    )


def test_builder_creates_expected_prospective_features():
    built = LightFeatureBuilder().transform(trajectory())
    sample_a = built[built["Sample"] == "A"].reset_index(drop=True)

    assert pd.isna(sample_a.loc[0, "delta_n50"])
    assert sample_a.loc[1, "delta_n50"] == 30
    assert sample_a.loc[2, "delta_n50"] == 20
    assert sample_a.loc[2, "n50_from_r1"] == 50
    assert sample_a.loc[2, "delta_n50_trend"] == -10
    assert sample_a.loc[2, "num_contigs_from_r1"] == -1
    assert sample_a.loc[2, "delta_gc"] == pytest.approx(-0.1)


def test_builder_is_invariant_to_unobserved_future_rounds():
    builder = LightFeatureBuilder()
    history = trajectory()
    prefix = history[history["round"] <= 2].copy()

    built_prefix = builder.transform(prefix)
    built_as_of = builder.transform_as_of(history, 2)

    pdt.assert_frame_equal(built_prefix, built_as_of)

    changed_future = history.copy()
    changed_future.loc[
        (changed_future["Sample"] == "A") & (changed_future["round"] == 3),
        ["n50", "num_contigs", "total_length", "gc"],
    ] = [99999, 99, 99999, 99.0]
    changed_as_of = builder.transform_as_of(changed_future, 2)
    pdt.assert_frame_equal(built_as_of, changed_as_of)


@pytest.mark.parametrize(
    "rounds, message",
    [
        ([2, 3], "start at round 1"),
        ([1, 3], "contiguous"),
        ([1, 1], "contiguous"),
        ([1.0, 1.5], "round must contain integers"),
    ],
)
def test_builder_rejects_invalid_round_sequences(rounds, message):
    frame = trajectory().iloc[: len(rounds)].copy()
    frame["round"] = rounds

    with pytest.raises(ValueError, match=message):
        LightFeatureBuilder().transform(frame)


def test_builder_requires_complete_numeric_inputs():
    frame = trajectory()
    frame.loc[0, "n50"] = None

    with pytest.raises(ValueError, match="complete numeric columns"):
        LightFeatureBuilder().transform(frame)


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("Sample", "", "non-empty string Sample"),
        ("Coverage_effective", 0, "coverage must be positive"),
        ("n50", float("inf"), "complete numeric columns"),
        ("num_contigs", 2.5, "num_contigs must contain integers"),
        ("total_length", 0, "count and length metrics must be positive"),
        ("gc", 100.1, "gc must be between"),
    ],
)
def test_builder_rejects_invalid_metric_domains(column, value, message):
    frame = trajectory()
    if isinstance(value, float) and pd.api.types.is_integer_dtype(frame[column]):
        frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        LightFeatureBuilder().transform(frame)
