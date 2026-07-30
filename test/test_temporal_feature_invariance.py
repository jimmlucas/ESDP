"""Tests that engineered features never depend on future polishing rounds."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest


def _load_feature_engineering_module():
    module_path = Path(__file__).resolve().parents[1] / "3_feature_engineering.py"
    spec = spec_from_file_location("esdp_feature_engineering", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FEATURE_ENGINEERING = _load_feature_engineering_module()


@pytest.fixture
def polishing_history():
    rows = []
    trajectories = {
        ("sample_a", "20X"): {
            "qv": [31.0, 33.0, 33.1, 33.11, 36.0],
            "busco_complete": [92.0, 94.0, 94.1, 94.1, 98.0],
            "n50": [1_000_000, 1_200_000, 1_205_000, 1_206_000, 1_500_000],
            "num_contigs": [8, 6, 6, 6, 4],
            "error_rate": [0.010, 0.007, 0.0069, 0.0069, 0.003],
            "assembly_frac": [0.95, 0.97, 0.971, 0.971, 0.99],
        },
        ("sample_b", "40X"): {
            "qv": [35.0, 35.2, 35.21, 35.21, 35.22],
            "busco_complete": [97.0, 97.1, 97.1, 97.1, 97.1],
            "n50": [2_000_000, 2_010_000, 2_011_000, 2_011_000, 2_012_000],
            "num_contigs": [3, 3, 3, 3, 3],
            "error_rate": [0.0040, 0.0039, 0.0039, 0.0039, 0.0039],
            "assembly_frac": [0.99, 0.991, 0.991, 0.991, 0.991],
        },
    }

    for (sample, coverage), metrics in trajectories.items():
        for index, round_number in enumerate(range(1, 6)):
            rows.append(
                {
                    "Sample": sample,
                    "Coverage": coverage,
                    "round": round_number,
                    "total_length": 5_000_000,
                    **{name: values[index] for name, values in metrics.items()},
                }
            )

    return pd.DataFrame(rows)


def _sorted_feature_table(df):
    return (
        FEATURE_ENGINEERING.engineer_features(df.copy())
        .sort_values(["Sample", "Coverage", "round"])
        .reset_index(drop=True)
    )


@pytest.mark.parametrize("cutoff_round", [1, 2, 3, 4, 5])
def test_features_are_invariant_to_future_rounds(polishing_history, cutoff_round):
    """Adding observations after a decision point must not change its features."""
    full = _sorted_feature_table(polishing_history)
    truncated_input = polishing_history[
        polishing_history["round"] <= cutoff_round
    ].copy()
    truncated = _sorted_feature_table(truncated_input)

    full_as_of_cutoff = (
        full[full["round"] <= cutoff_round]
        .reset_index(drop=True)
        .loc[:, truncated.columns]
    )

    pd.testing.assert_frame_equal(
        truncated,
        full_as_of_cutoff,
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_late_large_gain_does_not_change_earlier_plateau_state(polishing_history):
    """A large R5 gain must not redefine plateau thresholds for R1-R4."""
    through_r4 = _sorted_feature_table(
        polishing_history[polishing_history["round"] <= 4]
    )
    through_r5 = _sorted_feature_table(polishing_history)

    early_columns = ["Sample", "Coverage", "round", "is_plateau", "plateau_streak"]
    expected = through_r4[early_columns].reset_index(drop=True)
    actual = (
        through_r5[through_r5["round"] <= 4][early_columns]
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(expected, actual)


def test_first_round_does_not_claim_a_plateau(polishing_history):
    """A plateau requires observed improvement history beyond the first round."""
    first_round = _sorted_feature_table(
        polishing_history[polishing_history["round"] == 1]
    )

    assert first_round["is_plateau"].eq(0).all()
    assert first_round["plateau_streak"].eq(0).all()
