"""Parity tests for the canonical offline and inference feature paths."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

from esdp_decide import (
    PolishingMetrics,
    engineer_features_online,
    prepare_features,
)
from esdp_features import (
    FeatureBuilder,
    FeatureBuilderConfig,
    align_model_features,
)


def _load_offline_module():
    module_path = Path(__file__).resolve().parents[1] / "3_feature_engineering.py"
    spec = spec_from_file_location("offline_feature_engineering", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


OFFLINE = _load_offline_module()


def test_alignment_replaces_infinities_without_changing_metadata():
    aligned = align_model_features(
        pd.DataFrame(
            {
                "qv": [np.inf, -np.inf],
                "label": ["first", "second"],
            }
        ),
        ["qv", "label"],
    )

    assert aligned["qv"].isna().all()
    assert aligned["label"].tolist() == ["first", "second"]


def _history():
    return pd.DataFrame(
        [
            {
                "Sample": "sample_1",
                "Coverage": "40X",
                "round": 1,
                "qv": 32.0,
                "busco_complete": 94.0,
                "n50": 2_000_000,
                "num_contigs": 5,
                "error_rate": 0.008,
                "assembly_frac": 0.98,
                "total_length": 4_900_000,
            },
            {
                "Sample": "sample_1",
                "Coverage": "40X",
                "round": 2,
                "qv": 34.0,
                "busco_complete": 96.0,
                "n50": 2_100_000,
                "num_contigs": 4,
                "error_rate": 0.006,
                "assembly_frac": 0.99,
                "total_length": 4_910_000,
            },
        ]
    )


def test_offline_adapter_matches_canonical_builder():
    history = _history()
    expected = FeatureBuilder(
        FeatureBuilderConfig(plateau_relative_threshold=0.12)
    ).transform(history)
    actual = OFFLINE.engineer_features(history)

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_refactor_preserves_frozen_features_except_plateau_correction():
    """Moving formulas must not change unrelated values in the v1 dataset."""
    repository = Path(__file__).resolve().parents[1]
    raw = pd.read_csv(repository / "data/all_samples_polishing_metrics.csv")
    frozen = pd.read_csv(repository / "data/training_dataset_engineered.csv")
    current = FeatureBuilder(
        FeatureBuilderConfig(plateau_relative_threshold=0.12)
    ).transform(raw)

    excluded = {"is_plateau", "plateau_streak"}
    assert list(current.columns) == list(frozen.columns)
    comparable_columns = [
        column
        for column in current.columns
        if column in frozen.columns and column not in excluded
    ]

    pd.testing.assert_frame_equal(
        current[comparable_columns].reset_index(drop=True),
        frozen[comparable_columns].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_single_round_online_path_uses_canonical_formulas():
    metrics = PolishingMetrics(
        sample_id="sample_1",
        round=1,
        coverage=40.0,
        qv=32.0,
        busco_complete=94.0,
        n50=2_000_000,
        num_contigs=5,
        error_rate=0.008,
        total_length=4_900_000,
    )
    online = engineer_features_online(metrics)

    canonical_record = {
        "Sample": "sample_1",
        "Coverage": 40.0,
        "round": 1,
        "qv": 32.0,
        "busco_complete": 94.0,
        "n50": 2_000_000,
        "num_contigs": 5,
        "error_rate": 0.008,
        "total_length": 4_900_000,
    }
    canonical = FeatureBuilder().transform(
        pd.DataFrame([canonical_record])
    ).iloc[0]

    for feature in [
        "busco_per_contig",
        "n50_fraction",
        "completeness_score",
        "is_plateau",
        "plateau_streak",
    ]:
        assert online[feature] == canonical[feature]


def test_v1_model_alignment_is_shared_without_changing_its_contract():
    metrics = PolishingMetrics(
        sample_id="sample_1",
        round=1,
        coverage=40.0,
        qv=32.0,
        busco_complete=94.0,
        n50=2_000_000,
        num_contigs=5,
        error_rate=0.008,
        total_length=4_900_000,
    )
    feature_names = [
        "qv",
        "busco_per_contig",
        "n50_fraction",
        "delta_qv",
    ]

    expected = align_model_features(
        {
            "coverage": 40.0,
            "qv": 32.0,
            "busco_complete": 94.0,
            "n50": 2_000_000,
            "num_contigs": 5,
            "error_rate": 0.008,
            "total_length": 4_900_000,
        },
        feature_names,
    )
    actual = prepare_features(metrics, feature_names)

    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert pd.isna(actual.loc[0, "busco_per_contig"])
    assert pd.isna(actual.loc[0, "n50_fraction"])
    assert pd.isna(actual.loc[0, "delta_qv"])


def test_isolated_later_round_does_not_invent_temporal_features():
    metrics = PolishingMetrics(
        sample_id="sample_1",
        round=3,
        coverage=40.0,
        qv=35.0,
        busco_complete=96.0,
        n50=2_100_000,
        num_contigs=4,
        error_rate=0.006,
        total_length=4_910_000,
    )

    online = engineer_features_online(metrics)

    temporal_features = [
        "delta_qv",
        "delta_busco_complete",
        "cost_benefit_ratio",
        "score_improvement",
        "gain_cumulative",
        "score_improvement_trend",
        "is_plateau",
        "plateau_streak",
        "polishing_effectiveness",
        "qv_from_r1",
    ]
    assert all(pd.isna(online[feature]) for feature in temporal_features)


def test_polynomial_builder_adds_pairwise_interactions():
    raw = pd.DataFrame(
        {
            "qv": [30.0],
            "busco_complete": [90.0],
            "error_rate": [0.01],
        }
    )

    engineered = FeatureBuilder.add_polynomial_features(raw)

    assert engineered.loc[0, "interaction_qv*busco_complete"] == 2_700.0
    assert engineered.loc[0, "interaction_qv*error_rate"] == 0.3
    assert engineered.loc[0, "interaction_busco_complete*error_rate"] == 0.9
