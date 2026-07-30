#!/usr/bin/env python3
"""Generate the canonical prospective ESDP feature dataset."""

import logging

import pandas as pd
import yaml

from esdp_features import FeatureBuilder, FeatureBuilderConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

with open("config.yaml", "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

FEATURE_BUILDER = FeatureBuilder(
    FeatureBuilderConfig(
        plateau_relative_threshold=config["plateau"]["relative_threshold"],
    )
)


def load_data():
    """Load the merged polishing metrics dataset."""
    df = pd.read_csv(config["data"]["merged_csv"])
    logger.info("Loaded %s rows", len(df))
    return df


# Compatibility wrappers for scripts and tests that imported the v1 functions.
def add_basic_deltas(df):
    return FEATURE_BUILDER.add_basic_deltas(df)


def add_ratio_features(df):
    return FEATURE_BUILDER.add_ratio_features(df)


def add_cumulative_features(df):
    return FEATURE_BUILDER.add_cumulative_features(df)


def add_normalized_to_r1(df):
    return FEATURE_BUILDER.add_normalized_to_r1(df)


def add_trend_features(df):
    return FEATURE_BUILDER.add_trend_features(df)


def add_plateau_features(df):
    return FEATURE_BUILDER.add_plateau_features(df)


def add_polynomial_features(df, degree=2):
    return FEATURE_BUILDER.add_polynomial_features(df, degree=degree)


def add_domain_specific_features(df):
    return FEATURE_BUILDER.add_domain_specific_features(df)


def validate_features(df):
    return FEATURE_BUILDER.validate(df)


def engineer_features(df):
    """Build the complete prospective feature table."""
    return FEATURE_BUILDER.transform(df)


def main():
    """Run feature engineering and save the canonical dataset."""
    logger.info("=" * 60)
    logger.info("Starting Feature Engineering")
    logger.info("=" * 60)

    raw = load_data()
    engineered = engineer_features(raw)
    output_path = config["data"]["engineered_csv"]
    engineered.to_csv(output_path, index=False)

    logger.info("Original features: %s", len(raw.columns))
    logger.info("Engineered features: %s", len(engineered.columns))
    logger.info("Output saved to: %s", output_path)


if __name__ == "__main__":
    main()
