#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7_inference_pipeline.py - Production Inference Pipeline

Production-ready script for predicting optimal stopping rounds for new samples.

CORRECTIONS:
- Keeps internal model classes as 0-based (0,1,2)
- Returns human-readable/output classes as 1-based (1,2,3)
- Makes summary consistent with class_mapping
- Handles empty input CSV safely
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import logging
import yaml
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


class PolishingPredictor:
    """Production predictor for polishing round optimization."""

    def __init__(self, model_path=None, feature_names_path=None):
        """Initialize predictor with a bundled model pipeline artifact."""
        if model_path is None:
            model_path = Path(config["outputs"]["models_dir"]) / "best_model_pipeline.pkl"
        if feature_names_path is None:
            feature_names_path = Path(config["outputs"]["models_dir"]) / "feature_names.txt"

        logger.info(f"Loading model pipeline from {model_path}")
        self.model = joblib.load(model_path)

        # Try reading feature names from pipeline metadata; fallback to file
        self.feature_names = None
        if hasattr(self.model, "feature_names"):
            try:
                self.feature_names = list(self.model.feature_names)
            except Exception:
                self.feature_names = None

        if self.feature_names is None and feature_names_path and Path(feature_names_path).exists():
            logger.info(f"Loading feature names from {feature_names_path}")
            with open(feature_names_path, "r") as f:
                self.feature_names = [line.strip() for line in f if line.strip()]

        if self.feature_names is None:
            raise ValueError("Feature names not found in model artifact or feature names file")

        self.class_mapping = config["classes"]["class_mapping"]

        logger.info("Predictor initialized successfully")

    def prepare_features(self, sample_data):
        """
        Prepare feature vector from raw sample data.

        Important:
        - Keep missing values as NaN so the pipeline's imputer can handle them.
        - Keep the exact feature order expected by the model.
        """
        features = pd.DataFrame(np.nan, index=[0], columns=self.feature_names)

        for feature in self.feature_names:
            if feature in sample_data:
                features.at[0, feature] = sample_data[feature]

        # Replace infinities with NaN to be consistent with training/inference pipeline
        features = features.replace([np.inf, -np.inf], np.nan)

        return features

    def predict(self, sample_data, return_probabilities=False):
        """
        Predict optimal stopping class for a sample.

        Returns:
            predicted_class_index: internal model class (0,1,2)
            predicted_class: output class (1,2,3)
            class_name: human-readable class name
            probabilities/confidence: optional
        """
        X = self.prepare_features(sample_data)

        predicted = self.model.predict(X)
        predicted_class_index = int(predicted[0])   # 0,1,2 from model
        predicted_class = predicted_class_index + 1 # 1,2,3 for user/output

        class_name = self.class_mapping.get(predicted_class, str(predicted_class))

        result = {
            "predicted_class_index": predicted_class_index,
            "predicted_class": predicted_class,
            "class_name": class_name,
        }

        if return_probabilities and hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[0]

            try:
                class_labels = getattr(self.model, "classes_", None)
                if class_labels is None and hasattr(self.model, "named_steps") and "model" in self.model.named_steps:
                    class_labels = self.model.named_steps["model"].classes_
            except Exception:
                class_labels = None

            if class_labels is not None:
                # Convert model classes 0,1,2 -> output classes 1,2,3
                probs = {int(lbl) + 1: float(p) for lbl, p in zip(class_labels, probabilities)}
            else:
                probs = {i + 1: float(p) for i, p in enumerate(probabilities)}

            result["probabilities"] = probs
            result["confidence"] = float(max(probabilities))

        if hasattr(self.model, "model_version"):
            result["model_version"] = getattr(self.model, "model_version")

        return result

    def predict_from_csv(self, csv_path, output_path=None):
        """
        Predict for multiple samples from CSV.

        Args:
            csv_path: Path to input CSV with sample data
            output_path: Path to save predictions (optional)

        Returns:
            DataFrame with predictions
        """
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)

        if df.empty:
            logger.warning("Input CSV is empty. Returning empty DataFrame.")
            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Saved empty predictions file to {output_path}")
            return df

        predictions = []
        for _, row in df.iterrows():
            sample_data = row.to_dict()
            result = self.predict(sample_data, return_probabilities=True)
            predictions.append(result)

        df["predicted_class_index"] = [p["predicted_class_index"] for p in predictions]
        df["predicted_class"] = [p["predicted_class"] for p in predictions]
        df["predicted_class_name"] = [p["class_name"] for p in predictions]

        if all("confidence" in p for p in predictions):
            df["prediction_confidence"] = [p["confidence"] for p in predictions]

        if all("model_version" in p for p in predictions):
            df["model_version"] = [p["model_version"] for p in predictions]

        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")

        return df


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(description="Predict optimal polishing stopping round")
    parser.add_argument("--input", required=True, help="Input CSV file with sample data")
    parser.add_argument("--output", help="Output CSV file for predictions")
    parser.add_argument("--model", help="Path to bundled model pipeline (default: models/best_model_pipeline.pkl)")
    parser.add_argument("--features", help="Path to feature names file")

    args = parser.parse_args()

    predictor = PolishingPredictor(
        model_path=args.model,
        feature_names_path=args.features
    )

    results = predictor.predict_from_csv(args.input, args.output)

    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(results)}")

    if not results.empty and "predicted_class" in results.columns:
        class_counts = results["predicted_class"].value_counts().sort_index()
        for cls, count in class_counts.items():
            class_name = predictor.class_mapping.get(int(cls), str(cls))
            logger.info(f"  Class {cls} ({class_name}): {count} samples ({count/len(results)*100:.1f}%)")

        if "prediction_confidence" in results.columns:
            logger.info(f"\nAverage confidence: {results['prediction_confidence'].mean():.3f}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()