#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7_inference_pipeline.py - Production Inference Pipeline

Production-ready script for predicting optimal stopping rounds for new samples.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class PolishingPredictor:
    """Production predictor for polishing round optimization."""
    
    def __init__(self, model_path=None, feature_names_path=None):
        """Initialize predictor with a bundled model pipeline artifact."""
        if model_path is None:
            model_path = Path(config['outputs']['models_dir']) / 'best_model_pipeline.pkl'
        if feature_names_path is None:
            feature_names_path = Path(config['outputs']['models_dir']) / 'feature_names.txt'

        logger.info(f"Loading model pipeline from {model_path}")
        self.model = joblib.load(model_path)

        # Try to read feature names from the pipeline metadata, fallback to file
        self.feature_names = None
        if hasattr(self.model, 'feature_names'):
            try:
                self.feature_names = list(self.model.feature_names)
            except Exception:
                self.feature_names = None

        if self.feature_names is None and feature_names_path and Path(feature_names_path).exists():
            logger.info(f"Loading feature names from {feature_names_path}")
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f]

        if self.feature_names is None:
            raise ValueError("Feature names not found in model artifact or feature names file")

        self.class_mapping = config['classes']['class_mapping']

        logger.info("Predictor initialized successfully")
    
    def prepare_features(self, sample_data):
        """Prepare feature vector from raw sample data.

        Important: keep missing values as NaN so the pipeline's imputer can handle them.
        """
        # Create DataFrame with expected features initialized as NaN
        features = pd.DataFrame(np.nan, index=[0], columns=self.feature_names)

        # Fill in available features (leave others as NaN)
        for feature in self.feature_names:
            if feature in sample_data:
                val = sample_data[feature]
                # preserve NaN if input explicitly has null
                features.at[0, feature] = val

        return features
    
    def predict(self, sample_data, return_probabilities=False):
        """Predict optimal stopping class for a sample.
        
        Args:
            sample_data: Dictionary with sample features
            return_probabilities: If True, return class probabilities
        
        Returns:
            predicted_class: Integer class (1, 2, or 3)
            class_name: Human-readable class name
            probabilities: (optional) Class probabilities
        """
        # Prepare features
        X = self.prepare_features(sample_data)

        # Predict using the bundled pipeline (which contains imputer + scaler + model)
        predicted = self.model.predict(X)
        predicted_class = int(predicted[0])
        # Model was trained with 0-indexed labels; convert to 1-based for human-readable mapping
        class_idx = int(predicted_class) + 1
        class_name = self.class_mapping.get(class_idx, str(class_idx))

        result = {
            'predicted_class': int(predicted_class),
            'class_name': class_name,
        }

        # Probabilities and confidence (if available)
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            try:
                class_labels = getattr(self.model, 'classes_', None)
                if class_labels is None and hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
                    class_labels = self.model.named_steps['model'].classes_
            except Exception:
                class_labels = None

            if class_labels is not None:
                probs = {int(lbl): float(p) for lbl, p in zip(class_labels, probabilities)}
            else:
                probs = {i: float(p) for i, p in enumerate(probabilities)}

            result['probabilities'] = probs
            result['confidence'] = float(max(probabilities))

        # Model version if embedded
        if hasattr(self.model, 'model_version'):
            result['model_version'] = getattr(self.model, 'model_version')

        return result
    
    def predict_from_csv(self, csv_path, output_path=None):
        """Predict for multiple samples from CSV.
        
        Args:
            csv_path: Path to input CSV with sample data
            output_path: Path to save predictions (optional)
        
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Predict for each row
        predictions = []
        for idx, row in df.iterrows():
            sample_data = row.to_dict()
            result = self.predict(sample_data, return_probabilities=True)
            predictions.append(result)
        
        # Add predictions to dataframe
        df['predicted_class'] = [p['predicted_class'] for p in predictions]
        df['predicted_class_name'] = [p['class_name'] for p in predictions]
        if 'confidence' in predictions[0]:
            df['prediction_confidence'] = [p['confidence'] for p in predictions]
        
        # Save if requested
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")
        
        return df

def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(description='Predict optimal polishing stopping round')
    parser.add_argument('--input', required=True, help='Input CSV file with sample data')
    parser.add_argument('--output', help='Output CSV file for predictions')
    parser.add_argument('--model', help='Path to bundled model pipeline (default: models/best_model_pipeline.pkl)')
    parser.add_argument('--features', help='Path to feature names file')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PolishingPredictor(
        model_path=args.model,
        feature_names_path=args.features
    )
    
    # Make predictions
    results = predictor.predict_from_csv(args.input, args.output)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(results)}")
    
    class_counts = results['predicted_class'].value_counts().sort_index()
    for cls, count in class_counts.items():
        class_name = predictor.class_mapping[cls]
        logger.info(f"  Class {cls} ({class_name}): {count} samples ({count/len(results)*100:.1f}%)")
    
    if 'prediction_confidence' in results.columns:
        logger.info(f"\nAverage confidence: {results['prediction_confidence'].mean():.3f}")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()