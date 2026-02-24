#!/usr/bin/env python3
"""
Sensitivity Analysis: Confidence Threshold Impact

Evaluates how different confidence_threshold values affect:
- Model performance (accuracy, QWK, F1)
- Decision distribution (Early/Medium/Late)
- Conservative bias application rate
- Computational cost savings

This analysis justifies the default threshold=0.5 choice.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    cohen_kappa_score, mean_absolute_error, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ThresholdMetrics:
    """Metrics for a specific confidence threshold"""
    threshold: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    qwk: float
    mae: float

    # Decision distribution
    pct_early: float
    pct_medium: float
    pct_late: float

    # Conservative bias stats
    bias_applied_rate: float
    avg_confidence: float

    # Cost metrics (estimated)
    avg_rounds_saved: float
    cpu_reduction_pct: float

    # Safety metrics
    false_early_rate: float  # Predicted Early but should be Late
    false_negative_cost: float  # Weighted cost of wrong decisions


def load_data_and_model(
    data_path: str = "data/training_dataset_with_target.csv",
    model_path: str = "models/best_model_pipeline.pkl"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]:
    """Load test data and trained model"""

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Use same train/test split as training (80/20, random_state=42)
    from sklearn.model_selection import train_test_split

    # Load feature names from training (CRITICAL: must match exactly)
    feature_names_path = Path("models/feature_names.txt")
    if not feature_names_path.exists():
        raise FileNotFoundError(
            f"Feature names file not found: {feature_names_path}\n"
            "Please run: python 5_train_models.py to generate it."
        )

    with open(feature_names_path, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]

    logger.info(f"Loaded {len(feature_cols)} features from {feature_names_path}")

    # Verify all features exist in the dataset
    missing_features = set(feature_cols) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Missing features in dataset: {missing_features}\n"
            "Dataset may need to be regenerated with 3_feature_engineering.py"
        )

    X = df[feature_cols].copy()
    y = df['optimal_rounds_3class'].copy()

    # Convert to 0-indexed for sklearn
    y = y - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Test set size: {len(X_test)} samples")
    logger.info(f"Class distribution: {y_test.value_counts().sort_index().to_dict()}")

    # Load model
    logger.info(f"Loading model from {model_path}")
    model_pipeline = joblib.load(model_path)

    return X_train, X_test, y_train, y_test, model_pipeline


def apply_conservative_bias_vectorized(
    predictions: np.ndarray,
    confidences: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply conservative bias to predictions below confidence threshold

    Returns:
        adjusted_predictions: Modified predictions
        bias_applied: Boolean mask of where bias was applied
    """
    bias_applied = confidences < threshold
    adjusted = predictions.copy()
    adjusted[bias_applied] = np.minimum(adjusted[bias_applied] + 1, 2)

    return adjusted, bias_applied


def calculate_cost_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate computational cost savings

    Assumptions:
    - Class 0 (Early/R1): 0 extra rounds
    - Class 1 (Medium/R3): 2 extra rounds
    - Class 2 (Late/R5): 4 extra rounds
    - Each round costs ~1 CPU hour
    """
    # Map predictions to rounds saved
    rounds_map = {0: 4, 1: 2, 2: 0}  # Rounds saved vs always going to R5

    rounds_saved = np.array([rounds_map[p] for p in y_pred])
    avg_rounds_saved = rounds_saved.mean()

    # CPU reduction: baseline is always R5 (5 rounds)
    baseline_rounds = 5
    avg_rounds_used = baseline_rounds - avg_rounds_saved
    cpu_reduction_pct = (avg_rounds_saved / baseline_rounds) * 100

    return {
        'avg_rounds_saved': avg_rounds_saved,
        'cpu_reduction_pct': cpu_reduction_pct
    }


def calculate_safety_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate safety metrics (false early predictions)

    False early: Predicted class < True class (risky)
    """
    # False early rate: predicted Early (0) but true is Late (2)
    false_early_mask = (y_pred == 0) & (y_true == 2)
    false_early_rate = false_early_mask.sum() / len(y_true)

    # Weighted cost: penalize wrong decisions by distance
    cost = np.abs(y_pred - y_true).sum() / len(y_true)

    return {
        'false_early_rate': false_early_rate,
        'false_negative_cost': cost
    }


def evaluate_threshold(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_pipeline: object,
    threshold: float
) -> ThresholdMetrics:
    """Evaluate model performance at a specific confidence threshold"""

    # Get base predictions and probabilities
    y_pred_base = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)

    # Calculate confidence (max probability)
    confidences = y_proba.max(axis=1)

    # Apply conservative bias
    y_pred_adjusted, bias_applied = apply_conservative_bias_vectorized(
        y_pred_base, confidences, threshold
    )

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred_adjusted)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_adjusted)
    macro_f1 = f1_score(y_test, y_pred_adjusted, average='macro')
    qwk = cohen_kappa_score(y_test, y_pred_adjusted, weights='quadratic')
    mae = mean_absolute_error(y_test, y_pred_adjusted)

    # Decision distribution
    unique, counts = np.unique(y_pred_adjusted, return_counts=True)
    dist = dict(zip(unique, counts / len(y_pred_adjusted) * 100))
    pct_early = dist.get(0, 0)
    pct_medium = dist.get(1, 0)
    pct_late = dist.get(2, 0)

    # Conservative bias stats
    bias_applied_rate = bias_applied.sum() / len(bias_applied) * 100
    avg_confidence = confidences.mean()

    # Cost metrics
    cost_metrics = calculate_cost_metrics(y_test.values, y_pred_adjusted)

    # Safety metrics
    safety_metrics = calculate_safety_metrics(y_test.values, y_pred_adjusted)

    return ThresholdMetrics(
        threshold=threshold,
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        macro_f1=macro_f1,
        qwk=qwk,
        mae=mae,
        pct_early=pct_early,
        pct_medium=pct_medium,
        pct_late=pct_late,
        bias_applied_rate=bias_applied_rate,
        avg_confidence=avg_confidence,
        avg_rounds_saved=cost_metrics['avg_rounds_saved'],
        cpu_reduction_pct=cost_metrics['cpu_reduction_pct'],
        false_early_rate=safety_metrics['false_early_rate'],
        false_negative_cost=safety_metrics['false_negative_cost']
    )


def plot_sensitivity_analysis(
    results_df: pd.DataFrame,
    output_dir: str = "results"
):
    """Create visualization of sensitivity analysis"""

    Path(output_dir).mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # 1. Performance metrics
    ax = axes[0, 0]
    ax.plot(results_df['threshold'], results_df['accuracy'], 'o-', label='Accuracy', linewidth=2)
    ax.plot(results_df['threshold'], results_df['balanced_accuracy'], 's-', label='Balanced Acc', linewidth=2)
    ax.plot(results_df['threshold'], results_df['macro_f1'], '^-', label='Macro F1', linewidth=2)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics vs Confidence Threshold', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. QWK and MAE
    ax = axes[0, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(results_df['threshold'], results_df['qwk'], 'o-', color='green', label='QWK', linewidth=2)
    l2 = ax2.plot(results_df['threshold'], results_df['mae'], 's-', color='orange', label='MAE', linewidth=2)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('Quadratic Weighted Kappa', fontsize=12, color='green')
    ax2.set_ylabel('Mean Absolute Error', fontsize=12, color='orange')
    ax.set_title('Agreement (QWK) and Error (MAE)', fontsize=14, fontweight='bold')
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')
    ax.grid(True, alpha=0.3)

    # 3. Decision distribution
    ax = axes[1, 0]
    ax.plot(results_df['threshold'], results_df['pct_early'], 'o-', label='Early (R1)', linewidth=2)
    ax.plot(results_df['threshold'], results_df['pct_medium'], 's-', label='Medium (R3)', linewidth=2)
    ax.plot(results_df['threshold'], results_df['pct_late'], '^-', label='Late (R5)', linewidth=2)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Decision Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Conservative bias application
    ax = axes[1, 1]
    ax.plot(results_df['threshold'], results_df['bias_applied_rate'], 'o-', color='purple', linewidth=2)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('Bias Applied Rate (%)', fontsize=12)
    ax.set_title('Conservative Bias Application Rate', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. CPU reduction
    ax = axes[2, 0]
    ax.plot(results_df['threshold'], results_df['cpu_reduction_pct'], 'o-', color='blue', linewidth=2)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('CPU Reduction (%)', fontsize=12)
    ax.set_title('Computational Cost Savings', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Safety metrics
    ax = axes[2, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(results_df['threshold'], results_df['false_early_rate'] * 100, 'o-', 
                 color='red', label='False Early Rate', linewidth=2)
    l2 = ax2.plot(results_df['threshold'], results_df['false_negative_cost'], 's-', 
                  color='orange', label='Decision Cost', linewidth=2)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('False Early Rate (%)', fontsize=12, color='red')
    ax2.set_ylabel('Decision Cost (MAE)', fontsize=12, color='orange')
    ax.set_title('Safety Metrics', fontsize=14, fontweight='bold')
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / "sensitivity_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_path}")
    plt.close()


def main():
    """Run sensitivity analysis"""

    logger.info("="*60)
    logger.info("ESDP Sensitivity Analysis: Confidence Threshold")
    logger.info("="*60)

    # Load data and model
    X_train, X_test, y_train, y_test, model_pipeline = load_data_and_model()

    # Define thresholds to test
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    logger.info(f"\nTesting thresholds: {thresholds}")

    # Evaluate each threshold
    results = []
    for threshold in thresholds:
        logger.info(f"\nEvaluating threshold={threshold:.1f}")
        metrics = evaluate_threshold(X_test, y_test, model_pipeline, threshold)
        results.append(asdict(metrics))

        logger.info(f"  Accuracy: {metrics.accuracy:.3f}")
        logger.info(f"  QWK: {metrics.qwk:.3f}")
        logger.info(f"  CPU Reduction: {metrics.cpu_reduction_pct:.1f}%")
        logger.info(f"  Bias Applied: {metrics.bias_applied_rate:.1f}%")
        logger.info(f"  False Early Rate: {metrics.false_early_rate:.3f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "sensitivity_analysis.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved results to {output_path}")

    # Create visualization
    logger.info("\nGenerating plots...")
    plot_sensitivity_analysis(results_df, output_dir="results")

    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    # Find optimal threshold (maximize QWK while minimizing false early rate)
    results_df['score'] = results_df['qwk'] - results_df['false_early_rate']
    optimal_idx = results_df['score'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']

    logger.info(f"\nOptimal threshold (QWK - false_early_rate): {optimal_threshold:.1f}")
    logger.info(f"  Accuracy: {results_df.loc[optimal_idx, 'accuracy']:.3f}")
    logger.info(f"  QWK: {results_df.loc[optimal_idx, 'qwk']:.3f}")
    logger.info(f"  CPU Reduction: {results_df.loc[optimal_idx, 'cpu_reduction_pct']:.1f}%")
    logger.info(f"  False Early Rate: {results_df.loc[optimal_idx, 'false_early_rate']:.3f}")

    # Compare to default (0.5)
    default_idx = results_df[results_df['threshold'] == 0.5].index[0]
    logger.info(f"\nDefault threshold (0.5):")
    logger.info(f"  Accuracy: {results_df.loc[default_idx, 'accuracy']:.3f}")
    logger.info(f"  QWK: {results_df.loc[default_idx, 'qwk']:.3f}")
    logger.info(f"  CPU Reduction: {results_df.loc[default_idx, 'cpu_reduction_pct']:.1f}%")
    logger.info(f"  False Early Rate: {results_df.loc[default_idx, 'false_early_rate']:.3f}")

    logger.info("\n" + "="*60)
    logger.info("Sensitivity analysis completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
