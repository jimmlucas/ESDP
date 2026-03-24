#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_sensitivity_analysis.py

Sensitivity analysis for confidence threshold in ESDP.

- Reuses the SAME sample-level split saved by 5_train_models.py
- Evaluates decision units at trajectory level: Sample + Coverage
- Predicts from round-1 features only (as intended by ESDP design)
- Uses the bundled pipeline best_model_pipeline.pkl
- Saves outputs under outputs/results and outputs/plots
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
CPU_HOURS_PER_ROUND = 0.2684
RANDOM_STATE = 42
REFERENCE_THRESHOLD = 0.5

DATA_PATH = Path("data/training_dataset_with_target.csv")
MODEL_PATH = Path("models/best_model_pipeline.pkl")
FEATURES_PATH = Path("models/feature_names.txt")
SPLIT_CANDIDATES = [
    Path("outputs/train_test_split_samples.json"),
    Path("outputs/results/train_test_split_samples.json"),
]

RESULTS_DIR = Path("outputs/results")
PLOTS_DIR = Path("outputs/plots")


@dataclass
class ThresholdMetrics:
    threshold: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    qwk: float
    mae: float

    pct_early: float
    pct_medium: float
    pct_late: float

    bias_applied_rate: float
    avg_confidence: float

    avg_rounds_saved: float
    cpu_reduction_pct: float

    false_early_rate: float
    false_negative_cost: float


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_split_file() -> Path:
    for p in SPLIT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find train_test_split_samples.json in outputs/ or outputs/results/."
    )


def get_cov_col(df: pd.DataFrame) -> str:
    if "Coverage_effective" in df.columns:
        return "Coverage_effective"
    if "Coverage" in df.columns:
        return "Coverage"
    raise ValueError("Neither 'Coverage_effective' nor 'Coverage' found in dataframe.")


def load_feature_names() -> List[str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature names file not found: {FEATURES_PATH}")
    with open(FEATURES_PATH, "r") as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    return feature_cols


def load_test_trajectories(
    data_path: Path = DATA_PATH,
    split_file: Path | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, str]:
    """
    Load test trajectories using the exact sample-level split from training.

    Returns:
        df_test_full: all rows for test samples
        df_r1: only round-1 rows for each test trajectory Sample+Coverage
        y_true: true 0-indexed class for each trajectory (from round-1 row)
        cov_col: coverage column name
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    feature_cols = load_feature_names()
    logger.info(f"Loaded {len(feature_cols)} features from {FEATURES_PATH}")

    missing_features = set(feature_cols) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Missing features in dataset: {sorted(missing_features)}"
        )

    if "optimal_rounds_3class" not in df.columns:
        raise ValueError("Column 'optimal_rounds_3class' not found in dataset")

    cov_col = get_cov_col(df)

    if split_file is None:
        split_file = get_split_file()

    with open(split_file, "r") as f:
        split_info = json.load(f)

    test_samples = set(split_info["test_samples"])
    df_test = df[df["Sample"].astype(str).isin(test_samples)].copy()

    # Keep only complete trajectories with rounds 1, 3, 5
    grouped = df_test.groupby(["Sample", cov_col], dropna=False)
    valid_keys = []
    for key, g in grouped:
        rounds = set(g["round"].astype(int).tolist())
        if {1, 3, 5}.issubset(rounds):
            valid_keys.append(key)

    if not valid_keys:
        raise RuntimeError("No valid test trajectories with rounds 1, 3, and 5 found.")

    valid_df = pd.concat(
        [grouped.get_group(k) for k in valid_keys],
        axis=0
    ).copy()

    df_r1 = valid_df[valid_df["round"] == 1].copy()
    df_r1 = df_r1.sort_values(["Sample", cov_col]).reset_index(drop=True)

    y_true = (df_r1["optimal_rounds_3class"].astype(int) - 1).copy()

    logger.info(f"Test samples: {len(test_samples)}")
    logger.info(f"Valid test trajectories: {len(df_r1)}")
    logger.info(f"Class distribution: {y_true.value_counts().sort_index().to_dict()}")

    return valid_df, df_r1, y_true, cov_col


def apply_conservative_bias_vectorized(
    predictions: np.ndarray,
    confidences: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply conservative bias:
    if confidence < threshold, escalate one class (Early->Medium, Medium->Late, Late stays Late).
    """
    bias_applied = confidences < threshold
    adjusted = predictions.copy()
    adjusted[bias_applied] = np.minimum(adjusted[bias_applied] + 1, 2)
    return adjusted, bias_applied


def calculate_cost_metrics(y_pred: np.ndarray) -> Dict[str, float]:
    """
    Estimated cost savings relative to always running 5 rounds.

    Class mapping:
    0 -> Early  -> 1 round
    1 -> Medium -> 3 rounds
    2 -> Late   -> 5 rounds

    Rounds saved vs R5:
    0 -> 4
    1 -> 2
    2 -> 0
    """
    rounds_saved_map = {0: 4, 1: 2, 2: 0}
    rounds_saved = np.array([rounds_saved_map[int(p)] for p in y_pred], dtype=float)

    avg_rounds_saved = float(rounds_saved.mean())
    cpu_reduction_pct = float((avg_rounds_saved / 5.0) * 100.0)

    return {
        "avg_rounds_saved": avg_rounds_saved,
        "cpu_reduction_pct": cpu_reduction_pct,
    }


def calculate_safety_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Safety metrics.

    False early:
    predicted Early (0) when true class is Late (2).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    false_early_mask = (y_pred == 0) & (y_true == 2)
    false_early_rate = float(false_early_mask.sum() / len(y_true))

    false_negative_cost = float(np.abs(y_pred - y_true).mean())

    return {
        "false_early_rate": false_early_rate,
        "false_negative_cost": false_negative_cost,
    }


def evaluate_threshold(
    X_r1: pd.DataFrame,
    y_true: pd.Series,
    model_pipeline,
    threshold: float
) -> ThresholdMetrics:
    """Evaluate a specific confidence threshold on trajectory-level round-1 inputs."""
    y_pred_base = model_pipeline.predict(X_r1)
    y_proba = model_pipeline.predict_proba(X_r1)

    confidences = y_proba.max(axis=1)
    y_pred_adjusted, bias_applied = apply_conservative_bias_vectorized(
        y_pred_base,
        confidences,
        threshold
    )

    accuracy = float(accuracy_score(y_true, y_pred_adjusted))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred_adjusted))
    macro_f1 = float(f1_score(y_true, y_pred_adjusted, average="macro", zero_division=0))
    qwk = float(cohen_kappa_score(y_true, y_pred_adjusted, weights="quadratic"))
    mae = float(mean_absolute_error(y_true, y_pred_adjusted))

    unique, counts = np.unique(y_pred_adjusted, return_counts=True)
    dist = dict(zip(unique, counts / len(y_pred_adjusted) * 100.0))
    pct_early = float(dist.get(0, 0.0))
    pct_medium = float(dist.get(1, 0.0))
    pct_late = float(dist.get(2, 0.0))

    bias_applied_rate = float(bias_applied.mean() * 100.0)
    avg_confidence = float(confidences.mean())

    cost_metrics = calculate_cost_metrics(y_pred_adjusted)
    safety_metrics = calculate_safety_metrics(y_true.values, y_pred_adjusted)

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
        avg_rounds_saved=cost_metrics["avg_rounds_saved"],
        cpu_reduction_pct=cost_metrics["cpu_reduction_pct"],
        false_early_rate=safety_metrics["false_early_rate"],
        false_negative_cost=safety_metrics["false_negative_cost"],
    )

def plot_sensitivity_analysis(results_df: pd.DataFrame):
    """Create threshold-sensitivity plots in manuscript order (A–F)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # -------------------------------------------------------------------------
    # A. Computational cost savings
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(results_df["threshold"], results_df["cpu_reduction_pct"], "o-", color="blue", linewidth=2)
    ax.axvline(
        x=REFERENCE_THRESHOLD,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Reference threshold ({REFERENCE_THRESHOLD})"
    )
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("CPU reduction (%)")
    ax.set_title("Computational cost savings", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(-0.12, 1.05, "A", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")

    # -------------------------------------------------------------------------
    # B. Agreement and error
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(results_df["threshold"], results_df["qwk"], "o-", color="green", label="QWK", linewidth=2)
    l2 = ax2.plot(results_df["threshold"], results_df["mae"], "s-", color="orange", label="MAE", linewidth=2)
    ax.axvline(x=REFERENCE_THRESHOLD, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Quadratic weighted kappa", color="green")
    ax2.set_ylabel("Mean absolute error", color="orange")
    ax.set_title("Agreement and error", fontsize=11)
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="best")
    ax.grid(True, alpha=0.3)
    ax.text(-0.12, 1.05, "B", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")

    # -------------------------------------------------------------------------
    # C. Performance metrics
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(results_df["threshold"], results_df["accuracy"], "o-", label="Accuracy", linewidth=2)
    ax.plot(results_df["threshold"], results_df["balanced_accuracy"], "s-", label="Balanced Acc", linewidth=2)
    ax.plot(results_df["threshold"], results_df["macro_f1"], "^-", label="Macro F1", linewidth=2)
    ax.axvline(
        x=REFERENCE_THRESHOLD,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Reference threshold ({REFERENCE_THRESHOLD})"
    )
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")
    ax.set_title("Performance metrics", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(-0.12, 1.05, "C", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")

    # -------------------------------------------------------------------------
    # D. Safety metrics
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(
        results_df["threshold"],
        results_df["false_early_rate"] * 100,
        "o-",
        color="red",
        label="False early rate",
        linewidth=2
    )
    l2 = ax2.plot(
        results_df["threshold"],
        results_df["false_negative_cost"],
        "s-",
        color="orange",
        label="Decision error (MAE)",
        linewidth=2
    )
    ax.axvline(x=REFERENCE_THRESHOLD, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("False early rate (%)")
    ax2.set_ylabel("Decision error (MAE)")
    ax.set_title("Safety metrics", fontsize=11)
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="best")
    ax.grid(True, alpha=0.3)
    ax.text(-0.12, 1.05, "D", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")

    # -------------------------------------------------------------------------
    # E. Conservative-bias rate
    # -------------------------------------------------------------------------
    ax = axes[2, 0]
    ax.plot(results_df["threshold"], results_df["bias_applied_rate"], "o-", color="purple", linewidth=2)
    ax.axvline(
        x=REFERENCE_THRESHOLD,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Reference threshold ({REFERENCE_THRESHOLD})"
    )
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Bias applied rate (%)")
    ax.set_title("Conservative-bias rate", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(-0.12, 1.05, "E", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")

    # -------------------------------------------------------------------------
    # F. Decision distribution
    # -------------------------------------------------------------------------
    ax = axes[2, 1]
    ax.plot(results_df["threshold"], results_df["pct_early"], "o-", label="Early (R1)", linewidth=2)
    ax.plot(results_df["threshold"], results_df["pct_medium"], "s-", label="Medium (R3)", linewidth=2)
    ax.plot(results_df["threshold"], results_df["pct_late"], "^-", label="Late (R5)", linewidth=2)
    ax.axvline(
        x=REFERENCE_THRESHOLD,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Reference threshold ({REFERENCE_THRESHOLD})"
    )
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Decision distribution", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(-0.12, 1.05, "F", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")

    plt.tight_layout()
    output_path = PLOTS_DIR / "sensitivity_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("ESDP Sensitivity Analysis: Confidence Threshold")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load aligned test trajectories + model
    df_test_full, df_r1, y_true, cov_col = load_test_trajectories()
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info(f"Loading model from {MODEL_PATH}")

    feature_cols = load_feature_names()
    X_r1 = df_r1[feature_cols].copy().replace([np.inf, -np.inf], np.nan)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    logger.info(f"\nTesting thresholds: {thresholds}")

    results = []
    for threshold in thresholds:
        logger.info(f"\nEvaluating threshold={threshold:.1f}")
        metrics = evaluate_threshold(X_r1, y_true, model_pipeline, threshold)
        results.append(asdict(metrics))

        logger.info(f"  Accuracy: {metrics.accuracy:.3f}")
        logger.info(f"  QWK: {metrics.qwk:.3f}")
        logger.info(f"  CPU Reduction: {metrics.cpu_reduction_pct:.1f}%")
        logger.info(f"  Bias Applied: {metrics.bias_applied_rate:.1f}%")
        logger.info(f"  False Early Rate: {metrics.false_early_rate:.3f}")

    results_df = pd.DataFrame(results)

    output_csv = RESULTS_DIR / "sensitivity_analysis.csv"
    results_df.to_csv(output_csv, index=False)
    logger.info(f"\nSaved results to {output_csv}")

    logger.info("\nGenerating plots...")
    plot_sensitivity_analysis(results_df)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    # Composite criterion: maximize QWK while penalizing false early decisions
    results_df["score"] = results_df["qwk"] - results_df["false_early_rate"]
    optimal_idx = results_df["score"].idxmax()
    optimal_threshold = float(results_df.loc[optimal_idx, "threshold"])

    logger.info(f"\nOptimal threshold (QWK - false_early_rate): {optimal_threshold:.1f}")
    logger.info(f"  Accuracy: {results_df.loc[optimal_idx, 'accuracy']:.3f}")
    logger.info(f"  QWK: {results_df.loc[optimal_idx, 'qwk']:.3f}")
    logger.info(f"  CPU Reduction: {results_df.loc[optimal_idx, 'cpu_reduction_pct']:.1f}%")
    logger.info(f"  False Early Rate: {results_df.loc[optimal_idx, 'false_early_rate']:.3f}")

    default_idx = results_df.index[results_df["threshold"] == REFERENCE_THRESHOLD][0]
    logger.info(f"\nReference threshold ({REFERENCE_THRESHOLD}):")
    logger.info(f"  Accuracy: {results_df.loc[default_idx, 'accuracy']:.3f}")
    logger.info(f"  QWK: {results_df.loc[default_idx, 'qwk']:.3f}")
    logger.info(f"  CPU Reduction: {results_df.loc[default_idx, 'cpu_reduction_pct']:.1f}%")
    logger.info(f"  False Early Rate: {results_df.loc[default_idx, 'false_early_rate']:.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("Sensitivity analysis completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
