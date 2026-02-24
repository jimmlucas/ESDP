#!/usr/bin/env python3
"""
esdp_decide.py - Production-grade decision logic for ESDP

This module provides a clean, reusable decision function that:
1. Loads the bundled sklearn pipeline (imputer + scaler + model)
2. Applies domain-specific rules with configurable thresholds
3. Returns structured decisions with full transparency

Usage:
    from esdp_decide import decide, PolishingMetrics

    metrics = PolishingMetrics(sample_id="sample_001", ...)
    decision = decide(metrics, confidence_threshold=0.5)
    print(f"Recommended rounds: {decision.recommended_rounds}")
"""

import joblib
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class PolishingMetrics:
    """Input metrics for polishing decision (API contract)"""
    # Metadata
    sample_id: str
    genus: Optional[str] = None
    round: Optional[int] = None

    # Core assembly metrics
    coverage: Optional[float] = None
    coverage_effective: Optional[float] = None
    n50: Optional[float] = None
    qv: Optional[float] = None
    error_rate: Optional[float] = None
    busco_complete: Optional[float] = None
    num_contigs: Optional[int] = None
    total_length: Optional[int] = None

    # Raw read metrics
    raw_total_bp: Optional[float] = None
    raw_read_n50: Optional[float] = None
    raw_mean_read_len: Optional[float] = None

    # Assembly info metrics
    ai_num_contigs: Optional[int] = None
    ai_total_bp: Optional[int] = None
    ai_mean_cov: Optional[float] = None
    ai_median_cov: Optional[float] = None
    ai_cov_cv: Optional[float] = None
    ai_circular_n: Optional[int] = None
    ai_circular_bp_frac: Optional[float] = None
    ai_repeat_bp_frac: Optional[float] = None
    ai_longest_len: Optional[int] = None
    ai_longest_cov: Optional[float] = None

    # Polishing metrics
    polish_mean_contig_cov: Optional[float] = None
    align_err_consensus: Optional[float] = None
    align_err_polishing: Optional[float] = None

    # Flye metrics
    ovlp_div_initial: Optional[float] = None
    ovlp_median_div_first: Optional[float] = None
    mean_edge_coverage: Optional[float] = None

    # Additional features (can be computed or provided)
    delta_qv: Optional[float] = None
    delta_busco_complete: Optional[float] = None
    delta_error_rate: Optional[float] = None
    qv_improvement_rate: Optional[float] = None
    assembly_error: Optional[float] = None


@dataclass
class Decision:
    """Output decision (API contract with full transparency)"""
    sample_id: str
    recommended_rounds: int
    confidence: float
    reasoning: str
    warnings: List[str]
    rule_overrides: Dict[str, Any]  # NEW: explicit rule tracking
    model_version: str
    class_probabilities: Dict[str, float]


def engineer_features_online(metrics: PolishingMetrics) -> Dict[str, float]:
    """
    Compute derived features from raw metrics (online feature engineering).

    This mirrors the feature engineering done during training but operates
    on a single sample at inference time.
    """
    features = {}

    # Basic ratios
    if metrics.coverage and metrics.coverage_effective:
        features['coverage_efficiency'] = metrics.coverage_effective / metrics.coverage

    if metrics.n50 and metrics.total_length:
        features['n50_ratio'] = metrics.n50 / metrics.total_length

    if metrics.busco_complete and metrics.num_contigs:
        features['busco_per_contig'] = metrics.busco_complete / metrics.num_contigs

    # Assembly quality score
    if metrics.qv and metrics.busco_complete and metrics.num_contigs:
        features['assembly_quality_score'] = (
            metrics.qv * 0.4 + 
            metrics.busco_complete * 0.4 - 
            np.log1p(metrics.num_contigs) * 0.2
        )

    # Coverage metrics
    if metrics.ai_mean_cov and metrics.ai_median_cov:
        features['cov_mean_median_ratio'] = metrics.ai_mean_cov / metrics.ai_median_cov

    # Error metrics
    if metrics.error_rate and metrics.align_err_polishing:
        features['error_improvement'] = metrics.error_rate - metrics.align_err_polishing

    return features


def check_r1_quality(metrics: PolishingMetrics) -> tuple[bool, str]:
    """
    Check if R1 quality is already excellent (domain rule).

    Returns:
        (is_excellent, reason)
    """
    if metrics.qv and metrics.busco_complete:
        if metrics.qv >= 35 and metrics.busco_complete >= 95:
            return True, f"R1 quality excellent (QV={metrics.qv:.1f}, BUSCO={metrics.busco_complete:.1f}%)"

    return False, ""


def apply_conservative_bias(
    predicted_class: int,
    confidence: float,
    confidence_threshold: float
) -> tuple[int, bool, str]:
    """
    Apply conservative bias if confidence is below threshold.

    Args:
        predicted_class: 0-indexed class (0=Early, 1=Medium, 2=Late)
        confidence: Prediction confidence (0-1)
        confidence_threshold: Threshold for applying bias

    Returns:
        (adjusted_class, was_applied, reason)
    """
    if confidence < confidence_threshold:
        adjusted = min(predicted_class + 1, 2)  # Cap at class 2 (Late)
        reason = f"Confidence ({confidence:.3f}) below threshold ({confidence_threshold})"
        return adjusted, True, reason

    return predicted_class, False, ""


def prepare_features(
    metrics: PolishingMetrics,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Prepare feature vector from metrics.

    Args:
        metrics: Input polishing metrics
        feature_names: Expected feature names from training

    Returns:
        DataFrame with features in correct order
    """
    # Convert metrics to dict
    metrics_dict = asdict(metrics)

    # Remove metadata fields
    metadata_fields = ['sample_id', 'genus', 'round']
    for field in metadata_fields:
        metrics_dict.pop(field, None)

    # Create DataFrame with available features
    features = pd.DataFrame([metrics_dict])

    # Add missing features as NaN (pipeline will impute them)
    for feat in feature_names:
        if feat not in features.columns:
            features[feat] = np.nan

    # Reorder to match training
    features = features[feature_names]

    # Replace infinite values with NaN
    features = features.replace([np.inf, -np.inf], np.nan)

    return features


def decide(
    metrics: PolishingMetrics,
    model_path: str = "models/best_model_pipeline.pkl",
    confidence_threshold: float = 0.5,
    force_conservative: bool = False
) -> Decision:
    """
    Make polishing decision based on metrics.

    Decision hierarchy:
    1. ML model prediction
    2. Optional force_conservative override (always recommend R5)
    3. Automatic low-confidence override (escalate one tier)
    4. Domain-specific safety checks

    Args:
        metrics: Input polishing metrics
        model_path: Path to bundled model pipeline
        confidence_threshold: Threshold for automatic conservative bias (default: 0.5)
        force_conservative: If True, force recommendation to R5

    Returns:
        Decision object with recommendation and full transparency
    """
    # Load bundled pipeline
    pipeline = joblib.load(model_path)

    # Extract metadata
    feature_names = pipeline.feature_names if hasattr(pipeline, 'feature_names') else []
    model_version = pipeline.model_version if hasattr(pipeline, 'model_version') else "v1.0.0"

    # Prepare features as DataFrame
    X = prepare_features(metrics, feature_names)

    # Get prediction and probabilities
    y_pred = pipeline.predict(X)[0]  # 0-indexed class
    y_proba = pipeline.predict_proba(X)[0]

    # Map to 1-indexed rounds
    class_to_rounds = {0: 1, 1: 3, 2: 5}
    predicted_rounds = class_to_rounds[y_pred]
    confidence = float(y_proba[y_pred])

    # Initialize decision tracking
    warnings_list = []
    reasoning_parts = []
    rule_overrides = {
        "force_conservative": False,
        "low_confidence_override": False,
        "r1_quality_override": False,
        "confidence_threshold": confidence_threshold,
        "applied_threshold": None
    }

    # Class names for reasoning
    class_names = {0: "Early (R1-R2)", 1: "Medium (R3-R4)", 2: "Late (R5)"}
    base_reasoning = f"ML model predicts {class_names[y_pred]}"

    # --------------------------------------------------
    # Rule Layer (explicit and traceable)
    # --------------------------------------------------

    # Rule A: Force conservative (highest priority)
    if force_conservative:
        predicted_rounds = 5
        warnings_list.append("Force conservative override applied: recommend 5 rounds")
        reasoning_parts.append("Force conservative override triggered")
        rule_overrides["force_conservative"] = True

    # Rule B: Check R1 quality (second priority)
    elif metrics.round == 1:
        r1_excellent, r1_reason = check_r1_quality(metrics)
        if r1_excellent:
            predicted_rounds = 1
            warnings_list.append("R1 quality excellent: early stop recommended")
            reasoning_parts.append(r1_reason)
            rule_overrides["r1_quality_override"] = True

    # Rule C: Low confidence override (third priority)
    if not force_conservative and not rule_overrides["r1_quality_override"]:
        adjusted_class, was_applied, bias_reason = apply_conservative_bias(
            y_pred, confidence, confidence_threshold
        )
        if was_applied:
            predicted_rounds = class_to_rounds[adjusted_class]
            warnings_list.append(f"Conservative bias applied: recommend {predicted_rounds} rounds")
            reasoning_parts.append(bias_reason)
            rule_overrides["low_confidence_override"] = True
            rule_overrides["applied_threshold"] = confidence_threshold

    # Build final reasoning
    if reasoning_parts:
        full_reasoning = f"{base_reasoning}. {'. '.join(reasoning_parts)}. Recommend {predicted_rounds} rounds."
    else:
        full_reasoning = f"{base_reasoning}. Recommend {predicted_rounds} rounds."

    # Format probabilities
    class_probs = {
        "early_r1_r2": float(y_proba[0]),
        "medium_r3_r4": float(y_proba[1]),
        "late_r5": float(y_proba[2])
    }

    return Decision(
        sample_id=metrics.sample_id,
        recommended_rounds=predicted_rounds,
        confidence=confidence,
        reasoning=full_reasoning,
        warnings=warnings_list,
        rule_overrides=rule_overrides,
        model_version=model_version,
        class_probabilities=class_probs
    )


# ============================================================
# CLI for testing
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ESDP Decision Logic - Example Usage")
    print("=" * 60)

    # Example 1: Excellent R1 quality
    print("\nExample 1: Excellent R1 quality")
    print("-" * 60)
    metrics1 = PolishingMetrics(
        sample_id="sample_001",
        genus="Escherichia",
        round=1,
        coverage=50.0,
        coverage_effective=48.0,
        n50=4500000,
        qv=35.2,
        error_rate=0.0003,
        busco_complete=95.5,
        num_contigs=1,
        total_length=4800000
    )
    decision1 = decide(metrics1)
    print(f"Sample: {decision1.sample_id}")
    print(f"Recommended rounds: {decision1.recommended_rounds}")
    print(f"Confidence: {decision1.confidence:.3f}")
    print(f"Reasoning: {decision1.reasoning}")
    print(f"Warnings: {decision1.warnings}")
    print(f"Rule overrides: {decision1.rule_overrides}")

    # Example 2: Poor R1 quality (low confidence)
    print("\nExample 2: Poor R1 quality (low confidence)")
    print("-" * 60)
    metrics2 = PolishingMetrics(
        sample_id="sample_002",
        genus="Salmonella",
        round=1,
        coverage=30.0,
        coverage_effective=28.0,
        n50=2000000,
        qv=28.5,
        error_rate=0.0015,
        busco_complete=85.0,
        num_contigs=5,
        total_length=4900000
    )
    decision2 = decide(metrics2)
    print(f"Sample: {decision2.sample_id}")
    print(f"Recommended rounds: {decision2.recommended_rounds}")
    print(f"Confidence: {decision2.confidence:.3f}")
    print(f"Reasoning: {decision2.reasoning}")
    print(f"Warnings: {decision2.warnings}")
    print(f"Rule overrides: {decision2.rule_overrides}")

    # Example 3: Conservative mode
    print("\nExample 3: Conservative mode (force_conservative=True)")
    print("-" * 60)
    decision3 = decide(metrics1, force_conservative=True)
    print(f"Sample: {decision3.sample_id}")
    print(f"Recommended rounds: {decision3.recommended_rounds}")
    print(f"Confidence: {decision3.confidence:.3f}")
    print(f"Reasoning: {decision3.reasoning}")
    print(f"Warnings: {decision3.warnings}")
    print(f"Rule overrides: {decision3.rule_overrides}")

    # Example 4: Custom confidence threshold
    print("\nExample 4: Custom confidence threshold (0.6)")
    print("-" * 60)
    decision4 = decide(metrics2, confidence_threshold=0.6)
    print(f"Sample: {decision4.sample_id}")
    print(f"Recommended rounds: {decision4.recommended_rounds}")
    print(f"Confidence: {decision4.confidence:.3f}")
    print(f"Reasoning: {decision4.reasoning}")
    print(f"Warnings: {decision4.warnings}")
    print(f"Rule overrides: {decision4.rule_overrides}")

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
