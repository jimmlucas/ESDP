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
import pandas as pd

from esdp_features import (
    FeatureBuilder,
    FeatureBuilderConfig,
    align_model_features,
)


FEATURE_BUILDER = FeatureBuilder(FeatureBuilderConfig())


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
    """Compute canonical features for the history available to the v1 API.

    The v1 contract supplies one round at a time, so history-dependent
    features remain missing unless explicitly provided by the caller. The
    trajectory contract introduced in v2 will pass complete round history to
    the same :class:`FeatureBuilder`.
    """
    raw = asdict(metrics)
    record = {
        "Sample": metrics.sample_id,
        "Coverage": (
            metrics.coverage_effective
            if metrics.coverage_effective is not None
            else metrics.coverage
            if metrics.coverage is not None
            else "UNKNOWN"
        ),
        "round": metrics.round if metrics.round is not None else 1,
        **{
            key: value
            for key, value in raw.items()
            if key not in {"sample_id", "genus", "round"}
        },
    }
    canonical = FEATURE_BUILDER.transform(pd.DataFrame([record])).iloc[0].to_dict()

    # Preserve explicitly supplied historical values in the compatibility API.
    for key, value in raw.items():
        if key not in {"sample_id", "genus", "round"} and value is not None:
            canonical[key] = value

    return {
        key: value
        for key, value in canonical.items()
        if key not in {"Sample", "Coverage", "Coverage_effective", "round"}
    }


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
    # Preserve the published v1 model contract: only explicitly supplied
    # fields are aligned here. Canonical history-derived features are activated
    # together with the trajectory schema and a compatible v2 model artifact.
    raw = asdict(metrics)
    for metadata_field in ["sample_id", "genus", "round"]:
        raw.pop(metadata_field, None)
    return align_model_features(raw, feature_names)


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
