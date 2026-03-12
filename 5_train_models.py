#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5_train_models.py - Multi-Model Training with TRUE Pipeline Training

CORRECTIONS:
- Prevents data leakage by splitting at Sample level (not Sample+Coverage)
- Uses grouped sample split without invalid per-group stratification
- Trains models inside real pipelines from the start
- Handles NaN correctly with SimpleImputer
- Applies SMOTE only during training via imblearn Pipeline
- Eliminates feature-name warning mismatch between training and inference
- Saves bundled trained pipeline + legacy artifacts (model, scaler, imputer)
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Logging configuration ---
log_file = config["logging"]["file"]
log_level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
log_format = config["logging"]["format"]

Path(log_file).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=log_level,
    format=log_format,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available")

# Optional: SMOTE / imblearn pipeline
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    ImbPipeline = None
    logger.warning("imbalanced-learn not available - SMOTE disabled")

# Optional: Ordinal Regression
try:
    import mord
    HAS_MORD = True
except ImportError:
    HAS_MORD = False
    logger.warning("mord not available - ordinal regression disabled")

RANDOM_STATE = config["models"]["random_state"]

def load_data():
    """Load labeled dataset."""
    df = pd.read_csv(config["data"]["labeled_csv"])
    logger.info(f"Loaded {len(df)} rows")
    return df

def prepare_features(df):
    """
    Prepare feature matrix and labels.

    Important:
    - Features/labels are defined per Sample+Coverage trajectory upstream.
    - Split grouping is done here at Sample level to avoid leakage.
    """
    logger.info("Preparing features...")

    feature_candidates = [
        # Base metrics
        "n50", "qv", "error_rate", "busco_complete", "busco_fragmented",
        "busco_missing", "num_contigs", "total_length", "assembly_frac",
        "assembly_error",

        # Delta features
        "delta_qv", "delta_busco_complete", "delta_n50", "delta_num_contigs",
        "delta_error_rate", "delta_error_improvement", "delta_assembly_error",

        # Ratio features
        "qv_improvement_rate", "busco_per_contig", "n50_fraction",
        "cost_benefit_ratio",

        # Cumulative features
        "delta_qv_cumsum", "delta_busco_complete_cumsum",
        "score_improvement", "gain_cumulative",

        # Normalized to R1
        "qv_from_r1", "n50_from_r1", "error_rate_from_r1",
        "busco_complete_from_r1", "assembly_frac_from_r1",

        # Trend features
        "delta_qv_trend", "delta_busco_complete_trend", "score_improvement_trend",

        # Plateau features
        "is_plateau", "plateau_streak",

        # Domain-specific
        "completeness_score", "assembly_quality", "polishing_effectiveness",

        # Policy flags
        "r1_ok_group",

        # Flye features
        "coverage_est", "mean_edge_coverage", "align_err_consensus",
    ]

    features = [f for f in feature_candidates if f in df.columns]
    logger.info(f"Selected {len(features)} features")

    X = df[features].copy()

    if "optimal_rounds_3class" not in df.columns:
        raise ValueError("3-class labels not found! Run 4_label_optimal_round.py first")

    y = df["optimal_rounds_3class"].copy() - 1  # 1,2,3 -> 0,1,2

    # CRITICAL: split groups only by Sample to avoid leakage across coverages
    groups = df["Sample"].astype(str)

    # Replace inf with NaN; missing values handled later by imputer
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y, groups, features

def grouped_sample_split(X, y, groups, test_size=0.2, random_state=42):
    """
    Split by Sample (group), not by row, to prevent leakage between
    different coverages of the same biological sample.

    No stratification here because one Sample can contain multiple labels
    across different coverages/trajectories.
    """
    logger.info("Performing sample-level grouped split...")

    unique_groups = pd.Series(groups).drop_duplicates()

    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state
    )

    train_mask = groups.isin(train_groups)
    test_mask = groups.isin(test_groups)

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    logger.info(f"Train: {len(X_train)} rows, {pd.Series(train_groups).nunique()} samples")
    logger.info(f"Test: {len(X_test)} rows, {pd.Series(test_groups).nunique()} samples")
    logger.info(f"Train class distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
    logger.info(f"Test class distribution:\n{pd.Series(y_test).value_counts().sort_index()}")

    return X_train, X_test, y_train, y_test, train_groups, test_groups

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """Calculate classification and ordinal-aware metrics."""
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision, recall, f1_vals, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    mae = float(np.mean(np.abs(y_true - y_pred)))
    acc_pm1 = float(np.mean(np.abs(y_true - y_pred) <= 1))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "mae": mae,
        "accuracy_pm1": acc_pm1,
        "qwk": qwk,
    }

    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1_vals, support)):
        metrics[f"precision_class_{i+1}"] = float(p)
        metrics[f"recall_class_{i+1}"] = float(r)
        metrics[f"f1_class_{i+1}"] = float(f)
        metrics[f"support_class_{i+1}"] = int(s)

    targets = config["evaluation"]["target_metrics"]
    metrics["meets_targets"] = (
        balanced_acc >= targets["balanced_accuracy"] and
        macro_f1 >= targets["macro_f1"] and
        mae <= targets["mae"] and
        qwk >= targets["qwk"]
    )

    logger.info(f"\n{model_name} Metrics:")
    logger.info(f"  Accuracy: {acc:.3f}")
    logger.info(f"  Balanced Accuracy: {balanced_acc:.3f}")
    logger.info(f"  Macro F1: {macro_f1:.3f}")
    logger.info(f"  MAE: {mae:.3f}")
    logger.info(f"  Accuracy ±1: {acc_pm1:.3f}")
    logger.info(f"  QWK: {qwk:.3f}")
    logger.info(f"  Meets targets: {metrics['meets_targets']}")

    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plot and save confusion matrix (B/W-friendly, readable numbers)."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4.5, 4))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greys",
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        annot_kws={"size": 9}
    )
    ax.set_xticklabels(["Early", "Medium", "Late"], rotation=0)
    ax.set_yticklabels(["Early", "Medium", "Late"], rotation=0)

    plt.title(f"Confusion Matrix - {model_name}", fontsize=11)
    plt.ylabel("True label", fontsize=10)
    plt.xlabel("Predicted label", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix to {save_path}")

def plot_feature_importance_from_estimator(estimator, feature_names, model_name, save_path):
    """Plot top-20 feature importances (B/W-friendly horizontal bars)."""
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        importances = np.abs(coef).mean(axis=0) if getattr(coef, "ndim", 1) > 1 else np.abs(coef)
    else:
        logger.warning(f"Cannot extract feature importance for {model_name}")
        return

    if importances is None or len(importances) == 0:
        logger.warning(f"No importances available for {model_name}")
        return

    indices = np.argsort(importances)[-20:]
    top_importances = importances[indices]
    top_features = [feature_names[i] for i in indices]

    plt.figure(figsize=(5.5, 6))
    y_pos = np.arange(len(top_features))

    plt.barh(
        y_pos,
        top_importances,
        color="0.7",      # mid gray
        edgecolor="0.0",  # black border
        linewidth=0.8
    )
    plt.yticks(y_pos, top_features, fontsize=8)
    plt.xlabel("Importance", fontsize=10)
    plt.title(f"Top 20 feature importances – {model_name}", fontsize=11)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved feature importance to {save_path}")

def get_smote():
    """Return configured SMOTE object or None."""
    if not HAS_SMOTE or not config["imbalance"]["use_smote"]:
        return None

    k_neighbors = config["imbalance"]["smote_k_neighbors"]
    return SMOTE(
        sampling_strategy=config["imbalance"]["smote_sampling_strategy"],
        k_neighbors=k_neighbors,
        random_state=RANDOM_STATE
    )

def build_xgb_pipeline():
    """Build XGBoost pipeline."""
    if not HAS_XGBOOST:
        return None

    xgb = XGBClassifier(
        **config["models"]["xgboost"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss"
    )

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    smote = get_smote()
    if smote is not None:
        steps.append(("smote", smote))

    steps.append(("model", xgb))

    pipeline_cls = ImbPipeline if smote is not None else SkPipeline
    return pipeline_cls(steps)

def build_rf_pipeline():
    """Build Random Forest pipeline."""
    class_weights_config = config["imbalance"]["class_weights"]
    class_weights = {k - 1: v for k, v in class_weights_config.items()}

    rf = RandomForestClassifier(
        **config["models"]["random_forest"],
        class_weight=class_weights,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    smote = get_smote()
    if smote is not None:
        steps.append(("smote", smote))

    steps.append(("model", rf))

    pipeline_cls = ImbPipeline if smote is not None else SkPipeline
    return pipeline_cls(steps)

def build_ordinal_pipeline():
    """Build ordinal regression pipeline."""
    if not HAS_MORD:
        return None

    model = mord.LogisticAT(alpha=config["models"]["ordinal_regression"]["alpha"])

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    smote = get_smote()
    if smote is not None:
        steps.append(("smote", smote))

    steps.append(("model", model))

    pipeline_cls = ImbPipeline if smote is not None else SkPipeline
    return pipeline_cls(steps)

def fit_and_evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test, model_name):
    """Fit a pipeline and evaluate it."""
    logger.info(f"Training {model_name}...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, model_name)

    return pipeline, metrics, y_pred

def build_ensemble_pipeline():
    """
    Build ensemble pipeline.

    Ensemble uses RF + XGB only because mord.LogisticAT is not a sklearn classifier
    compatible with VotingClassifier in a reliable way.
    """
    estimators = []

    if HAS_XGBOOST:
        xgb = XGBClassifier(
            **config["models"]["xgboost"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss"
        )
        estimators.append(("xgboost", xgb))

    class_weights_config = config["imbalance"]["class_weights"]
    class_weights = {k - 1: v for k, v in class_weights_config.items()}
    rf = RandomForestClassifier(
        **config["models"]["random_forest"],
        class_weight=class_weights,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    estimators.append(("random_forest", rf))

    if len(estimators) < 2:
        return None

    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=-1
    )

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    smote = get_smote()
    if smote is not None:
        steps.append(("smote", smote))

    steps.append(("model", ensemble))

    pipeline_cls = ImbPipeline if smote is not None else SkPipeline
    return pipeline_cls(steps)

def extract_legacy_artifacts_from_pipeline(pipeline):
    """Extract fitted imputer, scaler, and final estimator from a trained pipeline."""
    named_steps = pipeline.named_steps

    imputer = named_steps.get("imputer", None)
    scaler = named_steps.get("scaler", None)
    model = named_steps.get("model", None)

    return imputer, scaler, model

def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Multi-Model Training Pipeline")
    logger.info("=" * 60)

    Path(config["outputs"]["models_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["plots_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["outputs"]["results_dir"]).mkdir(parents=True, exist_ok=True)

    # Load and prepare
    df = load_data()
    X, y, groups, feature_names = prepare_features(df)

    # Leakage-safe split by Sample
    X_train, X_test, y_train, y_test, train_groups, test_groups = grouped_sample_split(
        X, y, groups,
        test_size=config["models"]["test_size"],
        random_state=RANDOM_STATE
    )

    # Train models
    all_models = {}
    all_metrics = []

    # XGBoost
    xgb_pipeline = build_xgb_pipeline()
    if xgb_pipeline is not None:
        try:
            xgb_model, xgb_metrics, xgb_pred = fit_and_evaluate_pipeline(
                xgb_pipeline, X_train, y_train, X_test, y_test, "XGBoost"
            )
            all_models["xgboost"] = xgb_model
            all_metrics.append(xgb_metrics)
            plot_confusion_matrix(
                y_test, xgb_pred, "XGBoost",
                Path(config["outputs"]["plots_dir"]) / "cm_xgboost.png"
            )
            plot_feature_importance_from_estimator(
                xgb_model.named_steps["model"],
                feature_names,
                "XGBoost",
                Path(config["outputs"]["plots_dir"]) / "fi_xgboost.png"
            )
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")

    # Random Forest
    rf_pipeline = build_rf_pipeline()
    if rf_pipeline is not None:
        try:
            rf_model, rf_metrics, rf_pred = fit_and_evaluate_pipeline(
                rf_pipeline, X_train, y_train, X_test, y_test, "Random Forest"
            )
            all_models["random_forest"] = rf_model
            all_metrics.append(rf_metrics)
            plot_confusion_matrix(
                y_test, rf_pred, "Random Forest",
                Path(config["outputs"]["plots_dir"]) / "cm_random_forest.png"
            )
            plot_feature_importance_from_estimator(
                rf_model.named_steps["model"],
                feature_names,
                "Random Forest",
                Path(config["outputs"]["plots_dir"]) / "fi_random_forest.png"
            )
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")

    # Ordinal Regression
    ord_pipeline = build_ordinal_pipeline()
    if ord_pipeline is not None:
        try:
            ord_model, ord_metrics, ord_pred = fit_and_evaluate_pipeline(
                ord_pipeline, X_train, y_train, X_test, y_test, "Ordinal Regression"
            )
            all_models["ordinal"] = ord_model
            all_metrics.append(ord_metrics)
            plot_confusion_matrix(
                y_test, ord_pred, "Ordinal Regression",
                Path(config["outputs"]["plots_dir"]) / "cm_ordinal.png"
            )
        except Exception as e:
            logger.error(f"Ordinal regression training failed: {e}")

    # Ensemble
    ensemble_pipeline = build_ensemble_pipeline()
    if ensemble_pipeline is not None:
        try:
            ensemble_model, ensemble_metrics, ensemble_pred = fit_and_evaluate_pipeline(
                ensemble_pipeline, X_train, y_train, X_test, y_test, "Ensemble"
            )
            all_models["ensemble"] = ensemble_model
            all_metrics.append(ensemble_metrics)
            plot_confusion_matrix(
                y_test, ensemble_pred, "Ensemble",
                Path(config["outputs"]["plots_dir"]) / "cm_ensemble.png"
            )
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
    else:
        logger.warning("Not enough compatible models for ensemble.")

    if not all_metrics:
        raise RuntimeError("No models were successfully trained.")

    # Compare models
    metrics_df = pd.DataFrame(all_metrics)
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"\n{metrics_df.to_string(index=False)}")

    metrics_path = Path(config["outputs"]["results_dir"]) / "model_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to {metrics_path}")

    training_metrics_path = Path(config["outputs"]["results_dir"]) / "training_metrics.json"
    with open(training_metrics_path, "w") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2)
    logger.info(f"Saved training metrics JSON to {training_metrics_path}")

    # Save split info for reproducibility
    split_info = {
        "train_samples": sorted(pd.Series(train_groups).astype(str).tolist()),
        "test_samples": sorted(pd.Series(test_groups).astype(str).tolist()),
        "n_train_samples": int(pd.Series(train_groups).nunique()),
        "n_test_samples": int(pd.Series(test_groups).nunique()),
        "random_state": RANDOM_STATE,
        "test_size": config["models"]["test_size"]
    }
    split_info_path = Path(config["outputs"]["results_dir"]) / "train_test_split_samples.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Saved split info to {split_info_path}")

    # Also save a copy in outputs/ root for compatibility with other scripts
    split_info_root_path = Path("outputs") / "train_test_split_samples.json"
    with open(split_info_root_path, "w") as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Saved split info copy to {split_info_root_path}")

    # Select best model by balanced_accuracy
    best_model_name = metrics_df.loc[metrics_df["balanced_accuracy"].idxmax(), "model"]
    best_model_key = best_model_name.lower().replace(" ", "_")
    best_pipeline = all_models.get(best_model_key)

    logger.info(f"Best model: {best_model_name}")
    logger.info(f"  Balanced Accuracy: {metrics_df['balanced_accuracy'].max():.3f}")
    logger.info(
        f"  Macro F1: "
        f"{metrics_df.loc[metrics_df['model'] == best_model_name, 'macro_f1'].values[0]:.3f}"
    )

    if best_pipeline is not None:
        # Add metadata to fitted pipeline
        best_pipeline.feature_names = feature_names
        best_pipeline.model_version = "v1.1.0"
        best_pipeline.split_group_level = "Sample"
        best_pipeline.label_unit = "Sample+Coverage"
        best_pipeline.note = (
            "Models trained inside real pipelines. Train/test split performed at "
            "Sample level to prevent leakage across subsampled coverages of the same "
            "biological sample."
        )

        pipeline_path = Path(config["outputs"]["models_dir"]) / "best_model_pipeline.pkl"
        joblib.dump(best_pipeline, pipeline_path)
        logger.info(f"Saved model pipeline to {pipeline_path}")

        # Legacy artifacts
        imputer, scaler, model = extract_legacy_artifacts_from_pipeline(best_pipeline)

        model_path = Path(config["outputs"]["models_dir"]) / "best_model.pkl"
        scaler_path = Path(config["outputs"]["models_dir"]) / "scaler.pkl"
        imputer_path = Path(config["outputs"]["models_dir"]) / "imputer.pkl"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(imputer, imputer_path)

        logger.info(f"Saved best model to {model_path}")
        logger.info(f"Saved scaler to {scaler_path}")
        logger.info(f"Saved imputer to {imputer_path}")

        feature_path = Path(config["outputs"]["models_dir"]) / "feature_names.txt"
        with open(feature_path, "w") as f:
            f.write("\n".join(feature_names))
        logger.info(f"Saved feature names to {feature_path}")

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
