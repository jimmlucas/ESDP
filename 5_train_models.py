#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5_train_models.py - Multi-Model Training with Pipeline Bundling

PRODUCTION IMPROVEMENTS:
- Bundles imputer + scaler + model into single sklearn.Pipeline
- Handles NaN correctly (no fillna(0) - uses SimpleImputer)
- Saves feature_names and model_version as pipeline metadata
- Prevents training-serving skew
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import yaml
from collections import Counter
import json

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix,
    classification_report, cohen_kappa_score
)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# --- Logging configuration ---
log_file = config['logging']['file']
log_level = getattr(logging, config['logging']['level'].upper(), logging.INFO)
log_format = config['logging']['format']

# Ensure logs directory exists
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

# XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning("XGBoost not available")

# SMOTE
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    logging.warning("imbalanced-learn not available - SMOTE disabled")

# Ordinal Regression
try:
    import mord
    HAS_MORD = True
except ImportError:
    HAS_MORD = False
    logging.warning("mord not available - ordinal regression disabled")


RANDOM_STATE = config['models']['random_state']

def load_data():
    """Load labeled dataset."""
    df = pd.read_csv(config['data']['labeled_csv'])
    logger.info(f"Loaded {len(df)} rows")
    return df

def prepare_features(df):
    """
    Prepare feature matrix and labels.
    
    CRITICAL: Uses np.nan for missing values (not 0.0) to enable proper imputation.
    """
    logger.info("Preparing features...")
    
    # Select features
    feature_candidates = [
        # Base metrics
        'n50', 'qv', 'error_rate', 'busco_complete', 'busco_fragmented',
        'busco_missing', 'num_contigs', 'total_length', 'assembly_frac',
        'assembly_error',
        
        # Delta features
        'delta_qv', 'delta_busco_complete', 'delta_n50', 'delta_num_contigs',
        'delta_error_rate', 'delta_error_improvement', 'delta_assembly_error',
        
        # Ratio features
        'qv_improvement_rate', 'busco_per_contig', 'n50_fraction',
        'cost_benefit_ratio',
        
        # Cumulative features
        'delta_qv_cumsum', 'delta_busco_complete_cumsum',
        'score_improvement', 'gain_cumulative',
        
        # Normalized to R1
        'qv_from_r1', 'n50_from_r1', 'error_rate_from_r1',
        'busco_complete_from_r1', 'assembly_frac_from_r1',
        
        # Trend features
        'delta_qv_trend', 'delta_busco_complete_trend', 'score_improvement_trend',
        
        # Plateau features
        'is_plateau', 'plateau_streak',
        
        # Domain-specific
        'completeness_score', 'assembly_quality', 'polishing_effectiveness',
        
        # Policy flags
        'r1_ok_group',
        
        # Flye features
        'coverage_est', 'mean_edge_coverage', 'align_err_consensus',
    ]
    
    # Keep only available features
    features = [f for f in feature_candidates if f in df.columns]
    
    logger.info(f"Selected {len(features)} features")
    
    X = df[features].copy()
    
    # Use 3-class labels
    if 'optimal_rounds_3class' not in df.columns:
        raise ValueError("3-class labels not found! Run 4_label_optimal_round.py first")
    
    y = df['optimal_rounds_3class'].copy()
    
    # Convert to 0-indexed for sklearn compatibility
    y = y - 1  # Classes 1,2,3 -> 0,1,2
    
    # Group identifier
    cov_col = 'Coverage_effective' if 'Coverage_effective' in df.columns else 'Coverage'
    groups = df['Sample'].astype(str) + '|' + df[cov_col].astype(str)
    
    # ✅ CRITICAL: Replace infinite values with NaN (not 0)
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # ✅ CRITICAL: Keep NaN as NaN (don't fillna(0))
    # The pipeline's SimpleImputer will handle this
    
    return X, y, groups, features

def stratified_group_split(X, y, groups, test_size=0.2, random_state=42):
    """Split data by groups (not rows) to avoid leakage."""
    logger.info("Performing stratified group split...")
    
    # Get one label per group
    group_df = pd.DataFrame({'group': groups, 'label': y}).drop_duplicates('group')
    
    # Count samples per class
    class_counts = group_df['label'].value_counts().sort_index()
    logger.info(f"Groups per class:\n{class_counts}")
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    
    try:
        train_groups, test_groups = train_test_split(
            group_df['group'],
            test_size=test_size,
            stratify=group_df['label'],
            random_state=random_state
        )
    except ValueError as e:
        logger.warning(f"Stratification failed: {e}. Using random split.")
        train_groups, test_groups = train_test_split(
            group_df['group'],
            test_size=test_size,
            random_state=random_state
        )
    
    # Create masks
    train_mask = groups.isin(train_groups)
    test_mask = groups.isin(test_groups)
    
    X_train = X[train_mask].copy()
    X_test = X[test_mask].copy()
    y_train = y[train_mask].copy()
    y_test = y[test_mask].copy()
    
    logger.info(f"Train: {len(X_train)} rows, {train_groups.nunique()} groups")
    logger.info(f"Test: {len(X_test)} rows, {test_groups.nunique()} groups")
    logger.info(f"Train distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
    logger.info(f"Test distribution:\n{pd.Series(y_test).value_counts().sort_index()}")
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    """Apply SMOTE to balance classes."""
    if not HAS_SMOTE:
        logger.warning("SMOTE not available - skipping")
        return X_train, y_train
    
    logger.info("Applying SMOTE...")
    
    # Check if we have enough samples for SMOTE
    class_counts = pd.Series(y_train).value_counts()
    min_class = class_counts.min()
    
    if min_class < 2:
        logger.warning(f"Minimum class has only {min_class} samples - SMOTE not applicable")
        return X_train, y_train
    
    # Use k_neighbors = min(5, min_class - 1)
    k_neighbors = min(config['imbalance']['smote_k_neighbors'], min_class - 1)
    
    try:
        smote = SMOTE(
            sampling_strategy=config['imbalance']['smote_sampling_strategy'],
            k_neighbors=k_neighbors,
            random_state=RANDOM_STATE
        )
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        logger.info(f"Original distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
        logger.info(f"Resampled distribution:\n{pd.Series(y_resampled).value_counts().sort_index()}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        logger.error(f"SMOTE failed: {e}")
        return X_train, y_train

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier."""
    if not HAS_XGBOOST:
        logger.warning("XGBoost not available - skipping")
        return None, {}
    
    logger.info("Training XGBoost...")
    
    # Get class weights
    class_weights_config = config['imbalance']['class_weights']
    class_weights = {k-1: v for k, v in class_weights_config.items()}
    sample_weights = pd.Series(y_train).map(class_weights).fillna(1.0).values
    sample_weights = np.maximum(sample_weights, 0.1)
    
    # Train model
    model = XGBClassifier(
        **config['models']['xgboost'],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, "XGBoost")
    
    return model, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier."""
    logger.info("Training Random Forest...")
    
    # Get class weights
    class_weights_config = config['imbalance']['class_weights']
    class_weights = {k-1: v for k, v in class_weights_config.items()}
    
    model = RandomForestClassifier(
        **config['models']['random_forest'],
        class_weight=class_weights,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, "Random Forest")
    
    return model, metrics

def train_ordinal_regression(X_train, y_train, X_test, y_test):
    """Train ordinal regression model."""
    if not HAS_MORD:
        logger.warning("mord not available - skipping ordinal regression")
        return None, {}
    
    logger.info("Training Ordinal Regression...")
    
    try:
        model = mord.LogisticAT(alpha=config['models']['ordinal_regression']['alpha'])
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred, "Ordinal Regression")
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"Ordinal regression failed: {e}")
        return None, {}

def train_ensemble(models, X_train, y_train, X_test, y_test):
    """Train ensemble voting classifier."""
    logger.info("Training Ensemble...")
    
    available_models = []
    for name, model in models.items():
        if model is None:
            continue
        
        lname = name.lower()
        if "ordinal" in lname:
            logger.info(f"Skipping model '{name}' in ensemble (ordinal regression is not a sklearn classifier).")
            continue
        
        available_models.append((name, model))
    
    if len(available_models) < 2:
        logger.warning("Not enough compatible models for ensemble.")
        return None, {}
    
    ensemble = VotingClassifier(
        estimators=available_models,
        voting='soft',
        n_jobs=-1
    )
    
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, "Ensemble")
    
    return ensemble, metrics

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """Calculate comprehensive evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Ordinal metrics
    mae = np.mean(np.abs(y_true - y_pred))
    acc_pm1 = np.mean(np.abs(y_true - y_pred) <= 1)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    metrics = {
        'model': model_name,
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'mae': mae,
        'accuracy_pm1': acc_pm1,
        'qwk': qwk,
    }
    
    # Add per-class metrics
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        metrics[f'precision_class_{i+1}'] = p
        metrics[f'recall_class_{i+1}'] = r
        metrics[f'f1_class_{i+1}'] = f
        metrics[f'support_class_{i+1}'] = s
    
    # Check if meets targets
    targets = config['evaluation']['target_metrics']
    metrics['meets_targets'] = (
        balanced_acc >= targets['balanced_accuracy'] and
        macro_f1 >= targets['macro_f1'] and
        mae <= targets['mae'] and
        qwk >= targets['qwk']
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
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Early', 'Medium', 'Late'], 
                yticklabels=['Early', 'Medium', 'Late'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {save_path}")

def plot_feature_importance(model, feature_names, model_name, save_path):
    """Plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        logger.warning(f"Cannot extract feature importance for {model_name}")
        return
    
    # Get top 20 features
    indices = np.argsort(importances)[-20:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Feature Importances - {model_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved feature importance to {save_path}")

def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Multi-Model Training Pipeline")
    logger.info("=" * 60)
    
    # Create output directories
    Path(config['outputs']['models_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['outputs']['plots_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['outputs']['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    df = load_data()
    X, y, groups, feature_names = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = stratified_group_split(
        X, y, groups,
        test_size=config['models']['test_size'],
        random_state=RANDOM_STATE
    )
    
    # ✅ CRITICAL: Create preprocessing pipeline (imputer + scaler)
    logger.info("Creating preprocessing pipeline...")
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    # Fit imputer and scaler on training data
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Transform test data
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Apply SMOTE
    if config['imbalance']['use_smote']:
        X_train_scaled, y_train = apply_smote(X_train_scaled, y_train)
    
    # Convert back to DataFrame for models that need it
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Train models
    all_models = {}
    all_metrics = []
    
    # XGBoost
    xgb_model, xgb_metrics = train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
    if xgb_model:
        all_models['xgboost'] = xgb_model
        all_metrics.append(xgb_metrics)
        plot_confusion_matrix(y_test, xgb_model.predict(X_test_scaled), "XGBoost",
                    Path(config['outputs']['plots_dir']) / 'cm_xgboost.png')
        plot_feature_importance(xgb_model, feature_names, "XGBoost",
                    Path(config['outputs']['plots_dir']) / 'fi_xgboost.png')
    
    # Random Forest
    rf_model, rf_metrics = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    if rf_model:
        all_models['random_forest'] = rf_model
        all_metrics.append(rf_metrics)
        plot_confusion_matrix(y_test, rf_model.predict(X_test_scaled), "Random Forest",
                    Path(config['outputs']['plots_dir']) / 'cm_random_forest.png')
        plot_feature_importance(rf_model, feature_names, "Random Forest",
                    Path(config['outputs']['plots_dir']) / 'fi_random_forest.png')
    
    # Ordinal Regression
    ord_model, ord_metrics = train_ordinal_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    if ord_model:
        all_models['ordinal'] = ord_model
        all_metrics.append(ord_metrics)
        plot_confusion_matrix(y_test, ord_model.predict(X_test_scaled), "Ordinal Regression",
                    Path(config['outputs']['plots_dir']) / 'cm_ordinal.png')
    
    # Ensemble
    ensemble_model, ensemble_metrics = train_ensemble(all_models, X_train_scaled, y_train, X_test_scaled, y_test)
    if ensemble_model:
        all_models['ensemble'] = ensemble_model
        all_metrics.append(ensemble_metrics)
        plot_confusion_matrix(y_test, ensemble_model.predict(X_test_scaled), "Ensemble",
                    Path(config['outputs']['plots_dir']) / 'cm_ensemble.png')
    
    # Compare models
    metrics_df = pd.DataFrame(all_metrics)
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"\n{metrics_df.to_string()}")
    
    # Save metrics
    metrics_path = Path(config['outputs']['results_dir']) / 'model_comparison.csv'
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"\nSaved metrics to {metrics_path}")

    # Save metrics as JSON
    training_metrics_path = Path(config['outputs']['results_dir']) / 'training_metrics.json'
    with open(training_metrics_path, "w") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2)
    logger.info(f"Saved training metrics JSON to {training_metrics_path}")
    
    # Select best model
    best_model_name = metrics_df.loc[metrics_df['balanced_accuracy'].idxmax(), 'model']
    best_model = all_models.get(best_model_name.lower().replace(' ', '_'))
    
    logger.info(f"\nBest model: {best_model_name}")
    logger.info(f"  Balanced Accuracy: {metrics_df['balanced_accuracy'].max():.3f}")
    logger.info(f"  Macro F1: {metrics_df.loc[metrics_df['model'] == best_model_name, 'macro_f1'].values[0]:.3f}")
    
    # ✅ CRITICAL: Save bundled pipeline (imputer + scaler + model)
    if best_model:
        # Create pipeline
        pipeline = Pipeline([
            ('imputer', imputer),
            ('scaler', scaler),
            ('model', best_model)
        ])
        
        # Add metadata
        pipeline.feature_names = feature_names
        pipeline.model_version = "v1.0.0"
        
        # Save pipeline
        pipeline_path = Path(config['outputs']['models_dir']) / 'best_model_pipeline.pkl'
        joblib.dump(pipeline, pipeline_path)
        logger.info(f"Saved model pipeline to {pipeline_path}")
        
        # Also save legacy files for backward compatibility
        model_path = Path(config['outputs']['models_dir']) / 'best_model.pkl'
        scaler_path = Path(config['outputs']['models_dir']) / 'scaler.pkl'
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Saved best model to {model_path}")
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save feature names
        feature_path = Path(config['outputs']['models_dir']) / 'feature_names.txt'
        with open(feature_path, 'w') as f:
            f.write('\n'.join(feature_names))
        logger.info(f"Saved feature names to {feature_path}")
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()