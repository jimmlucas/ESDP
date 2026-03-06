#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8_evaluate_models.py - Evaluación formal y comparación con baselines.

CORREGIDO:
- Evita data leakage evaluando con split por Sample (no Sample+Coverage)
- Reutiliza exactamente el mismo split guardado por 5_train_models.py si existe
- Carga best_model_pipeline.pkl (imputer + scaler + model)
- Calcula métricas completas en test
- Compara contra baselines:
    * Baseline 1: siempre clase Late (clase 3 -> índice 2)
    * Baseline 2: umbral de QV (Early si QV > 30, si no Late)
    * Baseline 3: Random Forest usando features derivadas de R1
- Calcula intervalos de confianza por bootstrap
- Guarda resultados en outputs/baseline_comparison.csv
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
)

# -----------------------------------------------------------------------------
# Config y logging
# -----------------------------------------------------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

logging.basicConfig(
    level=getattr(logging, config.get("logging", {}).get("level", "INFO").upper(), logging.INFO),
    format=config.get(
        "logging", {}
    ).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger = logging.getLogger("evaluate_models")

RANDOM_STATE = config["models"]["random_state"]


# -----------------------------------------------------------------------------
# Utilidades de datos
# -----------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Carga el dataset etiquetado."""
    df = pd.read_csv(config["data"]["labeled_csv"])
    logger.info(f"Loaded {len(df)} rows")
    return df


def prepare_features(df: pd.DataFrame):
    """
    Construye X, y y groups de forma coherente con 5_train_models.py.

    - X: columnas usadas por el modelo entrenado (feature_names.txt)
    - y: optimal_rounds_3class recodificada a 0,1,2
    - groups: Sample (para evitar leakage entre coberturas de la misma muestra)
    """
    feature_path = Path(config["outputs"]["models_dir"]) / "feature_names.txt"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"No se encontró {feature_path}. Ejecuta primero 5_train_models.py."
        )

    feature_names = [
        line.strip()
        for line in feature_path.read_text().splitlines()
        if line.strip()
    ]
    logger.info(f"Using {len(feature_names)} features from feature_names.txt")

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(
            "Las siguientes features del modelo no están en el CSV etiquetado: "
            + ", ".join(missing)
        )

    X = df[feature_names].copy()

    if "optimal_rounds_3class" not in df.columns:
        raise ValueError("3-class labels not found! Run 4_label_optimal_round.py first")

    y = df["optimal_rounds_3class"].copy() - 1  # 1,2,3 -> 0,1,2

    # CRÍTICO: split por Sample
    groups = df["Sample"].astype(str)

    # Coherente con training: infinitos -> NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y, groups, feature_names


def grouped_sample_split(X, y, groups, test_size=0.2, random_state=42):
    """
    Split por Sample para evitar leakage entre coberturas derivadas
    del mismo aislado.
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
    logger.info(f"Train distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
    logger.info(f"Test distribution:\n{pd.Series(y_test).value_counts().sort_index()}")

    return X_train, X_test, y_train, y_test, train_groups, test_groups


def load_saved_split(groups: pd.Series):
    """
    Carga el split guardado por 5_train_models.py si existe.
    Devuelve train_groups y test_groups o (None, None) si no existe.
    """
    split_info_path = Path(config["outputs"]["results_dir"]) / "train_test_split_samples.json"

    if not split_info_path.exists():
        logger.warning(f"No split file found at {split_info_path}. Recomputing split.")
        return None, None

    with open(split_info_path, "r") as f:
        split_info = json.load(f)

    train_groups = pd.Series(split_info["train_samples"], dtype=str)
    test_groups = pd.Series(split_info["test_samples"], dtype=str)

    all_groups = set(pd.Series(groups).astype(str).unique())
    saved_groups = set(train_groups).union(set(test_groups))

    if not saved_groups.issubset(all_groups):
        logger.warning("Saved split contains samples not present in current dataset. Recomputing split.")
        return None, None

    logger.info(f"Loaded saved split from {split_info_path}")
    logger.info(f"Saved split: {len(train_groups)} train samples, {len(test_groups)} test samples")

    return train_groups, test_groups


def apply_saved_split(X, y, groups, train_groups, test_groups):
    """Aplica un split guardado previamente."""
    train_mask = groups.isin(train_groups)
    test_mask = groups.isin(test_groups)

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    logger.info("Applied saved sample-level split")
    logger.info(f"Train: {len(X_train)} rows, {pd.Series(train_groups).nunique()} samples")
    logger.info(f"Test: {len(X_test)} rows, {pd.Series(test_groups).nunique()} samples")
    logger.info(f"Train distribution:\n{pd.Series(y_train).value_counts().sort_index()}")
    logger.info(f"Test distribution:\n{pd.Series(y_test).value_counts().sort_index()}")

    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# Métricas
# -----------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """Calcula métricas completas."""
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

    for i in range(len(precision)):
        metrics[f"precision_class_{i+1}"] = float(precision[i])
        metrics[f"recall_class_{i+1}"] = float(recall[i])
        metrics[f"f1_class_{i+1}"] = float(f1_vals[i])
        metrics[f"support_class_{i+1}"] = int(support[i])

    targets = config["evaluation"]["target_metrics"]
    metrics["meets_targets"] = bool(
        (balanced_acc >= targets["balanced_accuracy"])
        and (macro_f1 >= targets["macro_f1"])
        and (mae <= targets["mae"])
        and (qwk >= targets["qwk"])
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


def _core_metrics(y_true, y_pred):
    """Métricas núcleo sin logging para bootstrap."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, random_state=42):
    """Bootstrap para balanced accuracy, macro-F1 y MAE."""
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    ba_vals, mf1_vals, mae_vals = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        m = _core_metrics(y_true[idx], y_pred[idx])
        ba_vals.append(m["balanced_accuracy"])
        mf1_vals.append(m["macro_f1"])
        mae_vals.append(m["mae"])

    def _summary(arr):
        arr = np.asarray(arr)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    return {
        "balanced_accuracy": _summary(ba_vals),
        "macro_f1": _summary(mf1_vals),
        "mae": _summary(mae_vals),
    }


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------

def predict_by_qv(X_test: pd.DataFrame, threshold: float = 30.0) -> np.ndarray:
    """
    Baseline por QV.
    Usa qv si existe entre las features seleccionadas.
    """
    if "qv" not in X_test.columns:
        raise ValueError("No se encontró la feature 'qv' en X_test")

    qv_vals = X_test["qv"].values
    return np.where(qv_vals > threshold, 0, 2)  # 0=Early, 2=Late


def get_r1_features(feature_names, X_train_columns):
    """
    Selecciona un subconjunto razonable de features ligadas a R1
    para el baseline R1-only.
    """
    r1_features = [
        c for c in feature_names
        if (
            c.endswith("_from_r1")
            or c in [
                "coverage_est",
                "mean_edge_coverage",
                "align_err_consensus",
                "r1_ok_group",
            ]
        )
    ]

    r1_features = [f for f in r1_features if f in X_train_columns]
    return r1_features


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Evaluación formal del mejor modelo y baselines")
    logger.info("=" * 60)

    # 1) Cargar datos y preparar features
    df = load_data()
    X, y, groups, feature_names = prepare_features(df)
    logger.info(f"Total filas: {len(X)} - Features: {len(feature_names)}")

    # 2) Reusar exactamente el split de training si existe
    train_groups, test_groups = load_saved_split(groups)

    if train_groups is not None and test_groups is not None:
        X_train, X_test, y_train, y_test = apply_saved_split(
            X, y, groups, train_groups, test_groups
        )
    else:
        test_size = config["models"]["test_size"]
        X_train, X_test, y_train, y_test, train_groups, test_groups = grouped_sample_split(
            X, y, groups, test_size=test_size, random_state=RANDOM_STATE
        )

    # 3) Cargar pipeline completo
    models_dir = Path(config["outputs"]["models_dir"])
    pipeline_path = models_dir / "best_model_pipeline.pkl"

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"No se encontró {pipeline_path}. Ejecuta primero 5_train_models.py."
        )

    pipeline = joblib.load(pipeline_path)
    logger.info(f"Cargado pipeline completo desde {pipeline_path}")

    # 4) Evaluar modelo principal
    y_pred_model = pipeline.predict(X_test)
    metrics_model = calculate_metrics(y_test, y_pred_model, model_name="Best_Model")

    # 5) Baseline 1: siempre Late
    y_baseline_late = np.full(shape=len(y_test), fill_value=2, dtype=int)
    metrics_late = calculate_metrics(
        y_test,
        y_baseline_late,
        model_name="Baseline_Always_Late",
    )

    # 6) Baseline 2: umbral de QV
    try:
        y_baseline_qv = predict_by_qv(X_test, threshold=30.0)
        metrics_qv = calculate_metrics(
            y_test,
            y_baseline_qv,
            model_name="Baseline_QV_Threshold_30",
        )
    except ValueError as e:
        logger.warning(f"No se pudo calcular baseline QV: {e}")
        metrics_qv = None

    # 7) Baseline 3: RF con features derivadas de R1
    r1_features = get_r1_features(feature_names, X_train.columns)

    if len(r1_features) == 0:
        logger.warning("No hay suficientes features tipo R1 para el baseline R1-only. Se omite.")
        metrics_r1 = None
    else:
        logger.info(f"Baseline R1-only usando {len(r1_features)} features: {r1_features}")

        # Coherente con training: imputar + escalar, sin fillna(0)
        imputer_r1 = SimpleImputer(strategy="median")
        scaler_r1 = StandardScaler()

        X_train_r1 = imputer_r1.fit_transform(X_train[r1_features])
        X_test_r1 = imputer_r1.transform(X_test[r1_features])

        X_train_r1 = scaler_r1.fit_transform(X_train_r1)
        X_test_r1 = scaler_r1.transform(X_test_r1)

        clf_r1 = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf_r1.fit(X_train_r1, y_train)
        y_baseline_r1 = clf_r1.predict(X_test_r1)

        metrics_r1 = calculate_metrics(
            y_test,
            y_baseline_r1,
            model_name="Baseline_R1_Only_RF",
        )

    # 8) Bootstrap para modelo principal
    logger.info("Calculando intervalos de confianza por bootstrap...")
    ci = bootstrap_ci(y_test, y_pred_model, n_bootstrap=1000, random_state=RANDOM_STATE)

    logger.info("\n" + "=" * 60)
    logger.info("BOOTSTRAP 95% CI - Best_Model")
    logger.info("=" * 60)
    for metric, stats in ci.items():
        logger.info(
            f"{metric}: mean={stats['mean']:.4f} sd={stats['std']:.4f} "
            f"CI95%=[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
        )

    # 9) Guardar resultados
    rows = [metrics_model, metrics_late]
    if metrics_qv is not None:
        rows.append(metrics_qv)
    if metrics_r1 is not None:
        rows.append(metrics_r1)

    results_df = pd.DataFrame(rows)

    out_dir = Path(config["outputs"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "baseline_comparison.csv"
    results_df.to_csv(out_path, index=False)

    # Guardar bootstrap CI
    ci_path = out_dir / "best_model_bootstrap_ci.json"
    with open(ci_path, "w") as f:
        json.dump(ci, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN COMPARATIVO (modelo vs baselines)")
    logger.info("=" * 60)
    logger.info(
        "\n" + results_df[
            ["model", "balanced_accuracy", "macro_f1", "mae", "qwk"]
        ].to_string(index=False)
    )
    logger.info(f"\nGuardado baseline_comparison.csv en {out_path}")
    logger.info(f"Guardado bootstrap CI en {ci_path}")


if __name__ == "__main__":
    main()