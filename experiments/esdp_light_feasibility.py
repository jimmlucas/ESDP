#!/usr/bin/env python3
"""Training-only feasibility study for prospective ESDP-light candidates.

This experiment never evaluates, tunes, or selects on the frozen held-out test
samples. It produces out-of-fold predictions for the 32 development samples
using sample-grouped cross-validation and prospective trajectory simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from esdp_light import (
    LIGHT_CORE_FEATURES,
    LIGHT_PROSPECTIVE_DERIVED_FEATURES,
    validate_frozen_sample_split,
)
from esdp_light_features import LIGHT_DYNAMIC_METRICS, LightFeatureBuilder


RANDOM_STATE = 42
N_SPLITS = 5
N_ESTIMATORS = 800
CONFIDENCE_THRESHOLD = 0.60
ACCEPTABLE_QV_LOSS = 0.5
ACCEPTABLE_BUSCO_LOSS = 1.0
N_BOOTSTRAP = 1000
EXPLORATORY_THRESHOLDS = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
CLASS_TO_MINIMUM_ROUND = {0: 1, 1: 3, 2: 5}
PRIMARY_BOOTSTRAP_METRICS = (
    "unsafe_early_stop",
    "quality_failure",
    "rounds_saved",
    "absolute_round_error",
)


@dataclass(frozen=True)
class Candidate:
    """One frozen feature and inference policy comparison."""

    name: str
    feature_names: tuple[str, ...]
    adaptive: bool
    description: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coverage_number(series: pd.Series) -> pd.Series:
    values = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"X$", "", regex=True)
    )
    numeric = pd.to_numeric(values, errors="raise")
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError("Coverage_effective must contain positive finite X values")
    return numeric.astype(float)


def prepare_training_frame(
    dataset_path: Path,
    split_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Load, guard, and causally rebuild development-only features."""
    frame = pd.read_csv(dataset_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validate_frozen_sample_split(
        frame,
        split["train_samples"],
        split["test_samples"],
    )

    train_samples = set(map(str, split["train_samples"]))
    test_samples = set(map(str, split["test_samples"]))
    training = frame[frame["Sample"].astype(str).isin(train_samples)].copy()
    observed_samples = set(training["Sample"].astype(str))
    if observed_samples != train_samples:
        missing = sorted(train_samples - observed_samples)
        raise ValueError(f"training split samples absent from dataset: {missing}")
    overlap = observed_samples & test_samples
    if overlap:
        raise RuntimeError(f"held-out test samples entered development: {sorted(overlap)}")

    required = {
        "Sample",
        "Coverage_effective",
        "round",
        "optimal_rounds_3class",
        "optimal_rounds_5class",
        "qv",
        "busco_complete",
        *LIGHT_DYNAMIC_METRICS,
    }
    missing_columns = sorted(required - set(training.columns))
    if missing_columns:
        raise ValueError(f"dataset missing experiment columns: {missing_columns}")

    training["Coverage_effective_numeric"] = _coverage_number(
        training["Coverage_effective"]
    )
    group_columns = ["Sample", "Coverage_effective"]
    for _, trajectory in training.groupby(group_columns, sort=False):
        rounds = sorted(trajectory["round"].astype(int).tolist())
        if rounds != [1, 2, 3, 4, 5]:
            raise ValueError(f"development trajectory lacks complete R1-R5: {rounds}")
        if trajectory["optimal_rounds_3class"].nunique() != 1:
            raise ValueError("3-class target changes within one trajectory")
        if trajectory["optimal_rounds_5class"].nunique() != 1:
            raise ValueError("5-class target changes within one trajectory")

    prospective_input = training[
        ["Sample", "round", *LIGHT_DYNAMIC_METRICS]
    ].copy()
    prospective_input["Coverage_effective"] = training[
        "Coverage_effective_numeric"
    ].to_numpy()
    prospective = LightFeatureBuilder().transform(prospective_input)
    prospective = prospective.rename(
        columns={"Coverage_effective": "Coverage_effective_numeric"}
    )
    derived = prospective[
        [
            "Sample",
            "Coverage_effective_numeric",
            "round",
            *LIGHT_PROSPECTIVE_DERIVED_FEATURES,
        ]
    ]
    training = training.drop(
        columns=[
            name
            for name in LIGHT_PROSPECTIVE_DERIVED_FEATURES
            if name in training.columns
        ]
    )
    training = training.merge(
        derived,
        on=["Sample", "Coverage_effective_numeric", "round"],
        how="left",
        validate="one_to_one",
    )
    if len(training) != len(prospective):
        raise RuntimeError("prospective feature merge changed development rows")

    training = training.sort_values(
        ["Sample", "Coverage_effective", "round"]
    ).reset_index(drop=True)
    audit = {
        "dataset_sha256": _sha256(dataset_path),
        "split_sha256": _sha256(split_path),
        "development_samples": len(train_samples),
        "held_out_samples_reserved": len(test_samples),
        "held_out_samples_used": [],
        "development_rows": len(training),
        "development_trajectories": int(
            training[group_columns].drop_duplicates().shape[0]
        ),
    }
    return training, audit


def define_candidates(repository: Path, frame: pd.DataFrame) -> tuple[Candidate, ...]:
    """Freeze comparisons before any cross-validation result is inspected."""
    legacy_features = tuple(
        line.strip()
        for line in (repository / "models" / "feature_names.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )
    light_core = (*LIGHT_CORE_FEATURES, *LIGHT_PROSPECTIVE_DERIVED_FEATURES)
    light_fasta = (*LIGHT_DYNAMIC_METRICS, *LIGHT_PROSPECTIVE_DERIVED_FEATURES)
    candidates = (
        Candidate(
            "legacy_full_retrospective",
            legacy_features,
            True,
            "Optimistic benchmark containing expensive and leakage-prone legacy features.",
        ),
        Candidate(
            "light_core_prospective",
            tuple(light_core),
            True,
            "Deployment-approved low-cost raw metrics plus causal temporal features.",
        ),
        Candidate(
            "light_fasta_prospective",
            tuple(light_fasta),
            True,
            "Currently operational FASTA-only metrics plus causal temporal features.",
        ),
        Candidate(
            "r1_core_only",
            tuple(LIGHT_CORE_FEATURES),
            False,
            "Low-cost raw metrics observed at R1 with a fixed planned final round.",
        ),
    )
    for candidate in candidates:
        missing = sorted(set(candidate.feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"{candidate.name} missing features: {missing}")
        if len(candidate.feature_names) != len(set(candidate.feature_names)):
            raise ValueError(f"{candidate.name} contains duplicate features")
    return candidates


def make_sample_folds(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign every biological sample to exactly one validation fold."""
    anchor = frame[frame["round"] == 1].copy()
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    assignments: dict[str, int] = {}
    y = anchor["optimal_rounds_3class"].astype(int) - 1
    groups = anchor["Sample"].astype(str)
    for fold, (_, validation_index) in enumerate(
        splitter.split(anchor, y, groups),
        start=1,
    ):
        validation_samples = set(groups.iloc[validation_index])
        for sample in validation_samples:
            if sample in assignments:
                raise RuntimeError(f"sample assigned to multiple folds: {sample}")
            assignments[sample] = fold
    all_samples = set(frame["Sample"].astype(str))
    if set(assignments) != all_samples:
        raise RuntimeError("fold assignment does not cover development samples")
    return pd.DataFrame(
        sorted(assignments.items()),
        columns=["Sample", "fold"],
    )


def build_model(random_state: int) -> Pipeline:
    """Use one frozen estimator for every feature-family comparison."""
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_depth=12,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight={0: 2.0, 1: 1.5, 2: 1.0},
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _round_class(round_number: int) -> int:
    if round_number <= 2:
        return 0
    if round_number <= 4:
        return 1
    return 2


def _trajectory_result(
    trajectory: pd.DataFrame,
    *,
    candidate: str,
    fold: int,
    selected_round: int,
    predicted_class: int,
    confidence: float | None,
) -> dict:
    trajectory = trajectory.sort_values("round")
    selected = trajectory[trajectory["round"] == selected_round].iloc[0]
    final = trajectory[trajectory["round"] == 5].iloc[0]
    true_class = int(final["optimal_rounds_3class"]) - 1
    optimal_round = int(final["optimal_rounds_5class"])
    qv_loss = float(final["qv"] - selected["qv"])
    busco_loss = float(final["busco_complete"] - selected["busco_complete"])
    premature_rounds = max(optimal_round - selected_round, 0)
    excess_rounds = max(selected_round - optimal_round, 0)
    return {
        "candidate": candidate,
        "fold": fold,
        "Sample": str(final["Sample"]),
        "Coverage_effective": str(final["Coverage_effective"]),
        "true_class": true_class,
        "optimal_round": optimal_round,
        "predicted_class_at_stop": predicted_class,
        "selected_class": _round_class(selected_round),
        "selected_round": selected_round,
        "confidence_at_stop": confidence,
        "unsafe_early_stop": int(premature_rounds > 0),
        "severe_unsafe_stop": int(premature_rounds >= 2),
        "premature_rounds": premature_rounds,
        "excess_rounds": excess_rounds,
        "absolute_round_error": abs(selected_round - optimal_round),
        "rounds_saved": 5 - selected_round,
        "qv_loss_vs_r5": qv_loss,
        "busco_loss_vs_r5": busco_loss,
        "quality_failure": int(
            qv_loss > ACCEPTABLE_QV_LOSS
            or busco_loss > ACCEPTABLE_BUSCO_LOSS
        ),
    }


def simulate_candidate(
    model: Pipeline,
    validation: pd.DataFrame,
    candidate: Candidate,
    fold: int,
) -> tuple[list[dict], list[dict]]:
    """Apply a conservative sequential stop rule to held-out fold trajectories."""
    results = []
    round_results = []
    for _, trajectory in validation.groupby(
        ["Sample", "Coverage_effective"],
        sort=True,
    ):
        trajectory = trajectory.sort_values("round")
        if candidate.adaptive:
            trajectory_predictions = []
            for _, row in trajectory.iterrows():
                current_round = int(row["round"])
                features = pd.DataFrame(
                    [[row[name] for name in candidate.feature_names]],
                    columns=candidate.feature_names,
                ).replace([np.inf, -np.inf], np.nan)
                predicted_class = int(model.predict(features)[0])
                probability = model.predict_proba(features)[0]
                confidence = float(probability[predicted_class])
                trajectory_predictions.append(
                    (current_round, predicted_class, confidence)
                )
                round_results.append(
                    {
                        "candidate": candidate.name,
                        "fold": fold,
                        "Sample": str(row["Sample"]),
                        "Coverage_effective": str(row["Coverage_effective"]),
                        "round": current_round,
                        "true_class": int(row["optimal_rounds_3class"]) - 1,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        **{
                            f"probability_class_{class_index}": float(
                                probability[class_index]
                            )
                            for class_index in range(3)
                        },
                    }
                )
            selected_round, predicted_class, confidence = trajectory_predictions[-1]
            for current_round, current_class, current_confidence in (
                trajectory_predictions
            ):
                eligible = current_round >= CLASS_TO_MINIMUM_ROUND[current_class]
                if current_round == 5 or (
                    eligible and current_confidence >= CONFIDENCE_THRESHOLD
                ):
                    selected_round = current_round
                    predicted_class = current_class
                    confidence = current_confidence
                    break
        else:
            row = trajectory[trajectory["round"] == 1].iloc[0]
            features = pd.DataFrame(
                [[row[name] for name in candidate.feature_names]],
                columns=candidate.feature_names,
            ).replace([np.inf, -np.inf], np.nan)
            predicted_class = int(model.predict(features)[0])
            probability = model.predict_proba(features)[0]
            confidence = float(probability[predicted_class])
            round_results.append(
                {
                    "candidate": candidate.name,
                    "fold": fold,
                    "Sample": str(row["Sample"]),
                    "Coverage_effective": str(row["Coverage_effective"]),
                    "round": 1,
                    "true_class": int(row["optimal_rounds_3class"]) - 1,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    **{
                        f"probability_class_{class_index}": float(
                            probability[class_index]
                        )
                        for class_index in range(3)
                    },
                }
            )
            selected_round = (
                CLASS_TO_MINIMUM_ROUND[predicted_class]
                if confidence >= CONFIDENCE_THRESHOLD
                else 5
            )
        results.append(
            _trajectory_result(
                trajectory,
                candidate=candidate.name,
                fold=fold,
                selected_round=selected_round,
                predicted_class=predicted_class,
                confidence=confidence,
            )
        )
    return results, round_results


def summarize_round_predictions(round_predictions: pd.DataFrame) -> pd.DataFrame:
    """Report discrimination and calibration independently of stop policy."""
    rows = []
    probability_columns = [f"probability_class_{index}" for index in range(3)]
    for (candidate, round_number), group in round_predictions.groupby(
        ["candidate", "round"],
        sort=True,
    ):
        y_true = group["true_class"].to_numpy(dtype=int)
        y_pred = group["predicted_class"].to_numpy(dtype=int)
        probabilities = group[probability_columns].to_numpy(dtype=float)
        one_hot = np.eye(3)[y_true]
        recalls = recall_score(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
        rows.append(
            {
                "candidate": candidate,
                "round": int(round_number),
                "n_predictions": len(group),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "min_class_recall": float(np.min(recalls)),
                "quadratic_weighted_kappa": cohen_kappa_score(
                    y_true,
                    y_pred,
                    weights="quadratic",
                ),
                "multiclass_brier": float(
                    np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
                ),
                "log_loss": log_loss(
                    y_true,
                    probabilities,
                    labels=[0, 1, 2],
                ),
                "mean_confidence": group["confidence"].mean(),
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    frame: pd.DataFrame,
    round_predictions: pd.DataFrame,
    candidates: tuple[Candidate, ...],
) -> pd.DataFrame:
    """Describe, but do not select, the development safety-savings frontier."""
    candidate_map = {candidate.name: candidate for candidate in candidates}
    trajectory_map = {
        (str(sample), str(coverage)): trajectory
        for (sample, coverage), trajectory in frame.groupby(
            ["Sample", "Coverage_effective"],
            sort=True,
        )
    }
    rows = []
    for threshold in EXPLORATORY_THRESHOLDS:
        threshold_results = []
        for candidate_name, candidate_predictions in round_predictions.groupby(
            "candidate",
            sort=True,
        ):
            candidate = candidate_map[candidate_name]
            for key, trajectory_predictions in candidate_predictions.groupby(
                ["Sample", "Coverage_effective"],
                sort=True,
            ):
                trajectory_predictions = trajectory_predictions.sort_values("round")
                if candidate.adaptive:
                    selected = trajectory_predictions.iloc[-1]
                    selected_round = 5
                    for _, prediction in trajectory_predictions.iterrows():
                        current_round = int(prediction["round"])
                        predicted_class = int(prediction["predicted_class"])
                        eligible = (
                            current_round
                            >= CLASS_TO_MINIMUM_ROUND[predicted_class]
                        )
                        if current_round == 5 or (
                            eligible and prediction["confidence"] >= threshold
                        ):
                            selected = prediction
                            selected_round = current_round
                            break
                else:
                    selected = trajectory_predictions.iloc[0]
                    selected_round = (
                        CLASS_TO_MINIMUM_ROUND[int(selected["predicted_class"])]
                        if selected["confidence"] >= threshold
                        else 5
                    )
                threshold_results.append(
                    _trajectory_result(
                        trajectory_map[(str(key[0]), str(key[1]))],
                        candidate=candidate_name,
                        fold=int(selected["fold"]),
                        selected_round=selected_round,
                        predicted_class=int(selected["predicted_class"]),
                        confidence=float(selected["confidence"]),
                    )
                )
        threshold_summary = summarize_predictions(
            pd.DataFrame(threshold_results)
        )
        threshold_summary.insert(1, "confidence_threshold", threshold)
        rows.append(threshold_summary)
    return pd.concat(rows, ignore_index=True)


def simulate_baseline(
    frame: pd.DataFrame,
    fold_map: dict[str, int],
    *,
    name: str,
    selected_round: int,
) -> list[dict]:
    results = []
    for _, trajectory in frame.groupby(
        ["Sample", "Coverage_effective"],
        sort=True,
    ):
        sample = str(trajectory.iloc[0]["Sample"])
        results.append(
            _trajectory_result(
                trajectory,
                candidate=name,
                fold=fold_map[sample],
                selected_round=selected_round,
                predicted_class=_round_class(selected_round),
                confidence=None,
            )
        )
    return results


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in predictions.groupby("candidate", sort=True):
        y_true = group["true_class"].to_numpy()
        y_selected = group["selected_class"].to_numpy()
        recalls = recall_score(
            y_true,
            y_selected,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
        rows.append(
            {
                "candidate": candidate,
                "n_trajectories": len(group),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_selected),
                "macro_f1": f1_score(
                    y_true,
                    y_selected,
                    average="macro",
                    zero_division=0,
                ),
                "min_class_recall": float(np.min(recalls)),
                "class_mae": mean_absolute_error(y_true, y_selected),
                "quadratic_weighted_kappa": cohen_kappa_score(
                    y_true,
                    y_selected,
                    weights="quadratic",
                ),
                "unsafe_early_stop_rate": group["unsafe_early_stop"].mean(),
                "severe_unsafe_stop_rate": group["severe_unsafe_stop"].mean(),
                "quality_failure_rate": group["quality_failure"].mean(),
                "exact_round_accuracy": (
                    group["selected_round"] == group["optimal_round"]
                ).mean(),
                "within_one_round_rate": (
                    group["absolute_round_error"] <= 1
                ).mean(),
                "mean_absolute_round_error": group[
                    "absolute_round_error"
                ].mean(),
                "mean_excess_rounds": group["excess_rounds"].mean(),
                "mean_premature_rounds": group["premature_rounds"].mean(),
                "mean_rounds_saved": group["rounds_saved"].mean(),
                "compute_reduction_fraction": group["rounds_saved"].mean() / 5,
                "mean_qv_loss_vs_r5": group["qv_loss_vs_r5"].mean(),
                "mean_busco_loss_vs_r5": group["busco_loss_vs_r5"].mean(),
                **{
                    f"selected_r{round_number}_fraction": (
                        group["selected_round"] == round_number
                    ).mean()
                    for round_number in range(1, 6)
                },
            }
        )
    return pd.DataFrame(rows).sort_values("candidate").reset_index(drop=True)


def _sample_bootstrap_ci(
    group: pd.DataFrame,
    column: str,
    *,
    random_state: int,
) -> dict[str, float]:
    per_sample = group.groupby("Sample")[column].mean()
    values = per_sample.to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    estimates = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(N_BOOTSTRAP)]
    )
    return {
        "estimate": float(group[column].mean()),
        "ci_lower": float(np.quantile(estimates, 0.025)),
        "ci_upper": float(np.quantile(estimates, 0.975)),
    }


def bootstrap_primary_metrics(predictions: pd.DataFrame) -> dict:
    output = {}
    for candidate_index, (candidate, group) in enumerate(
        predictions.groupby("candidate", sort=True)
    ):
        output[candidate] = {
            column: _sample_bootstrap_ci(
                group,
                column,
                random_state=RANDOM_STATE + candidate_index * 100 + metric_index,
            )
            for metric_index, column in enumerate(PRIMARY_BOOTSTRAP_METRICS)
        }
        sample_blocks = [
            sample_frame
            for _, sample_frame in group.groupby("Sample", sort=True)
        ]
        for metric_index, metric_name in enumerate(
            ("balanced_accuracy", "macro_f1"),
            start=len(PRIMARY_BOOTSTRAP_METRICS),
        ):
            rng = np.random.default_rng(
                RANDOM_STATE + candidate_index * 100 + metric_index
            )
            estimates = []
            for _ in range(N_BOOTSTRAP):
                selected_blocks = rng.choice(
                    len(sample_blocks),
                    size=len(sample_blocks),
                    replace=True,
                )
                sampled = pd.concat(
                    [sample_blocks[index] for index in selected_blocks],
                    ignore_index=True,
                )
                y_true = sampled["true_class"]
                y_pred = sampled["selected_class"]
                if metric_name == "balanced_accuracy":
                    estimate = balanced_accuracy_score(y_true, y_pred)
                else:
                    estimate = f1_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    )
                estimates.append(estimate)
            observed = (
                balanced_accuracy_score(group["true_class"], group["selected_class"])
                if metric_name == "balanced_accuracy"
                else f1_score(
                    group["true_class"],
                    group["selected_class"],
                    average="macro",
                    zero_division=0,
                )
            )
            output[candidate][metric_name] = {
                "estimate": float(observed),
                "ci_lower": float(np.quantile(estimates, 0.025)),
                "ci_upper": float(np.quantile(estimates, 0.975)),
            }
    return output


def endpoint_audit(predictions: pd.DataFrame) -> dict:
    """Expose disagreement between exact labels and partial QV/BUSCO outcomes."""
    always_r1 = predictions[predictions["candidate"] == "always_r1"].copy()
    late = always_r1[always_r1["optimal_round"] == 5]
    return {
        "n_trajectories": len(always_r1),
        "optimal_round_distribution": {
            str(int(round_number)): int(count)
            for round_number, count in always_r1["optimal_round"].value_counts(
                sort=False
            ).sort_index().items()
        },
        "r1_qv_busco_acceptable_fraction": float(
            1 - always_r1["quality_failure"].mean()
        ),
        "late_label_r1_qv_busco_acceptable_fraction": float(
            1 - late["quality_failure"].mean()
        ),
        "interpretation": (
            "QV/BUSCO loss versus R5 is only a partial safety endpoint; exact labels "
            "also encode error, assembly, R1 veto, and plateau criteria."
        ),
    }


def paired_bootstrap_comparisons(predictions: pd.DataFrame) -> dict:
    comparisons = (
        ("light_core_prospective", "r1_core_only"),
        ("light_core_prospective", "legacy_full_retrospective"),
        ("light_fasta_prospective", "light_core_prospective"),
    )
    output = {}
    keys = ["Sample", "Coverage_effective"]
    for comparison_index, (candidate, reference) in enumerate(comparisons):
        left = predictions[predictions["candidate"] == candidate]
        right = predictions[predictions["candidate"] == reference]
        merged = left.merge(right, on=keys, suffixes=("_candidate", "_reference"))
        if len(merged) != len(left) or len(merged) != len(right):
            raise RuntimeError(f"unpaired trajectory comparison: {candidate} vs {reference}")
        result = {}
        for metric_index, column in enumerate(PRIMARY_BOOTSTRAP_METRICS):
            merged["difference"] = (
                merged[f"{column}_candidate"] - merged[f"{column}_reference"]
            )
            per_sample = merged.groupby("Sample")["difference"].mean().to_numpy()
            rng = np.random.default_rng(
                RANDOM_STATE + comparison_index * 100 + metric_index
            )
            estimates = np.array(
                [
                    rng.choice(per_sample, size=len(per_sample), replace=True).mean()
                    for _ in range(N_BOOTSTRAP)
                ]
            )
            result[column] = {
                "difference_candidate_minus_reference": float(
                    merged["difference"].mean()
                ),
                "ci_lower": float(np.quantile(estimates, 0.025)),
                "ci_upper": float(np.quantile(estimates, 0.975)),
            }
        output[f"{candidate}_vs_{reference}"] = result
    return output


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    raise TypeError(f"cannot serialize {type(value).__name__}")


def run_experiment(
    *,
    repository: Path,
    dataset_path: Path,
    split_path: Path,
    output_dir: Path,
) -> dict:
    frame, audit = prepare_training_frame(dataset_path, split_path)
    candidates = define_candidates(repository, frame)
    folds = make_sample_folds(frame)
    fold_map = dict(zip(folds["Sample"], folds["fold"]))

    predictions: list[dict] = []
    round_predictions: list[dict] = []
    importance_rows: list[dict] = []
    all_samples = set(frame["Sample"].astype(str))
    for candidate in candidates:
        for fold in range(1, N_SPLITS + 1):
            validation_samples = {
                sample for sample, assigned_fold in fold_map.items() if assigned_fold == fold
            }
            training_samples = all_samples - validation_samples
            fit_frame = frame[frame["Sample"].astype(str).isin(training_samples)]
            if not candidate.adaptive:
                fit_frame = fit_frame[fit_frame["round"] == 1]
            validation = frame[
                frame["Sample"].astype(str).isin(validation_samples)
            ]
            model = build_model(RANDOM_STATE + fold)
            X_fit = fit_frame[list(candidate.feature_names)].replace(
                [np.inf, -np.inf],
                np.nan,
            )
            y_fit = fit_frame["optimal_rounds_3class"].astype(int) - 1
            model.fit(X_fit, y_fit)
            if tuple(model.classes_) != (0, 1, 2):
                raise RuntimeError(
                    f"{candidate.name} fold {fold} lacks one or more classes"
                )
            candidate_results, candidate_round_results = simulate_candidate(
                model,
                validation,
                candidate,
                fold,
            )
            predictions.extend(candidate_results)
            round_predictions.extend(candidate_round_results)

            imputer = model.named_steps["imputer"]
            feature_names = imputer.get_feature_names_out(candidate.feature_names)
            importances = model.named_steps["model"].feature_importances_
            importance_rows.extend(
                {
                    "candidate": candidate.name,
                    "fold": fold,
                    "feature": str(feature),
                    "importance": float(importance),
                }
                for feature, importance in zip(feature_names, importances)
            )

    predictions.extend(
        simulate_baseline(
            frame,
            fold_map,
            name="always_r5",
            selected_round=5,
        )
    )
    predictions.extend(
        simulate_baseline(
            frame,
            fold_map,
            name="always_r1",
            selected_round=1,
        )
    )
    prediction_frame = pd.DataFrame(predictions).sort_values(
        ["candidate", "fold", "Sample", "Coverage_effective"]
    )
    expected_trajectories = audit["development_trajectories"]
    counts = prediction_frame.groupby("candidate").size()
    if not counts.eq(expected_trajectories).all():
        raise RuntimeError(f"candidate prediction counts differ: {counts.to_dict()}")

    summary = summarize_predictions(prediction_frame)
    round_prediction_frame = pd.DataFrame(round_predictions).sort_values(
        ["candidate", "fold", "Sample", "Coverage_effective", "round"]
    )
    round_metrics = summarize_round_predictions(round_prediction_frame)
    sensitivity = threshold_sensitivity(
        frame,
        round_prediction_frame,
        candidates,
    )
    fold_metric_rows = []
    for (candidate_name, fold), group in prediction_frame.groupby(
        ["candidate", "fold"],
        sort=True,
    ):
        row = summarize_predictions(group).iloc[0].to_dict()
        row["candidate"] = candidate_name
        row["fold"] = int(fold)
        fold_metric_rows.append(row)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics = fold_metrics[
        ["candidate", "fold", *[c for c in fold_metrics if c not in {"candidate", "fold"}]]
    ]
    importance = pd.DataFrame(importance_rows)
    importance_summary = (
        importance.groupby(["candidate", "feature"])["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["candidate", "mean"], ascending=[True, False])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_dir / "oof_trajectory_predictions.csv", index=False)
    round_prediction_frame.to_csv(
        output_dir / "oof_round_predictions.csv",
        index=False,
    )
    summary.to_csv(output_dir / "candidate_summary.csv", index=False)
    round_metrics.to_csv(output_dir / "round_classification_metrics.csv", index=False)
    sensitivity.to_csv(output_dir / "threshold_sensitivity.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    folds.to_csv(output_dir / "sample_fold_assignments.csv", index=False)
    importance_summary.to_csv(output_dir / "feature_importance.csv", index=False)

    report = {
        "experiment_schema_version": "1.0.0",
        "scientific_status": "development-only feasibility; held-out test untouched",
        "audit": audit,
        "protocol": {
            "n_splits": N_SPLITS,
            "splitter": "StratifiedGroupKFold grouped by biological Sample",
            "random_state": RANDOM_STATE,
            "n_estimators": N_ESTIMATORS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "acceptable_qv_loss": ACCEPTABLE_QV_LOSS,
            "acceptable_busco_loss": ACCEPTABLE_BUSCO_LOSS,
            "bootstrap_replicates": N_BOOTSTRAP,
            "bootstrap_unit": "biological Sample",
            "exploratory_thresholds": list(EXPLORATORY_THRESHOLDS),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "candidates": {
            candidate.name: {
                "adaptive": candidate.adaptive,
                "description": candidate.description,
                "feature_names": list(candidate.feature_names),
            }
            for candidate in candidates
        },
        "summary": summary.to_dict(orient="records"),
        "round_classification_metrics": round_metrics.to_dict(orient="records"),
        "threshold_sensitivity": sensitivity.to_dict(orient="records"),
        "bootstrap_primary_metrics": bootstrap_primary_metrics(prediction_frame),
        "paired_bootstrap_comparisons": paired_bootstrap_comparisons(
            prediction_frame
        ),
        "endpoint_audit": endpoint_audit(prediction_frame),
    }
    (output_dir / "experiment_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/training_dataset_with_target.csv"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("outputs/train_test_split_samples.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/esdp_light_feasibility"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    dataset = args.dataset.resolve()
    split = args.split.resolve()
    output_dir = args.output_dir.resolve()
    report = run_experiment(
        repository=repository,
        dataset_path=dataset,
        split_path=split,
        output_dir=output_dir,
    )
    summary = pd.DataFrame(report["summary"])
    print(summary.to_string(index=False))
    print(f"\nResults: {output_dir}")
    print("Held-out test samples used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
