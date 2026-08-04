#!/usr/bin/env python3
"""Evaluate a future-material-benefit stopping endpoint for ESDP-light."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)

from experiments.esdp_light_binary_decision import (
    EXPLORATORY_STOP_THRESHOLDS,
    PRIMARY_STOP_THRESHOLD,
    build_binary_model,
    define_binary_candidates,
    rebuild_causal_quality_features,
)
from experiments.esdp_light_feasibility import (
    ACCEPTABLE_BUSCO_LOSS,
    ACCEPTABLE_QV_LOSS,
    N_BOOTSTRAP,
    N_ESTIMATORS,
    N_SPLITS,
    RANDOM_STATE,
    _json_default,
    bootstrap_primary_metrics,
    define_candidates,
    make_sample_folds,
    prepare_training_frame,
    summarize_predictions,
)


ENDPOINT_METRICS = {
    "qv": {"direction": 1.0, "tolerance": 0.05},
    "busco_complete": {"direction": 1.0, "tolerance": 1.0},
    "error_rate": {"direction": -1.0, "tolerance": 0.0005},
    "assembly_error": {"direction": -1.0, "tolerance": 0.01},
}
ENDPOINT_COLUMN = "material_optimal_round"
TARGET_COLUMN = "material_stop_eligible"


def _future_material_gains(
    current: pd.Series,
    future: pd.DataFrame,
    *,
    tolerance_scale: float,
) -> dict[str, bool]:
    """Return metrics with a future directional gain above tolerance."""
    gains = {}
    for metric, contract in ENDPOINT_METRICS.items():
        directional_gain = contract["direction"] * (
            future[metric] - float(current[metric])
        )
        threshold = contract["tolerance"] * tolerance_scale
        gains[metric] = bool((directional_gain > threshold).any())
    return gains


def add_material_benefit_endpoint(
    frame: pd.DataFrame,
    *,
    tolerance_scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add an absorbing target based on absence of material future gains."""
    if tolerance_scale <= 0:
        raise ValueError("tolerance_scale must be positive")
    missing = sorted(set(ENDPOINT_METRICS) - set(frame.columns))
    if missing:
        raise ValueError(f"material endpoint missing metrics: {missing}")

    parts = []
    audit_rows = []
    for (sample, coverage), trajectory in frame.groupby(
        ["Sample", "Coverage_effective"],
        sort=True,
    ):
        trajectory = trajectory.sort_values("round").copy()
        raw_eligible = []
        per_round_gains = []
        for _, current in trajectory.iterrows():
            future = trajectory[trajectory["round"] > current["round"]]
            gains = _future_material_gains(
                current,
                future,
                tolerance_scale=tolerance_scale,
            )
            per_round_gains.append(gains)
            raw_eligible.append(not any(gains.values()))

        first_stop_index = raw_eligible.index(True)
        optimal_round = int(trajectory.iloc[first_stop_index]["round"])
        trajectory[ENDPOINT_COLUMN] = optimal_round
        trajectory[TARGET_COLUMN] = (
            trajectory["round"] >= optimal_round
        ).astype(int)
        parts.append(trajectory)

        for row_index, (_, row) in enumerate(trajectory.iterrows()):
            audit_rows.append(
                {
                    "Sample": str(sample),
                    "Coverage_effective": str(coverage),
                    "round": int(row["round"]),
                    "raw_stop_eligible": int(raw_eligible[row_index]),
                    TARGET_COLUMN: int(row[TARGET_COLUMN]),
                    ENDPOINT_COLUMN: optimal_round,
                    "legacy_optimal_round": int(row["optimal_rounds_5class"]),
                    **{
                        f"future_gain_{metric}": int(gain)
                        for metric, gain in per_round_gains[row_index].items()
                    },
                }
            )

    endpoint_frame = pd.concat(parts, ignore_index=True)
    endpoint_audit = pd.DataFrame(audit_rows)
    return endpoint_frame, endpoint_audit


def collect_fold_predictions(
    model,
    validation: pd.DataFrame,
    candidate,
    fold: int,
) -> list[dict]:
    rows = []
    for _, row in validation[validation["round"] < 5].iterrows():
        features = pd.DataFrame(
            [[row[name] for name in candidate.feature_names]],
            columns=candidate.feature_names,
        ).replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "candidate": candidate.name.replace("_binary", "_material"),
                "fold": fold,
                "Sample": str(row["Sample"]),
                "Coverage_effective": str(row["Coverage_effective"]),
                "round": int(row["round"]),
                TARGET_COLUMN: int(row[TARGET_COLUMN]),
                "probability_stop": float(model.predict_proba(features)[0][1]),
            }
        )
    return rows


def _round_class(round_number: int) -> int:
    if round_number <= 2:
        return 0
    if round_number <= 4:
        return 1
    return 2


def trajectory_result(
    trajectory: pd.DataFrame,
    *,
    candidate: str,
    fold: int,
    selected_round: int,
    confidence: float | None,
) -> dict:
    trajectory = trajectory.sort_values("round")
    selected = trajectory[trajectory["round"] == selected_round].iloc[0]
    final = trajectory[trajectory["round"] == 5].iloc[0]
    optimal_round = int(final[ENDPOINT_COLUMN])
    qv_loss = float(final["qv"] - selected["qv"])
    busco_loss = float(final["busco_complete"] - selected["busco_complete"])
    premature_rounds = max(optimal_round - selected_round, 0)
    excess_rounds = max(selected_round - optimal_round, 0)
    return {
        "candidate": candidate,
        "fold": fold,
        "Sample": str(final["Sample"]),
        "Coverage_effective": str(final["Coverage_effective"]),
        "true_class": _round_class(optimal_round),
        "optimal_round": optimal_round,
        "predicted_class_at_stop": _round_class(selected_round),
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


def simulate_policy(
    frame: pd.DataFrame,
    round_predictions: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    trajectories = {
        (str(sample), str(coverage)): trajectory
        for (sample, coverage), trajectory in frame.groupby(
            ["Sample", "Coverage_effective"], sort=True
        )
    }
    results = []
    for (candidate, sample, coverage), predictions in round_predictions.groupby(
        ["candidate", "Sample", "Coverage_effective"], sort=True
    ):
        predictions = predictions.sort_values("round")
        selected_round = 5
        confidence = float(predictions.iloc[-1]["probability_stop"])
        for _, prediction in predictions.iterrows():
            if prediction["probability_stop"] >= threshold:
                selected_round = int(prediction["round"])
                confidence = float(prediction["probability_stop"])
                break
        results.append(
            trajectory_result(
                trajectories[(str(sample), str(coverage))],
                candidate=str(candidate),
                fold=int(predictions.iloc[0]["fold"]),
                selected_round=selected_round,
                confidence=confidence,
            )
        )
    return pd.DataFrame(results)


def summarize_rounds(round_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, round_number), group in round_predictions.groupby(
        ["candidate", "round"], sort=True
    ):
        y_true = group[TARGET_COLUMN].to_numpy(dtype=int)
        probability = group["probability_stop"].to_numpy(dtype=float)
        y_pred = (probability >= 0.5).astype(int)
        rows.append(
            {
                "candidate": candidate,
                "round": int(round_number),
                "n_predictions": len(group),
                "stop_eligible_fraction": float(y_true.mean()),
                "roc_auc": roc_auc_score(y_true, probability),
                "average_precision": average_precision_score(y_true, probability),
                "balanced_accuracy_at_0_5": balanced_accuracy_score(y_true, y_pred),
                "stop_recall_at_0_5": recall_score(
                    y_true, y_pred, pos_label=1, zero_division=0
                ),
                "continue_recall_at_0_5": recall_score(
                    y_true, y_pred, pos_label=0, zero_division=0
                ),
                "f1_stop_at_0_5": f1_score(
                    y_true, y_pred, pos_label=1, zero_division=0
                ),
                "brier": brier_score_loss(y_true, probability),
                "log_loss": log_loss(y_true, probability, labels=[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def endpoint_summary(endpoint_audit: pd.DataFrame) -> dict:
    trajectories = endpoint_audit[endpoint_audit["round"] == 1]
    return {
        "optimal_round_distribution": {
            str(int(round_number)): int(count)
            for round_number, count in trajectories[ENDPOINT_COLUMN]
            .value_counts(sort=False)
            .sort_index()
            .items()
        },
        "absorbing_stop_eligible_by_round": {
            str(int(round_number)): {
                "eligible": int(group[TARGET_COLUMN].sum()),
                "total": len(group),
                "fraction": float(group[TARGET_COLUMN].mean()),
            }
            for round_number, group in endpoint_audit.groupby("round", sort=True)
        },
        "future_gain_prevalence_by_metric": {
            metric: {
                str(int(round_number)): float(group[f"future_gain_{metric}"].mean())
                for round_number, group in endpoint_audit.groupby("round", sort=True)
            }
            for metric in ENDPOINT_METRICS
        },
        "raw_non_monotonic_trajectories": int(
            sum(
                not group["raw_stop_eligible"].is_monotonic_increasing
                for _, group in endpoint_audit.groupby(
                    ["Sample", "Coverage_effective"]
                )
            )
        ),
        "legacy_label_agreement": float(
            (
                trajectories[ENDPOINT_COLUMN]
                == trajectories["legacy_optimal_round"]
            ).mean()
        ),
        "legacy_label_mean_absolute_difference": float(
            (
                trajectories[ENDPOINT_COLUMN]
                - trajectories["legacy_optimal_round"]
            ).abs().mean()
        ),
    }


def tolerance_sensitivity(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for scale in (0.5, 1.0, 2.0):
        _, audit = add_material_benefit_endpoint(
            frame, tolerance_scale=scale
        )
        distribution = audit[audit["round"] == 1][ENDPOINT_COLUMN].value_counts()
        rows.append(
            {
                "tolerance_scale": scale,
                **{
                    f"optimal_r{round_number}": int(distribution.get(round_number, 0))
                    for round_number in range(1, 6)
                },
            }
        )
    return rows


def rejected_pareto_audit(frame: pd.DataFrame) -> dict:
    """Audit the non-monotonic Pareto alternative rejected before fitting."""
    raw_rows = []
    first_stop_rounds = []
    for (sample, coverage), trajectory in frame.groupby(
        ["Sample", "Coverage_effective"], sort=True
    ):
        trajectory = trajectory.sort_values("round")
        eligible_states = []
        for _, current in trajectory.iterrows():
            future = trajectory[trajectory["round"] > current["round"]]
            beneficial = False
            for _, later in future.iterrows():
                standardized = {
                    metric: contract["direction"]
                    * (float(later[metric]) - float(current[metric]))
                    / contract["tolerance"]
                    for metric, contract in ENDPOINT_METRICS.items()
                }
                if all(value >= -1.0 for value in standardized.values()) and any(
                    value > 1.0 for value in standardized.values()
                ):
                    beneficial = True
                    break
            eligible_states.append(not beneficial)
            raw_rows.append(
                {
                    "Sample": str(sample),
                    "Coverage_effective": str(coverage),
                    "round": int(current["round"]),
                    "stop_eligible": int(not beneficial),
                }
            )
        first_stop_rounds.append(eligible_states.index(True) + 1)

    raw = pd.DataFrame(raw_rows)
    distribution = pd.Series(first_stop_rounds).value_counts().sort_index()
    return {
        "status": "rejected before model fitting",
        "reason": (
            "cross-metric trade-offs create incomparable and non-monotonic "
            "raw stopping states"
        ),
        "raw_non_monotonic_trajectories": int(
            sum(
                not group["stop_eligible"].is_monotonic_increasing
                for _, group in raw.groupby(["Sample", "Coverage_effective"])
            )
        ),
        "earliest_stop_distribution": {
            str(int(round_number)): int(count)
            for round_number, count in distribution.items()
        },
    }


def add_controls(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    controls = []
    for selected_round, name in ((1, "always_r1"), (5, "always_r5")):
        for _, trajectory in frame.groupby(
            ["Sample", "Coverage_effective"], sort=True
        ):
            controls.append(
                trajectory_result(
                    trajectory,
                    candidate=name,
                    fold=0,
                    selected_round=selected_round,
                    confidence=None,
                )
            )
    controls_frame = pd.DataFrame(controls)
    controls_frame["confidence_at_stop"] = pd.to_numeric(
        controls_frame["confidence_at_stop"]
    )
    return pd.concat([predictions, controls_frame], ignore_index=True)


def paired_against_previous_endpoint(
    frame: pd.DataFrame,
    material_predictions: pd.DataFrame,
    previous_path: Path,
) -> dict:
    previous = pd.read_csv(previous_path)
    trajectory_map = {
        (str(sample), str(coverage)): trajectory
        for (sample, coverage), trajectory in frame.groupby(
            ["Sample", "Coverage_effective"], sort=True
        )
    }
    pairs = {
        "light_core_material": "light_core_binary",
        "light_fasta_material": "light_fasta_binary",
        "quality_causal_material": "quality_causal_binary",
        "legacy_rebuilt_material": "legacy_rebuilt_binary",
    }
    output = {}
    keys = ["Sample", "Coverage_effective"]
    for pair_index, (material_name, previous_name) in enumerate(pairs.items()):
        current = material_predictions[
            material_predictions["candidate"] == material_name
        ]
        old = previous[previous["candidate"] == previous_name]
        old_recomputed = []
        for _, row in old.iterrows():
            old_recomputed.append(
                trajectory_result(
                    trajectory_map[(str(row["Sample"]), str(row["Coverage_effective"]))],
                    candidate=previous_name,
                    fold=int(row["fold"]),
                    selected_round=int(row["selected_round"]),
                    confidence=float(row["confidence_at_stop"]),
                )
            )
        old_recomputed = pd.DataFrame(old_recomputed)
        merged = current.merge(old_recomputed, on=keys, suffixes=("_material", "_old"))
        result = {}
        for metric_index, metric in enumerate(
            ("unsafe_early_stop", "quality_failure", "rounds_saved", "absolute_round_error")
        ):
            differences = merged[f"{metric}_material"] - merged[f"{metric}_old"]
            per_sample = differences.groupby(merged["Sample"]).mean().to_numpy()
            rng = np.random.default_rng(RANDOM_STATE + pair_index * 100 + metric_index)
            estimates = np.array(
                [
                    rng.choice(per_sample, size=len(per_sample), replace=True).mean()
                    for _ in range(N_BOOTSTRAP)
                ]
            )
            result[metric] = {
                "material_minus_old_endpoint": float(differences.mean()),
                "ci_lower": float(np.quantile(estimates, 0.025)),
                "ci_upper": float(np.quantile(estimates, 0.975)),
            }
        output[f"{material_name}_vs_{previous_name}"] = result
    return output


def run_experiment(
    *,
    repository: Path,
    dataset_path: Path,
    split_path: Path,
    previous_predictions_path: Path,
    output_dir: Path,
) -> dict:
    frame, audit = prepare_training_frame(dataset_path, split_path)
    source = {
        candidate.name: candidate
        for candidate in define_candidates(repository, frame)
    }
    frame = rebuild_causal_quality_features(
        frame, source["legacy_full_retrospective"].feature_names
    )
    frame, endpoint_audit = add_material_benefit_endpoint(frame)
    candidates = define_binary_candidates(repository, frame)
    folds = make_sample_folds(frame)
    fold_map = dict(zip(folds["Sample"], folds["fold"]))
    samples = set(frame["Sample"].astype(str))

    round_rows = []
    importance_rows = []
    for candidate in candidates:
        for fold in range(1, N_SPLITS + 1):
            validation_samples = {
                sample for sample, assigned in fold_map.items() if assigned == fold
            }
            training_samples = samples - validation_samples
            fit_frame = frame[
                frame["Sample"].astype(str).isin(training_samples)
                & (frame["round"] < 5)
            ]
            validation = frame[
                frame["Sample"].astype(str).isin(validation_samples)
            ]
            y_fit = fit_frame[TARGET_COLUMN].astype(int)
            if set(y_fit) != {0, 1}:
                raise RuntimeError(f"{candidate.name} fold {fold} lacks a binary class")
            model = build_binary_model(RANDOM_STATE + fold)
            model.fit(
                fit_frame[list(candidate.feature_names)].replace(
                    [np.inf, -np.inf], np.nan
                ),
                y_fit,
            )
            round_rows.extend(
                collect_fold_predictions(model, validation, candidate, fold)
            )
            feature_names = model.named_steps["imputer"].get_feature_names_out(
                candidate.feature_names
            )
            importance_rows.extend(
                {
                    "candidate": candidate.name.replace("_binary", "_material"),
                    "fold": fold,
                    "feature": str(feature),
                    "importance": float(importance),
                }
                for feature, importance in zip(
                    feature_names,
                    model.named_steps["model"].feature_importances_,
                )
            )

    round_predictions = pd.DataFrame(round_rows).sort_values(
        ["candidate", "fold", "Sample", "Coverage_effective", "round"]
    )
    expected = audit["development_trajectories"]
    if not round_predictions.groupby(["candidate", "round"]).size().eq(expected).all():
        raise RuntimeError("material endpoint round predictions are incomplete")

    primary = simulate_policy(
        frame, round_predictions, threshold=PRIMARY_STOP_THRESHOLD
    )
    all_predictions = add_controls(frame, primary)
    summary = summarize_predictions(all_predictions)
    round_metrics = summarize_rounds(round_predictions)
    sensitivity_parts = []
    for threshold in EXPLORATORY_STOP_THRESHOLDS:
        predictions = simulate_policy(frame, round_predictions, threshold=threshold)
        part = summarize_predictions(predictions)
        part.insert(1, "stop_threshold", threshold)
        sensitivity_parts.append(part)
    threshold_results = pd.concat(sensitivity_parts, ignore_index=True)
    importance = pd.DataFrame(importance_rows)
    importance_summary = (
        importance.groupby(["candidate", "feature"])["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["candidate", "mean"], ascending=[True, False])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_audit.to_csv(output_dir / "material_endpoint_audit.csv", index=False)
    round_predictions.to_csv(output_dir / "oof_material_round_predictions.csv", index=False)
    all_predictions.to_csv(output_dir / "oof_material_trajectory_predictions.csv", index=False)
    summary.to_csv(output_dir / "material_candidate_summary.csv", index=False)
    round_metrics.to_csv(output_dir / "material_round_metrics.csv", index=False)
    threshold_results.to_csv(output_dir / "material_threshold_sensitivity.csv", index=False)
    importance_summary.to_csv(output_dir / "material_feature_importance.csv", index=False)
    folds.to_csv(output_dir / "sample_fold_assignments.csv", index=False)

    report = {
        "experiment_schema_version": "1.0.0",
        "scientific_status": "development-only endpoint study; held-out test untouched",
        "audit": audit,
        "protocol": {
            "endpoint": (
                "earliest round after which no future round improves any endpoint "
                "metric beyond its predeclared tolerance"
            ),
            "absorbing_target": f"{TARGET_COLUMN} = round >= {ENDPOINT_COLUMN}",
            "endpoint_metrics": ENDPOINT_METRICS,
            "training_rounds": [1, 2, 3, 4],
            "primary_stop_threshold": PRIMARY_STOP_THRESHOLD,
            "exploratory_stop_thresholds": list(EXPLORATORY_STOP_THRESHOLDS),
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "n_estimators": N_ESTIMATORS,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "endpoint_summary": endpoint_summary(endpoint_audit),
        "rejected_pareto_endpoint": rejected_pareto_audit(frame),
        "tolerance_sensitivity": tolerance_sensitivity(frame),
        "candidates": {
            candidate.name.replace("_binary", "_material"): {
                "description": candidate.description,
                "feature_names": list(candidate.feature_names),
            }
            for candidate in candidates
        },
        "summary": summary.to_dict(orient="records"),
        "round_metrics": round_metrics.to_dict(orient="records"),
        "threshold_sensitivity": threshold_results.to_dict(orient="records"),
        "bootstrap_primary_metrics": bootstrap_primary_metrics(primary),
        "paired_against_previous_endpoint": paired_against_previous_endpoint(
            frame, primary, previous_predictions_path
        ),
    }
    (output_dir / "material_experiment_report.json").write_text(
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
        "--previous-predictions",
        type=Path,
        default=Path(
            "outputs/esdp_light_binary_feasibility/oof_binary_trajectory_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/esdp_light_material_benefit"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_experiment(
        repository=Path(__file__).resolve().parents[1],
        dataset_path=args.dataset.resolve(),
        split_path=args.split.resolve(),
        previous_predictions_path=args.previous_predictions.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(pd.DataFrame(report["summary"]).to_string(index=False))
    print(f"\nResults: {args.output_dir.resolve()}")
    print("Held-out test samples used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
