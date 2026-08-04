#!/usr/bin/env python3
"""Direct CONTINUE/STOP_ELIGIBLE feasibility study for ESDP-light."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from esdp_features import FeatureBuilder
from esdp_light import LIGHT_CORE_FEATURES
from experiments.esdp_light_feasibility import (
    ACCEPTABLE_BUSCO_LOSS,
    ACCEPTABLE_QV_LOSS,
    N_BOOTSTRAP,
    N_ESTIMATORS,
    N_SPLITS,
    RANDOM_STATE,
    Candidate,
    _json_default,
    _trajectory_result,
    bootstrap_primary_metrics,
    define_candidates,
    make_sample_folds,
    prepare_training_frame,
    summarize_predictions,
)


PRIMARY_STOP_THRESHOLD = 0.70
EXPLORATORY_STOP_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
HISTORICAL_PREFIX = "historical__"
CAUSAL_QUALITY_FEATURES = tuple(
    dict.fromkeys(
        (
            *LIGHT_CORE_FEATURES,
            "qv",
            "error_rate",
            "busco_complete",
            "busco_fragmented",
            "busco_missing",
            "assembly_frac",
            "assembly_error",
            "delta_qv",
            "delta_busco_complete",
            "delta_error_rate",
            "delta_error_improvement",
            "delta_assembly_error",
            "qv_improvement_rate",
            "busco_per_contig",
            "n50_fraction",
            "cost_benefit_ratio",
            "delta_qv_cumsum",
            "delta_busco_complete_cumsum",
            "score_improvement",
            "gain_cumulative",
            "qv_from_r1",
            "n50_from_r1",
            "error_rate_from_r1",
            "busco_complete_from_r1",
            "assembly_frac_from_r1",
            "delta_qv_trend",
            "delta_busco_complete_trend",
            "score_improvement_trend",
            "completeness_score",
            "assembly_quality",
            "polishing_effectiveness",
        )
    )
)


def rebuild_causal_quality_features(
    frame: pd.DataFrame,
    legacy_features: tuple[str, ...],
) -> pd.DataFrame:
    """Recompute quality histories without legacy full-trajectory plateau leakage."""
    original = frame.sort_values(
        ["Sample", "Coverage_effective", "round"]
    ).reset_index(drop=True)
    rebuilt = FeatureBuilder().transform(original)
    rebuilt = rebuilt.sort_values(
        ["Sample", "Coverage_effective", "round"]
    ).reset_index(drop=True)
    missing = sorted(set(CAUSAL_QUALITY_FEATURES) - set(rebuilt.columns))
    if missing:
        raise ValueError(f"causal quality benchmark missing features: {missing}")
    for feature in legacy_features:
        rebuilt[f"{HISTORICAL_PREFIX}{feature}"] = original[feature].to_numpy()
    return rebuilt


def define_binary_candidates(
    repository: Path,
    frame: pd.DataFrame,
) -> tuple[Candidate, ...]:
    source = {candidate.name: candidate for candidate in define_candidates(repository, frame)}
    legacy_features = source["legacy_full_retrospective"].feature_names
    return (
        Candidate(
            "legacy_historical_leaky_binary",
            tuple(f"{HISTORICAL_PREFIX}{name}" for name in legacy_features),
            True,
            (
                "Invalid diagnostic using historical features before prospective "
                "plateau correction; never eligible for selection."
            ),
        ),
        Candidate(
            "legacy_rebuilt_binary",
            legacy_features,
            True,
            "Legacy feature family rebuilt with causal v2 temporal definitions.",
        ),
        Candidate(
            "quality_causal_binary",
            CAUSAL_QUALITY_FEATURES,
            True,
            (
                "Direct binary control using costly quality metrics rebuilt "
                "prospectively, without policy or plateau predictors."
            ),
        ),
        Candidate(
            "light_core_binary",
            source["light_core_prospective"].feature_names,
            True,
            "Direct binary formulation using approved low-cost core features.",
        ),
        Candidate(
            "light_fasta_binary",
            source["light_fasta_prospective"].feature_names,
            True,
            "Direct binary formulation using currently operational FASTA features.",
        ),
    )


def build_binary_model(random_state: int) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    max_depth=12,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _selected_class(round_number: int) -> int:
    if round_number <= 2:
        return 0
    if round_number <= 4:
        return 1
    return 2


def collect_fold_predictions(
    model: Pipeline,
    validation: pd.DataFrame,
    candidate: Candidate,
    fold: int,
) -> list[dict]:
    rows = []
    decisions = validation[validation["round"] < 5]
    for _, row in decisions.iterrows():
        features = pd.DataFrame(
            [[row[name] for name in candidate.feature_names]],
            columns=candidate.feature_names,
        ).replace([np.inf, -np.inf], np.nan)
        probability_stop = float(model.predict_proba(features)[0][1])
        rows.append(
            {
                "candidate": candidate.name,
                "fold": fold,
                "Sample": str(row["Sample"]),
                "Coverage_effective": str(row["Coverage_effective"]),
                "round": int(row["round"]),
                "stop_eligible": int(
                    row["round"] >= row["optimal_rounds_5class"]
                ),
                "probability_stop": probability_stop,
            }
        )
    return rows


def simulate_policy(
    frame: pd.DataFrame,
    round_predictions: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    trajectory_map = {
        (str(sample), str(coverage)): trajectory
        for (sample, coverage), trajectory in frame.groupby(
            ["Sample", "Coverage_effective"],
            sort=True,
        )
    }
    results = []
    for (candidate, sample, coverage), predictions in round_predictions.groupby(
        ["candidate", "Sample", "Coverage_effective"],
        sort=True,
    ):
        predictions = predictions.sort_values("round")
        selected_round = 5
        probability_stop = float(predictions.iloc[-1]["probability_stop"])
        fold = int(predictions.iloc[0]["fold"])
        for _, prediction in predictions.iterrows():
            if prediction["probability_stop"] >= threshold:
                selected_round = int(prediction["round"])
                probability_stop = float(prediction["probability_stop"])
                break
        results.append(
            _trajectory_result(
                trajectory_map[(str(sample), str(coverage))],
                candidate=str(candidate),
                fold=fold,
                selected_round=selected_round,
                predicted_class=_selected_class(selected_round),
                confidence=probability_stop,
            )
        )
    return pd.DataFrame(results)


def summarize_binary_rounds(round_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, round_number), group in round_predictions.groupby(
        ["candidate", "round"],
        sort=True,
    ):
        y_true = group["stop_eligible"].to_numpy(dtype=int)
        probability = group["probability_stop"].to_numpy(dtype=float)
        y_pred = (probability >= 0.5).astype(int)
        rows.append(
            {
                "candidate": candidate,
                "round": int(round_number),
                "n_predictions": len(group),
                "stop_eligible_fraction": y_true.mean(),
                "roc_auc": roc_auc_score(y_true, probability),
                "average_precision": average_precision_score(y_true, probability),
                "balanced_accuracy_at_0_5": balanced_accuracy_score(y_true, y_pred),
                "stop_recall_at_0_5": recall_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
                "continue_recall_at_0_5": recall_score(
                    y_true,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),
                "f1_stop_at_0_5": f1_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
                "brier": brier_score_loss(y_true, probability),
                "log_loss": log_loss(y_true, probability, labels=[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    frame: pd.DataFrame,
    round_predictions: pd.DataFrame,
) -> pd.DataFrame:
    summaries = []
    for threshold in EXPLORATORY_STOP_THRESHOLDS:
        predictions = simulate_policy(
            frame,
            round_predictions,
            threshold=threshold,
        )
        summary = summarize_predictions(predictions)
        summary.insert(1, "stop_threshold", threshold)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def endpoint_composition(frame: pd.DataFrame) -> dict:
    decisions = frame[frame["round"] < 5].copy()
    decisions["stop_eligible"] = (
        decisions["round"] >= decisions["optimal_rounds_5class"]
    ).astype(int)
    r1 = frame[frame["round"] == 1]
    return {
        "stop_eligible_by_round": {
            str(int(round_number)): {
                "eligible": int(group["stop_eligible"].sum()),
                "total": len(group),
                "fraction": float(group["stop_eligible"].mean()),
            }
            for round_number, group in decisions.groupby("round", sort=True)
        },
        "optimal_round_distribution": {
            str(int(round_number)): int(count)
            for round_number, count in r1["optimal_rounds_5class"].value_counts(
                sort=False
            ).sort_index().items()
        },
        "r1_ok_and_stable_count": int(
            ((r1["r1_ok_group"] == 1) & (r1["stable_all_group"] == 1)).sum()
        ),
    }


def paired_formulation_comparisons(
    binary_predictions: pd.DataFrame,
    three_class_path: Path,
) -> dict:
    three_class = pd.read_csv(three_class_path)
    comparisons = {
        "legacy_historical_leaky_binary": "legacy_full_retrospective",
        "light_core_binary": "light_core_prospective",
        "light_fasta_binary": "light_fasta_prospective",
    }
    metrics = (
        "unsafe_early_stop",
        "quality_failure",
        "rounds_saved",
        "absolute_round_error",
    )
    keys = ["Sample", "Coverage_effective"]
    output = {}
    for comparison_index, (binary_name, three_name) in enumerate(comparisons.items()):
        binary = binary_predictions[binary_predictions["candidate"] == binary_name]
        reference = three_class[three_class["candidate"] == three_name]
        merged = binary.merge(reference, on=keys, suffixes=("_binary", "_three_class"))
        if len(merged) != len(binary) or len(merged) != len(reference):
            raise RuntimeError(f"unpaired formulation comparison: {binary_name}")
        result = {}
        for metric_index, metric in enumerate(metrics):
            merged["difference"] = (
                merged[f"{metric}_binary"] - merged[f"{metric}_three_class"]
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
            result[metric] = {
                "binary_minus_three_class": float(merged["difference"].mean()),
                "ci_lower": float(np.quantile(estimates, 0.025)),
                "ci_upper": float(np.quantile(estimates, 0.975)),
            }
        output[f"{binary_name}_vs_{three_name}"] = result
    return output


def run_experiment(
    *,
    repository: Path,
    dataset_path: Path,
    split_path: Path,
    three_class_predictions_path: Path,
    output_dir: Path,
) -> dict:
    frame, audit = prepare_training_frame(dataset_path, split_path)
    source_candidates = {
        candidate.name: candidate
        for candidate in define_candidates(repository, frame)
    }
    frame = rebuild_causal_quality_features(
        frame,
        source_candidates["legacy_full_retrospective"].feature_names,
    )
    candidates = define_binary_candidates(repository, frame)
    folds = make_sample_folds(frame)
    fold_map = dict(zip(folds["Sample"], folds["fold"]))
    all_samples = set(frame["Sample"].astype(str))

    round_rows = []
    importance_rows = []
    for candidate in candidates:
        for fold in range(1, N_SPLITS + 1):
            validation_samples = {
                sample for sample, assigned in fold_map.items() if assigned == fold
            }
            training_samples = all_samples - validation_samples
            fit_frame = frame[
                frame["Sample"].astype(str).isin(training_samples)
                & (frame["round"] < 5)
            ].copy()
            validation = frame[
                frame["Sample"].astype(str).isin(validation_samples)
            ]
            y_fit = (
                fit_frame["round"] >= fit_frame["optimal_rounds_5class"]
            ).astype(int)
            if set(y_fit) != {0, 1}:
                raise RuntimeError(f"{candidate.name} fold {fold} lacks a binary class")
            model = build_binary_model(RANDOM_STATE + fold)
            X_fit = fit_frame[list(candidate.feature_names)].replace(
                [np.inf, -np.inf],
                np.nan,
            )
            model.fit(X_fit, y_fit)
            round_rows.extend(
                collect_fold_predictions(model, validation, candidate, fold)
            )

            feature_names = model.named_steps["imputer"].get_feature_names_out(
                candidate.feature_names
            )
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

    round_predictions = pd.DataFrame(round_rows).sort_values(
        ["candidate", "fold", "Sample", "Coverage_effective", "round"]
    )
    expected = audit["development_trajectories"]
    if not round_predictions.groupby(["candidate", "round"]).size().eq(expected).all():
        raise RuntimeError("binary round predictions are incomplete")

    primary_predictions = simulate_policy(
        frame,
        round_predictions,
        threshold=PRIMARY_STOP_THRESHOLD,
    )
    summary = summarize_predictions(primary_predictions)
    round_metrics = summarize_binary_rounds(round_predictions)
    sensitivity = threshold_sensitivity(frame, round_predictions)
    importance = pd.DataFrame(importance_rows)
    importance_summary = (
        importance.groupby(["candidate", "feature"])["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["candidate", "mean"], ascending=[True, False])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    round_predictions.to_csv(output_dir / "oof_binary_round_predictions.csv", index=False)
    primary_predictions.to_csv(
        output_dir / "oof_binary_trajectory_predictions.csv",
        index=False,
    )
    summary.to_csv(output_dir / "binary_candidate_summary.csv", index=False)
    round_metrics.to_csv(output_dir / "binary_round_metrics.csv", index=False)
    sensitivity.to_csv(output_dir / "binary_threshold_sensitivity.csv", index=False)
    importance_summary.to_csv(output_dir / "binary_feature_importance.csv", index=False)
    folds.to_csv(output_dir / "sample_fold_assignments.csv", index=False)

    report = {
        "experiment_schema_version": "1.0.0",
        "scientific_status": "development-only binary formulation; held-out test untouched",
        "audit": audit,
        "endpoint_composition": endpoint_composition(frame),
        "protocol": {
            "target": "STOP_ELIGIBLE = current_round >= optimal_rounds_5class",
            "training_rounds": [1, 2, 3, 4],
            "primary_stop_threshold": PRIMARY_STOP_THRESHOLD,
            "exploratory_stop_thresholds": list(EXPLORATORY_STOP_THRESHOLDS),
            "n_splits": N_SPLITS,
            "splitter": "same frozen sample-grouped folds as three-class experiment",
            "random_state": RANDOM_STATE,
            "n_estimators": N_ESTIMATORS,
            "acceptable_qv_loss": ACCEPTABLE_QV_LOSS,
            "acceptable_busco_loss": ACCEPTABLE_BUSCO_LOSS,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
        "candidates": {
            candidate.name: {
                "description": candidate.description,
                "feature_names": list(candidate.feature_names),
            }
            for candidate in candidates
        },
        "summary": summary.to_dict(orient="records"),
        "binary_round_metrics": round_metrics.to_dict(orient="records"),
        "threshold_sensitivity": sensitivity.to_dict(orient="records"),
        "bootstrap_primary_metrics": bootstrap_primary_metrics(primary_predictions),
        "paired_formulation_comparisons": paired_formulation_comparisons(
            primary_predictions,
            three_class_predictions_path,
        ),
    }
    (output_dir / "binary_experiment_report.json").write_text(
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
        "--three-class-predictions",
        type=Path,
        default=Path(
            "outputs/esdp_light_feasibility/oof_trajectory_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/esdp_light_binary_feasibility"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    report = run_experiment(
        repository=repository,
        dataset_path=args.dataset.resolve(),
        split_path=args.split.resolve(),
        three_class_predictions_path=args.three_class_predictions.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(pd.DataFrame(report["summary"]).to_string(index=False))
    print(f"\nResults: {args.output_dir.resolve()}")
    print("Held-out test samples used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
