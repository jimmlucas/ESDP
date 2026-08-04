#!/usr/bin/env python3
"""Audit whether ESDP-light endpoint tolerances are measurement-valid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.esdp_light_feasibility import prepare_training_frame


METRICS = ("qv", "busco_complete", "error_rate", "assembly_error")
TOLERANCES = {
    "qv": 0.05,
    "busco_complete": 1.0,
    "error_rate": 0.0005,
    "assembly_error": 0.01,
}
BUSCO_PROVENANCE_FIELDS = (
    "busco_version",
    "busco_lineage_dataset",
    "busco_lineage_creation_date",
    "busco_mode",
    "busco_options",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_audit(frame: pd.DataFrame) -> dict:
    expected_qv = -10.0 * np.log10(frame["error_rate"])
    qv_residual = frame["qv"] - expected_qv
    assembly_residual = frame["assembly_error"] - (
        frame["assembly_frac"] - 1.0
    ).abs()
    return {
        "qv_is_exact_error_transform": bool(qv_residual.abs().max() < 1e-12),
        "qv_formula": "qv = -10 * log10(error_rate)",
        "qv_error_max_absolute_residual": float(qv_residual.abs().max()),
        "assembly_error_is_exact_fraction_transform": bool(
            assembly_residual.abs().max() < 1e-12
        ),
        "assembly_error_formula": "assembly_error = abs(assembly_frac - 1)",
        "assembly_error_max_absolute_residual": float(
            assembly_residual.abs().max()
        ),
    }


def busco_audit(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    busco_n_values = sorted(int(value) for value in frame["busco_n"].unique())
    reconstructed_count = np.rint(
        frame["busco_complete"] * frame["busco_n"] / 100.0
    ).astype(int)
    reconstructed_percent = reconstructed_count / frame["busco_n"] * 100.0
    resolution = 100.0 / frame["busco_n"]

    ordered = frame.sort_values(
        ["Sample", "Coverage_effective", "round"]
    ).copy()
    ordered["busco_complete_count"] = reconstructed_count.loc[ordered.index]
    ordered["delta_busco_complete_count"] = ordered.groupby(
        ["Sample", "Coverage_effective"]
    )["busco_complete_count"].diff()

    trajectory_rows = []
    for (sample, coverage), trajectory in ordered.groupby(
        ["Sample", "Coverage_effective"], sort=True
    ):
        trajectory = trajectory.sort_values("round")
        r1 = trajectory.iloc[0]
        r5 = trajectory.iloc[-1]
        maximum_future = float(trajectory.iloc[1:]["busco_complete"].max())
        any_gain = maximum_future - float(r1["busco_complete"]) > TOLERANCES[
            "busco_complete"
        ]
        terminal_gain = (
            float(r5["busco_complete"]) - float(r1["busco_complete"])
            > TOLERANCES["busco_complete"]
        )
        trajectory_rows.append(
            {
                "Sample": str(sample),
                "Coverage_effective": str(coverage),
                "busco_n": int(r1["busco_n"]),
                "busco_resolution_percent": 100.0 / float(r1["busco_n"]),
                "r1_busco_complete": float(r1["busco_complete"]),
                "r5_busco_complete": float(r5["busco_complete"]),
                "r1_to_r5_change": float(
                    r5["busco_complete"] - r1["busco_complete"]
                ),
                "trajectory_range": float(
                    trajectory["busco_complete"].max()
                    - trajectory["busco_complete"].min()
                ),
                "any_future_gain_gt_1pp": int(any_gain),
                "terminal_gain_gt_1pp": int(terminal_gain),
                "transient_gain_only": int(any_gain and not terminal_gain),
            }
        )
    trajectories = pd.DataFrame(trajectory_rows)

    adjacent = ordered["delta_busco_complete_count"].dropna()
    return (
        {
            "busco_n_values": busco_n_values,
            "percent_resolution_values": sorted(
                float(value) for value in resolution.unique()
            ),
            "one_percent_threshold_requires_at_least_two_markers": bool(
                all(2 * value > 1.0 >= value for value in resolution.unique())
            ),
            "max_rounding_residual_percent": float(
                (frame["busco_complete"] - reconstructed_percent).abs().max()
            ),
            "adjacent_change_positive_fraction": float((adjacent > 0).mean()),
            "adjacent_change_zero_fraction": float((adjacent == 0).mean()),
            "adjacent_change_negative_fraction": float((adjacent < 0).mean()),
            "r1_any_future_gain_gt_1pp_fraction": float(
                trajectories["any_future_gain_gt_1pp"].mean()
            ),
            "r1_terminal_gain_gt_1pp_fraction": float(
                trajectories["terminal_gain_gt_1pp"].mean()
            ),
            "r1_transient_gain_only_fraction": float(
                trajectories["transient_gain_only"].mean()
            ),
            "mean_r1_to_r5_change": float(
                trajectories["r1_to_r5_change"].mean()
            ),
            "missing_provenance_fields": [
                field for field in BUSCO_PROVENANCE_FIELDS if field not in frame.columns
            ],
        },
        trajectories,
    )


def adjacent_change_audit(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["Sample", "Coverage_effective", "round"])
    rows = []
    for metric in METRICS:
        changes = ordered.groupby(["Sample", "Coverage_effective"])[metric].diff()
        changes = changes.dropna()
        rows.append(
            {
                "metric": metric,
                "tolerance": TOLERANCES[metric],
                "n_adjacent_changes": len(changes),
                "positive_fraction": float((changes > 0).mean()),
                "zero_fraction": float((changes == 0).mean()),
                "negative_fraction": float((changes < 0).mean()),
                "median_change": float(changes.median()),
                "p05_change": float(changes.quantile(0.05)),
                "p95_change": float(changes.quantile(0.95)),
                "absolute_change_exceeds_tolerance_fraction": float(
                    (changes.abs() > TOLERANCES[metric]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def replicate_audit(frame: pd.DataFrame) -> dict:
    keys = ["Sample", "Coverage_effective", "round"]
    repeated = frame.groupby(keys).size()
    return {
        "rows": len(frame),
        "unique_sample_coverage_rounds": int(frame[keys].drop_duplicates().shape[0]),
        "technical_replicate_cells": int((repeated > 1).sum()),
        "technical_repeatability_estimable": bool((repeated > 1).any()),
        "interpretation": (
            "Coverage trajectories are experimental conditions, not repeated "
            "measurements of the same assembly artifact."
        ),
    }


def materiality_contract(
    dependency: dict,
    busco: dict,
    replicates: dict,
) -> dict:
    return {
        "contract_version": "0.1.0-development",
        "release_gate": "blocked",
        "metrics": {
            "qv_error_pair": {
                "status": "invalid_redundant_and_provenance_ambiguous",
                "finding": dependency["qv_formula"],
                "required_action": (
                    "Use one same-round, versioned mapping-error measurement; "
                    "derive QV from it if needed, but never count both endpoints."
                ),
            },
            "busco_complete": {
                "status": "provisional",
                "finding": (
                    "The 1 percentage-point threshold resolves to at least two "
                    "of 116 markers, but 9.4% of trajectories show only a "
                    "transient >1-point gain."
                ),
                "required_action": (
                    "Store complete marker counts plus BUSCO version, lineage "
                    "dataset, creation date, mode, and options; justify whether "
                    "a transient two-marker gain is a material stopping outcome."
                ),
            },
            "assembly_fraction_error": {
                "status": "provisional_redundant_representation",
                "finding": dependency["assembly_error_formula"],
                "required_action": (
                    "Use one representation and record expected-genome-size "
                    "provenance; validate the 0.01 tolerance externally."
                ),
            },
        },
        "global_limitations": {
            "technical_repeatability_estimable": replicates[
                "technical_repeatability_estimable"
            ],
            "busco_provenance_complete": not busco["missing_provenance_fields"],
            "held_out_test_may_be_opened": False,
            "model_retraining_authorized": False,
            "nextflow_adaptive_stopping_authorized": False,
        },
    }


def run_audit(
    *,
    dataset_path: Path,
    split_path: Path,
    output_dir: Path,
) -> dict:
    frame, split_audit = prepare_training_frame(dataset_path, split_path)
    dependency = dependency_audit(frame)
    busco, busco_trajectories = busco_audit(frame)
    adjacent = adjacent_change_audit(frame)
    replicates = replicate_audit(frame)
    contract = materiality_contract(dependency, busco, replicates)

    output_dir.mkdir(parents=True, exist_ok=True)
    adjacent.to_csv(output_dir / "adjacent_change_audit.csv", index=False)
    busco_trajectories.to_csv(output_dir / "busco_trajectory_audit.csv", index=False)
    (output_dir / "materiality_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "audit_schema_version": "1.0.0",
        "scientific_status": "development-only measurement-validity gate",
        "dataset_sha256": sha256_file(dataset_path),
        "split_audit": split_audit,
        "dependency_audit": dependency,
        "busco_audit": busco,
        "replicate_audit": replicates,
        "adjacent_change_audit": adjacent.to_dict(orient="records"),
        "materiality_contract": contract,
        "external_contract_basis": {
            "source": "BUSCO official user guide",
            "url": "https://busco.ezlab.org/busco_userguide.html",
            "requirements_used": [
                "scores are percentages of lineage-specific marker genes",
                "report BUSCO and dependency versions",
                "report lineage dataset and creation date",
                "report BUSCO options and assessed assembly version",
            ],
        },
    }
    (output_dir / "measurement_validity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        default=Path("outputs/esdp_light_measurement_validity"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_audit(
        dataset_path=args.dataset.resolve(),
        split_path=args.split.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(report["materiality_contract"], indent=2, sort_keys=True))
    print(f"\nResults: {args.output_dir.resolve()}")
    print("Held-out test samples used: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
