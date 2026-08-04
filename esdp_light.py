"""Feature contracts and audit helpers for the ESDP-light research track."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Literal, Sequence

import pandas as pd


class MetricCostTier(str, Enum):
    """Incremental deployment cost for one metric family."""

    STATIC_ONCE = "STATIC_ONCE"
    ROUND_NATIVE = "ROUND_NATIVE"
    ALIGNMENT_REUSE = "ALIGNMENT_REUSE"
    EXPENSIVE_QC = "EXPENSIVE_QC"


@dataclass(frozen=True)
class MetricSpec:
    """Provenance and deployment status for one candidate raw metric."""

    name: str
    source: str
    cost_tier: MetricCostTier
    deployment_status: Literal["ready", "provisional", "forbidden"]
    notes: str = ""


METRIC_REGISTRY = (
    MetricSpec("coverage_est", "Flye log/params", MetricCostTier.STATIC_ONCE, "ready"),
    MetricSpec("raw_total_bp", "Flye log", MetricCostTier.STATIC_ONCE, "ready"),
    MetricSpec("raw_read_n50", "Flye log", MetricCostTier.STATIC_ONCE, "ready"),
    MetricSpec(
        "raw_n_reads",
        "raw-read summary",
        MetricCostTier.STATIC_ONCE,
        "provisional",
        "Populated in the dataset but not reconstructed by the checked-in collector.",
    ),
    MetricSpec("n50", "assembly FASTA statistics", MetricCostTier.ROUND_NATIVE, "ready"),
    MetricSpec(
        "num_contigs",
        "assembly FASTA statistics",
        MetricCostTier.ROUND_NATIVE,
        "provisional",
        "Currently collected through strict QUAST; requires a lightweight FASTA extractor.",
    ),
    MetricSpec(
        "total_length",
        "assembly FASTA statistics",
        MetricCostTier.ROUND_NATIVE,
        "provisional",
        "Currently collected through strict QUAST; requires a lightweight FASTA extractor.",
    ),
    MetricSpec(
        "gc",
        "assembly FASTA statistics",
        MetricCostTier.ROUND_NATIVE,
        "provisional",
        "Currently collected through strict QUAST; requires a lightweight FASTA extractor.",
    ),
    MetricSpec("min_overlap", "Flye log", MetricCostTier.STATIC_ONCE, "ready"),
    MetricSpec("overlap_based_coverage", "Flye log", MetricCostTier.STATIC_ONCE, "ready"),
    MetricSpec("ovlp_div_initial", "Flye log", MetricCostTier.STATIC_ONCE, "ready"),
    MetricSpec("ovlp_median_div_first", "Flye log", MetricCostTier.STATIC_ONCE, "ready"),
    MetricSpec("ovlp_median_div_last", "Flye log", MetricCostTier.STATIC_ONCE, "ready"),
    MetricSpec("mean_edge_coverage", "Flye log", MetricCostTier.STATIC_ONCE, "ready"),
    MetricSpec(
        "read_cov_cutoff",
        "Flye log",
        MetricCostTier.STATIC_ONCE,
        "provisional",
        "The checked-in Flye parser defines but does not extract this field.",
    ),
    MetricSpec(
        "unique_cov_threshold",
        "Flye log",
        MetricCostTier.STATIC_ONCE,
        "provisional",
        "The checked-in Flye parser defines but does not extract this field.",
    ),
    MetricSpec("asm_total_length", "Flye log", MetricCostTier.ROUND_NATIVE, "ready"),
    MetricSpec("asm_fragments", "Flye log", MetricCostTier.ROUND_NATIVE, "ready"),
    MetricSpec("asm_frag_N50", "Flye log", MetricCostTier.ROUND_NATIVE, "ready"),
    MetricSpec("asm_largest_frag", "Flye log", MetricCostTier.ROUND_NATIVE, "ready"),
    MetricSpec("asm_mean_coverage", "Flye log", MetricCostTier.ROUND_NATIVE, "ready"),
    MetricSpec("ai_num_contigs", "Flye assembly_info.txt", MetricCostTier.ROUND_NATIVE, "ready"),
    MetricSpec("ai_total_bp", "Flye assembly_info.txt", MetricCostTier.ROUND_NATIVE, "ready"),
    MetricSpec("ai_mean_cov", "Flye assembly_info.txt", MetricCostTier.ROUND_NATIVE, "ready"),
    MetricSpec("ai_median_cov", "Flye assembly_info.txt", MetricCostTier.ROUND_NATIVE, "ready"),
    MetricSpec("ai_cov_cv", "Flye assembly_info.txt", MetricCostTier.ROUND_NATIVE, "ready"),
    MetricSpec("ai_circular_n", "Flye assembly_info.txt", MetricCostTier.ROUND_NATIVE, "provisional", "Not reconstructed by the checked-in parser."),
    MetricSpec("ai_circular_bp_frac", "Flye assembly_info.txt", MetricCostTier.ROUND_NATIVE, "provisional", "Not reconstructed by the checked-in parser."),
    MetricSpec("ai_repeat_bp_frac", "Flye assembly_info.txt", MetricCostTier.ROUND_NATIVE, "provisional", "Not reconstructed by the checked-in parser."),
    MetricSpec("ai_low_cov_bp_frac", "Flye assembly_info.txt", MetricCostTier.ROUND_NATIVE, "provisional", "Not reconstructed by the checked-in parser."),
    MetricSpec("ai_short_bp_frac_10kb", "Flye assembly_info.txt", MetricCostTier.ROUND_NATIVE, "provisional", "Not reconstructed by the checked-in parser."),
    MetricSpec("ai_longest_len", "Flye assembly_info.txt", MetricCostTier.ROUND_NATIVE, "provisional", "Not reconstructed by the checked-in parser."),
    MetricSpec("ai_longest_cov", "Flye assembly_info.txt", MetricCostTier.ROUND_NATIVE, "provisional", "Not reconstructed by the checked-in parser."),
    MetricSpec(
        "polish_mean_contig_cov",
        "polishing alignment",
        MetricCostTier.ALIGNMENT_REUSE,
        "provisional",
        "Dataset values exist, but the checked-in collector does not reconstruct them.",
    ),
    MetricSpec(
        "align_err_consensus",
        "polishing alignment",
        MetricCostTier.ALIGNMENT_REUSE,
        "provisional",
        "Requires a versioned Minimap2/Samtools extraction contract.",
    ),
    MetricSpec(
        "align_err_polishing",
        "polishing alignment",
        MetricCostTier.ALIGNMENT_REUSE,
        "provisional",
        "Requires a versioned Minimap2/Samtools extraction contract.",
    ),
    MetricSpec(
        "mapping_error_rate",
        "post-polish Minimap2/Samtools alignment",
        MetricCostTier.ALIGNMENT_REUSE,
        "provisional",
        "New unambiguous field; must be measured against the polished assembly for the current round.",
    ),
)


LIGHT_CORE_FEATURES = tuple(
    spec.name for spec in METRIC_REGISTRY if spec.deployment_status == "ready"
)
LIGHT_ALIGNMENT_CANDIDATES = tuple(
    spec.name
    for spec in METRIC_REGISTRY
    if spec.cost_tier is MetricCostTier.ALIGNMENT_REUSE
)
LIGHT_PROSPECTIVE_DERIVED_FEATURES = (
    "delta_n50",
    "delta_num_contigs",
    "n50_from_r1",
)


FORBIDDEN_LIGHT_FEATURES = frozenset(
    {
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
        "cost_benefit_ratio",
        "delta_qv_cumsum",
        "delta_busco_complete_cumsum",
        "score_improvement",
        "gain_cumulative",
        "qv_from_r1",
        "error_rate_from_r1",
        "busco_complete_from_r1",
        "assembly_frac_from_r1",
        "delta_qv_trend",
        "delta_busco_complete_trend",
        "score_improvement_trend",
        "is_plateau",
        "plateau_streak",
        "completeness_score",
        "assembly_quality",
        "polishing_effectiveness",
        "r1_ok_group",
        "stable_all_group",
        "optimal_rounds_3class",
        "optimal_rounds_5class",
    }
)


class LightFeatureContractError(ValueError):
    """Raised when a candidate violates the low-cost feature boundary."""


def validate_light_feature_contract(
    feature_names: Sequence[str],
    *,
    allow_provisional_alignment: bool = False,
) -> tuple[str, ...]:
    """Validate uniqueness, provenance, and forbidden-feature exclusions."""
    names = tuple(feature_names)
    if not names:
        raise LightFeatureContractError("light feature set cannot be empty")
    if len(names) != len(set(names)):
        raise LightFeatureContractError("light feature names must be unique")

    forbidden = sorted(set(names) & FORBIDDEN_LIGHT_FEATURES)
    if forbidden:
        raise LightFeatureContractError(
            f"forbidden expensive or leakage-prone features: {forbidden}"
        )

    allowed = set(LIGHT_CORE_FEATURES) | set(LIGHT_PROSPECTIVE_DERIVED_FEATURES)
    if allow_provisional_alignment:
        allowed.update(LIGHT_ALIGNMENT_CANDIDATES)
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise LightFeatureContractError(
            f"features lack an approved deployment provenance: {unknown}"
        )
    return names


def audit_feature_availability(
    frame: pd.DataFrame,
    feature_names: Iterable[str],
) -> pd.DataFrame:
    """Summarize observed values without mutating or imputing the dataset."""
    rows = []
    total = len(frame)
    for name in feature_names:
        present = int(frame[name].notna().sum()) if name in frame else 0
        rows.append(
            {
                "feature": name,
                "present_rows": present,
                "total_rows": total,
                "availability_fraction": present / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def validate_frozen_sample_split(
    frame: pd.DataFrame,
    train_samples: Iterable[str],
    test_samples: Iterable[str],
) -> None:
    """Require a disjoint, exhaustive split at biological-sample level."""
    if "Sample" not in frame:
        raise ValueError("dataset requires a Sample column")
    observed = set(frame["Sample"].astype(str))
    train = set(map(str, train_samples))
    test = set(map(str, test_samples))
    overlap = train & test
    if overlap:
        raise ValueError(f"train/test sample overlap: {sorted(overlap)}")
    missing = observed - (train | test)
    unexpected = (train | test) - observed
    if missing or unexpected:
        raise ValueError(
            f"split does not match dataset; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
