#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4_label_optimal_round.py - Label Optimal Rounds with 3-CLASS SYSTEM

"""

import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Thresholds from config
R1_MIN_BUSCO = config['r1_thresholds']['min_busco']
R1_MAX_ASMERR = config['r1_thresholds']['max_assembly_error']
R1_MAX_ERR = config['r1_thresholds']['max_error_rate']
R1_MAX_CONTIGS = config['r1_thresholds']['max_contigs']
R1_MIN_COV_EST = config['r1_thresholds']['min_coverage_est']
R1_MAX_ALIGN_ERR_CONS = config['r1_thresholds']['max_align_err_cons']

EPS_QV = config['stability']['eps_qv']
EPS_ERROR = config['stability']['eps_error']
EPS_BUSCO = config['stability']['eps_busco']
EPS_ASMFRAC = config['stability']['eps_assembly_frac']
USE_ASMFRAC_IN_STABILITY = config['stability']['use_assembly_frac']

PLATEAU_REL_FRAC = config['plateau']['relative_threshold']

def load_data():
    """Load engineered dataset."""
    df = pd.read_csv(config['data']['engineered_csv'])
    logger.info(f"Loaded {len(df)} rows")
    return df

def _coverage_proxy(r1: pd.Series) -> float:
    """Get coverage estimate from various sources."""
    for k in ("coverage_est", "mean_edge_coverage", "polish_mean_contig_cov"):
        v = r1.get(k, np.nan)
        if pd.notna(v):
            try:
                return float(v)
            except Exception:
                pass
    return None

def assess_r1_and_stability(g: pd.DataFrame) -> tuple:
    """Assess R1 quality and stability across rounds."""
    g = g.sort_values("round").copy()
    r1 = g.iloc[0]

    # R1 quality metrics
    r1_busco = float(r1.get("busco_complete", np.nan))
    r1_asmerr = float(r1.get("assembly_error", np.nan))
    r1_err = float(r1.get("error_rate", np.nan))
    r1_contigs = float(r1.get("num_contigs", np.nan)) if "num_contigs" in g.columns else np.nan
    r1_qv = float(r1.get("qv", np.nan))
    r1_af = float(r1.get("assembly_frac", np.nan)) if "assembly_frac" in g.columns else np.nan

    # Core quality checks
    r1_ok = True
    if not np.isnan(r1_busco): r1_ok &= (r1_busco >= R1_MIN_BUSCO)
    if not np.isnan(r1_asmerr): r1_ok &= (r1_asmerr <= R1_MAX_ASMERR)
    if not np.isnan(r1_err): r1_ok &= (r1_err <= R1_MAX_ERR)
    if R1_MAX_CONTIGS is not None and not np.isnan(r1_contigs):
        r1_ok &= (r1_contigs <= R1_MAX_CONTIGS)

    # Flye-based veto
    covp = _coverage_proxy(r1)
    cov_ok = True if covp is None else (covp >= R1_MIN_COV_EST)
    aerr_cons = r1.get("align_err_consensus", np.nan)
    aerr_ok = True if pd.isna(aerr_cons) else (float(aerr_cons) <= R1_MAX_ALIGN_ERR_CONS)

    r1_ok_ext = bool(r1_ok and cov_ok and aerr_ok)

    # Stability across rounds
    if len(g) > 1:
        parts = []
        if "qv" in g.columns and not np.isnan(r1_qv):
            parts.append((g["qv"] - r1_qv).abs().max(skipna=True) <= EPS_QV)
        if "error_rate" in g.columns and not np.isnan(r1_err):
            parts.append((g["error_rate"] - r1_err).abs().max(skipna=True) <= EPS_ERROR)
        if "busco_complete" in g.columns and not np.isnan(r1_busco):
            parts.append((g["busco_complete"] - r1_busco).abs().max(skipna=True) <= EPS_BUSCO)
        if USE_ASMFRAC_IN_STABILITY and "assembly_frac" in g.columns and not np.isnan(r1_af):
            parts.append((g["assembly_frac"] - r1_af).abs().max(skipna=True) <= EPS_ASMFRAC)
        stability_ok = all(parts) if parts else True
    else:
        stability_ok = True

    return r1_ok_ext, stability_ok

def find_optimal_round_5class(group: pd.DataFrame) -> int:
    """Find optimal round (original 5-class system)."""
    g = group.sort_values("round").reset_index(drop=True).copy()

    if "assembly_error" not in g.columns and "assembly_frac" in g.columns:
        g["assembly_error"] = (g["assembly_frac"] - 1.0).abs()
    elif "assembly_error" not in g.columns:
        g["assembly_error"] = np.nan

    r1_ok_ext, stability_ok = assess_r1_and_stability(g)

    # Early-exit to R1
    if r1_ok_ext and stability_ok:
        return int(g["round"].iloc[0])

    # Plateau logic
    if 'score_improvement' not in g.columns:
        return int(g["round"].min() + (1 if not r1_ok_ext and g["round"].max() >= 2 else 0))
    
    max_gain = g["score_improvement"].max()
    if pd.isna(max_gain) or max_gain <= 0:
        return int(g["round"].min() + (1 if not r1_ok_ext and g["round"].max() >= 2 else 0))

    thr = max(PLATEAU_REL_FRAC * max_gain, 0.0)
    small = (g["score_improvement"].abs() < thr).tolist()

    for i in range(1, len(g)):
        if small[i]:
            candidate = int(g.loc[i, "round"])
            if candidate == 1 and not r1_ok_ext:
                return 2 if (g["round"].max() >= 2) else 1
            return candidate

    candidate = int(g["round"].iloc[-1])
    if candidate == 1 and not r1_ok_ext:
        return 2 if (g["round"].max() >= 2) else 1
    return candidate

def convert_to_3class(optimal_round_5class: int) -> int:
    """Convert 5-class labels to 3-class system.
    
    Original -> New:
    1, 2 -> 1 (Early)
    3, 4 -> 2 (Medium)
    5 -> 3 (Late)
    """
    mapping = config['classes']['original_to_new']
    return mapping.get(optimal_round_5class, 3)  # Default to Late if unknown

def compute_group_flags(group: pd.DataFrame) -> pd.Series:
    """Compute R1 quality and stability flags for group."""
    g = group.sort_values("round").copy()
    r1_ok_ext, stability_ok = assess_r1_and_stability(g)
    return pd.Series({
        "r1_ok_group": int(r1_ok_ext),
        "stable_all_group": int(stability_ok),
    })

def main():
    """Label optimal rounds with 3-class system."""
    logger.info("=" * 60)
    logger.info("Starting Optimal Round Labeling (3-CLASS SYSTEM)")
    logger.info("=" * 60)
    
    df = load_data()
    
    cov_col = "Coverage_effective" if "Coverage_effective" in df.columns else "Coverage"
    df["round"] = df["round"].astype(int)
    df[cov_col] = df[cov_col].astype(str).str.upper().str.replace("X$", "X", regex=True)
    df = df.sort_values(["Sample", cov_col, "round"]).reset_index(drop=True)
    
    # Ensure required columns
    needed_cols = ["Sample", cov_col, "round", "qv", "busco_complete", "error_rate"]
    missing = set(needed_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Find optimal rounds (5-class first, then convert)
    logger.info("Computing optimal rounds (5-class)...")
    optimal_rounds_5class = (
        df.groupby(["Sample", cov_col], sort=False)
        .apply(find_optimal_round_5class)
        .reset_index(name="optimal_rounds_5class")
    )
    
    # Convert to 3-class
    logger.info("Converting to 3-class system...")
    optimal_rounds_5class["optimal_rounds_3class"] = optimal_rounds_5class["optimal_rounds_5class"].apply(convert_to_3class)
    
    # Compute group flags
    logger.info("Computing group flags...")
    group_flags = (
        df.groupby(["Sample", cov_col], sort=False)
        .apply(compute_group_flags)
        .reset_index()
    )
    
    # Merge everything
    logger.info("Merging labels and flags...")
    
    # Standardize coverage column name for merge
    if cov_col == "Coverage_effective":
        optimal_rounds_5class = optimal_rounds_5class.rename(columns={cov_col: "Coverage"})
        group_flags = group_flags.rename(columns={cov_col: "Coverage"})
        merge_on = ["Sample", "Coverage"]
    else:
        merge_on = ["Sample", "Coverage"]
    
    df_final = df.merge(optimal_rounds_5class, left_on=["Sample", cov_col], right_on=merge_on, how="left")
    df_final = df_final.merge(group_flags, left_on=["Sample", cov_col], right_on=merge_on, how="left", suffixes=("", "_grp"))
    
    # Clean up duplicate coverage columns if any
    if cov_col == "Coverage_effective" and "Coverage_y" in df_final.columns:
        df_final = df_final.drop(columns=["Coverage_y"])
        if "Coverage_x" in df_final.columns:
            df_final = df_final.rename(columns={"Coverage_x": "Coverage"})
    
    # Save labeled dataset
    output_path = config['data']['labeled_csv']
    df_final.to_csv(output_path, index=False)
    
    # Report distribution
    dist_5class = (
        df_final[["Sample", cov_col, "optimal_rounds_5class"]]
        .drop_duplicates()["optimal_rounds_5class"]
        .value_counts()
        .sort_index()
    )
    
    dist_3class = (
        df_final[["Sample", cov_col, "optimal_rounds_3class"]]
        .drop_duplicates()["optimal_rounds_3class"]
        .value_counts()
        .sort_index()
    )
    
    logger.info("=" * 60)
    logger.info("Labeling Complete!")
    logger.info("=" * 60)
    logger.info("\nOriginal 5-class distribution (by groups):")
    for cls, count in dist_5class.items():
        logger.info(f"  Round {cls}: {count} groups")
    
    logger.info("\nNEW 3-class distribution (by groups):")
    class_names = config['classes']['class_mapping']
    for cls, count in dist_3class.items():
        logger.info(f"  Class {cls} ({class_names[cls]}): {count} groups")
    
    logger.info(f"\nOutput saved to: {output_path}")
    logger.info(f"Added columns: optimal_rounds_5class, optimal_rounds_3class, r1_ok_group, stable_all_group")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
