#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_csv_merge.py - Enhanced CSV Merger with Data Validation

"""

from pathlib import Path
import pandas as pd
import re
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/csv_merge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ====
# 1) Base directories configuration
# ====
base_paths = [
    Path("/PROJECTES/MICROBIOLOGIA/GITHUB/ont-paper/out_pipeline_Pseudomonas"),
    Path("/PROJECTES/MICROBIOLOGIA/GITHUB/ont-paper/out_pipeline_Staphylococcus"),
    Path("/PROJECTES/MICROBIOLOGIA/GITHUB/ont-paper/out_pipeline_Acinetobacter_baumanii"),
    Path("/PROJECTES/MICROBIOLOGIA/GITHUB/ont-paper/out_pipeline_Enterococcus"),
    Path("/PROJECTES/MICROBIOLOGIA/GITHUB/ont-paper/out_pipeline_Bacillus_subtilis"),
    Path("/PROJECTES/MICROBIOLOGIA/GITHUB/ont-paper/out_pipeline_Corynebacterium_glutamicum"),
    Path("/PROJECTES/MICROBIOLOGIA/GITHUB/ont-paper/out_pipeline_Vibrio"),
    Path("/PROJECTES/MICROBIOLOGIA/GITHUB/ont-paper/out_pipeline_Xanthomonas_campestris"),
    Path("/PROJECTES/MICROBIOLOGIA/GITHUB/ont-paper/out_pipeline_Paenibacillus"),
]

# ====
# 2) Utility functions
# ====
def _to_bp(num_str: str, unit: Optional[str] = None) -> int:
    """Convert string with optional unit (K, M, G) to base pairs."""
    s = str(num_str).replace(",", "").strip()
    if not s:
        return 0
    val = float(s)
    if unit:
        u = unit.lower()
        if u == "k": val *= 1_000
        elif u == "m": val *= 1_000_000
        elif u == "g": val *= 1_000_000_000
    return int(round(val))

def _parse_float(s: str) -> Optional[float]:
    """Safe float parser."""
    try:
        return float(str(s).strip())
    except Exception:
        return None

def _find_flye_log(flye_dir: Path) -> Optional[Path]:
    """Find flye.log regardless of case."""
    if not flye_dir.exists():
        return None
    for p in flye_dir.iterdir():
        if p.is_file() and re.match(r"(?i)^flye.*\.log$", p.name):
            return p
    return None

def validate_dataframe(df: pd.DataFrame, stage: str) -> bool:
    """Validate dataframe structure and content."""
    if df.empty:
        logger.warning(f"{stage}: DataFrame is empty")
        return False
    
    required_cols = ['Sample', 'Coverage', 'round']
    missing = set(required_cols) - set(df.columns)
    if missing:
        logger.error(f"{stage}: Missing required columns: {missing}")
        return False
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['Sample', 'Coverage', 'round']).sum()
    if duplicates > 0:
        logger.warning(f"{stage}: Found {duplicates} duplicate rows")
    
    # Check for nulls in key columns
    for col in required_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            logger.warning(f"{stage}: Found {null_count} null values in {col}")
    
    return True

# ====
# 3) Parsers (unchanged from original, but with logging)
# ====
def parse_flye_qc_from_log(flye_log: Path) -> Dict[str, Any]:
    """Extract metrics from flye.log."""
    out = {
        "expected_genome_size": None,
        "coverage_est": None,
        "coverage_est_src": "na",
        "raw_total_bp": None,
        "raw_read_n50": None,
        "raw_read_n90": None,
        "raw_mean_read_len": None,
        "raw_n_reads": None,
        "min_overlap": None,
        "overlap_based_coverage": None,
        "ovlp_div_initial": None,
        "ovlp_median_div_first": None,
        "ovlp_median_div_last": None,
        "mean_edge_coverage": None,
        "read_cov_cutoff": None,
        "unique_cov_threshold": None,
        "align_err_consensus": None,
        "align_err_polishing": None,
        "asm_total_length": None,
        "asm_fragments": None,
        "asm_frag_N50": None,
        "asm_largest_frag": None,
        "asm_mean_coverage": None,
        "polish_mean_contig_cov": None,
        "polish_selected_cov_threshold": None,
    }
    
    if not flye_log or not flye_log.exists():
        return out

    try:
        txt = flye_log.read_text(encoding="utf-8", errors="ignore")
        lines = txt.splitlines()

        # Parse genome size
        m = re.search(r"Input genome size:\s*([0-9,]+)", txt)
        if m:
            out["expected_genome_size"] = _to_bp(m.group(1))
        if out["expected_genome_size"] is None:
            m = re.search(r"--genome-size\s+([0-9]*\.?[0-9]+)\s*([kKmMgG]?)", txt)
            if m:
                out["expected_genome_size"] = _to_bp(m.group(1), m.group(2))

        # Parse coverage
        m = re.search(r"Estimated coverage:\s*([0-9]*\.?[0-9]+)", txt)
        if m:
            out["coverage_est"] = _parse_float(m.group(1))
            out["coverage_est_src"] = "log"

        # Parse read statistics
        m = re.search(r"Overlap-based coverage:\s*([0-9]*\.?[0-9]+)", txt)
        if m:
            out["overlap_based_coverage"] = _parse_float(m.group(1))

        m = re.search(r"Total read length:\s*([0-9,]+)", txt)
        if m:
            out["raw_total_bp"] = _to_bp(m.group(1))

        m = re.search(r"Reads N50/N90:\s*([0-9,]+)\s*/\s*([0-9,]+)", txt)
        if m:
            out["raw_read_n50"] = _to_bp(m.group(1))
            out["raw_read_n90"] = _to_bp(m.group(2))

        # Additional metrics...
        m = re.search(r"Minimum overlap set to\s*([0-9]+)", txt)
        if not m:
            m = re.search(r"Selected minimum overlap\s*([0-9]+)", txt)
        if m:
            out["min_overlap"] = int(m.group(1))

        # Divergence estimates
        m = re.search(r"Initial divergence estimate\s*:\s*([0-9]*\.?[0-9]+)", txt)
        if m:
            out["ovlp_div_initial"] = _parse_float(m.group(1))
        
        meds = [_parse_float(x) for x in re.findall(r"Median overlap divergence:\s*([0-9]*\.?[0-9]+)", txt)]
        meds = [x for x in meds if x is not None]
        if meds:
            out["ovlp_median_div_first"] = meds[0]
            out["ovlp_median_div_last"] = meds[-1]

        # Coverage metrics
        m = re.search(r"Mean edge coverage:\s*([0-9]*\.?[0-9]+)", txt)
        if m:
            out["mean_edge_coverage"] = _parse_float(m.group(1))

        # Assembly statistics
        bloc = re.search(r"Assembly statistics:\s*(.+?)(?:\n\[|\Z)", txt, flags=re.DOTALL)
        if bloc:
            b = bloc.group(1)
            m = re.search(r"Total length:\s*([0-9,]+)", b)
            if m: out["asm_total_length"] = _to_bp(m.group(1))
            m = re.search(r"Fragments:\s*([0-9,]+)", b)
            if m: out["asm_fragments"] = int(m.group(1).replace(",", ""))
            m = re.search(r"Fragments N50:\s*([0-9,]+)", b)
            if m: out["asm_frag_N50"] = _to_bp(m.group(1))
            m = re.search(r"Largest frg:\s*([0-9,]+)", b)
            if m: out["asm_largest_frag"] = _to_bp(m.group(1))
            m = re.search(r"Mean coverage:\s*([0-9]*\.?[0-9]+)", b)
            if m: out["asm_mean_coverage"] = _parse_float(m.group(1))

    except Exception as e:
        logger.error(f"Error parsing {flye_log}: {e}")
        
    return out

def parse_flye_params_json(params_json: Path) -> Dict[str, Any]:
    """Fallback: parse genome_size/coverage from params.json."""
    out = {}
    if not params_json or not params_json.exists():
        return out
    
    try:
        data = json.loads(params_json.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        logger.error(f"Error parsing {params_json}: {e}")
        return out

    gs = (data.get("genome_size") or data.get("genome-size") or data.get("input_genome_size"))
    if gs:
        try:
            out["expected_genome_size"] = int(gs)
        except Exception:
            out["expected_genome_size"] = _to_bp(str(gs))

    cov = (data.get("estimated_coverage") or data.get("coverage") or data.get("asm_coverage"))
    if cov is not None:
        c = _parse_float(cov)
        if c is not None:
            out["coverage_est"] = c
            out["coverage_est_src"] = "params"

    return out

def parse_assembly_info(assembly_info_path: Path, coverage_est: Optional[float]) -> Dict[str, Any]:
    """Parse assembly_info.txt for aggregate metrics."""
    out = {
        "ai_num_contigs": None,
        "ai_total_bp": None,
        "ai_mean_cov": None,
        "ai_median_cov": None,
        "ai_cov_cv": None,
        "ai_circular_n": None,
        "ai_circular_bp_frac": None,
        "ai_repeat_bp_frac": None,
        "ai_mult_gt1_n": None,
        "ai_mult_gt1_bp_frac": None,
        "ai_alt_groups_n": None,
        "ai_alt_bp_frac": None,
        "ai_low_cov_bp_frac": None,
        "ai_short_bp_frac_10kb": None,
        "ai_longest_len": None,
        "ai_longest_cov": None,
    }
    
    if not assembly_info_path.exists():
        return out

    try:
        df = pd.read_csv(assembly_info_path, sep=r"\s*\t\s*", engine="python", dtype=str)
        df.columns = [c.strip().lower().replace(".", "").replace(" ", "_") for c in df.columns]

        def pick(options):
            for c in options:
                if c in df.columns:
                    return c
            return None

        c_len = pick(["length"])
        c_cov = pick(["cov", "cov_"])
        
        if c_len: df[c_len] = pd.to_numeric(df[c_len], errors="coerce")
        if c_cov: df[c_cov] = pd.to_numeric(df[c_cov], errors="coerce")

        total_bp = df[c_len].sum(skipna=True) if c_len else None
        out["ai_num_contigs"] = int(df.shape[0])
        out["ai_total_bp"] = int(total_bp) if pd.notna(total_bp) else None

        if c_cov and df[c_cov].notna().any():
            mean_cov = float(df[c_cov].mean(skipna=True))
            med_cov = float(df[c_cov].median(skipna=True))
            std_cov = float(df[c_cov].std(skipna=True)) if df[c_cov].count() > 1 else 0.0
            out["ai_mean_cov"] = mean_cov
            out["ai_median_cov"] = med_cov
            out["ai_cov_cv"] = (std_cov / mean_cov) if mean_cov and mean_cov > 0 else None

    except Exception as e:
        logger.error(f"Error parsing {assembly_info_path}: {e}")

    return out

def parse_busco_metrics(busco_file: Path) -> Dict[str, Any]:
    """Parse BUSCO metrics."""
    metrics = {
        "busco_complete": None, "busco_single": None, "busco_duplicated": None,
        "busco_fragmented": None, "busco_missing": None, "busco_n": None,
    }
    if not busco_file.exists():
        return metrics
    
    try:
        with open(busco_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "C:" in line and "n:" in line:
                    mC = re.search(r"C:([\d\.]+)%", line)
                    mS = re.search(r"S:([\d\.]+)%", line)
                    mD = re.search(r"D:([\d\.]+)%", line)
                    mF = re.search(r"F:([\d\.]+)%", line)
                    mM = re.search(r"M:([\d\.]+)%", line)
                    mN = re.search(r"n:(\d+)", line)
                    if mC: metrics["busco_complete"] = float(mC.group(1))
                    if mS: metrics["busco_single"] = float(mS.group(1))
                    if mD: metrics["busco_duplicated"] = float(mD.group(1))
                    if mF: metrics["busco_fragmented"] = float(mF.group(1))
                    if mM: metrics["busco_missing"] = float(mM.group(1))
                    if mN: metrics["busco_n"] = int(mN.group(1))
                    break
    except Exception as e:
        logger.error(f"Error parsing {busco_file}: {e}")
        
    return metrics

def parse_quast_metrics(quast_file: Path) -> Dict[str, Any]:
    """Parse QUAST metrics."""
    metrics = {"l50": None, "auN": None}
    if not quast_file.exists():
        return metrics
    
    try:
        with open(quast_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if re.match(r"\s*L50\s+", line):
                    m = re.search(r"L50\s+(\d+)", line)
                    if m: metrics["l50"] = int(m.group(1))
                elif "auN" in line:
                    m = re.search(r"auN\s+([\d\.]+)", line)
                    if m: metrics["auN"] = float(m.group(1))
    except Exception as e:
        logger.error(f"Error parsing {quast_file}: {e}")
        
    return metrics

def parse_sample_and_coverage_from_folder(folder_name: str) -> tuple[str, str]:
    """Parse sample name and coverage from folder name."""
    name = folder_name.strip()
    low = name.lower()
    cov = None
    if "10x" in low: cov = "10X"
    elif "20x" in low: cov = "20X"
    elif "40x" in low: cov = "40X"
    elif "compl" in low or "full" in low: cov = "FULL"

    if cov is None:
        m = re.search(r"(.*)_(\d{1,3}x|compl|full)", low)
        if m:
            cov = m.group(2).upper().replace("COMPL", "FULL")
            sample = name[:len(m.group(1))]
            return sample, cov
        return name, "FULL"

    m2 = re.search(r"(.*)_(10x|20x|40x|compl|full)", low)
    sample = name[:len(m2.group(1))] if m2 else name
    return sample, cov

# ====
# 4) Main collection logic
# ====
def main():
    """Main data collection and merging pipeline."""
    logger.info("=" * 60)
    logger.info("Starting CSV merge pipeline")
    logger.info("=" * 60)
    
    df_list = []
    stats = {"total_samples": 0, "errors": 0, "warnings": 0}
    
    for genus_dir in base_paths:
        if not genus_dir.exists():
            logger.warning(f"Path not found: {genus_dir}")
            stats["warnings"] += 1
            continue

        logger.info(f"Processing: {genus_dir.name}")
        sample_count = 0
        
        for metrics_file in genus_dir.glob("**/polish/per_round_metrics.tsv"):
            try:
                sample_cov_folder = metrics_file.parents[1].name
                sample_id, coverage = parse_sample_and_coverage_from_folder(sample_cov_folder)

                # Parse Flye data
                flye_dir = metrics_file.parents[1] / "flye"
                flye_log = _find_flye_log(flye_dir)
                params_json = flye_dir / "params.json"

                flye_from_log = parse_flye_qc_from_log(flye_log) if flye_log else {}
                flye_from_params = parse_flye_params_json(params_json) if params_json.exists() else {}

                # Merge preferences
                expected_genome_size = flye_from_log.get("expected_genome_size") or flye_from_params.get("expected_genome_size")
                coverage_est = flye_from_log.get("coverage_est") if flye_from_log.get("coverage_est") is not None else flye_from_params.get("coverage_est")
                coverage_est_src = "log" if flye_from_log.get("coverage_est") is not None else ("params" if flye_from_params.get("coverage_est") is not None else "na")

                # Derive coverage if possible
                raw_total_bp = flye_from_log.get("raw_total_bp") or flye_from_params.get("raw_total_bp")
                if coverage_est is None and expected_genome_size and raw_total_bp:
                    try:
                        coverage_est = float(raw_total_bp) / float(expected_genome_size)
                        coverage_est_src = "derived"
                    except Exception:
                        pass

                # Parse assembly_info
                ai_path = flye_dir / "assembly_info.txt"
                ai = parse_assembly_info(ai_path, coverage_est)

                # Parse per_round_metrics
                df = pd.read_csv(metrics_file, sep="\t")

                # Parse BUSCO/QUAST per round
                busco_metrics_list, quast_metrics_list = [], []
                for r in df["round"]:
                    busco_dir = metrics_file.parent / f"busco_r{r}"
                    busco_files = list(busco_dir.glob("short_summary.specific.bacteria_odb12*.txt"))
                    busco_data = parse_busco_metrics(busco_files[0]) if busco_files else parse_busco_metrics(Path(""))
                    busco_metrics_list.append(busco_data)

                    quast_dir = metrics_file.parent / f"quast_r{r}"
                    quast_file = quast_dir / "report.txt"
                    quast_metrics_list.append(parse_quast_metrics(quast_file))

                df_extra = pd.concat([pd.DataFrame(busco_metrics_list),
                                     pd.DataFrame(quast_metrics_list)], axis=1)
                df = pd.concat([df, df_extra], axis=1)

                # Add metadata
                df["Sample"] = sample_id
                df["Coverage"] = coverage
                df["Genus"] = genus_dir.name.replace("out_pipeline_", "")

                # Add Flye fields
                for k, v in {
                    "expected_genome_size": expected_genome_size,
                    "coverage_est": coverage_est,
                    "coverage_est_src": coverage_est_src,
                    **{kk: flye_from_log.get(kk) for kk in [
                        "raw_total_bp", "raw_read_n50", "raw_read_n90", "raw_mean_read_len", "raw_n_reads",
                        "min_overlap", "overlap_based_coverage",
                        "ovlp_div_initial", "ovlp_median_div_first", "ovlp_median_div_last",
                        "mean_edge_coverage", "read_cov_cutoff", "unique_cov_threshold",
                        "align_err_consensus", "align_err_polishing",
                        "asm_total_length", "asm_fragments", "asm_frag_N50", "asm_largest_frag", "asm_mean_coverage",
                        "polish_mean_contig_cov", "polish_selected_cov_threshold",
                    ]}
                }.items():
                    df[k] = v

                # Add assembly_info fields
                for k, v in ai.items():
                    df[k] = v

                # Coverage_effective
                def _mk_label(cov, est):
                    if cov == "FULL" and est is not None:
                        try:
                            return f"{int(round(float(est)))}X"
                        except Exception:
                            return cov
                    return cov

                df["Coverage_effective"] = df.apply(
                    lambda r: _mk_label(r["Coverage"], r["coverage_est"]), axis=1
                )

                # Assembly fraction
                if expected_genome_size:
                    df["assembly_frac"] = df["total_length"].astype(float) / float(expected_genome_size)
                else:
                    df["assembly_frac"] = None

                df_list.append(df)
                sample_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {metrics_file}: {e}")
                stats["errors"] += 1
        
        logger.info(f"  Processed {sample_count} samples from {genus_dir.name}")
        stats["total_samples"] += sample_count

    # ====
    # 5) Merge and validate
    # ====
    if not df_list:
        logger.error("No data found! Check paths.")
        raise SystemExit("No per_round_metrics.tsv files found.")

    logger.info("Merging all data...")
    df_all = pd.concat(df_list, ignore_index=True)
    df_all.drop_duplicates(inplace=True)
    df_all = df_all[df_all["Sample"].notnull() & (df_all["Sample"] != "")]
    df_all = df_all.loc[:, ~df_all.columns.duplicated()]

    # Order columns
    expected_cols = [
        "Genus", "Sample", "Coverage", "Coverage_effective", "coverage_est", "coverage_est_src", "round",
        "n50", "qv", "indels_per_100kb", "error_rate",
        "busco_complete", "busco_single", "busco_duplicated",
        "busco_fragmented", "busco_missing", "busco_n",
        "l50", "auN", "num_contigs", "total_length", "gc",
        "expected_genome_size", "assembly_frac",
        "raw_total_bp", "raw_read_n50", "raw_read_n90", "raw_mean_read_len", "raw_n_reads",
        "min_overlap", "overlap_based_coverage",
        "ovlp_div_initial", "ovlp_median_div_first", "ovlp_median_div_last",
        "mean_edge_coverage", "read_cov_cutoff", "unique_cov_threshold",
        "align_err_consensus", "align_err_polishing",
        "asm_total_length", "asm_fragments", "asm_frag_N50", "asm_largest_frag", "asm_mean_coverage",
        "polish_mean_contig_cov", "polish_selected_cov_threshold",
        "ai_num_contigs", "ai_total_bp",
        "ai_mean_cov", "ai_median_cov", "ai_cov_cv",
        "ai_circular_n", "ai_circular_bp_frac",
        "ai_repeat_bp_frac",
        "ai_mult_gt1_n", "ai_mult_gt1_bp_frac",
        "ai_alt_groups_n", "ai_alt_bp_frac",
        "ai_low_cov_bp_frac", "ai_short_bp_frac_10kb",
        "ai_longest_len", "ai_longest_cov",
    ]

    final_cols = [c for c in expected_cols if c in df_all.columns]
    df_all = df_all[final_cols]

    # Convert to numeric
    num_cols = [c for c in df_all.columns if c not in ["Genus", "Sample", "Coverage", "Coverage_effective", "coverage_est_src"]]
    for col in num_cols:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    # Validate
    validate_dataframe(df_all, "Final merged data")

    # ====
    # 6) Save output
    # ====
    output_file = "data/all_samples_polishing_metrics.csv"
    df_all.to_csv(output_file, index=False)

    # Summary statistics
    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output file: {output_file}")
    logger.info(f"Total rows: {len(df_all)}")
    logger.info(f"Total samples processed: {stats['total_samples']}")
    logger.info(f"Unique Sample-Coverage groups: {df_all.groupby(['Sample', 'Coverage']).ngroups}")
    logger.info(f"Genera: {df_all['Genus'].nunique()}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Warnings: {stats['warnings']}")
    
    # Data quality summary
    logger.info("\nData Quality Summary:")
    logger.info(f"  Coverage estimate available: {df_all['coverage_est'].notna().sum()} / {len(df_all)}")
    logger.info(f"  BUSCO complete available: {df_all['busco_complete'].notna().sum()} / {len(df_all)}")
    logger.info(f"  Assembly fraction available: {df_all['assembly_frac'].notna().sum()} / {len(df_all)}")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    main()
