#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
9_benchmark_resources.py - Real-world resource saving analysis (FINAL)

Measures computational savings from ESDP early stopping decisions.

FINAL CORRECTIONS:
- Uses the SAME sample-level split saved by 5_train_models.py
- Benchmarks at trajectory level: Sample + Coverage (not Sample only)
- Prevents mixing rounds from different coverages of the same sample
- Uses best_model_pipeline.pkl directly
- Avoids legacy artifact mismatch warnings
- Produces bootstrap CIs, Wilcoxon tests, baseline comparisons, and plots

Metrics:
- CPU-hours saved per trajectory
- Quality loss (QV, BUSCO) from early stopping
- Efficiency (QV per CPU-hour)
- Baseline comparisons: Always R5, Always R1, Random
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION - REAL DVT METRICS
# ============================================================
CPU_HOURS_PER_ROUND = 0.2684
MEMORY_GB_PER_ROUND = 2.52

ACCEPTABLE_QV_LOSS = 0.5
ACCEPTABLE_BUSCO_LOSS = 1.0

N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42

DATA_PATH = Path("data/training_dataset_with_target.csv")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
PLOTS_DIR = OUTPUTS_DIR / "plots"


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def bootstrap_ci(
    data,
    statistic=np.mean,
    n_bootstrap=N_BOOTSTRAP,
    confidence=CONFIDENCE_LEVEL,
    random_state=RANDOM_STATE,
):
    """Bootstrap confidence interval for a statistic."""
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return np.nan, np.nan

    rng = np.random.RandomState(random_state)
    bootstrap_stats = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def format_ci(mean, lower, upper, decimals=2):
    """Format mean and confidence interval."""
    if np.isnan(mean):
        return "NA"
    return f"{mean:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"


def significance_stars(p):
    """Return significance stars."""
    if np.isnan(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def safe_wilcoxon(x, y):
    """Run Wilcoxon signed-rank test safely."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan, np.nan

    if np.allclose(x, y):
        return 0.0, 1.0

    try:
        stat, p = stats.wilcoxon(x, y)
        return float(stat), float(p)
    except Exception as e:
        logger.warning(f"Wilcoxon test failed: {e}")
        return np.nan, np.nan


def get_split_file():
    """Return the split file path, supporting both old and new locations."""
    candidates = [
        OUTPUTS_DIR / "train_test_split_samples.json",
        OUTPUTS_DIR / "results" / "train_test_split_samples.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find train_test_split_samples.json in outputs/ or outputs/results/."
    )


def get_cov_col(df):
    """Get coverage column name."""
    if "Coverage_effective" in df.columns:
        return "Coverage_effective"
    if "Coverage" in df.columns:
        return "Coverage"
    raise ValueError("Neither 'Coverage_effective' nor 'Coverage' found in dataframe.")


def load_feature_names():
    """Load feature names used by the trained model."""
    feature_path = MODELS_DIR / "feature_names.txt"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature names file: {feature_path}")
    return [line.strip() for line in feature_path.read_text().splitlines() if line.strip()]


# ============================================================
# MAIN BENCHMARK FUNCTION
# ============================================================
def run_benchmark():
    logger.info("=" * 60)
    logger.info("ESDP RESOURCE SAVING BENCHMARK (FINAL)")
    logger.info("=" * 60)
    logger.info(f"Cost model: {CPU_HOURS_PER_ROUND}h per round")
    logger.info(f"Baseline cost: {5 * CPU_HOURS_PER_ROUND}h (5 rounds)")
    logger.info(f"Bootstrap samples: {N_BOOTSTRAP}")
    logger.info(f"Confidence level: {CONFIDENCE_LEVEL * 100}%")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. Load data and trained pipeline
    # ============================================================
    logger.info("\nLoading data and model...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input data: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    feature_names = load_feature_names()

    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(
            "Missing required features in dataset: " + ", ".join(missing_features)
        )

    pipeline_path = MODELS_DIR / "best_model_pipeline.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Missing pipeline file: {pipeline_path}")

    pipeline = joblib.load(pipeline_path)
    cov_col = get_cov_col(df)

    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Unique samples: {df['Sample'].nunique()}")

    # ============================================================
    # 2. Load saved split from training
    # ============================================================
    split_file = get_split_file()
    with open(split_file, "r") as f:
        split_info = json.load(f)

    test_samples = set(split_info["test_samples"])
    df_test = df[df["Sample"].astype(str).isin(test_samples)].copy()

    logger.info(f"\nUsing TEST SET for benchmark: {len(test_samples)} samples")
    logger.info(f"Test rows: {len(df_test)}")

    # ============================================================
    # 3. Benchmark each trajectory: Sample + Coverage
    # ============================================================
    results = []

    grouped = df_test.groupby(["Sample", cov_col], dropna=False)

    for (sample, coverage), traj in grouped:
        traj = traj.sort_values("round").copy()

        required_rounds = {1, 3, 5}
        available_rounds = set(traj["round"].astype(int).tolist())

        if not required_rounds.issubset(available_rounds):
            logger.warning(
                f"Skipping trajectory {sample} | {coverage}: missing required rounds "
                f"(has {sorted(available_rounds)}, needs [1,3,5])"
            )
            continue

        r1 = traj[traj["round"] == 1].iloc[0]
        r3 = traj[traj["round"] == 3].iloc[0]
        r5 = traj[traj["round"] == 5].iloc[0]

        # Prepare R1 features for prediction using the trained pipeline directly
        X_r1 = pd.DataFrame([r1[feature_names]], columns=feature_names)
        X_r1 = X_r1.replace([np.inf, -np.inf], np.nan)

        y_pred = int(pipeline.predict(X_r1)[0])

        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_r1)[0]
            confidence = float(y_proba[y_pred])
        else:
            y_proba = None
            confidence = np.nan

        class_to_rounds = {0: 1, 1: 3, 2: 5}
        rec_rounds = class_to_rounds[y_pred]

        random_rounds = int(
            np.random.RandomState(
                abs(hash(f"{sample}|{coverage}|{RANDOM_STATE}")) % (2**32)
            ).choice([1, 3, 5])
        )

        metrics_rec = traj[traj["round"] == rec_rounds].iloc[0]
        metrics_random = traj[traj["round"] == random_rounds].iloc[0]

        # Resource cost
        cpu_r5 = 5 * CPU_HOURS_PER_ROUND
        cpu_r1 = 1 * CPU_HOURS_PER_ROUND
        cpu_esdp = rec_rounds * CPU_HOURS_PER_ROUND
        cpu_random = random_rounds * CPU_HOURS_PER_ROUND

        mem_r5 = 5 * MEMORY_GB_PER_ROUND
        mem_esdp = rec_rounds * MEMORY_GB_PER_ROUND

        # Quality metrics
        qv_r5 = float(r5["qv"])
        qv_r1 = float(r1["qv"])
        qv_esdp = float(metrics_rec["qv"])
        qv_random = float(metrics_random["qv"])

        busco_r5 = float(r5["busco_complete"])
        busco_r1 = float(r1["busco_complete"])
        busco_esdp = float(metrics_rec["busco_complete"])
        busco_random = float(metrics_random["busco_complete"])

        # Savings vs R5
        cpu_saved_esdp = cpu_r5 - cpu_esdp
        cpu_saved_r1 = cpu_r5 - cpu_r1
        cpu_saved_random = cpu_r5 - cpu_random

        # Quality loss vs R5
        qv_loss_esdp = qv_r5 - qv_esdp
        qv_loss_r1 = qv_r5 - qv_r1
        qv_loss_random = qv_r5 - qv_random

        busco_loss_esdp = busco_r5 - busco_esdp
        busco_loss_r1 = busco_r5 - busco_r1
        busco_loss_random = busco_r5 - busco_random

        # Efficiency
        eff_r5 = qv_r5 / cpu_r5
        eff_r1 = qv_r1 / cpu_r1
        eff_esdp = qv_esdp / cpu_esdp
        eff_random = qv_random / cpu_random

        acceptable_esdp = (
            (qv_loss_esdp <= ACCEPTABLE_QV_LOSS)
            and (busco_loss_esdp <= ACCEPTABLE_BUSCO_LOSS)
        )
        acceptable_r1 = (
            (qv_loss_r1 <= ACCEPTABLE_QV_LOSS)
            and (busco_loss_r1 <= ACCEPTABLE_BUSCO_LOSS)
        )

        true_class = (
            int(r1["optimal_rounds_3class"]) - 1
            if r1["optimal_rounds_3class"] in [1, 2, 3]
            else np.nan
        )

        results.append({
            "Sample": sample,
            "Coverage": coverage,
            "Trajectory_ID": f"{sample}|{coverage}",
            "Genus": r1["Genus"] if "Genus" in r1.index else "Unknown",

            "Predicted_Class": y_pred,
            "Recommended_Rounds": rec_rounds,
            "Random_Rounds": random_rounds,
            "Confidence": confidence,
            "True_Class": true_class,

            "CPU_R5_Hours": cpu_r5,
            "CPU_ESDP_Hours": cpu_esdp,
            "CPU_R1_Hours": cpu_r1,
            "CPU_Random_Hours": cpu_random,
            "CPU_Saved_ESDP": cpu_saved_esdp,
            "CPU_Saved_R1": cpu_saved_r1,
            "CPU_Saved_Random": cpu_saved_random,
            "CPU_Reduction_ESDP_Pct": (cpu_saved_esdp / cpu_r5) * 100,
            "CPU_Reduction_R1_Pct": (cpu_saved_r1 / cpu_r5) * 100,
            "CPU_Reduction_Random_Pct": (cpu_saved_random / cpu_r5) * 100,

            "Memory_R5_GB": mem_r5,
            "Memory_ESDP_GB": mem_esdp,
            "Memory_Saved_GB": mem_r5 - mem_esdp,

            "QV_R5": qv_r5,
            "QV_ESDP": qv_esdp,
            "QV_R1": qv_r1,
            "QV_Random": qv_random,
            "QV_Loss_ESDP": qv_loss_esdp,
            "QV_Loss_R1": qv_loss_r1,
            "QV_Loss_Random": qv_loss_random,

            "BUSCO_R5": busco_r5,
            "BUSCO_ESDP": busco_esdp,
            "BUSCO_R1": busco_r1,
            "BUSCO_Random": busco_random,
            "BUSCO_Loss_ESDP": busco_loss_esdp,
            "BUSCO_Loss_R1": busco_loss_r1,
            "BUSCO_Loss_Random": busco_loss_random,

            "Efficiency_R5": eff_r5,
            "Efficiency_ESDP": eff_esdp,
            "Efficiency_R1": eff_r1,
            "Efficiency_Random": eff_random,
            "Efficiency_Gain_ESDP_Pct": ((eff_esdp - eff_r5) / eff_r5) * 100,
            "Efficiency_Gain_R1_Pct": ((eff_r1 - eff_r5) / eff_r5) * 100,
            "Efficiency_Gain_Random_Pct": ((eff_random - eff_r5) / eff_r5) * 100,

            "Acceptable_Quality_ESDP": acceptable_esdp,
            "Acceptable_Quality_R1": acceptable_r1,
        })

    res_df = pd.DataFrame(results)

    if res_df.empty:
        raise RuntimeError("Benchmark produced no valid trajectories. Check rounds and test split.")

    # ============================================================
    # 4. Summary with confidence intervals
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS (with 95% CI)")
    logger.info("=" * 60)

    total_traj = len(res_df)
    total_samples = res_df["Sample"].nunique()

    cpu_saved_esdp_mean = res_df["CPU_Saved_ESDP"].mean()
    cpu_saved_esdp_ci = bootstrap_ci(res_df["CPU_Saved_ESDP"].values)

    cpu_reduction_esdp_mean = res_df["CPU_Reduction_ESDP_Pct"].mean()
    cpu_reduction_esdp_ci = bootstrap_ci(res_df["CPU_Reduction_ESDP_Pct"].values)

    logger.info("\n💰 RESOURCE SAVINGS (ESDP):")
    logger.info(f"  Total trajectories: {total_traj}")
    logger.info(f"  Unique test samples: {total_samples}")
    logger.info(f"  Total CPU saved: {res_df['CPU_Saved_ESDP'].sum():.1f}h")
    logger.info(
        f"  Avg CPU saved per trajectory: "
        f"{format_ci(cpu_saved_esdp_mean, *cpu_saved_esdp_ci)}h"
    )
    logger.info(
        f"  Avg CPU reduction: "
        f"{format_ci(cpu_reduction_esdp_mean, *cpu_reduction_esdp_ci)}%"
    )

    qv_loss_mean = res_df["QV_Loss_ESDP"].mean()
    qv_loss_ci = bootstrap_ci(res_df["QV_Loss_ESDP"].values)

    busco_loss_mean = res_df["BUSCO_Loss_ESDP"].mean()
    busco_loss_ci = bootstrap_ci(res_df["BUSCO_Loss_ESDP"].values)

    logger.info("\n🎯 QUALITY IMPACT (ESDP):")
    logger.info(f"  Avg QV loss: {format_ci(qv_loss_mean, *qv_loss_ci, decimals=4)} points")
    logger.info(f"  Avg BUSCO loss: {format_ci(busco_loss_mean, *busco_loss_ci)}%")
    logger.info(
        f"  Trajectories with ZERO QV loss: "
        f"{(res_df['QV_Loss_ESDP'] <= 0.01).sum()} / {total_traj}"
    )
    logger.info(
        f"  Trajectories with acceptable loss: "
        f"{res_df['Acceptable_Quality_ESDP'].sum()} / {total_traj}"
    )

    eff_gain_mean = res_df["Efficiency_Gain_ESDP_Pct"].mean()
    eff_gain_ci = bootstrap_ci(res_df["Efficiency_Gain_ESDP_Pct"].values)

    logger.info("\n⚡ EFFICIENCY GAINS (ESDP):")
    logger.info(f"  Avg efficiency gain: {format_ci(eff_gain_mean, *eff_gain_ci)}%")
    logger.info(f"  Best efficiency gain: {res_df['Efficiency_Gain_ESDP_Pct'].max():.1f}%")

    # ============================================================
    # 5. Statistical significance tests
    # ============================================================
    logger.info("\n📊 STATISTICAL SIGNIFICANCE:")

    stat_qv_r1, p_qv_r1 = safe_wilcoxon(res_df["QV_ESDP"], res_df["QV_R1"])
    stat_cpu_r1, p_cpu_r1 = safe_wilcoxon(res_df["CPU_ESDP_Hours"], res_df["CPU_R1_Hours"])

    logger.info("  ESDP vs Always R1:")
    logger.info(f"    QV difference: p={p_qv_r1:.4f} {significance_stars(p_qv_r1)}")
    logger.info(f"    CPU difference: p={p_cpu_r1:.4f} {significance_stars(p_cpu_r1)}")

    stat_qv_rand, p_qv_rand = safe_wilcoxon(res_df["QV_ESDP"], res_df["QV_Random"])
    stat_cpu_rand, p_cpu_rand = safe_wilcoxon(res_df["CPU_ESDP_Hours"], res_df["CPU_Random_Hours"])

    logger.info("  ESDP vs Random:")
    logger.info(f"    QV difference: p={p_qv_rand:.4f} {significance_stars(p_qv_rand)}")
    logger.info(f"    CPU difference: p={p_cpu_rand:.4f} {significance_stars(p_cpu_rand)}")

    # ============================================================
    # 6. Baseline comparison table
    # ============================================================
    logger.info("\n📋 BASELINE COMPARISON:")

    comparison = pd.DataFrame({
        "Strategy": ["Always R5 (Baseline)", "ESDP", "Always R1", "Random"],
        "Avg_CPU_Hours": [
            res_df["CPU_R5_Hours"].mean(),
            res_df["CPU_ESDP_Hours"].mean(),
            res_df["CPU_R1_Hours"].mean(),
            res_df["CPU_Random_Hours"].mean(),
        ],
        "Avg_QV": [
            res_df["QV_R5"].mean(),
            res_df["QV_ESDP"].mean(),
            res_df["QV_R1"].mean(),
            res_df["QV_Random"].mean(),
        ],
        "Avg_BUSCO": [
            res_df["BUSCO_R5"].mean(),
            res_df["BUSCO_ESDP"].mean(),
            res_df["BUSCO_R1"].mean(),
            res_df["BUSCO_Random"].mean(),
        ],
        "Avg_Efficiency": [
            res_df["Efficiency_R5"].mean(),
            res_df["Efficiency_ESDP"].mean(),
            res_df["Efficiency_R1"].mean(),
            res_df["Efficiency_Random"].mean(),
        ]
    })

    logger.info(f"\n{comparison.to_string(index=False)}")

    # ============================================================
    # 7. Decision breakdown
    # ============================================================
    logger.info("\n🔍 DECISION BREAKDOWN:")
    decision_counts = res_df["Recommended_Rounds"].value_counts().sort_index()
    for rounds, count in decision_counts.items():
        pct = (count / total_traj) * 100
        logger.info(f"  {rounds} rounds: {count} trajectories ({pct:.1f}%)")

    # ============================================================
    # 8. Visualizations
    # ============================================================
    logger.info("\nGenerating visualizations...")

    # Plot 1: Resource saving vs quality loss
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        res_df["CPU_Reduction_ESDP_Pct"],
        res_df["QV_Loss_ESDP"],
        c=res_df["Recommended_Rounds"],
        cmap="RdYlGn_r",
        s=90,
        alpha=0.7,
        edgecolors="black"
    )
    ax.axhline(
        ACCEPTABLE_QV_LOSS,
        color="red",
        linestyle="--",
        label=f"Acceptable QV loss threshold ({ACCEPTABLE_QV_LOSS})"
    )
    ax.axhline(
        0,
        color="green",
        linestyle="-",
        alpha=0.4,
        label="Zero QV loss"
    )
    ax.set_xlabel("CPU Resource Reduction (%)")
    ax.set_ylabel("Quality Loss (QV points)")
    ax.set_title("Resource-efficiency benchmark of ESDP")
    ax.grid(True, alpha=0.3)
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Recommended polishing rounds", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "resource_vs_quality.png", dpi=300)
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR / 'resource_vs_quality.png'}")

    # Plot 2: Baseline comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    strategies = ["R5\n(Baseline)", "ESDP", "R1\n(Always)", "Random"]
    cpu_means = comparison["Avg_CPU_Hours"].tolist()
    qv_means = comparison["Avg_QV"].tolist()

    bars1 = ax1.bar(strategies, cpu_means, alpha=0.8)
    ax1.set_ylabel("Average CPU Hours")
    ax1.set_title("Computational Cost Comparison")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.2f}h",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bars2 = ax2.bar(strategies, qv_means, alpha=0.8)
    ax2.set_ylabel("Average QV")
    ax2.set_title("Assembly Quality Comparison")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar in bars2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "baseline_comparison.png", dpi=300)
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR / 'baseline_comparison.png'}")

    # Plot 3: Efficiency comparison per trajectory
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(res_df))
    width = 0.25
    ax.bar(x - width, res_df["Efficiency_R5"], width, label="R5 (Baseline)", alpha=0.7)
    ax.bar(x, res_df["Efficiency_ESDP"], width, label="ESDP", alpha=0.7)
    ax.bar(x + width, res_df["Efficiency_R1"], width, label="R1 (Always)", alpha=0.7)
    ax.set_xlabel("Trajectory Index")
    ax.set_ylabel("Efficiency (QV per CPU-hour)")
    ax.set_title("Computational Efficiency: Baseline vs ESDP vs Always R1")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "efficiency_comparison.png", dpi=300)
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR / 'efficiency_comparison.png'}")

    # Plot 4: Decisions by genus
    fig, ax = plt.subplots(figsize=(12, 6))
    decision_by_genus = res_df.groupby(["Genus", "Recommended_Rounds"]).size().unstack(fill_value=0)
    decision_by_genus.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Genus")
    ax.set_ylabel("Number of Trajectories")
    ax.set_title("ESDP Decisions by Genus")
    ax.legend(title="Recommended Rounds")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "decisions_by_genus.png", dpi=300)
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR / 'decisions_by_genus.png'}")

    # Plot 5: Cost-benefit analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    res_df_sorted = res_df.sort_values("CPU_Saved_ESDP", ascending=False)
    cumulative_savings = res_df_sorted["CPU_Saved_ESDP"].cumsum()
    ax1.plot(range(len(cumulative_savings)), cumulative_savings, linewidth=2)
    ax1.fill_between(range(len(cumulative_savings)), cumulative_savings, alpha=0.3)
    ax1.set_xlabel("Number of Trajectories (sorted by savings)")
    ax1.set_ylabel("Cumulative CPU-Hours Saved")
    ax1.set_title("Cumulative Resource Savings")
    ax1.grid(True, alpha=0.3)

    ax2.hist(res_df["QV_Loss_ESDP"], bins=20, edgecolor="black", alpha=0.7)
    ax2.axvline(
        ACCEPTABLE_QV_LOSS,
        color="red",
        linestyle="--",
        label=f"Acceptable threshold ({ACCEPTABLE_QV_LOSS})"
    )
    ax2.axvline(0, color="green", linestyle="-", alpha=0.5, label="No loss")
    ax2.set_xlabel("QV Loss (points)")
    ax2.set_ylabel("Number of Trajectories")
    ax2.set_title("Distribution of Quality Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cost_benefit_analysis.png", dpi=300)
    plt.close()
    logger.info(f"  Saved: {PLOTS_DIR / 'cost_benefit_analysis.png'}")

    # ============================================================
    # 9. Save outputs
    # ============================================================
    output_path = OUTPUTS_DIR / "resource_benchmark_results.csv"
    res_df.to_csv(output_path, index=False)
    logger.info(f"\n💾 Results saved to: {output_path}")

    comparison_path = OUTPUTS_DIR / "baseline_comparison_table.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"Baseline comparison saved to: {comparison_path}")

    summary = {
        "total_trajectories": int(total_traj),
        "unique_test_samples": int(total_samples),
        "total_cpu_saved_hours": float(res_df["CPU_Saved_ESDP"].sum()),
        "avg_cpu_saved_hours": float(cpu_saved_esdp_mean),
        "avg_cpu_saved_ci_lower": float(cpu_saved_esdp_ci[0]),
        "avg_cpu_saved_ci_upper": float(cpu_saved_esdp_ci[1]),
        "avg_cpu_reduction_pct": float(cpu_reduction_esdp_mean),
        "avg_cpu_reduction_ci_lower": float(cpu_reduction_esdp_ci[0]),
        "avg_cpu_reduction_ci_upper": float(cpu_reduction_esdp_ci[1]),
        "avg_qv_loss": float(qv_loss_mean),
        "avg_qv_loss_ci_lower": float(qv_loss_ci[0]),
        "avg_qv_loss_ci_upper": float(qv_loss_ci[1]),
        "avg_busco_loss": float(busco_loss_mean),
        "avg_busco_loss_ci_lower": float(busco_loss_ci[0]),
        "avg_busco_loss_ci_upper": float(busco_loss_ci[1]),
        "trajectories_zero_qv_loss": int((res_df["QV_Loss_ESDP"] <= 0.01).sum()),
        "trajectories_acceptable_loss": int(res_df["Acceptable_Quality_ESDP"].sum()),
        "avg_efficiency_gain_pct": float(eff_gain_mean),
        "avg_efficiency_gain_ci_lower": float(eff_gain_ci[0]),
        "avg_efficiency_gain_ci_upper": float(eff_gain_ci[1]),
        "wilcoxon_qv_vs_r1_pvalue": float(p_qv_r1) if not np.isnan(p_qv_r1) else np.nan,
        "wilcoxon_cpu_vs_r1_pvalue": float(p_cpu_r1) if not np.isnan(p_cpu_r1) else np.nan,
        "wilcoxon_qv_vs_random_pvalue": float(p_qv_rand) if not np.isnan(p_qv_rand) else np.nan,
        "wilcoxon_cpu_vs_random_pvalue": float(p_cpu_rand) if not np.isnan(p_cpu_rand) else np.nan,
    }

    summary_df = pd.DataFrame([summary])
    summary_path = OUTPUTS_DIR / "resource_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to: {summary_path}")

    # ============================================================
    # 10. Publication-ready summary table
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PUBLICATION-READY SUMMARY TABLE")
    logger.info("=" * 60)

    pub_table = pd.DataFrame({
        "Metric": [
            "CPU Hours Saved (mean ± 95% CI)",
            "CPU Reduction % (mean ± 95% CI)",
            "QV Loss (mean ± 95% CI)",
            "BUSCO Loss % (mean ± 95% CI)",
            "Efficiency Gain % (mean ± 95% CI)",
            "Trajectories with Zero QV Loss",
            "Trajectories with Acceptable Loss",
            "Statistical significance (QV vs Always R1)",
        ],
        "Value": [
            format_ci(cpu_saved_esdp_mean, *cpu_saved_esdp_ci) + "h",
            format_ci(cpu_reduction_esdp_mean, *cpu_reduction_esdp_ci) + "%",
            format_ci(qv_loss_mean, *qv_loss_ci, decimals=4),
            format_ci(busco_loss_mean, *busco_loss_ci) + "%",
            format_ci(eff_gain_mean, *eff_gain_ci) + "%",
            f"{(res_df['QV_Loss_ESDP'] <= 0.01).sum()} / {total_traj} "
            f"({(res_df['QV_Loss_ESDP'] <= 0.01).sum() / total_traj * 100:.1f}%)",
            f"{res_df['Acceptable_Quality_ESDP'].sum()} / {total_traj} "
            f"({res_df['Acceptable_Quality_ESDP'].sum() / total_traj * 100:.1f}%)",
            f"p={p_qv_r1:.4f} {significance_stars(p_qv_r1)}",
        ],
    })

    logger.info(f"\n{pub_table.to_string(index=False)}")

    pub_table_path = OUTPUTS_DIR / "publication_summary_table.csv"
    pub_table.to_csv(pub_table_path, index=False)
    logger.info(f"\nPublication table saved to: {pub_table_path}")

    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK COMPLETE!")
    logger.info("=" * 60)

    return res_df, summary, comparison, pub_table


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    results_df, summary, comparison, pub_table = run_benchmark()