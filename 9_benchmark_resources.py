#!/usr/bin/env python3
"""
9_benchmark_resources.py - Real-world resource saving analysis (ENHANCED)

Measures actual computational cost savings from ESDP early stopping decisions.

NEW FEATURES:
- Bootstrap confidence intervals (95% CI)
- Statistical significance tests (Wilcoxon signed-rank)
- Baseline comparisons (Always R1, Always R5, Random)
- Publication-ready summary table
- Memory usage tracking (if available)

Metrics:
- CPU-hours saved per sample
- Quality loss (QV, BUSCO) from early stopping
- Cost-benefit ratio (quality per CPU-hour)
- Pareto efficiency analysis
- Statistical significance vs baselines

Real cost baseline (from DVT_trace.txt analysis):
- CPU per round: 0.2684 hours (16.1 minutes)
- Memory per round: 2.52 GB (peak RSS)
- Baseline (5 rounds): 1.342 CPU-hours, 12.6 GB-hours
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import logging
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION - REAL DVT METRICS
# ============================================================
# Values extracted from DVT_trace.txt (72 tasks analyzed)
CPU_HOURS_PER_ROUND = 0.2684  # Real measured CPU time
MEMORY_GB_PER_ROUND = 2.52    # Real peak RSS memory

# Quality loss thresholds (acceptable degradation)
ACCEPTABLE_QV_LOSS = 0.5  # 0.5 QV points
ACCEPTABLE_BUSCO_LOSS = 1.0  # 1% BUSCO completeness

# Bootstrap parameters
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def bootstrap_ci(data, statistic=np.mean, n_bootstrap=N_BOOTSTRAP, confidence=CONFIDENCE_LEVEL):
    """
    Calculate bootstrap confidence interval for a statistic.

    Args:
        data: Array-like data
        statistic: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)

    Returns:
        (lower_bound, upper_bound)
    """
    bootstrap_stats = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return lower, upper


def format_ci(mean, lower, upper, decimals=2):
    """Format mean with confidence interval"""
    return f"{mean:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"


# ============================================================
# MAIN BENCHMARK FUNCTION
# ============================================================
def run_benchmark():
    """
    Run comprehensive resource benchmark comparing:
    - Baseline: Always run 5 rounds
    - ESDP: Stop at model recommendation
    - Always R1: Always stop at round 1
    - Random: Random stopping (1, 3, or 5)
    """
    logger.info("="*60)
    logger.info("ESDP RESOURCE SAVING BENCHMARK (ENHANCED)")
    logger.info("="*60)
    logger.info(f"Cost model: {CPU_HOURS_PER_ROUND}h per round")
    logger.info(f"Baseline cost: {5 * CPU_HOURS_PER_ROUND}h (5 rounds)")
    logger.info(f"Bootstrap samples: {N_BOOTSTRAP}")
    logger.info(f"Confidence level: {CONFIDENCE_LEVEL*100}%")

    # ============================================================
    # 1. Load data and model
    # ============================================================
    logger.info("\nLoading data and model...")
    df = pd.read_csv("data/training_dataset_with_target.csv")
    pipeline = joblib.load("models/best_model_pipeline.pkl")
    feature_names = pipeline.feature_names

    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Unique samples: {df['Sample'].nunique()}")

    # ============================================================
    # 2. Perform stratified split (same as training)
    # ============================================================
    from sklearn.model_selection import GroupShuffleSplit

    # Get unique groups and their labels
    sample_labels = df.groupby('Sample')['optimal_rounds_3class'].first()
    samples = sample_labels.index.values
    labels = sample_labels.values

    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(samples, labels, groups=samples))

    train_samples = samples[train_idx]
    test_samples = samples[test_idx]

    logger.info(f"\nUsing TEST SET for benchmark: {len(test_samples)} samples")

    # ============================================================
    # 3. Run benchmark on test samples
    # ============================================================
    results = []

    for sample in test_samples:
        sample_data = df[df['Sample'] == sample].sort_values('round')

        # Skip if sample doesn't have all 5 rounds
        if len(sample_data) < 5:
            logger.warning(f"Skipping {sample}: only {len(sample_data)} rounds")
            continue

        # Get R1 data for prediction
        r1_data = sample_data[sample_data['round'] == 1].iloc[0]

        # Prepare features
        X = pd.DataFrame([r1_data[feature_names]])
        X = X.replace([np.inf, -np.inf], np.nan)

        # Get ESDP prediction
        y_pred = pipeline.predict(X)[0]
        y_proba = pipeline.predict_proba(X)[0]

        # Map class to recommended rounds
        class_to_rounds = {0: 1, 1: 3, 2: 5}
        rec_rounds = class_to_rounds[y_pred]
        confidence = float(y_proba[y_pred])

        # Get metrics at different stopping points
        metrics_r1 = sample_data[sample_data['round'] == 1].iloc[0]
        metrics_r3 = sample_data[sample_data['round'] == 3].iloc[0]
        metrics_r5 = sample_data[sample_data['round'] == 5].iloc[0]
        metrics_rec = sample_data[sample_data['round'] == rec_rounds].iloc[0]

        # Random baseline (uniform choice)
        random_rounds = np.random.choice([1, 3, 5])
        metrics_random = sample_data[sample_data['round'] == random_rounds].iloc[0]

        # Calculate resource costs for each strategy
        cpu_r5 = 5 * CPU_HOURS_PER_ROUND
        cpu_r1 = 1 * CPU_HOURS_PER_ROUND
        cpu_esdp = rec_rounds * CPU_HOURS_PER_ROUND
        cpu_random = random_rounds * CPU_HOURS_PER_ROUND

        mem_r5 = 5 * MEMORY_GB_PER_ROUND
        mem_esdp = rec_rounds * MEMORY_GB_PER_ROUND

        # Quality metrics
        qv_r5 = metrics_r5['qv']
        qv_r1 = metrics_r1['qv']
        qv_esdp = metrics_rec['qv']
        qv_random = metrics_random['qv']

        busco_r5 = metrics_r5['busco_complete']
        busco_r1 = metrics_r1['busco_complete']
        busco_esdp = metrics_rec['busco_complete']
        busco_random = metrics_random['busco_complete']

        # Calculate savings vs baseline (R5)
        cpu_saved_esdp = cpu_r5 - cpu_esdp
        cpu_saved_r1 = cpu_r5 - cpu_r1
        cpu_saved_random = cpu_r5 - cpu_random

        # Quality loss vs baseline (R5)
        qv_loss_esdp = qv_r5 - qv_esdp
        qv_loss_r1 = qv_r5 - qv_r1
        qv_loss_random = qv_r5 - qv_random

        busco_loss_esdp = busco_r5 - busco_esdp
        busco_loss_r1 = busco_r5 - busco_r1
        busco_loss_random = busco_r5 - busco_random

        # Efficiency (quality per CPU-hour)
        eff_r5 = qv_r5 / cpu_r5
        eff_r1 = qv_r1 / cpu_r1
        eff_esdp = qv_esdp / cpu_esdp
        eff_random = qv_random / cpu_random

        # Determine if quality loss is acceptable
        acceptable_esdp = (qv_loss_esdp <= ACCEPTABLE_QV_LOSS) and (busco_loss_esdp <= ACCEPTABLE_BUSCO_LOSS)
        acceptable_r1 = (qv_loss_r1 <= ACCEPTABLE_QV_LOSS) and (busco_loss_r1 <= ACCEPTABLE_BUSCO_LOSS)

        results.append({
            'Sample': sample,
            'Genus': r1_data['Genus'],
            'Predicted_Class': y_pred,
            'Recommended_Rounds': rec_rounds,
            'Random_Rounds': random_rounds,
            'Confidence': confidence,

            # Resource metrics (ESDP)
            'CPU_R5_Hours': cpu_r5,
            'CPU_ESDP_Hours': cpu_esdp,
            'CPU_R1_Hours': cpu_r1,
            'CPU_Random_Hours': cpu_random,
            'CPU_Saved_ESDP': cpu_saved_esdp,
            'CPU_Saved_R1': cpu_saved_r1,
            'CPU_Saved_Random': cpu_saved_random,
            'CPU_Reduction_ESDP_Pct': (cpu_saved_esdp / cpu_r5) * 100,
            'CPU_Reduction_R1_Pct': (cpu_saved_r1 / cpu_r5) * 100,

            # Memory metrics
            'Memory_R5_GB': mem_r5,
            'Memory_ESDP_GB': mem_esdp,
            'Memory_Saved_GB': mem_r5 - mem_esdp,

            # Quality metrics
            'QV_R5': qv_r5,
            'QV_ESDP': qv_esdp,
            'QV_R1': qv_r1,
            'QV_Random': qv_random,
            'QV_Loss_ESDP': qv_loss_esdp,
            'QV_Loss_R1': qv_loss_r1,
            'QV_Loss_Random': qv_loss_random,

            'BUSCO_R5': busco_r5,
            'BUSCO_ESDP': busco_esdp,
            'BUSCO_R1': busco_r1,
            'BUSCO_Random': busco_random,
            'BUSCO_Loss_ESDP': busco_loss_esdp,
            'BUSCO_Loss_R1': busco_loss_r1,
            'BUSCO_Loss_Random': busco_loss_random,

            # Efficiency metrics
            'Efficiency_R5': eff_r5,
            'Efficiency_ESDP': eff_esdp,
            'Efficiency_R1': eff_r1,
            'Efficiency_Random': eff_random,
            'Efficiency_Gain_ESDP_Pct': ((eff_esdp - eff_r5) / eff_r5) * 100,
            'Efficiency_Gain_R1_Pct': ((eff_r1 - eff_r5) / eff_r5) * 100,

            # Decision quality
            'Acceptable_Quality_ESDP': acceptable_esdp,
            'Acceptable_Quality_R1': acceptable_r1,
            'Optimal_Round_True': r1_data['optimal_rounds_3class']
        })

    res_df = pd.DataFrame(results)

    # ============================================================
    # 4. Statistical Summary with Confidence Intervals
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK RESULTS (with 95% CI)")
    logger.info("="*60)

    total_samples = len(res_df)

    # CPU savings
    cpu_saved_esdp_mean = res_df['CPU_Saved_ESDP'].mean()
    cpu_saved_esdp_ci = bootstrap_ci(res_df['CPU_Saved_ESDP'].values)

    cpu_reduction_esdp_mean = res_df['CPU_Reduction_ESDP_Pct'].mean()
    cpu_reduction_esdp_ci = bootstrap_ci(res_df['CPU_Reduction_ESDP_Pct'].values)

    logger.info(f"\n💰 RESOURCE SAVINGS (ESDP):")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Total CPU saved: {res_df['CPU_Saved_ESDP'].sum():.1f}h")
    logger.info(f"  Avg CPU saved per sample: {format_ci(cpu_saved_esdp_mean, *cpu_saved_esdp_ci)}h")
    logger.info(f"  Avg CPU reduction: {format_ci(cpu_reduction_esdp_mean, *cpu_reduction_esdp_ci)}%")

    # Quality impact
    qv_loss_mean = res_df['QV_Loss_ESDP'].mean()
    qv_loss_ci = bootstrap_ci(res_df['QV_Loss_ESDP'].values)

    busco_loss_mean = res_df['BUSCO_Loss_ESDP'].mean()
    busco_loss_ci = bootstrap_ci(res_df['BUSCO_Loss_ESDP'].values)

    logger.info(f"\n🎯 QUALITY IMPACT (ESDP):")
    logger.info(f"  Avg QV loss: {format_ci(qv_loss_mean, *qv_loss_ci, decimals=4)} points")
    logger.info(f"  Avg BUSCO loss: {format_ci(busco_loss_mean, *busco_loss_ci)}%")
    logger.info(f"  Samples with ZERO QV loss: {(res_df['QV_Loss_ESDP'] <= 0.01).sum()} / {total_samples}")
    logger.info(f"  Samples with acceptable loss: {res_df['Acceptable_Quality_ESDP'].sum()} / {total_samples}")

    # Efficiency
    eff_gain_mean = res_df['Efficiency_Gain_ESDP_Pct'].mean()
    eff_gain_ci = bootstrap_ci(res_df['Efficiency_Gain_ESDP_Pct'].values)

    logger.info(f"\n⚡ EFFICIENCY GAINS (ESDP):")
    logger.info(f"  Avg efficiency gain: {format_ci(eff_gain_mean, *eff_gain_ci)}%")
    logger.info(f"  Best efficiency gain: {res_df['Efficiency_Gain_ESDP_Pct'].max():.1f}%")

    # ============================================================
    # 5. Statistical Significance Tests
    # ============================================================
    logger.info(f"\n📊 STATISTICAL SIGNIFICANCE:")

    # Wilcoxon signed-rank test: ESDP vs Always R1
    stat_qv, p_qv = stats.wilcoxon(res_df['QV_ESDP'], res_df['QV_R1'])
    stat_cpu, p_cpu = stats.wilcoxon(res_df['CPU_ESDP_Hours'], res_df['CPU_R1_Hours'])

    logger.info(f"  ESDP vs Always R1:")
    logger.info(f"    QV difference: p={p_qv:.4f} {'***' if p_qv < 0.001 else '**' if p_qv < 0.01 else '*' if p_qv < 0.05 else 'ns'}")
    logger.info(f"    CPU difference: p={p_cpu:.4f} {'***' if p_cpu < 0.001 else '**' if p_cpu < 0.01 else '*' if p_cpu < 0.05 else 'ns'}")

    # ESDP vs Random
    stat_qv_rand, p_qv_rand = stats.wilcoxon(res_df['QV_ESDP'], res_df['QV_Random'])
    stat_cpu_rand, p_cpu_rand = stats.wilcoxon(res_df['CPU_ESDP_Hours'], res_df['CPU_Random_Hours'])

    logger.info(f"  ESDP vs Random:")
    logger.info(f"    QV difference: p={p_qv_rand:.4f} {'***' if p_qv_rand < 0.001 else '**' if p_qv_rand < 0.01 else '*' if p_qv_rand < 0.05 else 'ns'}")
    logger.info(f"    CPU difference: p={p_cpu_rand:.4f} {'***' if p_cpu_rand < 0.001 else '**' if p_cpu_rand < 0.01 else '*' if p_cpu_rand < 0.05 else 'ns'}")

    # ============================================================
    # 6. Baseline Comparison Table
    # ============================================================
    logger.info(f"\n📋 BASELINE COMPARISON:")

    comparison = pd.DataFrame({
        'Strategy': ['Always R5 (Baseline)', 'ESDP', 'Always R1', 'Random'],
        'Avg_CPU_Hours': [
            res_df['CPU_R5_Hours'].mean(),
            res_df['CPU_ESDP_Hours'].mean(),
            res_df['CPU_R1_Hours'].mean(),
            res_df['CPU_Random_Hours'].mean()
        ],
        'Avg_QV': [
            res_df['QV_R5'].mean(),
            res_df['QV_ESDP'].mean(),
            res_df['QV_R1'].mean(),
            res_df['QV_Random'].mean()
        ],
        'Avg_BUSCO': [
            res_df['BUSCO_R5'].mean(),
            res_df['BUSCO_ESDP'].mean(),
            res_df['BUSCO_R1'].mean(),
            res_df['BUSCO_Random'].mean()
        ],
        'Avg_Efficiency': [
            res_df['Efficiency_R5'].mean(),
            res_df['Efficiency_ESDP'].mean(),
            res_df['Efficiency_R1'].mean(),
            res_df['Efficiency_Random'].mean()
        ]
    })

    logger.info(f"\n{comparison.to_string(index=False)}")

    # ============================================================
    # 7. Decision Breakdown
    # ============================================================
    logger.info(f"\n🔍 DECISION BREAKDOWN:")
    decision_counts = res_df['Recommended_Rounds'].value_counts().sort_index()
    for rounds, count in decision_counts.items():
        pct = (count / total_samples) * 100
        logger.info(f"  {rounds} rounds: {count} samples ({pct:.1f}%)")

    # ============================================================
    # 8. Visualizations
    # ============================================================
    logger.info("\nGenerating visualizations...")

    # Create output directory
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)

    # Plot 1: Resource Saving vs Quality Loss (Pareto frontier)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        res_df['CPU_Reduction_ESDP_Pct'],
        res_df['QV_Loss_ESDP'],
        c=res_df['Recommended_Rounds'],
        cmap='RdYlGn_r',
        s=100,
        alpha=0.6,
        edgecolors='black'
    )
    ax.axhline(y=ACCEPTABLE_QV_LOSS, color='red', linestyle='--', 
               label=f'Acceptable QV loss ({ACCEPTABLE_QV_LOSS})')
    ax.axhline(y=0, color='green', linestyle='-', alpha=0.3, label='No quality loss')
    ax.set_xlabel('CPU Resource Reduction (%)', fontsize=12)
    ax.set_ylabel('Quality Loss (QV points)', fontsize=12)
    ax.set_title('ESDP Performance: Resource Saving vs Quality Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Recommended Rounds', rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig("outputs/plots/resource_vs_quality.png", dpi=300)
    logger.info("  Saved: outputs/plots/resource_vs_quality.png")

    # Plot 2: Baseline Comparison (Bar chart)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    strategies = ['R5\n(Baseline)', 'ESDP', 'R1\n(Always)', 'Random']
    cpu_means = [
        res_df['CPU_R5_Hours'].mean(),
        res_df['CPU_ESDP_Hours'].mean(),
        res_df['CPU_R1_Hours'].mean(),
        res_df['CPU_Random_Hours'].mean()
    ]
    qv_means = [
        res_df['QV_R5'].mean(),
        res_df['QV_ESDP'].mean(),
        res_df['QV_R1'].mean(),
        res_df['QV_Random'].mean()
    ]

    # CPU comparison
    bars1 = ax1.bar(strategies, cpu_means, color=['gray', 'green', 'red', 'orange'], alpha=0.7)
    ax1.set_ylabel('Average CPU Hours', fontsize=12)
    ax1.set_title('Computational Cost Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}h',
                ha='center', va='bottom', fontsize=10)

    # QV comparison
    bars2 = ax2.bar(strategies, qv_means, color=['gray', 'green', 'red', 'orange'], alpha=0.7)
    ax2.set_ylabel('Average QV', fontsize=12)
    ax2.set_title('Assembly Quality Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("outputs/plots/baseline_comparison.png", dpi=300)
    logger.info("  Saved: outputs/plots/baseline_comparison.png")

    # Plot 3: Efficiency comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(res_df))
    width = 0.25
    ax.bar(x - width, res_df['Efficiency_R5'], width, label='R5 (Baseline)', alpha=0.7)
    ax.bar(x, res_df['Efficiency_ESDP'], width, label='ESDP', alpha=0.7)
    ax.bar(x + width, res_df['Efficiency_R1'], width, label='R1 (Always)', alpha=0.7)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Efficiency (QV per CPU-hour)', fontsize=12)
    ax.set_title('Computational Efficiency: Baseline vs ESDP vs Always R1', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("outputs/plots/efficiency_comparison.png", dpi=300)
    logger.info("  Saved: outputs/plots/efficiency_comparison.png")

    # Plot 4: Decision distribution by genus
    fig, ax = plt.subplots(figsize=(12, 6))
    decision_by_genus = res_df.groupby(['Genus', 'Recommended_Rounds']).size().unstack(fill_value=0)
    decision_by_genus.plot(kind='bar', stacked=True, ax=ax, colormap='RdYlGn_r')
    ax.set_xlabel('Genus', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('ESDP Decisions by Genus', fontsize=14, fontweight='bold')
    ax.legend(title='Recommended Rounds')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("outputs/plots/decisions_by_genus.png", dpi=300)
    logger.info("  Saved: outputs/plots/decisions_by_genus.png")

    # Plot 5: Cost-Benefit Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Cumulative savings
    res_df_sorted = res_df.sort_values('CPU_Saved_ESDP', ascending=False)
    cumulative_savings = res_df_sorted['CPU_Saved_ESDP'].cumsum()
    ax1.plot(range(len(cumulative_savings)), cumulative_savings, linewidth=2)
    ax1.fill_between(range(len(cumulative_savings)), cumulative_savings, alpha=0.3)
    ax1.set_xlabel('Number of Samples (sorted by savings)', fontsize=12)
    ax1.set_ylabel('Cumulative CPU-Hours Saved', fontsize=12)
    ax1.set_title('Cumulative Resource Savings', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Quality distribution
    ax2.hist(res_df['QV_Loss_ESDP'], bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(x=ACCEPTABLE_QV_LOSS, color='red', linestyle='--', 
                label=f'Acceptable threshold ({ACCEPTABLE_QV_LOSS})')
    ax2.axvline(x=0, color='green', linestyle='-', alpha=0.5, label='No loss')
    ax2.set_xlabel('QV Loss (points)', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Distribution of Quality Loss', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("outputs/plots/cost_benefit_analysis.png", dpi=300)
    logger.info("  Saved: outputs/plots/cost_benefit_analysis.png")

    # ============================================================
    # 9. Save results
    # ============================================================
    output_path = "outputs/resource_benchmark_results.csv"
    res_df.to_csv(output_path, index=False)
    logger.info(f"\n💾 Results saved to: {output_path}")

    # Save baseline comparison
    comparison.to_csv("outputs/baseline_comparison_table.csv", index=False)
    logger.info(f"Baseline comparison saved to: outputs/baseline_comparison_table.csv")

    # Save summary statistics with CI
    summary = {
        'total_samples': total_samples,
        'total_cpu_saved_hours': res_df['CPU_Saved_ESDP'].sum(),
        'avg_cpu_saved_hours': cpu_saved_esdp_mean,
        'avg_cpu_saved_ci_lower': cpu_saved_esdp_ci[0],
        'avg_cpu_saved_ci_upper': cpu_saved_esdp_ci[1],
        'avg_cpu_reduction_pct': cpu_reduction_esdp_mean,
        'avg_cpu_reduction_ci_lower': cpu_reduction_esdp_ci[0],
        'avg_cpu_reduction_ci_upper': cpu_reduction_esdp_ci[1],
        'avg_qv_loss': qv_loss_mean,
        'avg_qv_loss_ci_lower': qv_loss_ci[0],
        'avg_qv_loss_ci_upper': qv_loss_ci[1],
        'avg_busco_loss': busco_loss_mean,
        'avg_busco_loss_ci_lower': busco_loss_ci[0],
        'avg_busco_loss_ci_upper': busco_loss_ci[1],
        'samples_zero_qv_loss': (res_df['QV_Loss_ESDP'] <= 0.01).sum(),
        'samples_acceptable_loss': res_df['Acceptable_Quality_ESDP'].sum(),
        'avg_efficiency_gain_pct': eff_gain_mean,
        'avg_efficiency_gain_ci_lower': eff_gain_ci[0],
        'avg_efficiency_gain_ci_upper': eff_gain_ci[1],
        'wilcoxon_qv_vs_r1_pvalue': p_qv,
        'wilcoxon_cpu_vs_r1_pvalue': p_cpu,
        'wilcoxon_qv_vs_random_pvalue': p_qv_rand,
        'wilcoxon_cpu_vs_random_pvalue': p_cpu_rand
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("outputs/resource_benchmark_summary.csv", index=False)
    logger.info(f"Summary saved to: outputs/resource_benchmark_summary.csv")

    # ============================================================
    # 10. Publication-Ready Table
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("PUBLICATION-READY SUMMARY TABLE")
    logger.info("="*60)

    pub_table = pd.DataFrame({
        'Metric': [
            'CPU Hours Saved (mean ± 95% CI)',
            'CPU Reduction % (mean ± 95% CI)',
            'QV Loss (mean ± 95% CI)',
            'BUSCO Loss % (mean ± 95% CI)',
            'Efficiency Gain % (mean ± 95% CI)',
            'Samples with Zero QV Loss',
            'Samples with Acceptable Loss',
            'Statistical Significance (vs R1)'
        ],
        'Value': [
            format_ci(cpu_saved_esdp_mean, *cpu_saved_esdp_ci) + 'h',
            format_ci(cpu_reduction_esdp_mean, *cpu_reduction_esdp_ci) + '%',
            format_ci(qv_loss_mean, *qv_loss_ci, decimals=4),
            format_ci(busco_loss_mean, *busco_loss_ci) + '%',
            format_ci(eff_gain_mean, *eff_gain_ci) + '%',
            f"{(res_df['QV_Loss_ESDP'] <= 0.01).sum()} / {total_samples} ({(res_df['QV_Loss_ESDP'] <= 0.01).sum()/total_samples*100:.1f}%)",
            f"{res_df['Acceptable_Quality_ESDP'].sum()} / {total_samples} ({res_df['Acceptable_Quality_ESDP'].sum()/total_samples*100:.1f}%)",
            f"p={p_qv:.4f} {'***' if p_qv < 0.001 else '**' if p_qv < 0.01 else '*' if p_qv < 0.05 else 'ns'}"
        ]
    })

    logger.info(f"\n{pub_table.to_string(index=False)}")

    pub_table.to_csv("outputs/publication_summary_table.csv", index=False)
    logger.info(f"\nPublication table saved to: outputs/publication_summary_table.csv")

    logger.info("\n" + "="*60)
    logger.info("BENCHMARK COMPLETE!")
    logger.info("="*60)

    return res_df, summary, comparison, pub_table


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    results_df, summary, comparison, pub_table = run_benchmark()