#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_exploratory_analysis.py - Comprehensive Exploratory Data Analysis

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import yaml

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Ensure output directories
Path(config['outputs']['plots_dir']).mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the merged dataset."""
    df = pd.read_csv(config['data']['merged_csv'])
    logger.info(f"Loaded {len(df)} rows from {config['data']['merged_csv']}")
    return df

def analyze_distributions(df):
    """Analyze and plot distributions of key metrics."""
    logger.info("Analyzing metric distributions...")
    
    metrics = ['qv', 'error_rate', 'busco_complete', 'n50', 'num_contigs']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric in df.columns:
            df[metric].hist(bins=50, ax=axes[idx], edgecolor='black')
            axes[idx].set_title(f'Distribution of {metric}')
            axes[idx].set_xlabel(metric)
            axes[idx].set_ylabel('Frequency')
            
            # Add statistics
            mean_val = df[metric].mean()
            median_val = df[metric].median()
            axes[idx].axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[idx].axvline(median_val, color='g', linestyle='--', label=f'Median: {median_val:.2f}')
            axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(Path(config['outputs']['plots_dir']) / 'metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved metric_distributions.png")

def analyze_by_round(df):
    """Analyze metrics progression across rounds."""
    logger.info("Analyzing metrics by round...")
    
    metrics = ['qv', 'busco_complete', 'error_rate']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(metrics):
        if metric in df.columns:
            df.groupby('round')[metric].mean().plot(ax=axes[idx], marker='o', linewidth=2)
            axes[idx].set_title(f'{metric} by Round')
            axes[idx].set_xlabel('Round')
            axes[idx].set_ylabel(f'Mean {metric}')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(config['outputs']['plots_dir']) / 'metrics_by_round.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved metrics_by_round.png")

def analyze_coverage_groups(df):
    """Analyze differences across coverage groups."""
    logger.info("Analyzing coverage groups...")
    
    cov_col = 'Coverage_effective' if 'Coverage_effective' in df.columns else 'Coverage'
    
    # Count samples per coverage
    cov_counts = df.groupby([cov_col, 'Sample']).size().reset_index().groupby(cov_col).size()
    
    plt.figure(figsize=(10, 6))
    cov_counts.plot(kind='bar')
    plt.title('Number of Samples per Coverage Group')
    plt.xlabel('Coverage')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(config['outputs']['plots_dir']) / 'coverage_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved coverage_distribution.png")

def analyze_genus_distribution(df):
    """Analyze genus distribution."""
    if 'Genus' not in df.columns:
        logger.warning("Genus column not found")
        return
    
    logger.info("Analyzing genus distribution...")
    
    genus_counts = df.groupby(['Genus', 'Sample']).size().reset_index().groupby('Genus').size()
    
    plt.figure(figsize=(12, 6))
    genus_counts.plot(kind='bar', color='steelblue')
    plt.title('Number of Samples per Genus')
    plt.xlabel('Genus')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(Path(config['outputs']['plots_dir']) / 'genus_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved genus_distribution.png")

def analyze_quality_metrics(df):
    """Analyze quality metric correlations."""
    logger.info("Analyzing quality metric correlations...")
    
    metrics = ['qv', 'error_rate', 'busco_complete', 'n50', 'num_contigs', 'assembly_frac']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) < 2:
        logger.warning("Not enough metrics for correlation analysis")
        return
    
    corr_matrix = df[available_metrics].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Quality Metrics')
    plt.tight_layout()
    plt.savefig(Path(config['outputs']['plots_dir']) / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved correlation_matrix.png")

def analyze_improvement_trends(df):
    """Analyze improvement trends across rounds."""
    logger.info("Analyzing improvement trends...")
    
    # Group by Sample and Coverage
    cov_col = 'Coverage_effective' if 'Coverage_effective' in df.columns else 'Coverage'
    
    # Calculate improvement from R1 to R5
    improvements = []
    for (sample, cov), group in df.groupby(['Sample', cov_col]):
        group = group.sort_values('round')
        if len(group) >= 2:
            r1 = group.iloc[0]
            r_last = group.iloc[-1]
            
            improvements.append({
                'sample': sample,
                'coverage': cov,
                'rounds': len(group),
                'qv_improvement': r_last['qv'] - r1['qv'] if 'qv' in group.columns else None,
                'busco_improvement': r_last['busco_complete'] - r1['busco_complete'] if 'busco_complete' in group.columns else None,
                'error_improvement': r1['error_rate'] - r_last['error_rate'] if 'error_rate' in group.columns else None,
            })
    
    imp_df = pd.DataFrame(improvements)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, metric in enumerate(['qv_improvement', 'busco_improvement', 'error_improvement']):
        if metric in imp_df.columns:
            imp_df[metric].hist(bins=30, ax=axes[idx], edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {metric}')
            axes[idx].set_xlabel(metric)
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(0, color='r', linestyle='--', label='No change')
            axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(Path(config['outputs']['plots_dir']) / 'improvement_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved improvement_distributions.png")
    
    return imp_df

def generate_summary_statistics(df):
    """Generate and save summary statistics."""
    logger.info("Generating summary statistics...")
    
    summary = {
        'Total Rows': len(df),
        'Unique Samples': df['Sample'].nunique(),
        'Unique Genera': df['Genus'].nunique() if 'Genus' in df.columns else 'N/A',
        'Rounds': sorted(df['round'].unique().tolist()) if 'round' in df.columns else 'N/A',
        'Coverage Groups': df['Coverage'].unique().tolist() if 'Coverage' in df.columns else 'N/A',
    }
    
    # Quality metrics summary
    metrics = ['qv', 'error_rate', 'busco_complete', 'n50']
    for metric in metrics:
        if metric in df.columns:
            summary[f'{metric}_mean'] = df[metric].mean()
            summary[f'{metric}_std'] = df[metric].std()
            summary[f'{metric}_min'] = df[metric].min()
            summary[f'{metric}_max'] = df[metric].max()
    
    # Save to file
    summary_path = Path(config['outputs']['results_dir']) / 'eda_summary.txt'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Saved summary to {summary_path}")
    
    return summary

def main():
    """Run all EDA analyses."""
    logger.info("=" * 60)
    logger.info("Starting Exploratory Data Analysis")
    logger.info("=" * 60)
    
    df = load_data()
    
    analyze_distributions(df)
    analyze_by_round(df)
    analyze_coverage_groups(df)
    analyze_genus_distribution(df)
    analyze_quality_metrics(df)
    imp_df = analyze_improvement_trends(df)
    summary = generate_summary_statistics(df)
    
    logger.info("=" * 60)
    logger.info("EDA Complete!")
    logger.info(f"Total plots generated: 7")
    logger.info(f"Plots saved to: {config['outputs']['plots_dir']}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
