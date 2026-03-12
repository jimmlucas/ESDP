#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_feature_engineering.py - Advanced Feature Engineering

"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from sklearn.preprocessing import PolynomialFeatures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def load_data():
    """Load the merged dataset."""
    df = pd.read_csv(config['data']['merged_csv'])
    logger.info(f"Loaded {len(df)} rows")
    return df

def add_basic_deltas(df):
    """Add delta features (changes from previous round)."""
    logger.info("Adding basic delta features...")
    
    cov_col = 'Coverage_effective' if 'Coverage_effective' in df.columns else 'Coverage'
    df = df.sort_values(['Sample', cov_col, 'round']).reset_index(drop=True)
    
    # Delta features
    for metric in ['qv', 'busco_complete', 'n50', 'num_contigs', 'error_rate']:
        if metric in df.columns:
            df[f'delta_{metric}'] = df.groupby(['Sample', cov_col])[metric].diff()
            
    # For error_rate, decreasing is good, so flip sign
    if 'delta_error_rate' in df.columns:
        df['delta_error_improvement'] = -df['delta_error_rate']
    
    # Assembly error
    if 'assembly_frac' in df.columns:
        df['assembly_error'] = (df['assembly_frac'] - 1.0).abs()
        df['delta_assembly_error'] = df.groupby(['Sample', cov_col])['assembly_error'].diff()
    
    return df

def add_ratio_features(df):
    """Add ratio and efficiency features."""
    logger.info("Adding ratio features...")
    
    # QV improvement rate (per round)
    if 'delta_qv' in df.columns:
        df['qv_improvement_rate'] = df['delta_qv'] / df['round']
    
    # BUSCO efficiency (BUSCO per contig)
    if 'busco_complete' in df.columns and 'num_contigs' in df.columns:
        df['busco_per_contig'] = df['busco_complete'] / (df['num_contigs'] + 1)  # +1 to avoid div by 0
    
    # Assembly efficiency
    if 'n50' in df.columns and 'total_length' in df.columns:
        df['n50_fraction'] = df['n50'] / (df['total_length'] + 1)
    
    # Cost-benefit ratio (improvement per computational cost)
    # Assume cost proportional to round number
    if 'delta_qv' in df.columns and 'delta_busco_complete' in df.columns:
        improvement = (
            df['delta_qv'].fillna(0) * 0.5 + 
            df['delta_busco_complete'].fillna(0) * 0.5
        )
        df['cost_benefit_ratio'] = improvement / df['round']
    
    return df

def add_cumulative_features(df):
    """Add cumulative features."""
    logger.info("Adding cumulative features...")
    
    cov_col = 'Coverage_effective' if 'Coverage_effective' in df.columns else 'Coverage'
    
    # Cumulative improvements
    for metric in ['delta_qv', 'delta_busco_complete']:
        if metric in df.columns:
            df[f'{metric}_cumsum'] = df.groupby(['Sample', cov_col])[metric].cumsum()
    
    # Score improvement (weighted combination)
    if all(c in df.columns for c in ['delta_qv', 'delta_busco_complete']):
        df['score_improvement'] = (
            df['delta_qv'].fillna(0) * 0.4 +
            df['delta_busco_complete'].fillna(0) * 0.3 +
            df.get('delta_error_improvement', 0).fillna(0) * 0.2 +
            df.get('delta_assembly_error', 0).fillna(0) * 0.1
        )
        df['gain_cumulative'] = df.groupby(['Sample', cov_col])['score_improvement'].cumsum()
    
    return df

def add_normalized_to_r1(df):
    """Normalize features to R1 values."""
    logger.info("Adding R1-normalized features...")
    
    cov_col = 'Coverage_effective' if 'Coverage_effective' in df.columns else 'Coverage'
    
    for metric in ['qv', 'n50', 'error_rate', 'busco_complete', 'assembly_frac']:
        if metric in df.columns:
            # Get R1 value for each group
            r1_values = df[df['round'] == 1].set_index(['Sample', cov_col])[metric]
            
            # Merge back and calculate difference
            df[f'{metric}_from_r1'] = df.apply(
                lambda row: row[metric] - r1_values.get((row['Sample'], row[cov_col]), np.nan)
                if (row['Sample'], row[cov_col]) in r1_values.index else np.nan,
                axis=1
            )
    
    return df

def add_trend_features(df):
    """Add trend/momentum features (second derivatives)."""
    logger.info("Adding trend features...")
    
    cov_col = 'Coverage_effective' if 'Coverage_effective' in df.columns else 'Coverage'
    
    # Second derivatives (acceleration/deceleration)
    for metric in ['delta_qv', 'delta_busco_complete', 'score_improvement']:
        if metric in df.columns:
            df[f'{metric}_trend'] = df.groupby(['Sample', cov_col])[metric].diff()
    
    return df

def add_plateau_features(df):
    """Add plateau detection features."""
    logger.info("Adding plateau features...")
    
    cov_col = 'Coverage_effective' if 'Coverage_effective' in df.columns else 'Coverage'
    
    if 'score_improvement' not in df.columns:
        return df
    
    def detect_plateau(group):
        group = group.sort_values('round').copy()
        max_gain = group['score_improvement'].max()
        threshold = config['plateau']['relative_threshold'] * max_gain if max_gain > 0 else 0
        
        group['is_plateau'] = (group['score_improvement'].abs() < threshold).astype(int)
        
        # Plateau streak
        streak, run = [], 0
        for val in group['is_plateau']:
            run = run + 1 if val == 1 else 0
            streak.append(run)
        group['plateau_streak'] = streak
        
        return group
    
    df = df.groupby(['Sample', cov_col], group_keys=False).apply(detect_plateau)
    
    return df

def add_polynomial_features(df, degree=2):
    """Add polynomial features for key metrics (optional, can be computationally expensive)."""
    logger.info(f"Adding polynomial features (degree={degree})...")
    
    # Select key features for polynomial expansion
    poly_features = ['qv', 'busco_complete', 'error_rate']
    available_features = [f for f in poly_features if f in df.columns]
    
    if len(available_features) < 2:
        logger.warning("Not enough features for polynomial expansion")
        return df
    
    # Only apply to numeric columns
    X_poly = df[available_features].copy()
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
    X_poly_transformed = poly.fit_transform(X_poly.fillna(0))
    
    # Get feature names
    poly_feature_names = poly.get_feature_names_out(available_features)
    
    # Add only interaction terms (not individual features which we already have)
    interaction_mask = ['*' in name for name in poly_feature_names]
    interaction_names = [name for name, is_interaction in zip(poly_feature_names, interaction_mask) if is_interaction]
    interaction_features = X_poly_transformed[:, interaction_mask]
    
    # Add to dataframe
    for idx, name in enumerate(interaction_names):
        df[f'interaction_{name}'] = interaction_features[:, idx]
    
    logger.info(f"Added {len(interaction_names)} interaction features")
    
    return df

def add_domain_specific_features(df):
    """Add domain-specific features based on biological knowledge."""
    logger.info("Adding domain-specific features...")
    
    # Completeness indicator (high BUSCO + low error)
    if 'busco_complete' in df.columns and 'error_rate' in df.columns:
        df['completeness_score'] = (
            df['busco_complete'] / 100.0 * (1 - df['error_rate'])
        )
    
    # Assembly quality score
    if all(c in df.columns for c in ['busco_complete', 'num_contigs', 'assembly_frac']):
        df['assembly_quality'] = (
            df['busco_complete'] / 100.0 * 
            (1 / (df['num_contigs'] + 1)) * 
            (1 - (df['assembly_frac'] - 1.0).abs())
        )
    
    # Polishing effectiveness (how much improvement per round)
    if 'score_improvement' in df.columns:
        df['polishing_effectiveness'] = df['score_improvement'] / (df['round'] + 1)
    
    return df

def validate_features(df):
    """Validate engineered features."""
    logger.info("Validating features...")
    
    # Check for infinite values
    inf_cols = []
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
    
    if inf_cols:
        logger.warning(f"Found infinite values in: {inf_cols}")
        for col in inf_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Report missing values
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    
    if len(missing_summary) > 0:
        logger.info(f"Missing values summary (top 10):\n{missing_summary.head(10)}")
    
    return df

def main():
    """Run feature engineering pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Feature Engineering")
    logger.info("=" * 60)
    
    df = load_data()
    original_cols = len(df.columns)
    
    # Apply feature engineering steps
    df = add_basic_deltas(df)
    df = add_ratio_features(df)
    df = add_cumulative_features(df)
    df = add_normalized_to_r1(df)
    df = add_trend_features(df)
    df = add_plateau_features(df)
    df = add_domain_specific_features(df)
    
    # Optional: Add polynomial features (can be slow)
    # df = add_polynomial_features(df, degree=2)
    
    df = validate_features(df)
    
    # Save engineered dataset
    output_path = config['data']['engineered_csv']
    df.to_csv(output_path, index=False)
    
    new_cols = len(df.columns)
    logger.info("=" * 60)
    logger.info("Feature Engineering Complete!")
    logger.info(f"Original features: {original_cols}")
    logger.info(f"New features: {new_cols}")
    logger.info(f"Added: {new_cols - original_cols} features")
    logger.info(f"Output saved to: {output_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
