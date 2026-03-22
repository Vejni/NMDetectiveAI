"""
Interreplicate correlation of ASE NMD efficiency.

This script measures the consistency of ASE NMD efficiency measurements across
biological replicates (different samples with the same PTC mutation). 

Uses split-half reliability: randomly splits replicates into two groups,
calculates mean NMD efficiency per group, and correlates across PTCs.
Repeats the splitting N_SPLITS times for robust estimates.

Requires raw data:
    - data/raw/PTC/somatic_TCGA.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from NMD.config import RAW_DATA_DIR
from NMD.manuscript.output import get_paths
from NMD.data.preprocessing import (
    apply_strict_val_expression_filter,
    create_rule_labels,
    impute_rna_halflife,
    apply_lenient_expression_filter,
    apply_splice_site_filter,
    apply_vaf_filter,
    apply_frameshift_correction,
    apply_threshold_filter,
    apply_transcript_length_filter,
    center_nmd_efficiency,
    apply_regression_correction,
)
from NMD.data.DatasetConfig import DatasetConfig


# Configuration
SCRIPT_NAME = "data_interreplicate_correlation"
MIN_REPLICATES = 3
N_SPLITS = 100  # Number of random splits for split-half reliability
RANDOM_SEED = 42

# Paths
RAW_DATA_FILE = RAW_DATA_DIR / "PTC" / "somatic_TCGA.txt"
GERMLINE_TCGA_FILE = RAW_DATA_DIR / "PTC" / "germline_TCGA.txt"
GTEX_FILE = RAW_DATA_DIR / "PTC" / "GTEx.txt"
PLOT_TITLE = "Interreplicate correlation \n of ASE NMD efficiency"
PLOT_TITLE_FONTSIZE = 18

def load_and_preprocess_data() -> pd.DataFrame:
    """
    Load raw data and apply preprocessing steps up to (but not including) aggregation.
    
    Returns:
        DataFrame with preprocessed individual variant observations
    """
    logger.info(f"Loading raw data from {RAW_DATA_FILE}")
    df = pd.read_csv(RAW_DATA_FILE, sep='\t')
    logger.info(f"Loaded {len(df)} variants")
    
    # Create default config (no somatic overlap removal for somatic data)
    config = DatasetConfig(somatic_overlap_removal="none")
    
    # Apply preprocessing steps
    logger.info("Applying preprocessing steps...")
    
    # Step 1: Create rule labels and impute halflife
    df = create_rule_labels(df)
    df = impute_rna_halflife(df)
    
    # Step 2: Expression/CV filtering
    if config.apply_expression_filter:
        logger.info("Applying expression/CV filter...")
        df = apply_lenient_expression_filter(df, config)
    
    # Step 3: Splice site filtering
    if config.apply_splice_filter:
        logger.info("Applying splice site filter...")
        df = apply_splice_site_filter(df, config)
    
    # Step 4: Frameshift correction
    if config.apply_frameshift_correction:
        logger.info("Applying frameshift correction...")
        df = apply_frameshift_correction(df)
    
    # Step 5: Fill NA values
    df["exons_length_postPTC"] = df["exons_length_postPTC"].fillna(0)
    df["exons_length_prePTC"] = df["exons_length_prePTC"].fillna(0)
    df["UTR3s_length"] = df["UTR3s_length"].fillna(0)
    df["UTR5s_length"] = df["UTR5s_length"].fillna(0)
    
    # Step 6: Center NMD efficiency
    if config.center_nmd_efficiency:
        logger.info("Centering NMD efficiency...")
        df = center_nmd_efficiency(df)
    
    # Step 7: Regression correction
    if config.apply_regression_correction:
        logger.info("Applying regression correction...")
        df = apply_regression_correction(df, config)
    
    logger.info(f"Preprocessing complete. Retained {len(df)} variants")
    
    return df


def identify_replicate_ptcs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify PTCs with multiple replicate observations.
    
    Args:
        df: Preprocessed DataFrame with individual variants
        
    Returns:
        DataFrame with PTCs that have >= MIN_REPLICATES observations
    """
    # Define PTC identifier
    ptc_cols = ['gene_id', 'transcript_id', 'chr', 'start_pos', 'Ref', 'Alt']
    
    # Count replicates per PTC
    replicate_counts = df.groupby(ptc_cols).size().reset_index(name='n_replicates')
    
    # Filter for PTCs with sufficient replicates
    replicate_ptcs = replicate_counts[replicate_counts['n_replicates'] >= MIN_REPLICATES]
    
    logger.info(f"Found {len(replicate_ptcs)} PTCs with >= {MIN_REPLICATES} replicates")
    logger.info(f"Replicate distribution: min={replicate_ptcs['n_replicates'].min()}, "
                f"max={replicate_ptcs['n_replicates'].max()}, "
                f"median={replicate_ptcs['n_replicates'].median()}")
    
    # Merge back with original data
    df_replicates = df.merge(replicate_ptcs, on=ptc_cols, how='inner')
    
    logger.info(f"Total replicate observations: {len(df_replicates)}")
    
    return df_replicates


def calculate_split_half_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate split-half reliability by randomly splitting replicates and correlating means.
    
    Args:
        df: DataFrame with replicate PTCs
        
    Returns:
        DataFrame with split-half results (mean values for plotting) and correlation stats
    """
    np.random.seed(RANDOM_SEED)
    
    ptc_cols = ['gene_id', 'transcript_id', 'chr', 'start_pos', 'Ref', 'Alt']
    
    # Determine NMD group for each PTC
    ptc_groups = df.groupby(ptc_cols).agg({
        'ASE_NMD_efficiency_TPM': list,
        'n_replicates': 'first',
        'Last_Exon': 'first',
        'Penultimate_Exon': 'first',
        'Start_Prox': 'first',
        'Long_Exon': 'first',
    }).reset_index()
    
    # Categorize PTCs as triggering or evading
    def get_nmd_group(row):
        if row['Last_Exon'] == 1 or row['Penultimate_Exon'] == 1 or row['Start_Prox'] == 1 or row['Long_Exon'] == 1:
            return 'NMD evading (any rule)'
        else:
            return 'NMD triggering (no rule)'
    
    ptc_groups['nmd_group'] = ptc_groups.apply(get_nmd_group, axis=1)
    
    # Perform N_SPLITS random splits
    all_splits = []
    for split_idx in tqdm(range(N_SPLITS), desc="Performing split-half splits"):
        split_results = []
        
        for _, row in ptc_groups.iterrows():
            values = np.array(row['ASE_NMD_efficiency_TPM'])
            
            # Randomly split replicates
            n_half = max(1, len(values) // 2)
            shuffled_idx = np.random.permutation(len(values))
            group1_idx = shuffled_idx[:n_half]
            group2_idx = shuffled_idx[n_half:]
            
            split_results.append({
                'nmd_group': row['nmd_group'],
                'n_replicates': len(values),
                'mean1': np.mean(values[group1_idx]),
                'mean2': np.mean(values[group2_idx]),
                'ptc_id': f"{row['gene_id']}_{row['transcript_id']}_{row['chr']}_{row['start_pos']}",
                'split_idx': split_idx
            })
        
        all_splits.extend(split_results)
    
    splits_df = pd.DataFrame(all_splits)
    
    # Calculate correlation per group across all splits
    group_correlations = []
    for group in ['NMD triggering (no rule)', 'NMD evading (any rule)']:
        group_data = splits_df[splits_df['nmd_group'] == group]
        if len(group_data) >= 2:
            corr, pval = spearmanr(group_data['mean1'], group_data['mean2'])
            group_correlations.append({
                'nmd_group': group,
                'correlation': corr,
                'pvalue': pval,
                'n_ptcs': len(group_data) // N_SPLITS,  # Number of unique PTCs
                'n_observations': len(group_data)  # Total across all splits
            })
    
    corr_df = pd.DataFrame(group_correlations)
    logger.info(f"Calculated split-half correlations across {N_SPLITS} splits")
    
    return splits_df, corr_df


def plot_combined_figure(splits_df: pd.DataFrame, corr_df: pd.DataFrame,
                        germline_df: pd.DataFrame = None, gtex_df: pd.DataFrame = None,
                        somatic_df: pd.DataFrame = None):
    """
    Create combined figure with 3 scatter plots:
    1. Split-half reliability
    2. Somatic TCGA vs Germline TCGA
    3. Somatic TCGA vs GTEx
    
    Args:
        splits_df: DataFrame with split-half mean values
        corr_df: DataFrame with correlation statistics per group
        germline_df: DataFrame with germline TCGA data (preprocessed, with variant_id and NMD group)
        gtex_df: DataFrame with GTEx data (preprocessed, with variant_id and NMD group)
        somatic_df: DataFrame with somatic TCGA data (preprocessed, with variant_id)
    """
    # Define colors
    colors = {'NMD triggering (no rule)': '#e74c3c', 'NMD evading (any rule)': '#3498db'}
    
    # Create figure with 3 rows
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))
    
    # ===== PLOT 1: Split-half reliability =====
    ax = axes[0]
    ax.set_title(PLOT_TITLE, fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold', pad=15)
    
    # Plot density for each group using kernel density estimation
    for group in ['NMD evading (any rule)', 'NMD triggering (no rule)']:  # Evading first so triggering is on top
        group_data = splits_df[splits_df['nmd_group'] == group]
        sns.kdeplot(
            x=group_data['mean1'], 
            y=group_data['mean2'],
            color=colors[group],
            alpha=0.6,
            fill=True,
            levels=5,
            ax=ax,
            label=group
        )
    
    # Add diagonal line (perfect correlation)
    min_val = min(splits_df['mean1'].min(), splits_df['mean2'].min())
    max_val = max(splits_df['mean1'].max(), splits_df['mean2'].max())
    ax.plot([min_val, max_val], [min_val, max_val],
           'k--', alpha=0.5, linewidth=1.5, zorder=0)
    
    # Calculate overall correlation (across both groups)
    overall_corr, _ = spearmanr(splits_df['mean1'], splits_df['mean2'])
    
    # Add single correlation text
    n_total_ptcs = len(splits_df) // N_SPLITS
    ax.text(0.05, 0.95, f'ρ = {overall_corr:.3f}\nn = {int(n_total_ptcs)} PTCs',
           transform=ax.transAxes, fontsize=14,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # Labels and styling
    ax.set_xlabel('Mean ASE NMD efficiency (Group 1)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean ASE NMD efficiency (Group 2)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)
    
    # Create custom legend
    handles = [mpatches.Patch(color=colors[group], label=group, alpha=0.6) for group in ['NMD evading (any rule)', 'NMD triggering (no rule)']]
    ax.legend(handles=handles, fontsize=16, loc='lower right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle=':')
    ax.set_aspect('equal', adjustable='box')
    
    # ===== PLOT 2: Somatic vs Germline TCGA =====
    germline_merged = None
    gtex_merged = None
    
    if germline_df is not None and somatic_df is not None:
        ax = axes[1]
        
        # First aggregate each dataset by variant_id (take median)
        somatic_agg = somatic_df.groupby('variant_id').agg({
            'ASE_NMD_efficiency_TPM': 'median'
        }).reset_index()
        
        germline_agg = germline_df.groupby(['variant_id', 'nmd_group']).agg({
            'ASE_NMD_efficiency_TPM': 'median'
        }).reset_index()
        
        # Now merge the aggregated data
        merged = germline_agg.merge(
            somatic_agg, 
            on='variant_id', 
            suffixes=('_germline', '_somatic')
        )
        germline_merged = merged
        
        if len(merged) > 0:
            # Plot density for each group using kernel density estimation
            for group in ['NMD evading (any rule)', 'NMD triggering (no rule)']:  # Evading first so triggering is on top
                group_data = merged[merged['nmd_group'] == group]
                sns.kdeplot(
                    x=group_data['ASE_NMD_efficiency_TPM_somatic'], 
                    y=group_data['ASE_NMD_efficiency_TPM_germline'],
                    color=colors[group],
                    alpha=0.6,
                    fill=True,
                    levels=5,
                    ax=ax,
                    label=group
                )
            
            # Add diagonal line
            min_val = min(merged['ASE_NMD_efficiency_TPM_somatic'].min(), 
                         merged['ASE_NMD_efficiency_TPM_germline'].min())
            max_val = max(merged['ASE_NMD_efficiency_TPM_somatic'].max(), 
                         merged['ASE_NMD_efficiency_TPM_germline'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'k--', alpha=0.5, linewidth=1.5, zorder=0)
            
            # Calculate overall correlation
            corr, _ = spearmanr(merged['ASE_NMD_efficiency_TPM_somatic'], 
                                  merged['ASE_NMD_efficiency_TPM_germline'])
            
            # Add text
            ax.text(0.05, 0.95, f'ρ = {corr:.3f}\nn = {len(merged)}',
                   transform=ax.transAxes, fontsize=14,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
        
        ax.set_ylabel('ASE NMD efficiency (Germline TCGA)', fontsize=14, fontweight='bold')
        ax.set_xlabel('ASE NMD efficiency (Somatic TCGA)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=16)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, linestyle=':')
        ax.set_aspect('equal', adjustable='box')
    
    # ===== PLOT 3: Somatic vs GTEx =====
    if gtex_df is not None and somatic_df is not None:
        ax = axes[2]
        
        # First aggregate each dataset by variant_id (take median)
        somatic_agg = somatic_df.groupby('variant_id').agg({
            'ASE_NMD_efficiency_TPM': 'median'
        }).reset_index()
        
        gtex_agg = gtex_df.groupby(['variant_id', 'nmd_group']).agg({
            'ASE_NMD_efficiency_TPM': 'median'
        }).reset_index()
        
        # Now merge the aggregated data
        merged = gtex_agg.merge(
            somatic_agg, 
            on='variant_id', 
            suffixes=('_gtex', '_somatic')
        )
        gtex_merged = merged
        
        if len(merged) > 0:
            # Plot density for each group using kernel density estimation
            for group in ['NMD evading (any rule)', 'NMD triggering (no rule)']:  # Evading first so triggering is on top
                group_data = merged[merged['nmd_group'] == group]
                sns.kdeplot(
                    x=group_data['ASE_NMD_efficiency_TPM_somatic'], 
                    y=group_data['ASE_NMD_efficiency_TPM_gtex'],
                    color=colors[group],
                    alpha=0.6,
                    fill=True,
                    levels=5,
                    ax=ax,
                    label=group
                )
            
            # Add diagonal line
            min_val = min(merged['ASE_NMD_efficiency_TPM_somatic'].min(), 
                         merged['ASE_NMD_efficiency_TPM_gtex'].min())
            max_val = max(merged['ASE_NMD_efficiency_TPM_somatic'].max(), 
                         merged['ASE_NMD_efficiency_TPM_gtex'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'k--', alpha=0.5, linewidth=1.5, zorder=0)
            
            # Calculate overall correlation
            corr, _ = spearmanr(merged['ASE_NMD_efficiency_TPM_somatic'], 
                                  merged['ASE_NMD_efficiency_TPM_gtex'])
            
            # Add text
            ax.text(0.05, 0.95, f'ρ = {corr:.3f}\nn = {len(merged)}',
                   transform=ax.transAxes, fontsize=14,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
        
        ax.set_xlabel('ASE NMD efficiency (Somatic TCGA)', fontsize=14, fontweight='bold')
        ax.set_ylabel('ASE NMD efficiency (GTEx)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=16)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3, linestyle=':')
        ax.set_aspect('equal', adjustable='box')
    
    # Synchronize axis limits for all plots to align them
    # Collect limits from all axes, but handle cases where some axes may not have data
    all_xlims = []
    all_ylims = []
    
    for i, ax in enumerate(axes):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Only include limits that are not the default matplotlib auto-range
        # Default auto-range is typically (0, 1) for empty plots, but let's be more robust
        if xlim[0] != xlim[1] and ylim[0] != ylim[1]:  # Only if ranges are meaningful
            all_xlims.append(xlim)
            all_ylims.append(ylim)
    
    if all_xlims and all_ylims:  # Only if we have at least one valid axis
        global_xlim = (min(x[0] for x in all_xlims), max(x[1] for x in all_xlims))
        global_ylim = (min(y[0] for y in all_ylims), max(y[1] for y in all_ylims))
        
        # Add small padding to prevent points from being cut off
        x_range = global_xlim[1] - global_xlim[0]
        y_range = global_ylim[1] - global_ylim[0]
        x_padding = x_range * 0.05
        y_padding = y_range * 0.05
        
        global_xlim = (global_xlim[0] - x_padding, global_xlim[1] + x_padding)
        global_ylim = (global_ylim[0] - y_padding, global_ylim[1] + y_padding)
        
        # Apply to all axes
        for ax in axes:
            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)
    
    plt.tight_layout()
    
    return fig


def load_and_preprocess_germline_data(file_path: Path) -> pd.DataFrame:
    """
    Load and preprocess germline/GTEx data using same steps as somatic.
    
    Args:
        file_path: Path to germline or GTEx raw data file
        
    Returns:
        DataFrame with preprocessed variants including NMD group classification
    """
    logger.info(f"Loading germline data from {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    logger.info(f"Loaded {len(df)} variants")
    
    # Create default config
    config = DatasetConfig(somatic_overlap_removal="none")
    
    # Determine variant type
    var_type = "gtex" if "GTEx" in str(file_path) else "germline"
    
    # Apply same preprocessing steps as somatic (without aggregation)
    logger.info("Applying preprocessing steps...")
    
    df = create_rule_labels(df)
    df = impute_rna_halflife(df)
    
    # Skip expression/CV filtering for germline (somatic only)
    
    # Splice site filtering
    if config.apply_splice_filter:
        logger.info("Applying splice site filter...")
        df = apply_splice_site_filter(df, config)
    
    # VAF filtering (germline only)
    if config.apply_vaf_filter and var_type == "germline":
        logger.info("Applying VAF filter...")
        df = apply_vaf_filter(df, config)
    
    # Frameshift correction
    if config.apply_frameshift_correction:
        logger.info("Applying frameshift correction...")
        df = apply_frameshift_correction(df)
    
    # Fill NA values
    df["exons_length_postPTC"] = df["exons_length_postPTC"].fillna(0)
    df["exons_length_prePTC"] = df["exons_length_prePTC"].fillna(0)
    df["UTR3s_length"] = df["UTR3s_length"].fillna(0)
    df["UTR5s_length"] = df["UTR5s_length"].fillna(0)
    
    # Center NMD efficiency
    if config.center_nmd_efficiency:
        logger.info("Centering NMD efficiency...")
        df = center_nmd_efficiency(df)
    
    # Regression correction
    if config.apply_regression_correction:
        logger.info("Applying regression correction...")
        df = apply_regression_correction(df, config)
    
    # Threshold filtering (remove extreme values)
    if config.apply_threshold_filter:
        logger.info("Applying threshold filter...")
        df = apply_threshold_filter(df, config)
    
    # Transcript length filtering
    if config.apply_transcript_length_filter:
        logger.info("Applying transcript length filter...")
        df = apply_transcript_length_filter(df, var_type, config)
    
    # Add NMD group classification
    def get_nmd_group(row):
        if row['Last_Exon'] == 1 or row['Penultimate_Exon'] == 1 or row['Start_Prox'] == 1 or row['Long_Exon'] == 1:
            return 'NMD evading (any rule)'
        else:
            return 'NMD triggering (no rule)'
    
    df['nmd_group'] = df.apply(get_nmd_group, axis=1)
    
    logger.info(f"Preprocessing complete. Retained {len(df)} variants")
    
    return df


def prepare_overlap_data(somatic_df: pd.DataFrame):
    """
    Prepare germline and GTEx data for overlap comparison.
    
    Args:
        somatic_df: Preprocessed somatic TCGA DataFrame
        
    Returns:
        Tuple of (germline_df, gtex_df) or None if files don't exist
    """
    # Create variant_id for somatic data
    somatic_df = somatic_df.copy()
    somatic_df['variant_id'] = (somatic_df['chr'].astype(str) + ':' + 
                                  somatic_df['start_pos'].astype(str) + ':' + 
                                  somatic_df['Ref'] + ':' + somatic_df['Alt'])
    
    germline_df = None
    gtex_df = None
    
    # Process germline TCGA
    if GERMLINE_TCGA_FILE.exists():
        logger.info("Loading and preprocessing germline TCGA...")
        germline_df = load_and_preprocess_germline_data(GERMLINE_TCGA_FILE)
        germline_df['variant_id'] = (germline_df['chr'].astype(str) + ':' + 
                                      germline_df['start_pos'].astype(str) + ':' + 
                                      germline_df['Ref'] + ':' + germline_df['Alt'])
        
        # Find overlaps
        somatic_variants = set(somatic_df['variant_id'])
        overlap_mask = germline_df['variant_id'].isin(somatic_variants)
        n_overlap = overlap_mask.sum()
        
        if n_overlap > 0:
            logger.info(f"Found {n_overlap} overlapping variants with germline TCGA")
            # Keep only overlapping variants
            germline_df = germline_df[overlap_mask].copy()
        else:
            logger.warning("No overlapping variants found with germline TCGA")
            germline_df = None
    else:
        logger.warning(f"Germline TCGA file not found: {GERMLINE_TCGA_FILE}")
    
    # Process GTEx
    if GTEX_FILE.exists():
        logger.info("Loading and preprocessing GTEx...")
        gtex_df = load_and_preprocess_germline_data(GTEX_FILE)
        gtex_df['variant_id'] = (gtex_df['chr'].astype(str) + ':' + 
                                  gtex_df['start_pos'].astype(str) + ':' + 
                                  gtex_df['Ref'] + ':' + gtex_df['Alt'])
        
        # Find overlaps
        somatic_variants = set(somatic_df['variant_id'])
        overlap_mask = gtex_df['variant_id'].isin(somatic_variants)
        n_overlap = overlap_mask.sum()
        
        if n_overlap > 0:
            logger.info(f"Found {n_overlap} overlapping variants with GTEx")
            # Keep only overlapping variants
            gtex_df = gtex_df[overlap_mask].copy()
        else:
            logger.warning("No overlapping variants found with GTEx")
            gtex_df = None
    else:
        logger.warning(f"GTEx file not found: {GTEX_FILE}")
    
    return germline_df, gtex_df, somatic_df


def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = True,
):
    """Generate interreplicate correlation figure.

    Args:
        figure_label: Panel label when called from the manuscript app.
        figure_number: Figure number when called from the manuscript app.
        regenerate: If False and source data exists, skip processing.
    """
    paths = get_paths(
        script_name=SCRIPT_NAME,
        figure_label=figure_label,
        figure_number=figure_number,
        source_data_ext=".xlsx",
    )
    logger.info("Starting interreplicate correlation analysis")
    
    # Check if we can skip processing and load from existing source data
    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        
        # Load data from Excel
        splits_df = pd.read_excel(paths.source_data, sheet_name='Panel_A_Split_Half')
        
        # Try to load optional panels
        try:
            germline_merged = pd.read_excel(paths.source_data, sheet_name='Panel_B_Somatic_Germline')
        except:
            germline_merged = None
        
        try:
            gtex_merged = pd.read_excel(paths.source_data, sheet_name='Panel_C_Somatic_GTEx')
        except:
            gtex_merged = None
        
        # Create a simple corr_df (not saved, so we'll skip it for figure generation)
        corr_df = None
        
        logger.info("Loaded source data successfully")
    else:
        logger.info(f"Configuration:")
        logger.info(f"  MIN_REPLICATES: {MIN_REPLICATES}")
        logger.info(f"  N_SPLITS: {N_SPLITS}")
        logger.info(f"  RANDOM_SEED: {RANDOM_SEED}")
        
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Identify replicate PTCs
        df_replicates = identify_replicate_ptcs(df)
        
        # Calculate split-half correlations
        splits_df, corr_df = calculate_split_half_correlations(df_replicates)
        
        # Prepare germline overlap data
        logger.info("\nPreparing germline overlap data...")
        germline_df, gtex_df, somatic_df_with_id = prepare_overlap_data(df)
        
        # Save source data to Excel with multiple sheets (one per panel)
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(paths.source_data, engine='openpyxl') as writer:
            # Panel A: Split-half reliability (only the aggregate splits)
            panel_a = splits_df[['nmd_group', 'mean1', 'mean2']].copy()
            panel_a.to_excel(writer, sheet_name='Panel_A_Split_Half', index=False)
            
            # Panel B: Somatic vs Germline TCGA
            if germline_df is not None and somatic_df_with_id is not None:
                somatic_agg = somatic_df_with_id.groupby('variant_id').agg({
                    'ASE_NMD_efficiency_TPM': 'median'
                }).reset_index()
                germline_agg = germline_df.groupby('variant_id').agg({
                    'ASE_NMD_efficiency_TPM': 'median',
                    'nmd_group': 'first'
                }).reset_index()
                panel_b = germline_agg.merge(
                    somatic_agg, 
                    on='variant_id', 
                    suffixes=('_germline', '_somatic')
                )[['ASE_NMD_efficiency_TPM_somatic', 'ASE_NMD_efficiency_TPM_germline', 'nmd_group']]
                panel_b.to_excel(writer, sheet_name='Panel_B_Somatic_Germline', index=False)
                germline_merged = panel_b
            else:
                germline_merged = None
            
            # Panel C: Somatic vs GTEx
            if gtex_df is not None and somatic_df_with_id is not None:
                somatic_agg = somatic_df_with_id.groupby('variant_id').agg({
                    'ASE_NMD_efficiency_TPM': 'median'
                }).reset_index()
                gtex_agg = gtex_df.groupby('variant_id').agg({
                    'ASE_NMD_efficiency_TPM': 'median',
                    'nmd_group': 'first'
                }).reset_index()
                panel_c = gtex_agg.merge(
                    somatic_agg, 
                    on='variant_id', 
                    suffixes=('_gtex', '_somatic')
                )[['ASE_NMD_efficiency_TPM_somatic', 'ASE_NMD_efficiency_TPM_gtex', 'nmd_group']]
                panel_c.to_excel(writer, sheet_name='Panel_C_Somatic_GTEx', index=False)
                gtex_merged = panel_c
            else:
                gtex_merged = None
        
        logger.info(f"Saved source data to {paths.source_data}")
    
    # Create and save figure
    # When loading from source data, we need to reconstruct the data format expected by plot_combined_figure
    if not regenerate and paths.source_data.exists():
        # Reconstruct germline_df and gtex_df from merged data for plotting
        if germline_merged is not None:
            # Create a mock germline_df with variant_id and nmd_group for plotting
            germline_df = germline_merged.rename(columns={
                'ASE_NMD_efficiency_TPM_germline': 'ASE_NMD_efficiency_TPM'
            })[['ASE_NMD_efficiency_TPM', 'nmd_group']].copy()
            germline_df['variant_id'] = germline_merged.index  # Dummy variant_id
        else:
            germline_df = None
        
        if gtex_merged is not None:
            # Create a mock gtex_df with variant_id and nmd_group for plotting
            gtex_df = gtex_merged.rename(columns={
                'ASE_NMD_efficiency_TPM_gtex': 'ASE_NMD_efficiency_TPM'
            })[['ASE_NMD_efficiency_TPM', 'nmd_group']].copy()
            gtex_df['variant_id'] = gtex_merged.index  # Dummy variant_id
        else:
            gtex_df = None
        
        # Create a mock somatic_df_with_id
        if germline_merged is not None or gtex_merged is not None:
            # Use whichever is available to get somatic values
            source = germline_merged if germline_merged is not None else gtex_merged
            somatic_col = 'ASE_NMD_efficiency_TPM_somatic'
            somatic_df_with_id = pd.DataFrame({
                'variant_id': source.index,
                'ASE_NMD_efficiency_TPM': source[somatic_col]
            })
        else:
            somatic_df_with_id = None
    
    fig = plot_combined_figure(splits_df, corr_df, germline_df, gtex_df, somatic_df_with_id)
    fig.savefig(paths.figure_png, dpi=300, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Saved figure to {paths.figure_png}")
    plt.close(fig)
    
    logger.success("Interreplicate correlation analysis complete!")


if __name__ == "__main__":
    main()
