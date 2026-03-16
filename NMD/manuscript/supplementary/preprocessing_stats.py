"""
Combined preprocessing statistics for all PTC datasets.

This script generates a combined figure showing preprocessing steps for
somatic_TCGA, germline_TCGA, and GTEx datasets with variant counts and
NMD efficiency changes across preprocessing steps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from NMD.config import (
    TABLES_DIR,
    FIGURES_DIR,
    CONTRASTING_2_COLOURS
)


# ============================================================================
# CONFIGURATION - Define all paths and parameters here
# ============================================================================

# Paths
SOMATIC_STATS_FILE = TABLES_DIR / "data" / "PTC" / "somatic_TCGA_preprocessing_stats.csv"
GERMLINE_STATS_FILE = TABLES_DIR / "data" / "PTC" / "germline_TCGA_preprocessing_stats.csv"
GTEX_STATS_FILE = TABLES_DIR / "data" / "PTC" / "GTEx_preprocessing_stats.csv"
OUTPUT_TABLE = TABLES_DIR / "manuscript" / "supplementary" / "preprocessing_stats.xlsx"
OUTPUT_FIGURE = FIGURES_DIR / "manuscript" / "supplementary" / "preprocessing_stats.png"
OUTPUT_FIGURE_PDF = FIGURES_DIR / "manuscript" / "supplementary" / "preprocessing_stats.pdf"
PLOT_TITLE = "Combined preprocessing statistics for PTC datasets"

# Plot aesthetics
COLOR_SNV = CONTRASTING_2_COLOURS[1]  # '#022778' - dark blue
COLOR_INDEL = CONTRASTING_2_COLOURS[0]  # '#ff9e9d' - light pink

FIGURE_SIZE = (20, 22)  # For 3 datasets vertically stacked
DPI = 300


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_data():
    """Load and combine preprocessing statistics. Returns DataFrame for plotting."""
    
    logger.info("Starting data processing...")
    
    # Load datasets
    logger.info(f"Loading somatic TCGA stats from {SOMATIC_STATS_FILE}")
    somatic_stats = pd.read_csv(SOMATIC_STATS_FILE)
    somatic_stats['dataset'] = 'somatic_TCGA'
    logger.info(f"Loaded {len(somatic_stats)} preprocessing steps")
    
    logger.info(f"Loading germline TCGA stats from {GERMLINE_STATS_FILE}")
    germline_stats = pd.read_csv(GERMLINE_STATS_FILE)
    germline_stats['dataset'] = 'germline_TCGA'
    logger.info(f"Loaded {len(germline_stats)} preprocessing steps")
    
    logger.info(f"Loading GTEx stats from {GTEX_STATS_FILE}")
    gtex_stats = pd.read_csv(GTEX_STATS_FILE)
    gtex_stats['dataset'] = 'GTEx'
    logger.info(f"Loaded {len(gtex_stats)} preprocessing steps")
    
    # Combine datasets
    df = pd.concat([somatic_stats, germline_stats, gtex_stats], ignore_index=True)
    logger.info(f"Combined dataset: {len(df)} total preprocessing records")
    
    # Save combined data as XLSX with sheets per dataset
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    
    # Delete existing file if it exists to avoid corruption issues
    if OUTPUT_TABLE.exists():
        OUTPUT_TABLE.unlink()
        logger.info(f"Deleted existing table file: {OUTPUT_TABLE}")
    
    with pd.ExcelWriter(OUTPUT_TABLE, engine='openpyxl') as writer:
        for dataset in ['somatic_TCGA', 'germline_TCGA', 'GTEx']:
            dataset_df = df[df['dataset'] == dataset].copy()
            sheet_name = dataset.replace('_', ' ').title().replace(' ', '_')
            dataset_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"  Saved {len(dataset_df)} rows for {sheet_name}")
    
    logger.info(f"Saved combined data to {OUTPUT_TABLE}")
    
    return {
        'somatic_TCGA': somatic_stats,
        'germline_TCGA': germline_stats,
        'GTEx': gtex_stats
    }


# ============================================================================
# PLOTTING
# ============================================================================

def plot_combined_preprocessing_steps(datasets_dict):
    """
    Plot preprocessing steps for all datasets in a single figure.
    
    Creates a 3-row figure with one row per dataset. Each row has:
    - Left axis: Stacked bar chart showing SNV and indel counts at each step
    - Right axis: Line plots showing mean NMDeff for last exon and triggering groups with CI
    
    Args:
        datasets_dict: Dictionary with dataset names as keys and preprocessing DataFrames as values
    """
    # Sort datasets in desired order
    dataset_order = ['somatic_TCGA', 'germline_TCGA', 'GTEx']
    datasets = [(name, datasets_dict[name]) for name in dataset_order if name in datasets_dict]
    
    fig, axes = plt.subplots(len(datasets), 1, figsize=FIGURE_SIZE)
    
    # Ensure axes is always an array
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, (dataset_name, preprocessing_df) in enumerate(datasets):
        ax1 = axes[idx]
        
        # Set up the x-axis
        x_pos = np.arange(len(preprocessing_df))
        step_names = preprocessing_df['step'].values
        
        # Left axis: Stacked bar chart for variant counts
        width = 0.6
        snv_counts = preprocessing_df['n_snvs'].values
        indel_counts = preprocessing_df['n_indels'].values
        total_counts = snv_counts + indel_counts
        
        # Create stacked bars
        bars_snv = ax1.bar(x_pos, snv_counts, width, label='SNVs', 
                           color=COLOR_SNV, alpha=0.8)
        bars_indel = ax1.bar(x_pos, indel_counts, width, bottom=snv_counts, 
                            label='Indels', color=COLOR_INDEL, alpha=0.8)
        
        # Add total variant count on top of each bar
        for i, (x, total) in enumerate(zip(x_pos, total_counts)):
            ax1.text(x, total + ax1.get_ylim()[1] * 0.02, f'{int(total):,}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Only add x-axis label on the last subplot
        if idx == len(datasets) - 1:
            ax1.set_xlabel('Preprocessing Step', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Number of Variants', fontsize=16, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(step_names, rotation=45, ha='right', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='black', labelsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Right axis: Line plots for NMDeff with CI
        ax2 = ax1.twinx()
        
        # Plot last exon (evading) - SNVs
        last_exon_snv_means = preprocessing_df['last_exon_snv_mean'].values
        last_exon_snv_ci_lower = preprocessing_df['last_exon_snv_ci_lower'].values
        last_exon_snv_ci_upper = preprocessing_df['last_exon_snv_ci_upper'].values
        
        ax2.plot(x_pos, last_exon_snv_means, 'o-', color=COLOR_SNV, linewidth=2.5, 
                markersize=8, label='Last Exon SNVs', zorder=10, alpha=0.9)
        ax2.fill_between(x_pos, last_exon_snv_ci_lower, last_exon_snv_ci_upper, 
                        alpha=0.2, color=COLOR_SNV, zorder=5)
        
        # Plot last exon (evading) - Indels
        last_exon_indel_means = preprocessing_df['last_exon_indel_mean'].values
        last_exon_indel_ci_lower = preprocessing_df['last_exon_indel_ci_lower'].values
        last_exon_indel_ci_upper = preprocessing_df['last_exon_indel_ci_upper'].values
        
        ax2.plot(x_pos, last_exon_indel_means, 'o--', color=COLOR_INDEL, linewidth=2.5, 
                markersize=8, label='Last Exon Indels', zorder=10, alpha=0.9)
        ax2.fill_between(x_pos, last_exon_indel_ci_lower, last_exon_indel_ci_upper, 
                        alpha=0.2, color=COLOR_INDEL, zorder=5)
        
        # Plot triggering - SNVs
        triggering_snv_means = preprocessing_df['triggering_snv_mean'].values
        triggering_snv_ci_lower = preprocessing_df['triggering_snv_ci_lower'].values
        triggering_snv_ci_upper = preprocessing_df['triggering_snv_ci_upper'].values
        
        ax2.plot(x_pos, triggering_snv_means, 's-', color=COLOR_SNV, linewidth=2.5, 
                markersize=8, label='Triggering SNVs', zorder=10, alpha=0.6)
        ax2.fill_between(x_pos, triggering_snv_ci_lower, triggering_snv_ci_upper, 
                        alpha=0.15, color=COLOR_SNV, zorder=5)
        
        # Plot triggering - Indels
        triggering_indel_means = preprocessing_df['triggering_indel_mean'].values
        triggering_indel_ci_lower = preprocessing_df['triggering_indel_ci_lower'].values
        triggering_indel_ci_upper = preprocessing_df['triggering_indel_ci_upper'].values
        
        ax2.plot(x_pos, triggering_indel_means, 's--', color=COLOR_INDEL, linewidth=2.5, 
                markersize=8, label='Triggering Indels', zorder=10, alpha=0.6)
        ax2.fill_between(x_pos, triggering_indel_ci_lower, triggering_indel_ci_upper, 
                        alpha=0.15, color=COLOR_INDEL, zorder=5)
        
        ax2.set_ylabel('Mean NMD Efficiency', fontsize=16, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='black', labelsize=14)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Combine legends and place in lower left
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', 
                  fontsize=14, framealpha=0.95)
        
        # Add dataset label as row title
        ax1.text(-0.08, 0.5, dataset_name.replace('_', ' ').upper(),
                transform=ax1.transAxes,
                fontsize=20,
                fontweight='bold',
                va='center',
                rotation=90)
    
    plt.suptitle(PLOT_TITLE, fontsize=18, y=0.92)
    plt.tight_layout(h_pad=1.5)
    return fig


def plot_from_tables(datasets_dict):
    """Generate plots from processed data."""
    logger.info("Generating plots...")
    
    fig = plot_combined_preprocessing_steps(datasets_dict)
    
    # Save figure
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=DPI, bbox_inches='tight')
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches='tight')
    logger.info(f"Saved figure to {OUTPUT_FIGURE}")
    logger.info(f"Saved PDF to {OUTPUT_FIGURE_PDF}")
    
    plt.close(fig)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Preprocessing Statistics Analysis")
    logger.info("=" * 80)
    
    # Check if individual tables exist
    if not (SOMATIC_STATS_FILE.exists() and GERMLINE_STATS_FILE.exists() and GTEX_STATS_FILE.exists()):
        logger.error("One or more preprocessing stats files not found!")
        logger.error(f"Expected files:")
        logger.error(f"  - {SOMATIC_STATS_FILE}")
        logger.error(f"  - {GERMLINE_STATS_FILE}")
        logger.error(f"  - {GTEX_STATS_FILE}")
        logger.error("Please run the data preprocessing first.")
        return
    
    # Process data
    datasets_dict = process_data()
    
    # Generate plots
    plot_from_tables(datasets_dict)
    
    logger.success("Preprocessing statistics analysis complete!")


if __name__ == "__main__":
    main()
