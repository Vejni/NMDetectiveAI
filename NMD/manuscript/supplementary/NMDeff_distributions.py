"""
NMDeff distributions across datasets.

This script generates distribution plots showing NMDeff (normalized NMD efficiency)
across three datasets (somatic_TCGA, germline_TCGA, GTEx) with multiple visualizations:
1. Overall distribution with NMD-triggering vs evading groups
2. Boxplots by NMD rule groups
3. Gene-level mean distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from NMD.config import (
    INTERIM_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    CONTRASTING_2_COLOURS
)


# ============================================================================
# CONFIGURATION - Define all paths and parameters here
# ============================================================================

# Paths
SOMATIC_TCGA_FILE = INTERIM_DATA_DIR / "PTC" / "somatic_TCGA.csv"
GERMLINE_TCGA_FILE = INTERIM_DATA_DIR / "PTC" / "germline_TCGA.csv"
GTEX_FILE = INTERIM_DATA_DIR / "PTC" / "GTEx.csv"
OUTPUT_TABLE = TABLES_DIR / "manuscript" / "supplementary" / "NMDeff_distributions.xlsx"
OUTPUT_FIGURE = FIGURES_DIR / "manuscript" / "supplementary" / "NMDeff_distributions.png"
OUTPUT_FIGURE_PDF = FIGURES_DIR / "manuscript" / "supplementary" / "NMDeff_distributions.pdf"
PLOT_TITLE = "Processed NMD efficiency distributions across datasets"

# Analysis parameters
VALUE_COL = "NMDeff_Norm"  # Column to analyze

# Plot aesthetics
COLOR_TRIGGERING = CONTRASTING_2_COLOURS[1]  # '#022778' - dark blue
COLOR_EVADING = CONTRASTING_2_COLOURS[0]     # '#ff9e9d' - light pink
COLOR_HISTOGRAM = '#555555'                   # Gray for histogram background
EVADING_GRADIENT = ['#ff9e9d', '#ffb3b3', '#ffc8c8', '#ffdddd']  # Light to lighter pink

FIGURE_SIZE = (24, 18)  # For 3 datasets × 3 columns
DPI = 300


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_data():
    """Load and combine datasets. Returns DataFrame for plotting."""
    
    logger.info("Starting data processing...")
    
    # Load datasets
    logger.info(f"Loading somatic TCGA from {SOMATIC_TCGA_FILE}")
    somatic_TCGA = pd.read_csv(SOMATIC_TCGA_FILE)
    logger.info(f"Loaded {len(somatic_TCGA)} somatic PTCs")
    
    logger.info(f"Loading germline TCGA from {GERMLINE_TCGA_FILE}")
    germline_TCGA = pd.read_csv(GERMLINE_TCGA_FILE)
    logger.info(f"Loaded {len(germline_TCGA)} germline TCGA PTCs")
    
    logger.info(f"Loading GTEx from {GTEX_FILE}")
    GTEx = pd.read_csv(GTEX_FILE)
    logger.info(f"Loaded {len(GTEx)} GTEx PTCs")
    
    # Add dataset labels
    somatic_TCGA["dataset"] = "somatic_TCGA"
    germline_TCGA["dataset"] = "germline_TCGA"
    GTEx["dataset"] = "GTEx"
    
    somatic_TCGA["germline_vs_somatic"] = "somatic"
    germline_TCGA["germline_vs_somatic"] = "germline"
    GTEx["germline_vs_somatic"] = "germline"
    
    # Combine datasets
    df = pd.concat([somatic_TCGA, germline_TCGA, GTEx], ignore_index=True)
    logger.info(f"Combined dataset: {len(df)} total PTCs")
    
    # Save combined data as XLSX with sheets per dataset
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    
    # Delete existing file if it exists to avoid corruption issues
    if OUTPUT_TABLE.exists():
        OUTPUT_TABLE.unlink()
        logger.info(f"Deleted existing table file: {OUTPUT_TABLE}")
    
    # Save only columns needed for plotting - check which exist
    required_columns = ['dataset', 'NMDeff', 'NMDeff_Norm', 'Last_Exon', 
                       'Penultimate_Exon', 'Start_Prox', 'Long_Exon']
    optional_columns = ['gene_symbol', 'gene', 'gene_name']
    
    plot_columns = required_columns.copy()
    for col in optional_columns:
        if col in df.columns:
            plot_columns.append(col)
            break  # Only add the first gene column found
    
    logger.info(f"Saving columns: {plot_columns}")
    
    # Try xlsxwriter engine instead of openpyxl
    try:
        with pd.ExcelWriter(OUTPUT_TABLE, engine='xlsxwriter') as writer:
            for dataset in ['somatic_TCGA', 'germline_TCGA', 'GTEx']:
                dataset_df = df[df['dataset'] == dataset][plot_columns].copy()
                sheet_name = dataset.replace('_', ' ').title().replace(' ', '_')
                dataset_df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"  Saved {len(dataset_df)} rows for {sheet_name}")
        logger.info(f"Saved combined data to {OUTPUT_TABLE}")
    except ImportError:
        # Fallback to separate CSV files if xlsxwriter not available
        logger.warning("xlsxwriter not available, saving as separate CSV files")
        csv_dir = OUTPUT_TABLE.parent / OUTPUT_TABLE.stem
        csv_dir.mkdir(exist_ok=True)
        for dataset in ['somatic_TCGA', 'germline_TCGA', 'GTEx']:
            dataset_df = df[df['dataset'] == dataset][plot_columns].copy()
            csv_file = csv_dir / f"{dataset}.csv"
            dataset_df.to_csv(csv_file, index=False)
            logger.info(f"  Saved {len(dataset_df)} rows to {csv_file}")
        logger.info(f"Saved data to {csv_dir}")
    except Exception as e:
        logger.error(f"Failed to save Excel file: {e}")
        raise
    
    return df


# ============================================================================
# PLOTTING
# ============================================================================

def plot_nmd_analysis(df, col):
    """
    Create comprehensive NMD analysis plots for different datasets.
    
    Args:
        df: DataFrame containing NMD data with 'dataset' column
        col: Column name to analyze (e.g., 'NMDeff_Norm')
    """
    # Define the datasets
    datasets = sorted(df['dataset'].unique(), reverse=True)  # Reverse for desired order
    
    # Calculate global x-axis limits for normalized column
    col_data = df[col].dropna()
    x_min, x_max = col_data.min(), col_data.max()
    x_margin = (x_max - x_min) * 0.05
    x_limits_norm = (x_min - x_margin, x_max + x_margin)
    
    # Calculate global x-axis limits for non-normalized NMDeff
    nmdeff_data = df['NMDeff'].dropna()
    nmdeff_min, nmdeff_max = nmdeff_data.min(), nmdeff_data.max()
    nmdeff_margin = (nmdeff_max - nmdeff_min) * 0.05
    x_limits_nmdeff = (nmdeff_min - nmdeff_margin, nmdeff_max + nmdeff_margin)
    
    # Calculate global y-axis limits for boxplots
    all_boxplot_data = []
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        all_boxplot_data.extend(dataset_df[col].dropna().values)
    
    if all_boxplot_data:
        y_min, y_max = min(all_boxplot_data), max(all_boxplot_data)
        y_margin = (y_max - y_min) * 0.05
        y_limits = (y_min - y_margin, y_max + y_margin)
    else:
        y_limits = None
    
    # Create figure
    fig, axes = plt.subplots(len(datasets), 3, figsize=FIGURE_SIZE)
    
    # Ensure axes is 2D even with single dataset
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    
    # Plot for each dataset
    for i, dataset in enumerate(datasets):
        dataset_df = df[df['dataset'] == dataset]
        
        # Define NMD groups
        evading_mask = dataset_df["Last_Exon"] == 1
        triggering_mask = (
            (dataset_df["Last_Exon"] == 0) & 
            (dataset_df["Penultimate_Exon"] == 0) & 
            (dataset_df["Start_Prox"] == 0) & 
            (dataset_df["Long_Exon"] == 0)
        )
        
        evading_group = dataset_df[evading_mask]
        triggering_group = dataset_df[triggering_mask]
        
        # ====================================================================
        # Column 0: NMDeff (non-normalized) distribution with KDE
        # ====================================================================
        ax_nmdeff = axes[i, 0]
        
        # Histogram for all data
        ax_nmdeff.hist(
            dataset_df['NMDeff'].dropna(), 
            bins=30, 
            alpha=0.3,
            color=COLOR_HISTOGRAM, 
            density=True, 
            label='All variants'
        )
        
        # KDE for triggering group
        if len(triggering_group) > 0:
            sns.kdeplot(
                data=triggering_group, 
                x='NMDeff', 
                ax=ax_nmdeff,
                label='NMD-triggering', 
                color=COLOR_TRIGGERING,
                bw_adjust=0.5, 
                linewidth=3
            )
            triggering_mean = triggering_group['NMDeff'].mean()
            ax_nmdeff.axvline(
                x=triggering_mean, 
                color=COLOR_TRIGGERING,
                linestyle='--', 
                alpha=0.8, 
                linewidth=2
            )
        
        # KDE for evading group
        if len(evading_group) > 0:
            sns.kdeplot(
                data=evading_group, 
                x='NMDeff', 
                ax=ax_nmdeff,
                label='Last Exon (NMD-evading)', 
                color=COLOR_EVADING,
                bw_adjust=0.5, 
                linewidth=3
            )
            evading_mean = evading_group['NMDeff'].mean()
            ax_nmdeff.axvline(
                x=evading_mean, 
                color=COLOR_EVADING,
                linestyle='--', 
                alpha=0.8, 
                linewidth=2
            )
        
        ax_nmdeff.set_xlabel('NMDeff', fontsize=14)
        ax_nmdeff.set_ylabel('Density', fontsize=14)
        ax_nmdeff.legend(fontsize=12)
        ax_nmdeff.grid(True, alpha=0.3)
        ax_nmdeff.set_xlim(x_limits_nmdeff)
        ax_nmdeff.tick_params(labelsize=12)
        
        # ====================================================================
        # Column 1: NMDeff_Norm distribution with KDE
        # ====================================================================
        ax_dist = axes[i, 1]
        
        # Histogram for all data
        ax_dist.hist(
            dataset_df[col].dropna(), 
            bins=30, 
            alpha=0.3,
            color=COLOR_HISTOGRAM, 
            density=True, 
            label='All variants'
        )
        
        # KDE for triggering group
        if len(triggering_group) > 0:
            sns.kdeplot(
                data=triggering_group, 
                x=col, 
                ax=ax_dist,
                label='NMD-triggering', 
                color=COLOR_TRIGGERING,
                bw_adjust=0.5, 
                linewidth=3
            )
            triggering_mean = triggering_group[col].mean()
            ax_dist.axvline(
                x=triggering_mean, 
                color=COLOR_TRIGGERING,
                linestyle='--', 
                alpha=0.8, 
                linewidth=2
            )
        
        # KDE for evading group
        if len(evading_group) > 0:
            sns.kdeplot(
                data=evading_group, 
                x=col, 
                ax=ax_dist,
                label='Last Exon (NMD-evading)', 
                color=COLOR_EVADING,
                bw_adjust=0.5, 
                linewidth=3
            )
            evading_mean = evading_group[col].mean()
            ax_dist.axvline(
                x=evading_mean, 
                color=COLOR_EVADING,
                linestyle='--', 
                alpha=0.8, 
                linewidth=2
            )
        
        ax_dist.set_xlabel(col, fontsize=14)
        ax_dist.set_ylabel('Density', fontsize=14)
        ax_dist.legend(fontsize=12)
        ax_dist.grid(True, alpha=0.3)
        ax_dist.set_xlim(x_limits_norm)
        ax_dist.tick_params(labelsize=12)
        
        # ====================================================================
        # Column 2: Boxplot by NMD rule groups
        # ====================================================================
        ax_box = axes[i, 2]
        
        # Define all rule groups
        last_exon = dataset_df[dataset_df["Last_Exon"] == 1]
        penultimate_exon = dataset_df[
            (dataset_df["Last_Exon"] == 0) & 
            (dataset_df["Penultimate_Exon"] == 1)
        ]
        start_prox = dataset_df[
            (dataset_df["Last_Exon"] == 0) & 
            (dataset_df["Penultimate_Exon"] == 0) & 
            (dataset_df["Start_Prox"] == 1)
        ]
        long_exon = dataset_df[
            (dataset_df["Last_Exon"] == 0) & 
            (dataset_df["Penultimate_Exon"] == 0) & 
            (dataset_df["Start_Prox"] == 0) & 
            (dataset_df["Long_Exon"] == 1)
        ]
        nmd_triggering = dataset_df[
            (dataset_df["Last_Exon"] == 0) & 
            (dataset_df["Penultimate_Exon"] == 0) & 
            (dataset_df["Start_Prox"] == 0) & 
            (dataset_df["Long_Exon"] == 0)
        ]
        
        groups_data = [
            last_exon[col].dropna(),
            penultimate_exon[col].dropna(),
            start_prox[col].dropna(),
            long_exon[col].dropna(),
            nmd_triggering[col].dropna()
        ]
        
        group_labels = [
            'Last Exon',
            'Penultimate\nExon',
            'Start\nProximal',
            'Long Exon',
            'NMD\nTriggering'
        ]
        colors = EVADING_GRADIENT + [COLOR_TRIGGERING]
        
        bp = ax_box.boxplot(
            groups_data, 
            labels=group_labels, 
            patch_artist=True
        )
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax_box.set_ylabel(col, fontsize=14)
        ax_box.tick_params(axis='x', rotation=45, labelsize=11)
        ax_box.tick_params(axis='y', labelsize=12)
        ax_box.grid(True, alpha=0.3)
        
        if y_limits:
            ax_box.set_ylim(y_limits)
        
        # Add dataset label as row title
        axes[i, 0].text(
            -0.15, 
            0.5, 
            dataset.replace('_', ' ').upper(),
            transform=axes[i, 0].transAxes,
            fontsize=16,
            fontweight='bold',
            va='center',
            rotation=90
        )
    
    plt.suptitle(PLOT_TITLE, fontsize=18, y=1.02)
    plt.tight_layout()
    return fig


def plot_from_table(df):
    """Generate plots from processed data."""
    logger.info("Generating plots...")
    
    fig = plot_nmd_analysis(df, VALUE_COL)
    
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
    logger.info("NMDeff Distributions Analysis")
    logger.info("=" * 80)
    
    # Check if table already exists
    if OUTPUT_TABLE.exists():
        try:
            logger.info(f"Loading existing data from {OUTPUT_TABLE}")
            # Read all sheets and combine
            all_sheets = pd.read_excel(OUTPUT_TABLE, sheet_name=None, engine='openpyxl')
            df = pd.concat(all_sheets.values(), ignore_index=True)
        except Exception as e:
            logger.warning(f"Failed to load existing table ({e}), regenerating...")
            if OUTPUT_TABLE.exists():
                OUTPUT_TABLE.unlink()
            df = process_data()
    else:
        logger.info("Processing data (table not found)...")
        df = process_data()
    
    # Generate plots
    plot_from_table(df)
    
    logger.success("NMDeff distributions analysis complete!")


if __name__ == "__main__":
    main()
