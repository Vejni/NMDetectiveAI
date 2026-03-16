#!/usr/bin/env python3
"""
Manuscript figure Fig4c: Penultimate exon fit quality histogram

Horizontal histogram showing distribution of R² values from sigmoid curve fits
across all genes, with highlighted genes marked by dashed lines.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path

from NMD.config import (
    PROJ_ROOT,
    COLOURS,
    CONTRASTING_3_COLOURS,
)
from NMD.manuscript.output import get_paths

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input paths (from supplementary analysis)
SUPPL_TABLE = PROJ_ROOT / "manuscript" / "supplementary" / "tables" / "penultimate_exon_fits.csv"

# Highlighted genes to mark with dashed lines
HIGHLIGHTED_GENES = ["PTEN", "TP53", "BRCA2"]

GENE_COLORS = {
    "PTEN": CONTRASTING_3_COLOURS[2],  # Dark blue
    "TP53": COLOURS[1],  # Yellow/gold
    "BRCA2": COLOURS[0],  # Orange
}

# Plot parameters
FIGURE_SIZE = (4, 5)  # Vertical orientation
DPI = 300
BINS = 500
HIST_COLOR = COLOURS[0]  # Orange
HIGHLIGHT_COLOR = COLOURS[2]  # Pink/red

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_fit_data():
    """
    Load 4PL fit parameters from supplementary analysis.
    
    Returns:
        DataFrame with fitted parameters including R² values
    """
    if not SUPPL_TABLE.exists():
        logger.error(f"Fit data not found: {SUPPL_TABLE}")
        logger.error("Please run penultimate_exon_genome_wide.py first to generate the data.")
        raise FileNotFoundError(f"Missing required file: {SUPPL_TABLE}")
    
    logger.info(f"Loading fit data from {SUPPL_TABLE}")
    df = pd.read_csv(SUPPL_TABLE)
    
    logger.info(f"Loaded {len(df)} transcripts with fit data")
    return df


def plot_r2_histogram(df, highlighted_genes):
    """
    Create horizontal histogram of R² values with highlighted genes.
    
    Args:
        df: DataFrame with 'r2' and 'gene_name' columns
        highlighted_genes: List of gene names to highlight with dashed lines
    
    Returns:
        matplotlib figure and dict with data for export
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Get R² values
    r2_values = df['r2'].dropna()
    
    # Create horizontal histogram
    counts, bins, patches = ax.hist(
        r2_values, 
        bins=BINS, 
        orientation='horizontal',
        color=HIST_COLOR,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add highlighted genes as horizontal dashed lines
    gene_r2_values = {}
    for gene in highlighted_genes:
        gene_data = df[df['gene_name'] == gene]
        if len(gene_data) > 0:
            r2_val = gene_data['r2'].iloc[0]
            gene_r2_values[gene] = r2_val
            ax.axhline(
                r2_val, 
                color=GENE_COLORS.get(gene, HIGHLIGHT_COLOR), 
                linestyle='--', 
                linewidth=2,
                alpha=0.8,
                label=f'{gene} (R²={r2_val:.3f})'
            )
        else:
            logger.warning(f"Gene {gene} not found in fit data")
    
    # Add median line
    median_r2 = r2_values.median()
    ax.axhline(
        median_r2,
        color='black',
        linestyle=':',
        linewidth=1.5,
        alpha=0.6,
        label=f'Median (R²={median_r2:.3f})'
    )
    
    # Labels and formatting
    ax.set_ylabel('R² of 4PL curve fit', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of genes', fontsize=14, fontweight='bold')
    ax.set_ylim(0.7, 1)
    ax.legend(loc='lower right', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='both')
    plt.title('Distribution of R² values for\npenultimate exon 4PL fits\n', fontsize=16, fontweight='bold')
    
    
    # Add statistics text
    stats_text = (
        f'n = {len(r2_values)}\n'
        f'Mean: {r2_values.mean():.3f}\n'
        f'Median: {median_r2:.3f}\n'
        f'R² ≥ 0.7: {(r2_values >= 0.7).sum()} ({100*(r2_values >= 0.7).sum()/len(r2_values):.1f}%)\n'
        f'R² ≥ 0.9: {(r2_values >= 0.9).sum()} ({100*(r2_values >= 0.9).sum()/len(r2_values):.1f}%)'
    )
    ax.text(
        0.15, 0.75, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=14
    )
    
    plt.tight_layout()
    
    # Prepare data for export
    bin_centers = (bins[:-1] + bins[1:]) / 2
    export_data = pd.DataFrame({
        'r2_bin_center': bin_centers,
        'count': counts
    })
    
    # Add highlighted gene R² values
    highlight_df = pd.DataFrame([
        {'gene': gene, 'r2': r2}
        for gene, r2 in gene_r2_values.items()
    ])
    
    return fig, {'histogram': export_data, 'highlighted_genes': highlight_df}


def main(
    figure_label: str = "c",
    figure_number: str = "Fig4",
    regenerate: bool = False,
):
    """
    Main function to generate Fig4c.
    
    Args:
        figure_label: Panel label (default: "c")
        figure_number: Figure number (default: "Fig4")
        regenerate: Force regeneration of source data
    """
    # Get output paths
    paths = get_paths(
        script_name="penultimate_exon_fit_quality",
        figure_label=figure_label,
        figure_number=figure_number,
        source_data_ext=".xlsx",
    )
    
    # Check if we need to regenerate
    if not regenerate and paths.source_data.exists() and paths.figure_png.exists():
        logger.info(f"Output files already exist. Skipping regeneration.")
        logger.info(f"  Figure: {paths.figure_png}")
        logger.info(f"  Data: {paths.source_data}")
        logger.info("Use --regenerate to force regeneration.")
        return
    
    # Load fit data
    logger.info("Loading penultimate exon fit data...")
    df = load_fit_data()
    
    # Generate plot
    logger.info("Creating R² histogram...")
    fig, data = plot_r2_histogram(df, HIGHLIGHTED_GENES)
    
    # Save figure
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    logger.info(f"PNG saved: {paths.figure_png}")
    
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"PDF saved: {paths.figure_pdf}")
    
    plt.close()
    
    # Save source data
    paths.source_data.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(paths.source_data, engine='openpyxl') as writer:
        data['histogram'].to_excel(writer, sheet_name='r2_histogram', index=False)
        data['highlighted_genes'].to_excel(writer, sheet_name='highlighted_genes', index=False)
    
    logger.info(f"Source data saved: {paths.source_data}")
    logger.success(f"Fig{figure_number}{figure_label} completed successfully!")


if __name__ == "__main__":
    main()
