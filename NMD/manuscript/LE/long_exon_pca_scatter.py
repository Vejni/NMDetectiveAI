"""
PCA scatter plot of long exon NMD efficiency curves.

Creates a scatter plot showing the first two principal components,
colored by exon length on a log scale, with highlighted extreme points
and specific gene-transcript pairs of interest.

Note: This script uses pre-computed PCA results from NMD.analysis.long_exon_pca_analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from NMD.config import TABLES_DIR, COLOURS
from NMD.manuscript.output import get_paths


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_NAME = "long_exon_pca_scatter"
PCA_SCORES_FILE = TABLES_DIR / "exon_analysis" / "long_exon_pca_scores.csv"
PCA_COMPONENTS_FILE = TABLES_DIR / "exon_analysis" / "long_exon_pca_components.csv"

# Gene-transcript pairs to highlight on PCA
HIGHLIGHT_GENES = [
    ("BRCA2", "ENST00000380152.7", 9),
    ("BRCA2", "ENST00000380152.7", 10),
    ("BRCA2", "ENST00000380152.7", 13),
    ("BRCA1", "ENST00000357654.7", 9),
    ("TP53", "ENST00000269305.9", 6),
    ("SMARCA4", "ENST00000344626.8", 3),
    ("NOTCH1", "ENST00000277541.6", 25),
    ("NOTCH1", "ENST00000277541.6", 26),
]

# PCA parameters
N_INTERPOLATION_POINTS = 50

# Figure settings
FIGURE_SIZE = (10, 8)
DPI = 300
COLORMAP = LinearSegmentedColormap.from_list("exon_length", ["#022778", "#ff9e9d"])
MARKER_SIZE = 30
MARKER_ALPHA = 0.7
HIGHLIGHT_COLOR = 'red'
EXTREME_COLOR = 'orange'


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_data():
    """Load pre-computed PCA scores and variance ratios."""
    logger.info(f"Loading PCA scores from {PCA_SCORES_FILE}")
    
    if not PCA_SCORES_FILE.exists():
        raise FileNotFoundError(
            f"PCA scores not found at {PCA_SCORES_FILE}. "
            "Run NMD.analysis.long_exon_pca_analysis first."
        )
    
    if not PCA_COMPONENTS_FILE.exists():
        raise FileNotFoundError(
            f"PCA components not found at {PCA_COMPONENTS_FILE}. "
            "Run NMD.analysis.long_exon_pca_analysis first."
        )
    
    # Load PC scores
    scores_df = pd.read_csv(PCA_SCORES_FILE)
    logger.info(f"Loaded PC scores for {len(scores_df)} exons")
    
    # Load PC components to get variance ratios
    components_df = pd.read_csv(PCA_COMPONENTS_FILE)
    # Variance ratios are in the last row (relative_position == -1)
    variance_row = components_df[components_df['relative_position'] == -1].iloc[0]
    variance_ratios = [variance_row['PC1_loading'], variance_row['PC2_loading']]
    
    logger.info(f"Variance explained: PC1={variance_ratios[0]*100:.1f}%, "
               f"PC2={variance_ratios[1]*100:.1f}%")
    
    # Prepare source data
    source_data = scores_df[['gene_name', 'transcript_id', 'exon_idx', 
                             'exon_length', 'PC1', 'PC2']].copy()
    
    return source_data, variance_ratios


# ============================================================================
# PLOTTING
# ============================================================================

def plot_pca_scatter(source_data, variance_ratios):
    """Create PCA scatter plot."""
    logger.info("Creating PCA scatter plot...")
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    pc1 = source_data['PC1'].values
    pc2 = source_data['PC2'].values
    exon_lengths = source_data['exon_length'].values
    
    # Main scatter plot
    scatter = ax.scatter(pc1, pc2, c=exon_lengths, cmap=COLORMAP, 
                        s=MARKER_SIZE, alpha=MARKER_ALPHA, 
                        norm=LogNorm(vmin=exon_lengths.min(), vmax=exon_lengths.max()),
                        edgecolors='none', zorder=2)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Exon length (nt)', fontsize=14, fontweight='bold')
    
    # Find and plot extreme points
    pc1_min_idx = np.argmin(pc1)
    pc1_max_idx = np.argmax(pc1)
    pc2_min_idx = np.argmin(pc2)
    pc2_max_idx = np.argmax(pc2)
    
    extreme_indices = {pc1_min_idx, pc1_max_idx, pc2_min_idx, pc2_max_idx}
    
    for idx in extreme_indices:
        ax.scatter(pc1[idx], pc2[idx], s=MARKER_SIZE*3, 
                  facecolors='none', edgecolors=EXTREME_COLOR, 
                  linewidths=2.5, zorder=4)
        gene_name = source_data.iloc[idx]['gene_name']
        exon_num = int(source_data.iloc[idx]['exon_idx'])
        label = f"{gene_name} ex{exon_num}"
        ax.annotate(label, 
                   xy=(pc1[idx], pc2[idx]),
                   xytext=(0, 0), textcoords='offset points',
                   fontsize=10, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                            edgecolor=EXTREME_COLOR, alpha=0.9, linewidth=1.5),
                   zorder=5)
    
    # Highlight specified genes
    if HIGHLIGHT_GENES:
        for gene_name, transcript_id, exon_idx in HIGHLIGHT_GENES:
            mask = ((source_data['gene_name'] == gene_name) & 
                   (source_data['transcript_id'] == transcript_id) &
                   (source_data['exon_idx'] == exon_idx))
            
            if mask.any():
                idx = np.where(mask)[0][0]
                ax.scatter(pc1[idx], pc2[idx], s=MARKER_SIZE*3, 
                          facecolors='none', edgecolors=HIGHLIGHT_COLOR, 
                          linewidths=2.5, zorder=4)
                label = f"{gene_name} ex{exon_idx}"
                ax.annotate(label, 
                           xy=(pc1[idx], pc2[idx]),
                           xytext=(0, 0), textcoords='offset points',
                           fontsize=10, fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                    edgecolor=HIGHLIGHT_COLOR, alpha=0.9, linewidth=1.5),
                           zorder=5)
    
    ax.set_xlabel(f'PC1 ({variance_ratios[0]*100:.1f}% variance)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel(f'PC2 ({variance_ratios[1]*100:.1f}% variance)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('PCA of long exon NMD efficiency curves', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = True,
):
    """Generate PCA scatter plot figure.

    Args:
        figure_label: Panel label when called from the manuscript app.
        figure_number: Figure number when called from the manuscript app.
        regenerate: If False and source data exists, skip processing.
    """
    paths = get_paths(
        script_name=SCRIPT_NAME,
        figure_label=figure_label,
        figure_number=figure_number,
    )
    
    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        source_data = pd.read_csv(paths.source_data)
        # Variance ratios stored in last row as metadata
        variance_ratios = [source_data.iloc[-1]['PC1'], source_data.iloc[-1]['PC2']]
        source_data = source_data.iloc[:-1].copy()
    else:
        logger.info("Loading PCA results...")
        source_data, variance_ratios = process_data()
        
        # Save source data with variance ratios as metadata row
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        metadata_row = pd.DataFrame({
            'gene_name': ['VARIANCE_RATIOS'],
            'transcript_id': [''],
            'exon_idx': [0],
            'exon_length': [0],
            'PC1': [variance_ratios[0]],
            'PC2': [variance_ratios[1]]
        })
        combined_data = pd.concat([source_data, metadata_row], ignore_index=True)
        combined_data.to_csv(paths.source_data, index=False)
        logger.info(f"Source data saved to {paths.source_data}")
    
    # Create and save figure
    fig = plot_pca_scatter(source_data, variance_ratios)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)
    
    logger.success("PCA scatter plot complete!")


if __name__ == "__main__":
    main()
