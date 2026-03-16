"""
Principal component curves for long exon NMD efficiency.

Shows how the first four principal components vary across 
relative position within exons, representing different curve types.

Note: This script uses pre-computed PCA results from NMD.analysis.long_exon_pca_analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from NMD.config import TABLES_DIR
from NMD.manuscript.output import get_paths


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_NAME = "long_exon_pca_curves"
PCA_COMPONENTS_FILE = TABLES_DIR / "exon_analysis" / "long_exon_pca_components.csv"

# PCA parameters
N_PCS_TO_SHOW = 2

# Figure settings
FIGURE_SIZE = (6, 4)
DPI = 300


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_data():
    """Load pre-computed PCA components."""
    logger.info(f"Loading PCA components from {PCA_COMPONENTS_FILE}")
    
    if not PCA_COMPONENTS_FILE.exists():
        raise FileNotFoundError(
            f"PCA components not found at {PCA_COMPONENTS_FILE}. "
            "Run NMD.analysis.long_exon_pca_analysis first."
        )
    
    # Load PC components
    components_df = pd.read_csv(PCA_COMPONENTS_FILE)
    
    # Separate data from metadata (variance ratios in last row with relative_position == -1)
    variance_row = components_df[components_df['relative_position'] == -1]
    source_data = components_df[components_df['relative_position'] != -1].copy()
    
    # Add variance explained columns from metadata row
    if len(variance_row) > 0:
        variance_row = variance_row.iloc[0]
        for i in range(N_PCS_TO_SHOW):
            source_data[f'PC{i+1}_variance_explained'] = variance_row[f'PC{i+1}_loading']
    
    logger.info(f"Loaded PC loadings for {len(source_data)} positions")
    
    return source_data


# ============================================================================
# PLOTTING
# ============================================================================

def plot_pc_curves(source_data):
    """Create principal component curves plot."""
    logger.info("Creating PC curves plot...")
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    x = source_data['relative_position'].values
    colours = ['#ff9e9d', '#022778']
    
    for i in range(N_PCS_TO_SHOW):
        pc_curve = source_data[f'PC{i+1}_loading'].values
        variance = source_data[f'PC{i+1}_variance_explained'].iloc[0]
        color = colours[i]
        
        ax.plot(x, pc_curve, linewidth=2.5, color=color, 
               label=f'PC{i+1} ({variance*100:.1f}%)', alpha=0.85)
    
    ax.set_xlabel('Relative position in exon', fontsize=14, fontweight='bold')
    ax.set_ylabel('Component loading', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='center left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Curve type across principal components', 
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
    """Generate PC curves figure.

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
    else:
        logger.info("Loading PCA results...")
        source_data = process_data()
        
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        source_data.to_csv(paths.source_data, index=False)
        logger.info(f"Source data saved to {paths.source_data}")
    
    # Create and save figure
    fig = plot_pc_curves(source_data)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)
    
    logger.success("PC curves plot complete!")


if __name__ == "__main__":
    main()
