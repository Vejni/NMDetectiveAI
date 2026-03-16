"""
Spearman correlations between exon features and principal components.

Shows how exon length, exon number, and linear fit R2 
correlate with the first four principal components of NMD efficiency curves.

Note: This script uses pre-computed PCA results from NMD.analysis.long_exon_pca_analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

from NMD.config import TABLES_DIR, COLOURS
from NMD.manuscript.output import get_paths


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_NAME = "long_exon_pca_correlations"
PCA_SCORES_FILE = TABLES_DIR / "exon_analysis" / "long_exon_pca_scores.csv"

# PCA parameters
N_PCS = 2

# Figure settings
FIGURE_SIZE = (6, 4)
DPI = 300


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_data():
    """Load pre-computed PCA scores and calculate correlations with exon features."""
    logger.info(f"Loading PCA scores from {PCA_SCORES_FILE}")
    
    if not PCA_SCORES_FILE.exists():
        raise FileNotFoundError(
            f"PCA scores not found at {PCA_SCORES_FILE}. "
            "Run NMD.analysis.long_exon_pca_analysis first."
        )
    
    scores_df = pd.read_csv(PCA_SCORES_FILE)
    logger.info(f"Loaded PC scores for {len(scores_df)} exons")
    
    # Calculate relative squared error (RMSE normalized by range)
    # Since predictions are roughly 0-1, we normalize by sqrt(variance) for interpretability
    scores_df['linear_rse'] = np.sqrt(scores_df['linear_mse'])
    
    # Calculate Spearman correlations
    features = ['exon_length', 'exon_idx', 'linear_r2', 'linear_rse']
    feature_labels = ['Exon length', 'Exon number', 'Linear fit R²', 'Linear fit RSE']
    pcs = [f'PC{i+1}' for i in range(N_PCS)]
    
    correlations = []
    for feature, label in zip(features, feature_labels):
        for pc in pcs:
            corr, pval = spearmanr(scores_df[feature], scores_df[pc])
            correlations.append({
                'feature': label,
                'PC': pc,
                'correlation': corr,
                'p_value': pval
            })
    
    source_data = pd.DataFrame(correlations)
    return source_data


# ============================================================================
# PLOTTING
# ============================================================================

def plot_correlations(source_data):
    """Create correlation barplot."""
    logger.info("Creating PC correlations plot...")
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    pcs = [f'PC{i+1}' for i in range(N_PCS)]
    features = ['Exon length', 'Exon number', 'Linear fit R²', 'Linear fit RSE']
    x_positions = np.arange(len(pcs))
    width = 0.2  # Width for 4 bars
    
    # Get correlations for each feature
    colors = [COLOURS[0], COLOURS[1], COLOURS[2], COLOURS[3]] if len(COLOURS) > 3 else ['C0', 'C1', 'C2', 'C3']
    
    for i, (feature, color) in enumerate(zip(features, colors)):
        corrs = source_data[source_data['feature'] == feature]['correlation'].values
        offset = (i - 1.5) * width  # Center the bars
        bars = ax.bar(x_positions + offset, corrs, width, 
                      label=feature, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Principal Component', fontsize=14, fontweight='bold')
    ax.set_ylabel('Spearman correlation', fontsize=14, fontweight='bold')
    ax.set_title('Correlation of exon features with PCs', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pcs)
    ax.legend(fontsize=11, loc='best')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    
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
    """Generate PC correlations figure.

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
        logger.info("Loading PCA results and computing correlations...")
        source_data = process_data()
        
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        source_data.to_csv(paths.source_data, index=False)
        logger.info(f"Source data saved to {paths.source_data}")
    
    # Create and save figure
    fig = plot_correlations(source_data)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)
    
    logger.success("PC correlations plot complete!")


if __name__ == "__main__":
    main()
