"""
Extreme examples from PCA analysis of long exon NMD efficiency curves.

Shows NMD efficiency curves for exons at the extremes of PC1 and PC2,
illustrating the different curve types captured by the principal components.

Note: This script uses pre-computed PCA results from NMD.analysis.long_exon_pca_analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

from NMD.config import TABLES_DIR
from NMD.manuscript.output import get_paths


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_NAME = "long_exon_pca_examples"
PCA_SCORES_FILE = TABLES_DIR / "exon_analysis" / "long_exon_pca_scores.csv"
LE_DIR = TABLES_DIR / "LE"

# Figure settings
FIGURE_SIZE = (18, 5)
DPI = 300
MIN_COLORS = ['#ff9e9d', '#ff6b6b', '#cc0000']  # Light to dark red shades
MAX_COLORS = ['#022778', '#0044aa', '#0066cc']  # Dark to light blue shades
GAUSSIAN_SIGMA = 1
N_EXTREMES = 3  # Number of extreme examples per direction

# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_exon_data(gene_name, transcript_id, exon_idx):
    """Load the actual data for a specific exon."""
    exon_file = LE_DIR / f"{gene_name}_{transcript_id}_exon{exon_idx}.csv"
    if not exon_file.exists():
        raise FileNotFoundError(f"Exon data not found: {exon_file}")
    return pd.read_csv(exon_file)


def process_data():
    """Load pre-computed PCA scores and find extreme examples."""
    logger.info(f"Loading PCA scores from {PCA_SCORES_FILE}")
    
    if not PCA_SCORES_FILE.exists():
        raise FileNotFoundError(
            f"PCA scores not found at {PCA_SCORES_FILE}. "
            "Run NMD.analysis.long_exon_pca_analysis first."
        )
    
    scores_df = pd.read_csv(PCA_SCORES_FILE)
    logger.info(f"Loaded PC scores for {len(scores_df)} exons")
    
    # Find extreme points
    pc1 = scores_df['PC1'].values
    pc2 = scores_df['PC2'].values
    
    # Get indices of N most extreme values
    pc1_min_indices = np.argsort(pc1)[:N_EXTREMES]
    pc1_max_indices = np.argsort(pc1)[-N_EXTREMES:]
    pc2_min_indices = np.argsort(pc2)[:N_EXTREMES]
    pc2_max_indices = np.argsort(pc2)[-N_EXTREMES:]
    
    extreme_indices = pc1_min_indices.tolist() + pc1_max_indices.tolist() + pc2_min_indices.tolist() + pc2_max_indices.tolist()
    extreme_types = (['PC1 min'] * N_EXTREMES + ['PC1 max'] * N_EXTREMES + 
                     ['PC2 min'] * N_EXTREMES + ['PC2 max'] * N_EXTREMES)
    
    # Collect extreme example data
    extreme_data = {}
    
    for idx, extreme_type in zip(extreme_indices, extreme_types):
        row = scores_df.iloc[idx]
        gene_name = row['gene_name']
        transcript_id = row['transcript_id']
        exon_idx = int(row['exon_idx'])
        exon_length = row['exon_length']
        
        try:
            exon_data = load_exon_data(gene_name, transcript_id, exon_idx)
            x = exon_data['ptc_position'].values
            y = exon_data['prediction'].values
            
            # Apply gaussian smoothing
            y_smooth = gaussian_filter1d(y, sigma=GAUSSIAN_SIGMA)
            
            # Normalize x-axis to 0-1 based on actual data range
            x_norm = (x - x.min()) / (x.max() - x.min())
            
            df_entry = pd.DataFrame({
                'gene_name': [gene_name],
                'transcript_id': [transcript_id],
                'exon_idx': [exon_idx],
                'exon_length': [exon_length],
                'ptc_position_normalized': [x_norm],
                'NMD_efficiency': [y_smooth]
            })
            
            if extreme_type not in extreme_data:
                extreme_data[extreme_type] = []
            extreme_data[extreme_type].append(df_entry)
            
            logger.info(f"Found {extreme_type}: {gene_name} exon {exon_idx}")
            
        except Exception as e:
            logger.warning(f"Could not load extreme example {extreme_type}: {e}")
    
    # Convert lists to DataFrames
    for key in extreme_data:
        extreme_data[key] = pd.concat(extreme_data[key], ignore_index=True)
    
    return extreme_data


# ============================================================================
# PLOTTING
# ============================================================================

def plot_extreme_examples(extreme_data):
    """Create 1x2 grid showing 3 min and 3 max examples for PC1 and PC2."""
    logger.info("Creating extreme examples plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    pcs = ['PC1', 'PC2']
    
    # First pass: plot data and collect global y-limits
    global_ymin = np.inf
    global_ymax = -np.inf
    
    for col, pc in enumerate(pcs):
        ax = axes[col]
        
        # Collect all data for this PC
        all_data = []
        for i in range(1, N_EXTREMES + 1):
            min_type = f'{pc} min'
            max_type = f'{pc} max'
            
            if min_type in extreme_data:
                df_min = extreme_data[min_type]
                if len(df_min) >= i:
                    all_data.append((df_min.iloc[i-1], 'min', i))
            
            if max_type in extreme_data:
                df_max = extreme_data[max_type]
                if len(df_max) >= i:
                    all_data.append((df_max.iloc[i-1], 'max', i))
        
        # Plot each example with distinct colors per rank
        for row_data, direction, rank in all_data:
            x = row_data['ptc_position_normalized']
            y = row_data['NMD_efficiency']
            gene = row_data['gene_name']
            exon = row_data['exon_idx']
            length = row_data['exon_length']
            
            # Select color based on direction and rank (1-indexed)
            if direction == 'min':
                color = MIN_COLORS[rank - 1]
            else:
                color = MAX_COLORS[rank - 1]
            
            ax.scatter(x, y, color=color, s=15, alpha=0.6, zorder=2)
            ax.plot(x, y, color=color, linewidth=2, alpha=0.9, zorder=3,
                   label=f'{gene} ex{exon} ({int(length)} nt)')
            
            global_ymin = min(global_ymin, y.min())
            global_ymax = max(global_ymax, y.max())
        
        ax.set_xlabel('Normalized PTC position', fontsize=14, fontweight='bold')
        if col == 0:
            ax.set_ylabel('NMD efficiency', fontsize=14, fontweight='bold')
        ax.set_title(pc, fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
        ax.set_xlim(0, 1)
    
    fig.suptitle('Examples of extreme curve types captured by PCs', fontsize=18, fontweight='bold')
    
    # Second pass: apply consistent y-limits to all plots
    for ax in axes:
        ax.set_ylim(global_ymin, global_ymax)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
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
    """Generate extreme examples figure.

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
    
    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        extreme_data = {}
        with pd.ExcelFile(paths.source_data) as xls:
            for sheet_name in xls.sheet_names:
                extreme_data[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
    else:
        logger.info("Loading PCA results and finding extreme examples...")
        extreme_data = process_data()
        
        # Save source data as xlsx with one sheet per extreme
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(paths.source_data, engine='openpyxl') as writer:
            for extreme_type, df in extreme_data.items():
                sheet_name = extreme_type.replace(' ', '_')
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info(f"Source data saved to {paths.source_data}")
    
    # Create and save figure
    fig = plot_extreme_examples(extreme_data)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)
    
    logger.success("Extreme examples plot complete!")


if __name__ == "__main__":
    main()
