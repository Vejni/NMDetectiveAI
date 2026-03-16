"""
Hyperparameter sweep results visualization.

This script generates a line plot showing validation loss across different
hyperparameter configurations, highlighting the selected NMDetectiveAI model.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
INPUT_TABLE = TABLES_DIR / "manuscript" / "supplementary" / "hyperparameter_sweep.csv"
OUTPUT_FIGURE = FIGURES_DIR / "manuscript" / "supplementary" / "hyperparameter_sweep.png"
OUTPUT_FIGURE_PDF = FIGURES_DIR / "manuscript" / "supplementary" / "hyperparameter_sweep.pdf"
PLOT_TITLE = "Hyperparameter search results"

# Plot parameters
SELECTED_ID = 11  # NMDetectiveAI configuration
COLOR_SELECTED = CONTRASTING_2_COLOURS[1]  # Dark blue
COLOR_OTHER = CONTRASTING_2_COLOURS[0]
FIGURE_SIZE = (10, 6)
DPI = 300


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_data():
    """Load and prepare sweep data for plotting."""
    
    logger.info("Starting data processing...")
    
    # Load sweep results
    logger.info(f"Loading sweep results from {INPUT_TABLE}")
    df = pd.read_csv(INPUT_TABLE)
    logger.info(f"Loaded {len(df)} hyperparameter configurations")
    
    return df


# ============================================================================
# PLOTTING
# ============================================================================

def plot_from_table(df):
    """Generate scatter plot with running minimum from processed data."""
    logger.info("Generating plot...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Get column names
    id_col = df.columns[0]  # "Index"
    loss_col = df.columns[1]  # "Best Val Loss"
    
    # Calculate running minimum
    running_min = df[loss_col].cummin()
    
    # Plot all points as scatter
    ax.scatter(df[id_col], df[loss_col], 
               color=COLOR_OTHER, s=60, alpha=0.6, 
               label='Other configurations', zorder=2)
    
    # Plot running minimum line
    ax.plot(df[id_col], running_min,
            color='black', linewidth=2, linestyle='--',
            alpha=0.7, label='Running minimum', zorder=3)
    
    # Highlight selected configuration
    selected_row = df[df[id_col] == SELECTED_ID]
    if not selected_row.empty:
        ax.scatter(selected_row[id_col], selected_row[loss_col],
                   color=COLOR_SELECTED, s=200, marker='o',
                   edgecolors='white', linewidths=2,
                   label='NMDetectiveAI', zorder=5)
        
        # Add annotation with increased y offset
        ax.annotate('NMDetectiveAI',
                   xy=(selected_row[id_col].values[0], selected_row[loss_col].values[0]),
                   xytext=(10, 30), textcoords='offset points',
                   fontsize=14, fontweight='bold',
                   color=COLOR_SELECTED,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor=COLOR_SELECTED, linewidth=2),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                 color=COLOR_SELECTED, linewidth=2))
        
        logger.info(f"Selected configuration (ID={SELECTED_ID}): "
                   f"Val Loss = {selected_row[loss_col].values[0]:.6f}")
    
    # Formatting
    ax.set_xlabel('Configuration ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Validation Loss', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Set x-axis to show integer ticks
    ax.set_xticks(range(0, len(df) + 1, 5))
    
    plt.title(PLOT_TITLE, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
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
    logger.info("Hyperparameter Sweep Visualization")
    logger.info("=" * 80)
    
    # Check if table already exists
    logger.info("Processing data (table not found)...")
    df = process_data()
    
    # Generate plot
    plot_from_table(df)
    
    logger.success("Hyperparameter sweep visualization complete!")


if __name__ == "__main__":
    main()
