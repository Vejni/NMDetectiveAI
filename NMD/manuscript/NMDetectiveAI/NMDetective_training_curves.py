"""
Training curves plot for NMDetectiveAI model.

This script generates training and validation loss curves, along with
validation correlation curves for three model variants:
- NMDetectiveAI (astral-sweep-11)
- PTC_LP (pretrained low parameters)
- PTC_from_random (random initialization)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from loguru import logger

from NMD.config import (
    INTERIM_DATA_DIR,
    CONTRASTING_3_COLOURS,
)
from NMD.manuscript.output import get_paths

# ============================================================================
# CONFIGURATION - Define all paths and parameters here
# ============================================================================

# Script identity (used for default standalone output filenames)
SCRIPT_NAME = "NMDetective_training_curves"

# Data paths
DATA_FILE = INTERIM_DATA_DIR / "training" / "wandb_export_2026-01-19T14_58_48.290+01_00.csv"

# Smoothing parameter (sigma for gaussian filter)
SMOOTHING_SIGMA = 1.0  # Adjust to control smoothness of curves

# Plot aesthetics
COLORS = {
    'nmdetective_ai': CONTRASTING_3_COLOURS[2],  # '#022778' - dark blue
    'ptc_lp': CONTRASTING_3_COLOURS[0],  # '#ff9e9d' - pink/red
    'ptc_random': CONTRASTING_3_COLOURS[1],  # '#2d8b4d' - green
}

ALPHA = 0.8
LINE_WIDTH = 2.5
FIGURE_SIZE = (8, 5)
DPI = 300

# Font sizes
AXIS_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 16
ANNOTATION_FONTSIZE = 16
TICK_FONTSIZE = 14
PLOT_TITLE = "Generalization performance during training"
PLOT_TITLE_FONTSIZE = 18

# Labels for legend
LABELS = {
    'nmdetective_ai': 'NMDetectiveAI',
    'ptc_lp': 'Orthrus probing only',
    'ptc_random': 'Mamba no pretraining'
}

# Training parameters
MAX_STEP = 60  # Cut graph at step 100
EARLY_STOPPING_STEP = 42  # Step where early stopping occurred


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_data():
    """Load and process training data."""
    
    logger.info(f"Loading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    logger.info(f"Loaded {len(df)} training steps")
    
    # Filter to MAX_STEP
    df = df[df['Step'] <= MAX_STEP]
    logger.info(f"Filtered to {len(df)} steps (up to step {MAX_STEP})")
    
    # Extract relevant columns and rename for clarity
    df_processed = pd.DataFrame({
        'step': df['Step'],
        # NMDetectiveAI (astral-sweep-11)
        'nmdetective_ai_val_correlation': df['astral-sweep-11 - val_correlation'],
        # PTC_LP
        'ptc_lp_val_correlation': df['PTC_LP - val_correlation'],
        # PTC_from_random
        'ptc_random_val_correlation': df['PTC_from_random - val_correlation'],
    })
    
    # Check if we should use smoothed columns (MIN/MAX)
    use_smoothed = False
    if 'astral-sweep-11 - val_correlation__MIN' in df.columns:
        logger.info("Detected MIN/MAX columns for smoothing")
        if not df['astral-sweep-11 - val_correlation'].equals(df['astral-sweep-11 - val_correlation__MIN']):
            use_smoothed = True
            logger.info("Using pre-smoothed MIN/MAX columns")
            df_processed['nmdetective_ai_val_correlation_smooth'] = df['astral-sweep-11 - val_correlation__MIN']
            df_processed['ptc_lp_val_correlation_smooth'] = df['PTC_LP - val_correlation__MIN']
            df_processed['ptc_random_val_correlation_smooth'] = df['PTC_from_random - val_correlation__MIN']

    if not use_smoothed:
        logger.info(f"Applying Gaussian smoothing with sigma={SMOOTHING_SIGMA}")
        for col, smooth_col in [
            ('nmdetective_ai_val_correlation', 'nmdetective_ai_val_correlation_smooth'),
            ('ptc_lp_val_correlation', 'ptc_lp_val_correlation_smooth'),
            ('ptc_random_val_correlation', 'ptc_random_val_correlation_smooth'),
        ]:
            if df_processed[col].notna().sum() > 0:
                mask = df_processed[col].notna()
                smoothed = df_processed[col].copy()
                if mask.sum() > SMOOTHING_SIGMA:
                    smoothed[mask] = gaussian_filter1d(df_processed.loc[mask, col], sigma=SMOOTHING_SIGMA)
                df_processed[smooth_col] = smoothed
            else:
                df_processed[smooth_col] = df_processed[col]
    
    logger.info(f"Data processing complete. {len(df_processed)} steps ready for plotting")

    # select only necessary columns for plotting
    df_processed = df_processed[[
        'step',
        'nmdetective_ai_val_correlation_smooth',
        'ptc_lp_val_correlation_smooth',
        'ptc_random_val_correlation_smooth'
    ]]
    
    return df_processed


# ============================================================================
# PLOTTING
# ============================================================================

def plot_from_table(df_plot):
    """Generate training curves plot from processed data."""
    
    logger.info("Creating training curves plot...")
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)

    # Plot only validation correlation curves (single y-axis)
    # NMDetectiveAI validation correlation
    mask_ai = ~df_plot['nmdetective_ai_val_correlation_smooth'].isna()
    ax.plot(
        df_plot.loc[mask_ai, 'step'],
        df_plot.loc[mask_ai, 'nmdetective_ai_val_correlation_smooth'],
        color=COLORS['nmdetective_ai'],
        alpha=ALPHA,
        linewidth=LINE_WIDTH,
        label=f'{LABELS["nmdetective_ai"]}',
        linestyle='-'
    )

    # PTC_LP validation correlation
    mask_lp = ~df_plot['ptc_lp_val_correlation_smooth'].isna()
    ax.plot(
        df_plot.loc[mask_lp, 'step'],
        df_plot.loc[mask_lp, 'ptc_lp_val_correlation_smooth'],
        color=COLORS['ptc_lp'],
        alpha=ALPHA,
        linewidth=LINE_WIDTH,
        label=f'{LABELS["ptc_lp"]}',
        linestyle='--'
    )

    # PTC_from_random validation correlation
    mask_random = ~df_plot['ptc_random_val_correlation_smooth'].isna()
    ax.plot(
        df_plot.loc[mask_random, 'step'],
        df_plot.loc[mask_random, 'ptc_random_val_correlation_smooth'],
        color=COLORS['ptc_random'],
        alpha=ALPHA,
        linewidth=LINE_WIDTH,
        label=f'{LABELS["ptc_random"]}',
        linestyle=':'
    )

    # Add vertical line for early stopping
    ax.axvline(x=EARLY_STOPPING_STEP, color='gray', linestyle='-.', linewidth=1.5, alpha=0.7, zorder=1)

    # Add text annotation for early stopping
    y_pos = ax.get_ylim()[1] * 0.85  # Position at 85% of y-axis height
    ax.text(
        EARLY_STOPPING_STEP + 2, y_pos,
        'Early stopping\n(~ 3 epochs)',
        fontsize=ANNOTATION_FONTSIZE,
        color='gray',
        verticalalignment='top'
    )

    # Formatting
    ax.set_xlabel('Training step', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')
    ax.set_ylabel('Validation correlation', fontsize=AXIS_LABEL_FONTSIZE, fontweight='bold')

    ax.legend(loc='right', bbox_to_anchor=(0.6, 0.35), frameon=True, framealpha=0.9, fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0, MAX_STEP)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.set_title(PLOT_TITLE, fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = True,
):
    """Generate the training-curves figure.

    Args:
        figure_label: Panel label (e.g. "Fig2a") when called from the
            manuscript app.  *None* → standalone mode.
        figure_number: Figure number (e.g. "Fig2") when called from the
            manuscript app.
        regenerate: If *False* and source data already exists, skip the
            data-processing step and plot directly from the saved table.
    """
    paths = get_paths(
        script_name=SCRIPT_NAME,
        figure_label=figure_label,
        figure_number=figure_number,
        source_data_ext=".csv",
    )

    logger.info("Starting training curves plot generation")

    # Process or load source data
    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data: {paths.source_data}")
        df_plot = pd.read_csv(paths.source_data)
    else:
        logger.info("Processing data...")
        df_plot = process_data()
        df_plot.to_csv(paths.source_data, index=False)
        logger.info(f"Source data saved to: {paths.source_data}")

    # Generate plots
    fig = plot_from_table(df_plot)

    # Save figures
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    logger.info(f"PNG saved to: {paths.figure_png}")

    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"PDF saved to: {paths.figure_pdf}")

    plt.close()
    logger.success("Done!")


if __name__ == "__main__":
    main()
