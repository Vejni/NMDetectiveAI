#!/usr/bin/env python3
"""
Manuscript figure Fig4d: Penultimate exon prediction curves

Shows confidence interval over all penultimate exon predictions from -200nt to 0
from last exon junction, with fitted logistic curves for 3 highlighted genes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path
from genome_kit import Genome
from scipy.optimize import curve_fit
from tqdm import tqdm

from NMD.config import (
    PROJ_ROOT,
    TABLES_DIR,
    COLOURS,
    CONTRASTING_3_COLOURS,
    GENCODE_VERSION,
)
from NMD.manuscript.output import get_paths

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input paths
GW_PREDICTIONS_DIR = TABLES_DIR / "GW"
SUPPL_TABLE = PROJ_ROOT / "manuscript" / "supplementary" / "tables" / "penultimate_exon_fits.csv"

# Highlighted genes to show fitted curves
HIGHLIGHTED_GENES = ["PTEN", "TP53", "BRCA2"]
GENE_COLORS = {
    "PTEN": CONTRASTING_3_COLOURS[2],  # Dark blue
    "TP53": COLOURS[1],  # Yellow/gold
    "BRCA2": COLOURS[0],  # Orange
}

# Analysis window
PE_WINDOW_START = -199  # Start of window (nt from junction)
PE_WINDOW_END = -1  # End of window (at junction)

# Plot parameters
FIGURE_SIZE = (10, 5)
DPI = 300
CI_LEVEL = 0.95  # 95% confidence interval
CI_COLOR = 'lightgray'
CI_ALPHA = 0.5
MEDIAN_COLOR = 'black'
MEDIAN_LINEWIDTH = 2
CURVE_LINEWIDTH = 2.5
JUNCTION_COLOR = COLOURS[2]  # Pink/red
RULE_55NT_COLOR = COLOURS[2]  # Same as junction

# ============================================================================
# FUNCTIONS
# ============================================================================

def four_param_logistic(x, a, d, c, b):
    """
    4-parameter logistic (4PL) function.
    
    Args:
        x: independent variable (positions)
        a: minimum asymptote
        d: maximum asymptote
        c: inflection point (EC50)
        b: slope
    
    Returns:
        y values for x
    """
    x = np.array(x, dtype=float)
    c = float(c) if c != 0 else 1e-8
    return d + (a - d) / (1.0 + (x / c) ** b)


def load_all_pe_predictions():
    """
    Load all penultimate exon predictions from genome-wide analysis.
    
    Returns:
        dict mapping position (relative to junction) to list of predictions
    """
    logger.info("Loading all penultimate exon predictions...")
    
    # Initialize genome
    genome = Genome(GENCODE_VERSION)
    
    # Find all prediction files
    prediction_files = list(GW_PREDICTIONS_DIR.glob("*_ptc_predictions.csv"))
    logger.info(f"Found {len(prediction_files)} prediction files")
    
    # Dictionary to store predictions by position
    predictions_by_position = {}
    
    for pred_file in tqdm(prediction_files, desc="Loading predictions"):
        try:
            # Load predictions
            df_pred = pd.read_csv(pred_file)
            
            if len(df_pred) == 0:
                continue
            
            transcript_id = df_pred['transcript_id'].iloc[0]
            
            # Get transcript
            try:
                transcript = genome.transcripts[transcript_id]
            except KeyError:
                transcript_id_base = transcript_id.split('.')[0]
                try:
                    transcript = genome.transcripts[transcript_id_base]
                except KeyError:
                    continue
            
            # Get penultimate exon region
            pe_info = get_penultimate_exon_region(transcript)
            if pe_info is None:
                continue
            
            penultimate_junction, pe_start, pe_end = pe_info
            
            # Filter predictions to window of interest
            positions_rel = df_pred['ptc_position'].values - penultimate_junction
            mask = (positions_rel >= PE_WINDOW_START) & (positions_rel <= PE_WINDOW_END)
            
            if not mask.any():
                continue
            
            # Add predictions to dictionary
            for pos_rel, pred in zip(positions_rel[mask], df_pred['prediction'].values[mask]):
                pos_rel = int(pos_rel)
                if pos_rel not in predictions_by_position:
                    predictions_by_position[pos_rel] = []
                predictions_by_position[pos_rel].append(pred)
        
        except Exception as e:
            logger.debug(f"Error processing {pred_file.name}: {e}")
            continue
    
    logger.info(f"Loaded predictions for {len(predictions_by_position)} unique positions")
    return predictions_by_position


def get_penultimate_exon_region(transcript):
    """
    Get the penultimate exon region boundaries in transcript coordinates.
    
    Args:
        transcript: genome_kit Transcript object
    
    Returns:
        tuple: (penultimate_junction, pe_region_start, pe_region_end)
               or None if transcript doesn't have sufficient exons
    """
    # Check if transcript has CDS
    if not transcript.cdss or len(transcript.cdss) == 0:
        return None
    
    # Get 5'UTR length
    utr5_length = sum(len(exon) for exon in transcript.utr5s) if transcript.utr5s else 0
    
    # Get exon boundaries in transcript coordinates
    cumulative = 0
    exon_boundaries = []
    for exon in transcript.exons:
        cumulative += len(exon)
        if cumulative > utr5_length:  # Only count exons with CDS
            exon_boundaries.append(cumulative)
    
    # Need at least 3 exons with CDS
    if len(exon_boundaries) < 3:
        return None
    
    # Penultimate junction is the start of the last exon
    penultimate_junction = exon_boundaries[-2] + 1
    
    # Return junction and full region (will filter by window later)
    return penultimate_junction, utr5_length + 1, penultimate_junction - 1


def calculate_ci_bounds(predictions_by_position, ci_level=0.95, position_step=3):
    """
    Calculate confidence interval bounds across all predictions with smoothing.
    
    Args:
        predictions_by_position: dict mapping position to list of predictions
        ci_level: Confidence level (default 0.95)
        position_step: Step size for regular grid (default 3 to match codon spacing)
    
    Returns:
        DataFrame with position, median, lower_ci, upper_ci columns
    """
    # Get all available positions
    all_positions = sorted(predictions_by_position.keys())
    if len(all_positions) == 0:
        return pd.DataFrame(columns=['position', 'median', 'lower_ci', 'upper_ci'])
    
    # Create regular grid from min to max position
    min_pos = min(all_positions)
    max_pos = max(all_positions)
    grid_positions = list(range(min_pos, max_pos + 1, position_step))
    
    medians = []
    lower_bounds = []
    upper_bounds = []
    
    alpha = 1 - ci_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # For each grid position, collect predictions from nearby positions
    smoothing_window = position_step  # Window size for collecting nearby predictions
    
    for grid_pos in grid_positions:
        # Collect predictions from positions within the smoothing window
        nearby_predictions = []
        for actual_pos, preds in predictions_by_position.items():
            if abs(actual_pos - grid_pos) <= smoothing_window / 2:
                nearby_predictions.extend(preds)
        
        if len(nearby_predictions) > 0:
            preds_array = np.array(nearby_predictions)
            medians.append(np.median(preds_array))
            lower_bounds.append(np.percentile(preds_array, lower_percentile))
            upper_bounds.append(np.percentile(preds_array, upper_percentile))
        else:
            # If no predictions nearby, use NaN (will be interpolated)
            medians.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
    
    # Create DataFrame
    df = pd.DataFrame({
        'position': grid_positions,
        'median': medians,
        'lower_ci': lower_bounds,
        'upper_ci': upper_bounds
    })
    
    # Interpolate missing values to create smooth curves
    df = df.interpolate(method='linear', limit_direction='both')
    
    # Remove any remaining NaN values at edges
    df = df.dropna()
    
    return df


def load_gene_fit_params(gene_name):
    """
    Load fitted 4PL parameters for a specific gene.
    
    Args:
        gene_name: Name of the gene
    
    Returns:
        dict with fitted parameters or None if not found
    """
    if not SUPPL_TABLE.exists():
        logger.warning(f"Fit data not found: {SUPPL_TABLE}")
        return None
    
    df = pd.read_csv(SUPPL_TABLE)
    gene_data = df[df['gene_name'] == gene_name]
    
    if len(gene_data) == 0:
        logger.warning(f"Gene {gene_name} not found in fit data")
        return None
    
    row = gene_data.iloc[0]
    
    # Get the x_min and x_max to denormalize the inflection point
    x_min = row['x_min']
    x_max = row['x_max']
    
    return {
        'a': row['a'],
        'd': row['d'],
        'c': row['c'],  # This is normalized c
        'c_absolute': row.get('c_absolute', row['c']),  # Use absolute if available, otherwise normalized
        'b': row['b'],
        'r2': row['r2'],
        'x_min': x_min,
        'x_max': x_max
    }


def plot_pe_curves_with_ci(df_ci, highlighted_genes):
    """
    Plot confidence interval and highlighted gene curves.
    
    Args:
        df_ci: DataFrame with CI bounds
        highlighted_genes: List of gene names to plot
    
    Returns:
        matplotlib figure and dict with data for export
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Plot CI band
    ax.fill_between(
        df_ci['position'],
        df_ci['lower_ci'],
        df_ci['upper_ci'],
        color=CI_COLOR,
        alpha=CI_ALPHA,
        label=f'{int(CI_LEVEL*100)}% CI (all genes)'
    )
    
    # Plot median line
    ax.plot(
        df_ci['position'],
        df_ci['median'],
        color=MEDIAN_COLOR,
        linewidth=MEDIAN_LINEWIDTH,
        linestyle='--',
        label='Median',
        zorder=3
    )
    
    # Plot fitted curves for highlighted genes
    gene_curves = {}
    for gene in highlighted_genes:
        params = load_gene_fit_params(gene)
        
        if params is None:
            continue
        
        # Generate curve from fitted parameters
        # Need to reconstruct the positions based on x_min and x_max
        x_min = params['x_min']
        x_max = params['x_max']
        x_range = x_max - x_min
        
        # Create x values in the original coordinate space
        x_positions = np.linspace(x_min, x_max, 200)
        
        # Normalize x for the 4PL function (as was done during fitting)
        x_norm = (x_positions - x_min) / x_range
        
        # Calculate y values
        y_values = four_param_logistic(
            x_norm,
            params['a'],
            params['d'],
            params['c'],
            params['b']
        )
        
        # Plot
        color = GENE_COLORS.get(gene, COLOURS[0])
        ax.plot(
            x_positions,
            y_values,
            color=color,
            linewidth=CURVE_LINEWIDTH,
            label=f'{gene} (MAX={params["a"]:.2f}, MIN={params["d"]:.2f}, I={params["c_absolute"]:.0f}nt, ER={params["b"]:.1f})',
            zorder=4
        )
        
        # Store for export
        gene_curves[gene] = pd.DataFrame({
            'position': x_positions,
            'prediction': y_values
        })
    
    # Add vertical lines for junction and 55nt rule
    ax.axvline(0, color=JUNCTION_COLOR, linestyle='--', linewidth=2, 
               alpha=0.8, label='Last exon junction', zorder=2)
    ax.axvline(-55, color=RULE_55NT_COLOR, linestyle=':', linewidth=2,
               alpha=0.8, label='55 nt rule', zorder=2)
    
    # Labels and formatting
    ax.set_xlabel('Distance from last exon junction (nt)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Predicted NMD efficiency', fontsize=16, fontweight='bold')
    ax.set_xlim(PE_WINDOW_START, PE_WINDOW_END)
    ax.set_ylim(-1, 1)
    ax.legend(loc='lower left', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # increase font size for ticks
    ax.tick_params(labelsize=14)

    plt.title('Predicted penultimate exon NMD evasion curves genomewide', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    # Prepare export data
    export_data = {
        'ci_bounds': df_ci,
        **{f'{gene}_curve': curve for gene, curve in gene_curves.items()}
    }
    
    return fig, export_data


def main(
    figure_label: str = "d",
    figure_number: str = "Fig4",
    regenerate: bool = False,
):
    """
    Main function to generate Fig4d.
    
    Args:
        figure_label: Panel label (default: "d")
        figure_number: Figure number (default: "Fig4")
        regenerate: Force regeneration of source data
    """
    # Get output paths
    paths = get_paths(
        script_name="penultimate_exon_curves",
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
    
    # Load all PE predictions and calculate CI
    logger.info("Loading penultimate exon predictions...")
    predictions_by_position = load_all_pe_predictions()
    
    logger.info("Calculating confidence intervals...")
    df_ci = calculate_ci_bounds(predictions_by_position, ci_level=CI_LEVEL, position_step=3)
    
    # Generate plot
    logger.info("Creating PE curves plot...")
    fig, data = plot_pe_curves_with_ci(df_ci, HIGHLIGHTED_GENES)
    
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
        for sheet_name, df in data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    logger.info(f"Source data saved: {paths.source_data}")
    logger.success(f"Fig{figure_number}{figure_label} completed successfully!")


if __name__ == "__main__":
    main()
