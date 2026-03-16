#!/usr/bin/env python3
"""
Supplementary figure: Genome-wide penultimate exon 4PL parameter distributions

Analyzes NMDetective-AI predictions across all MANE transcripts in penultimate exon regions.
Fits 4-parameter logistic curves and visualizes parameter distributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
from genome_kit import Genome
from scipy.optimize import curve_fit
from tqdm import tqdm

from NMD.config import (
    TABLES_DIR,
    FIGURES_DIR,
    GENCODE_VERSION,
    COLOURS,
    PROJ_ROOT,
    CONTRASTING_3_COLOURS
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/output paths
GW_PREDICTIONS_DIR = TABLES_DIR / "GW"
OUTPUT_TABLE = TABLES_DIR / "penultimate_exon_genome_wide.xlsx"
SUPPL_TABLE = PROJ_ROOT / "manuscript" / "tables" / "penultimate_exon_fits.csv"
OUTPUT_FIGURE = FIGURES_DIR / "manuscript" / "supplementary" / "penultimate_exon_4pl_distributions.png"
OUTPUT_FIGURE_PDF = FIGURES_DIR / "manuscript" / "supplementary" / "penultimate_exon_4pl_distributions.pdf"

# Example genes to plot
EXAMPLE_GENES = ["BRCA2", "TP53"]  # Genes to show as examples\n\n# Colors\nMODEL_AI_COLOR = CONTRASTING_3_COLOURS[2]  # '#022778' - dark blue

# Analysis parameters
MIN_EXONS = 3  # Minimum number of coding exons required
MIN_POINTS_FOR_FIT = 10  # Minimum points in penultimate exon to attempt fit
PE_WINDOW_NT = 200  # Nucleotides upstream of last exon junction to analyze

# Plot parameters
FIGURE_SIZE = (14, 10)  # Larger to accommodate example plots
DPI = 300


# ============================================================================
# FUNCTIONS
# ============================================================================

def four_param_logistic(x, a, d, c, b):
    """
    4-parameter logistic (4PL) function.
    y = d + (a - d) / (1 + (x / c)**b)
    
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
    
    # Need at least 3 exons with CDS (so we have a penultimate exon)
    if len(exon_boundaries) < MIN_EXONS:
        return None
    
    # Penultimate junction is the start of the last exon
    penultimate_junction = exon_boundaries[-2] + 1
    
    # Define penultimate exon region: PE_WINDOW_NT nucleotides upstream of junction
    pe_region_start = max(penultimate_junction - PE_WINDOW_NT, utr5_length + 1)
    pe_region_end = penultimate_junction - 1  # Up to but not including junction
    
    return penultimate_junction, pe_region_start, pe_region_end


def fit_4pl_to_penultimate_exon(positions, predictions, penultimate_junction):
    """
    Fit 4-parameter logistic curve to penultimate exon predictions.
    
    Args:
        positions: PTC positions (nucleotide coordinates)
        predictions: NMDetective-AI predictions
        penultimate_junction: Position of the penultimate junction
    
    Returns:
        dict with fitted parameters and goodness of fit, or None if fit fails
    """
    if len(positions) < MIN_POINTS_FOR_FIT:
        return None
    
    try:
        # Normalize x to [0,1] for numerical stability
        x_min = float(np.min(positions))
        x_max = float(np.max(positions))
        x_range = x_max - x_min
        
        if x_range == 0:
            return None
        
        x_norm = (positions - x_min) / x_range
        
        # Initial parameter guesses
        a0 = float(np.min(predictions))
        d0 = float(np.max(predictions))
        c0 = 0.5  # Inflection point in middle
        b0 = 1.0
        
        # Parameter bounds
        a_lower = float(np.min(predictions) - np.abs(np.min(predictions)))
        a_upper = float(np.max(predictions) + np.abs(np.max(predictions)))
        d_lower = a_lower
        d_upper = a_upper
        
        bounds_lower = [a_lower, d_lower, 1e-6, 0.01]
        bounds_upper = [a_upper, d_upper, 1.0, 20.0]
        
        # Fit
        popt, pcov = curve_fit(
            four_param_logistic, x_norm, predictions,
            p0=[a0, d0, c0, b0],
            bounds=(bounds_lower, bounds_upper),
            maxfev=20000
        )
        
        # Calculate R²
        y_pred = four_param_logistic(x_norm, *popt)
        ss_res = np.sum((predictions - y_pred) ** 2)
        ss_tot = np.sum((predictions - np.mean(predictions)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Convert inflection point back to position relative to junction
        # Since we're now fitting on relative coordinates, c_absolute is just the fitted value
        c_relative_to_junction = x_min + popt[2] * x_range
        
        return {
            'a': popt[0],  # min asymptote
            'd': popt[1],  # max asymptote
            'c': popt[2],  # inflection point (normalized)
            'c_absolute': c_relative_to_junction,  # inflection point (nt from junction, negative = upstream)
            'b': popt[3],  # slope
            'r2': r2,
            'n_points': len(positions),
            'x_min': x_min,
            'x_max': x_max
        }
    
    except Exception as e:
        logger.debug(f"4PL fit failed: {e}")
        return None


def process_genome_wide_predictions():
    """
    Process all genome-wide predictions, extract penultimate exon data,
    and fit 4PL curves.
    
    Returns:
        DataFrame with fitted parameters for each transcript
    """
    logger.info("Loading genome...")
    genome = Genome(GENCODE_VERSION)
    
    # Find all prediction files
    prediction_files = list(GW_PREDICTIONS_DIR.glob("*_ptc_predictions.csv"))
    logger.info(f"Found {len(prediction_files)} prediction files")
    
    results = []
    skipped_no_exons = 0
    skipped_no_data = 0
    skipped_fit_failed = 0
    
    for pred_file in tqdm(prediction_files, desc="Processing transcripts"):
        try:
            # Load predictions
            df_pred = pd.read_csv(pred_file)
            
            if len(df_pred) == 0:
                continue
            
            gene_name = df_pred['gene_name'].iloc[0]
            transcript_id = df_pred['transcript_id'].iloc[0]
            
            # Get transcript from genome
            try:
                transcript = genome.transcripts[transcript_id]
            except KeyError:
                # Try without version number
                transcript_id_base = transcript_id.split('.')[0]
                try:
                    transcript = genome.transcripts[transcript_id_base]
                except KeyError:
                    logger.debug(f"Transcript {transcript_id} not found in genome")
                    continue
            
            # Get penultimate exon region
            pe_info = get_penultimate_exon_region(transcript)
            if pe_info is None:
                skipped_no_exons += 1
                continue
            
            penultimate_junction, pe_start, pe_end = pe_info
            
            # Filter predictions to penultimate exon region
            pe_mask = (df_pred['ptc_position'] >= pe_start) & (df_pred['ptc_position'] <= pe_end)
            df_pe = df_pred[pe_mask]
            
            if len(df_pe) < MIN_POINTS_FOR_FIT:
                skipped_no_data += 1
                continue
            
            # Fit 4PL curve
            positions = df_pe['ptc_position'].values
            predictions = df_pe['prediction'].values
            
            # Convert positions to relative coordinates for fitting
            positions_relative = positions - penultimate_junction
            
            fit_result = fit_4pl_to_penultimate_exon(positions_relative, predictions, penultimate_junction)
            
            if fit_result is None:
                skipped_fit_failed += 1
                continue
            
            # Store results
            results.append({
                'gene_name': gene_name,
                'transcript_id': transcript_id,
                'penultimate_junction': penultimate_junction,
                'pe_start': pe_start,
                'pe_end': pe_end,
                'num_exons': df_pred['num_exons'].iloc[0] if 'num_exons' in df_pred.columns else None,
                'cds_length': df_pred['cds_length'].iloc[0] if 'cds_length' in df_pred.columns else None,
                **fit_result
            })
            
        except Exception as e:
            logger.debug(f"Error processing {pred_file.name}: {e}")
            continue
    
    logger.info(f"Successfully fitted {len(results)} transcripts")
    logger.info(f"Skipped: {skipped_no_exons} (insufficient exons), "
                f"{skipped_no_data} (insufficient data), "
                f"{skipped_fit_failed} (fit failed)")
    
    return pd.DataFrame(results)


def load_example_penultimate_data(gene_name, genome, df_full=None):
    """
    Load prediction data for a specific gene's penultimate exon region.
    
    Args:
        gene_name: Name of the gene
        genome: genome_kit Genome object
        df_full: Full DataFrame with fitted parameters (optional)
    
    Returns:
        tuple: (positions, predictions, penultimate_junction, fit_params) or None if not found
    """
    # Find prediction file for this gene
    pred_files = list(GW_PREDICTIONS_DIR.glob(f"{gene_name}_*_ptc_predictions.csv"))
    
    if len(pred_files) == 0:
        logger.warning(f"No prediction file found for {gene_name}")
        return None
    
    # Load first matching file
    df_pred = pd.read_csv(pred_files[0])
    
    if len(df_pred) == 0:
        return None
    
    transcript_id = df_pred['transcript_id'].iloc[0]
    
    # Get transcript
    try:
        transcript = genome.transcripts[transcript_id]
    except KeyError:
        transcript_id_base = transcript_id.split('.')[0]
        try:
            transcript = genome.transcripts[transcript_id_base]
        except KeyError:
            return None
    
    # Get penultimate exon region
    pe_info = get_penultimate_exon_region(transcript)
    if pe_info is None:
        return None
    
    penultimate_junction, pe_start, pe_end = pe_info
    
    # Filter to penultimate exon region
    pe_mask = (df_pred['ptc_position'] >= pe_start) & (df_pred['ptc_position'] <= pe_end)
    df_pe = df_pred[pe_mask]
    
    if len(df_pe) < MIN_POINTS_FOR_FIT:
        return None
    
    # Convert positions to distance from junction (negative = upstream)
    positions_from_junction = df_pe['ptc_position'].values - penultimate_junction
    predictions = df_pe['prediction'].values
    
    # Get fitted parameters if available
    fit_params = None
    if df_full is not None:
        # Find this gene's fitted parameters
        gene_mask = df_full['gene_name'] == gene_name
        if gene_mask.any():
            gene_fit = df_full[gene_mask].iloc[0]
            fit_params = {
                'A': gene_fit['a'],
                'D': gene_fit['d'],
                'C': gene_fit['c_absolute'],   # junction-relative, for display
                'c_norm': gene_fit['c'],        # normalized c used during fitting
                'B': gene_fit['b'],
                'r2': gene_fit['r2'],
                'x_min': gene_fit['x_min'],     # normalization bounds from fitting
                'x_max': gene_fit['x_max'],
            }
    
    return positions_from_junction, predictions, penultimate_junction, fit_params


def plot_parameter_distributions(df, df_full=None):
    """
    Create a 3x2 subplot showing example PE curves and distributions of 4PL parameters.
    
    Args:
        df: DataFrame with fitted parameters for plotting
        df_full: Full DataFrame with all metadata (for finding examples)
    
    Returns:
        matplotlib figure and dict with data for each panel
    """
    fig = plt.figure(figsize=FIGURE_SIZE)
    
    # Create grid: 2 example plots on top row, 4 parameter distributions below
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
    
    # Load genome for example plots
    genome = Genome(GENCODE_VERSION)
    
    # Dictionary to store data for Excel export
    panel_data = {}
    
    # Panel labels (a, b, c, ...)
    panel_labels = iter('abcdef')

    # Top row: Example penultimate exon plots
    for idx, gene_name in enumerate(EXAMPLE_GENES):
        ax = fig.add_subplot(gs[0, idx])
        ax.text(-0.05, 1.05, next(panel_labels), transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
        
        example_data = load_example_penultimate_data(gene_name, genome, df_full)
        
        if example_data is not None:
            positions, predictions, junction, fit_params = example_data
            
            # Plot data
            ax.scatter(positions, predictions, c=COLOURS[1], s=20, alpha=0.6, label='Predictions')
            ax.axvline(0, color=COLOURS[2], linestyle='--', linewidth=1.5, 
                      alpha=0.8, label='Last exon junction')
            ax.axvline(-55, color=COLOURS[2], linestyle=':', linewidth=1.5,
                      alpha=0.8, label='55 nt rule')
            
            # Plot fitted curve if available
            if fit_params is not None:
                # Generate curve points in junction-relative coordinates
                x_curve = np.linspace(positions.min(), positions.max(), 100)
                # Normalize using the exact bounds from fitting
                x_range = fit_params['x_max'] - fit_params['x_min']
                x_norm = (x_curve - fit_params['x_min']) / x_range
                # Apply 4PL with the stored normalized c
                y_curve = four_param_logistic(
                    x_norm, fit_params['A'], fit_params['D'],
                    fit_params['c_norm'], fit_params['B']
                )
                ax.plot(x_curve, y_curve, color=CONTRASTING_3_COLOURS[2], linewidth=2,
                       label=f'4PL fit (R²={fit_params["r2"]:.2f})')
                # Add parameter text
                param_text = (
                    f'A={fit_params["A"]:.2f}\n'
                    f'D={fit_params["D"]:.2f}\n'
                    f'C={fit_params["C"]:.0f}\n'
                    f'B={fit_params["B"]:.2f}'
                )
                ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Distance from last exon junction (nt)', fontsize=10)
            ax.set_ylabel('NMD efficiency', fontsize=10)
            ax.set_title(f'{gene_name} penultimate exon', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            
            # Store data for Excel
            panel_data[f'Example_{gene_name}'] = pd.DataFrame({
                'position_from_junction': positions,
                'nmd_efficiency': predictions
            })
        else:
            ax.text(0.5, 0.5, f'{gene_name}\ndata not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{gene_name} penultimate exon', fontsize=11, fontweight='bold')
    
    # Bottom 2 rows: Parameter distribution histograms (2x2 grid)
    axes = [fig.add_subplot(gs[i+1, j]) for i in range(2) for j in range(2)]
    
    # Parameter names and descriptions
    params = [
        ('a', 'Minimum Asymptote (A)', 'Lower plateau'),
        ('d', 'Maximum Asymptote (D)', 'Upper plateau'),
        ('c_absolute', 'Inflection Point (C)', 'Distance upstream from junction (nt)'),
        ('b', 'Slope (B)', 'Steepness')
    ]
    
    # Filter out extreme outliers for better visualization (keep 99% of data)
    for idx, (param, title, xlabel) in enumerate(params):
        ax = axes[idx]
        ax.text(-0.05, 1.05, next(panel_labels), transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
        
        data = df[param].dropna()
        
        # Remove extreme outliers (beyond 1st and 99th percentile)
        q01 = data.quantile(0.01)
        q99 = data.quantile(0.99)
        data_filtered = data[(data >= q01) & (data <= q99)]
        
        # Plot histogram
        ax.hist(data_filtered, bins=50, color=COLOURS[0], alpha=0.7, edgecolor='black')
        
        # Add median line
        median_val = data_filtered.median()
        ax.axvline(median_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Median: {median_val:.2f}')
        
        # Statistics text
        stats_text = (f'n = {len(data_filtered)}\n'
                     f'Mean: {data_filtered.mean():.2f}\n'
                     f'Std: {data_filtered.std():.2f}')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Store histogram data for Excel
        hist_counts, hist_bins = np.histogram(data_filtered, bins=50)
        bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        panel_data[f'Param_{param}'] = pd.DataFrame({
            'bin_center': bin_centers,
            'count': hist_counts
        })
    
    return fig, panel_data


def main():
    """Main function to process data and generate plots."""
    
    # Create output directories
    SUPPL_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if parameter table already exists
    if False: #OUTPUT_TABLE.exists():
        logger.info(f"Parameter table already exists: {OUTPUT_TABLE}")
        # Load parameter data from Excel (just need the parameter columns for plotting)
        df_plot = pd.read_excel(OUTPUT_TABLE, sheet_name='Param_a')  # Read any sheet to get count
        df_plot = pd.DataFrame()  # Will reconstruct from parameter sheets
        
        # Load each parameter's histogram data (we'll use this to reconstruct distributions)
        # Actually, for plotting we need the raw parameter values, not histograms
        # So we should keep the parameter values separate
        logger.warning("Excel file exists but contains histogram data. Need to reload full data for plotting.")
        
        # Try to load full results for statistics and re-plotting
        if SUPPL_TABLE.exists():
            logger.info("Loading full results for statistics and plotting...")
            df_full = pd.read_csv(SUPPL_TABLE)
            df_plot = df_full[['a', 'd', 'c_absolute', 'b']].copy()
        else:
            logger.error("Full results table not found, cannot regenerate plot")
            return
    else:
        logger.info("Parameter table not found. Processing genome-wide predictions...")
        df_full = process_genome_wide_predictions()
        
        if len(df_full) == 0:
            logger.error("No data to save!")
            return
        
        # Will save OUTPUT_TABLE after plotting (need panel data)
        df_plot = df_full[['a', 'd', 'c_absolute', 'b']].copy()
        
        # Save full results for supplementary table
        df_full.to_csv(SUPPL_TABLE, index=False)
        logger.info(f"Full results saved to: {SUPPL_TABLE}")
    
    # Use df_plot for plotting, df_full for statistics
    df = df_plot
    
    # Generate plots
    logger.info("Generating parameter distribution plots...")
    fig, panel_data = plot_parameter_distributions(df, df_full)
    
    # Save figures
    fig.savefig(OUTPUT_FIGURE, dpi=DPI, bbox_inches='tight')
    logger.info(f"PNG figure saved to: {OUTPUT_FIGURE}")
    
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches='tight')
    logger.info(f"PDF figure saved to: {OUTPUT_FIGURE_PDF}")
    
    plt.close()
    
    # Save panel data as Excel with multiple sheets
    logger.info("Saving panel data as Excel...")
    with pd.ExcelWriter(OUTPUT_TABLE, engine='openpyxl') as writer:
        for sheet_name, data in panel_data.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info(f"Panel data saved to: {OUTPUT_TABLE}")
    logger.info(f"Saved {len(panel_data)} panels: {list(panel_data.keys())}")
    
    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)
    logger.info(f"Total transcripts fitted: {len(df_full)}")
    logger.info(f"\nParameter statistics:")
    for param in ['a', 'd', 'c_absolute', 'b']:
        data = df[param].dropna()
        logger.info(f"\n{param}:")
        logger.info(f"  Mean ± SD: {data.mean():.3f} ± {data.std():.3f}")
        logger.info(f"  Median: {data.median():.3f}")
        logger.info(f"  Range: [{data.min():.3f}, {data.max():.3f}]")
    
    # R² statistics from full data
    if 'r2' in df_full.columns:
        r2_data = df_full['r2'].dropna()
        logger.info(f"\nR²:")
        logger.info(f"  Mean ± SD: {r2_data.mean():.3f} ± {r2_data.std():.3f}")
        logger.info(f"  Median: {r2_data.median():.3f}")
        logger.info(f"  Range: [{r2_data.min():.3f}, {r2_data.max():.3f}]")
        
        # Report quality metrics
        good_fits = df_full[df_full['r2'] >= 0.7]
        excellent_fits = df_full[df_full['r2'] >= 0.9]
        logger.info(f"\nFit quality:")
        logger.info(f"  R² ≥ 0.7: {len(good_fits)} ({100*len(good_fits)/len(df_full):.1f}%)")
        logger.info(f"  R² ≥ 0.9: {len(excellent_fits)} ({100*len(excellent_fits)/len(df_full):.1f}%)")
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
