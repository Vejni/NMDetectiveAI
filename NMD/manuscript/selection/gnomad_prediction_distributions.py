"""
Figure 7a: Distribution of NMDetectiveAI predictions on rare gnomAD variants.
Shows 3-component Gaussian mixture model fit for rare stop-gain variants.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from sklearn.mixture import GaussianMixture

from NMD.config import (
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    COLOURS
)

FONT_SIZES = dict(
    panel_label=20,
    axis_label=16,
    tick_label=14,
    legend=14,
)


def load_and_prepare_data():
    """
    Load gnomAD variant predictions for rare variants.
    
    Returns:
        Dictionary with data for rare variants
    """
    data = {}
    
    for var_type in ['rare']:
        # Load predictions
        file_path = (
            PROCESSED_DATA_DIR 
            / "gnomad_v4.1" 
            / f"annotated_{var_type}" 
            / f"gnomad.v4.1.all_chromosomes.{var_type}_stopgain_snv.mane.annotated_with_predictions.tsv"
        )
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading {var_type} variant predictions from {file_path}")
        df = pd.read_csv(file_path, sep='\t')
        
        # Filter for successfully processed predictions
        df_filtered = df[df['NMDetectiveAI_status'] == 'processed'].copy()
        predictions = df_filtered['NMDetectiveAI_prediction'].values
        
        # Fit 3-component GMM
        logger.info(f"Fitting 3-component GMM for {var_type} variants...")
        predictions_2d = predictions.reshape(-1, 1)
        gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full')
        gmm.fit(predictions_2d)
        
        # Get component parameters
        means = gmm.means_.flatten()
        vars = gmm.covariances_.flatten()
        weights = gmm.weights_
        idx_sorted = np.argsort(means)
        
        mu1, mu2, mu3 = means[idx_sorted]
        var1, var2, var3 = vars[idx_sorted]
        w1, w2, w3 = weights[idx_sorted]
        
        # Calculate thresholds
        from scipy.optimize import brentq
        from scipy import stats
        
        def diff_gaussians_12(x):
            g1 = w1 * stats.norm.pdf(x, mu1, np.sqrt(var1))
            g2 = w2 * stats.norm.pdf(x, mu2, np.sqrt(var2))
            return g1 - g2
        
        def diff_gaussians_23(x):
            g2 = w2 * stats.norm.pdf(x, mu2, np.sqrt(var2))
            g3 = w3 * stats.norm.pdf(x, mu3, np.sqrt(var3))
            return g2 - g3
        
        try:
            threshold1 = brentq(diff_gaussians_12, mu1, mu2)
        except ValueError:
            threshold1 = (mu1 + mu2) / 2
        
        try:
            threshold2 = brentq(diff_gaussians_23, mu2, mu3)
        except ValueError:
            threshold2 = (mu2 + mu3) / 2
        
        # Classify predictions
        pred_class = np.zeros(len(predictions), dtype=int)
        pred_class[predictions >= threshold1] = 1
        pred_class[predictions >= threshold2] = 2
        
        data[var_type] = {
            'predictions': predictions,
            'gmm': gmm,
            'thresholds': (threshold1, threshold2),
            'components': {
                'means': (mu1, mu2, mu3),
                'stds': (np.sqrt(var1), np.sqrt(var2), np.sqrt(var3)),
                'weights': (w1, w2, w3)
            },
            'pred_class': pred_class
        }
        
        logger.info(f"{var_type.capitalize()} variants: {len(predictions)} samples")
        logger.info(f"  Threshold 1: {threshold1:.3f}, Threshold 2: {threshold2:.3f}")
    
    return data


def save_source_data(data, output_file):
    """
    Save source data as CSV file.
    
    Args:
        data: Dictionary with data for rare variants
        output_file: Full path to the output CSV file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process rare variant data only
    if True:
        var_type = 'rare'
        var_data = data[var_type]
        predictions = var_data['predictions']
        pred_class = var_data['pred_class']
        mu1, mu2, mu3 = var_data['components']['means']
        std1, std2, std3 = var_data['components']['stds']
        w1, w2, w3 = var_data['components']['weights']
        threshold1, threshold2 = var_data['thresholds']
        
        # Distribution data (histogram bins + GMM components)
        hist, bin_edges = np.histogram(predictions, bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # GMM density at bin centers
        x = np.linspace(predictions.min(), predictions.max(), 200)
        x_2d = x.reshape(-1, 1)
        logprob = var_data['gmm'].score_samples(x_2d)
        pdf = np.exp(logprob)
        
        dist_df = pd.DataFrame({
            'bin_center': bin_centers,
            'histogram_density': hist
        })
        
        gmm_df = pd.DataFrame({
            'x': x,
            'total_density': pdf
        })
        
        # Component parameters
        comp_df = pd.DataFrame({
            'component': ['NMD-evading', 'Intermediate', 'NMD-triggering'],
            'mean': [mu1, mu2, mu3],
            'std': [std1, std2, std3],
            'weight': [w1, w2, w3]
        })
        
        thresh_df = pd.DataFrame({
            'threshold': ['evading/intermediate', 'intermediate/triggering'],
            'value': [threshold1, threshold2]
        })
        
        # Combine all data into single CSV
        # Add section headers for clarity
        result_data = []
        
        # Histogram section
        result_data.append(pd.DataFrame({'section': ['HISTOGRAM_DATA']}))
        result_data.append(dist_df)
        result_data.append(pd.DataFrame())  # blank row
        
        # GMM density section
        result_data.append(pd.DataFrame({'section': ['GMM_DENSITY']}))
        result_data.append(gmm_df)
        result_data.append(pd.DataFrame())  # blank row
        
        # Component parameters section
        result_data.append(pd.DataFrame({'section': ['GMM_COMPONENTS']}))
        result_data.append(comp_df)
        result_data.append(pd.DataFrame())  # blank row
        
        # Thresholds section
        result_data.append(pd.DataFrame({'section': ['THRESHOLDS']}))
        result_data.append(thresh_df)
        
        final_df = pd.concat(result_data, ignore_index=True)
        final_df.to_csv(output_file, index=False)
    
    logger.info(f"Saved source data to {output_file}")


def create_figure(data):
    """
    Create single panel figure showing rare variant distribution with GMM fit.
    
    Args:
        data: Dictionary with data for rare variants
    
    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    
    # Process rare variants only
    for var_type in ['rare']:
        var_data = data[var_type]
        predictions = var_data['predictions']
        gmm = var_data['gmm']
        threshold1, threshold2 = var_data['thresholds']
        pred_class = var_data['pred_class']
        mu1, mu2, mu3 = var_data['components']['means']
        
        # Distribution with GMM
        # Histogram
        var_label = f"Variants (n={len(predictions):,})"
        ax.hist(predictions, bins=100, density=True, alpha=0.6, color='gray', label=var_label)
        
        # GMM components
        x = np.linspace(predictions.min(), predictions.max(), 1000)
        x_2d = x.reshape(-1, 1)
        
        logprob = gmm.score_samples(x_2d)
        responsibilities = gmm.predict_proba(x_2d)
        pdf = np.exp(logprob)
        ax.plot(x, pdf, 'k-', linewidth=2.5, label='Mixture', zorder=10)
        
        # Individual components
        pdf_individual = responsibilities * pdf[:, np.newaxis]
        # Use NMD evading/triggering color palette
        colors = [COLOURS[2], COLOURS[1], COLOURS[6]]  # NMD-evading (pink), Intermediate (yellow), NMD-triggering (blue)
        labels = ['NMD evading', 'Intermediate', 'NMD triggering']
        
        means = gmm.means_.flatten()
        idx_sorted = np.argsort(means)
        
        for i, idx in enumerate(idx_sorted):
            ax.plot(x, pdf_individual[:, idx], '--', color=colors[i], 
                    linewidth=2, label=f'{labels[i]}', alpha=0.8)
        
        # Thresholds - use evading/triggering colors
        ax.axvline(threshold1, color=COLOURS[2], linestyle='--', linewidth=2, 
                   alpha=0.7, zorder=5)
        ax.axvline(threshold2, color=COLOURS[6], linestyle='--', linewidth=2, 
                   alpha=0.7, zorder=5)
        
        # Add threshold labels next to the lines (slightly to the right)
        ax.text(threshold1 + 0.02, ax.get_ylim()[1] * 0.95, f'{threshold1:.2f}', 
                ha='left', va='top', fontsize=FONT_SIZES['legend'], 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        ax.text(threshold2 + 0.02, ax.get_ylim()[1] * 0.95, f'{threshold2:.2f}', 
                ha='left', va='top', fontsize=FONT_SIZES['legend'], 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('NMDetective-AI prediction', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
        ax.set_ylabel('Density', fontsize=FONT_SIZES['axis_label'], fontweight='bold')
        ax.tick_params(labelsize=FONT_SIZES['tick_label'])
        ax.legend(fontsize=FONT_SIZES['legend'], frameon=True, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.45))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.title("Distribution of NMDetective-AI predictions\nfor rare gnomAD PTC variants", fontsize=FONT_SIZES['panel_label'], fontweight='bold')
    plt.tight_layout()
    
    # Adjust subplot parameters to make room for the legend below
    plt.subplots_adjust(bottom=0.25)
    
    return fig


def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = True,
):
    """Generate Fig7a: rare gnomAD variant prediction distribution.

    Args:
        figure_label: Panel label when called from the manuscript app.
        figure_number: Figure number when called from the manuscript app.
        regenerate: If False and source data exists, skip processing.
    """
    logger.info("Generating Fig7a: gnomAD rare variant prediction distributions")
    
    from NMD.manuscript.output import get_paths
    
    # Use manuscript app output structure if called from app
    if figure_label is not None and figure_number is not None:
        paths = get_paths(
            script_name="gnomad_prediction_distributions",
            figure_label=figure_label,
            figure_number=figure_number,
        )
        source_data_path = paths.source_data
        figure_png_path = paths.figure_png
        figure_pdf_path = paths.figure_pdf
    else:
        # Standalone mode
        fig_output_dir = FIGURES_DIR / "manuscript" / "selection"
        table_output_dir = TABLES_DIR / "manuscript" / "selection"
        source_data_path = table_output_dir / "gnomad_prediction_distributions.csv"
        figure_png_path = fig_output_dir / "gnomad_prediction_distributions.png"
        figure_pdf_path = fig_output_dir / "gnomad_prediction_distributions.pdf"
        
        fig_output_dir.mkdir(parents=True, exist_ok=True)
        table_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we should regenerate data
    if not regenerate and source_data_path.exists():
        logger.info(f"Source data {source_data_path} already exists, skipping data processing")
        logger.info("Note: Data will still be loaded for plotting")
    
    # Load and prepare data (always needed for plotting)
    data = load_and_prepare_data()
    
    # Save source data if it doesn't exist or if regenerating
    if regenerate or not source_data_path.exists():
        save_source_data(data, source_data_path)
    
    # Create and save figure
    fig = create_figure(data)
    figure_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_png_path, dpi=300, bbox_inches='tight')
    fig.savefig(figure_pdf_path, bbox_inches='tight')
    logger.info(f"Figure saved to {figure_png_path}")
    plt.close(fig)
    
    logger.success("Figure 7a generation complete!")



if __name__ == "__main__":
    main()
