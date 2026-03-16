"""
Supplementary figure: Distribution of NMDetectiveAI predictions on gnomAD variants.
Shows 3-component Gaussian mixture model fits for rare and common variants.
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
    panel_label=16,
    axis_label=14,
    tick_label=12,
    legend=12,
)


def load_and_prepare_data():
    """
    Load gnomAD variant predictions for both rare and common variants.
    
    Returns:
        Dictionary with data for both variant types
    """
    data = {}
    
    for var_type in ['rare', 'common']:
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


def save_source_data(data, output_dir):
    """
    Save source data for each panel as separate tabs in XLSX file.
    
    Args:
        data: Dictionary with data for both variant types
        output_dir: Output directory for tables
    """
    output_file = output_dir / "gnomad_prediction_distributions.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for var_type in ['rare', 'common']:
            var_data = data[var_type]
            predictions = var_data['predictions']
            pred_class = var_data['pred_class']
            mu1, mu2, mu3 = var_data['components']['means']
            std1, std2, std3 = var_data['components']['stds']
            w1, w2, w3 = var_data['components']['weights']
            threshold1, threshold2 = var_data['thresholds']
            
            # Panel A/C: Distribution data (histogram bins + GMM components)
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
            
            # Combine distribution data
            panel_letter = 'a' if var_type == 'rare' else 'c'
            sheet_name = f"{panel_letter}_{var_type}_distribution"
            
            # Write with spacing
            dist_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
            gmm_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(dist_df)+2)
            comp_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(dist_df)+len(gmm_df)+4)
            thresh_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(dist_df)+len(gmm_df)+len(comp_df)+6)
            
            # Panel B/D: Boxplot data
            panel_letter = 'b' if var_type == 'rare' else 'd'
            box_data = []
            
            for cls, label in enumerate(['NMD-evading', 'Intermediate', 'NMD-triggering']):
                cls_predictions = predictions[pred_class == cls]
                box_data.append({
                    'class': label,
                    'n': len(cls_predictions),
                    'min': np.min(cls_predictions),
                    'q1': np.percentile(cls_predictions, 25),
                    'median': np.median(cls_predictions),
                    'q3': np.percentile(cls_predictions, 75),
                    'max': np.max(cls_predictions),
                    'mean': np.mean(cls_predictions),
                    'std': np.std(cls_predictions)
                })
            
            box_df = pd.DataFrame(box_data)
            box_df.to_excel(writer, sheet_name=f"{panel_letter}_{var_type}_boxplot", index=False)
    
    logger.info(f"Saved source data to {output_file}")


def create_figure(data, output_dir):
    """
    Create 2x2 figure with distribution and boxplots for rare and common variants.
    
    Args:
        data: Dictionary with data for both variant types
        output_dir: Output directory for figures
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    panel_labels = ['a', 'b', 'c', 'd']
    var_types = ['rare', 'common']
    
    for row_idx, var_type in enumerate(var_types):
        var_data = data[var_type]
        predictions = var_data['predictions']
        gmm = var_data['gmm']
        threshold1, threshold2 = var_data['thresholds']
        pred_class = var_data['pred_class']
        mu1, mu2, mu3 = var_data['components']['means']
        
        # Left column: Distribution with GMM
        ax = axes[row_idx, 0]
        
        # Histogram
        ax.hist(predictions, bins=100, density=True, alpha=0.6, color='gray', label='Data')
        
        # GMM components
        x = np.linspace(predictions.min(), predictions.max(), 1000)
        x_2d = x.reshape(-1, 1)
        
        logprob = gmm.score_samples(x_2d)
        responsibilities = gmm.predict_proba(x_2d)
        pdf = np.exp(logprob)
        ax.plot(x, pdf, 'k-', linewidth=2.5, label='Mixture', zorder=10)
        
        # Individual components
        pdf_individual = responsibilities * pdf[:, np.newaxis]
        colors = [COLOURS[0], COLOURS[2], COLOURS[1]]
        labels = ['NMD-evading', 'Intermediate', 'NMD-triggering']
        
        means = gmm.means_.flatten()
        idx_sorted = np.argsort(means)
        
        for i, idx in enumerate(idx_sorted):
            ax.plot(x, pdf_individual[:, idx], '--', color=colors[i], 
                    linewidth=2, label=f'{labels[i]}', alpha=0.8)
        
        # Thresholds
        ax.axvline(threshold1, color='orange', linestyle='--', linewidth=2, 
                   label=f'T1={threshold1:.2f}', alpha=0.7, zorder=5)
        ax.axvline(threshold2, color='red', linestyle='--', linewidth=2, 
                   label=f'T2={threshold2:.2f}', alpha=0.7, zorder=5)
        
        ax.set_xlabel('NMDetectiveAI prediction', fontsize=FONT_SIZES['axis_label'])
        ax.set_ylabel('Density', fontsize=FONT_SIZES['axis_label'])
        ax.tick_params(labelsize=FONT_SIZES['tick_label'])
        ax.legend(fontsize=FONT_SIZES['legend'], frameon=True, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add panel label
        ax.text(-0.15, 1.05, panel_labels[row_idx * 2], transform=ax.transAxes,
                fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='top')
        
        # Add variant type label
        var_label = f"{var_type.capitalize()} variants (n={len(predictions):,})"
        ax.text(0.98, 0.97, var_label, transform=ax.transAxes,
                fontsize=FONT_SIZES['axis_label'], ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5))
        
        # Right column: Boxplot
        ax = axes[row_idx, 1]
        
        data_to_plot = [
            predictions[pred_class == 0],
            predictions[pred_class == 1],
            predictions[pred_class == 2]
        ]
        
        bp = ax.boxplot(data_to_plot, 
                        labels=['NMD-\nevading', 'Inter-\nmediate', 'NMD-\ntriggering'],
                        patch_artist=True, widths=0.6,
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='black'))
        
        bp['boxes'][0].set_facecolor(COLOURS[0])
        bp['boxes'][1].set_facecolor(COLOURS[2])
        bp['boxes'][2].set_facecolor(COLOURS[1])
        
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        # Thresholds
        ax.axhline(threshold1, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(threshold2, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_ylabel('NMDetectiveAI prediction', fontsize=FONT_SIZES['axis_label'])
        ax.set_xlabel('Predicted class', fontsize=FONT_SIZES['axis_label'])
        ax.tick_params(labelsize=FONT_SIZES['tick_label'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add panel label
        ax.text(-0.15, 1.05, panel_labels[row_idx * 2 + 1], transform=ax.transAxes,
                fontsize=FONT_SIZES['panel_label'], fontweight='bold', va='top')
        
        # Add class counts - position below axis with more space
        for i, (cls_data, x_pos) in enumerate(zip(data_to_plot, [1, 2, 3])):
            n = len(cls_data)
            pct = 100 * n / len(predictions)
            # Position text below the axis with adequate spacing
            y_pos = ax.get_ylim()[0] - 0.25 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(x_pos, y_pos,
                    f'n={n:,}\n({pct:.1f}%)', 
                    ha='center', va='top', fontsize=FONT_SIZES['legend']-1)
    
    plt.tight_layout()
    
    # Adjust subplot parameters to make room for annotations
    plt.subplots_adjust(bottom=0.15, hspace=0.3)
    
    # Save figures
    output_file_png = output_dir / "gnomad_prediction_distributions.png"
    fig.savefig(output_file_png, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure to {output_file_png}")
    
    output_file_pdf = output_dir / "gnomad_prediction_distributions.pdf"
    fig.savefig(output_file_pdf, bbox_inches='tight')
    logger.info(f"Saved figure to {output_file_pdf}")
    
    plt.close()


def main():
    """Main function to generate supplementary figure."""
    logger.info("Generating supplementary figure: gnomAD prediction distributions")
    
    # Setup output directories
    fig_output_dir = FIGURES_DIR / "manuscript" / "supplementary"
    table_output_dir = TABLES_DIR / "manuscript" / "supplementary"
    
    fig_output_dir.mkdir(parents=True, exist_ok=True)
    table_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if table already exists
    table_file = table_output_dir / "gnomad_prediction_distributions.xlsx"
    
    if table_file.exists():
        logger.info(f"Table {table_file} already exists, skipping data processing")
        logger.info("Loading data for plotting only...")
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Save source data
    if not table_file.exists():
        save_source_data(data, table_output_dir)
    
    # Create figure
    create_figure(data, fig_output_dir)
    
    logger.success("Supplementary figure generation complete!")


if __name__ == "__main__":
    main()
