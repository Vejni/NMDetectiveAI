"""
Analyze distribution of NMDetectiveAI predictions on gnomAD variants.
Fit mixture models to find optimal threshold for binarizing predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.optimize import brentq
import typer

from NMD.config import PROCESSED_DATA_DIR, FIGURES_DIR, COLOURS

app = typer.Typer()


def load_gnomad_predictions(var_type: str = "common") -> pd.DataFrame:
    """
    Load gnomAD variants with predictions.
    
    Args:
        var_type: Type of variants ("common" or "rare")
    
    Returns:
        DataFrame with predictions
    """
    file_path = (
        PROCESSED_DATA_DIR 
        / "gnomad_v4.1" 
        / f"annotated_{var_type}" 
        / f"gnomad.v4.1.all_chromosomes.{var_type}_stopgain_snv.mane.annotated_with_predictions.tsv"
    )
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    
    # Filter for successfully processed predictions
    df_filtered = df[df['NMDetectiveAI_status'] == 'processed'].copy()
    logger.info(f"Loaded {len(df_filtered)} variants with predictions (from {len(df)} total)")
    
    return df_filtered


def fit_gaussian_mixture(predictions: np.ndarray) -> tuple:
    """
    Fit 3-component Gaussian mixture model to predictions.
    Components represent: NMD-evading, Intermediate, NMD-triggering.
    
    Args:
        predictions: Array of predictions
    
    Returns:
        Tuple of (fitted model, BIC, AIC)
    """
    predictions_2d = predictions.reshape(-1, 1)
    
    gmm = GaussianMixture(
        n_components=3,
        random_state=42,
        covariance_type='full'
    )
    gmm.fit(predictions_2d)
    
    bic = gmm.bic(predictions_2d)
    aic = gmm.aic(predictions_2d)
    
    logger.info(f"3-component GMM: BIC={bic:.2f}, AIC={aic:.2f}")
    
    return gmm, bic, aic


def find_thresholds_from_gmm(gmm: GaussianMixture) -> tuple:
    """
    Find two thresholds from 3-component Gaussian mixture model.
    Returns intersection points between adjacent components.
    
    Args:
        gmm: Fitted 3-component Gaussian mixture model
    
    Returns:
        Tuple of (threshold1, threshold2, component_params)
        threshold1: boundary between NMD-evading and Intermediate
        threshold2: boundary between Intermediate and NMD-triggering
    """
    if gmm.n_components != 3:
        raise ValueError("This method requires exactly 3 components")
    
    # Get parameters of the three components
    means = gmm.means_.flatten()
    vars = gmm.covariances_.flatten()
    weights = gmm.weights_
    
    # Sort by mean (lowest to highest)
    idx_sorted = np.argsort(means)
    
    mu1, mu2, mu3 = means[idx_sorted]
    var1, var2, var3 = vars[idx_sorted]
    w1, w2, w3 = weights[idx_sorted]
    
    logger.info(f"Component 1 (NMD-evading): mean={mu1:.3f}, std={np.sqrt(var1):.3f}, weight={w1:.3f}")
    logger.info(f"Component 2 (Intermediate): mean={mu2:.3f}, std={np.sqrt(var2):.3f}, weight={w2:.3f}")
    logger.info(f"Component 3 (NMD-triggering): mean={mu3:.3f}, std={np.sqrt(var3):.3f}, weight={w3:.3f}")
    
    # Find intersection between components 1 and 2
    def diff_gaussians_12(x):
        g1 = w1 * stats.norm.pdf(x, mu1, np.sqrt(var1))
        g2 = w2 * stats.norm.pdf(x, mu2, np.sqrt(var2))
        return g1 - g2
    
    try:
        threshold1 = brentq(diff_gaussians_12, mu1, mu2)
        logger.info(f"Threshold 1 (evading/intermediate): {threshold1:.3f}")
    except ValueError:
        threshold1 = (mu1 + mu2) / 2
        logger.warning(f"No intersection found between components 1-2, using midpoint: {threshold1:.3f}")
    
    # Find intersection between components 2 and 3
    def diff_gaussians_23(x):
        g2 = w2 * stats.norm.pdf(x, mu2, np.sqrt(var2))
        g3 = w3 * stats.norm.pdf(x, mu3, np.sqrt(var3))
        return g2 - g3
    
    try:
        threshold2 = brentq(diff_gaussians_23, mu2, mu3)
        logger.info(f"Threshold 2 (intermediate/triggering): {threshold2:.3f}")
    except ValueError:
        threshold2 = (mu2 + mu3) / 2
        logger.warning(f"No intersection found between components 2-3, using midpoint: {threshold2:.3f}")
    
    component_params = (
        mu1, mu2, mu3,
        np.sqrt(var1), np.sqrt(var2), np.sqrt(var3),
        w1, w2, w3
    )
    
    return threshold1, threshold2, component_params


def calculate_metrics_at_thresholds(predictions: np.ndarray, labels: np.ndarray, threshold1: float, threshold2: float) -> dict:
    """
    Calculate classification metrics with two thresholds (3-class system).
    Maps 3-class predictions to binary for comparison with rule-based labels.
    
    Args:
        predictions: Array of predictions
        labels: Ground truth labels (True for NMD-triggering)
        threshold1: First threshold (evading/intermediate boundary)
        threshold2: Second threshold (intermediate/triggering boundary)
    
    Returns:
        Dictionary of metrics
    """
    # Create 3-class predictions
    pred_class = np.zeros(len(predictions), dtype=int)
    pred_class[predictions >= threshold1] = 1  # Intermediate
    pred_class[predictions >= threshold2] = 2  # Triggering
    
    # For binary comparison: treat intermediate as evading (conservative)
    pred_labels_conservative = predictions >= threshold2
    
    # For binary comparison: treat intermediate as triggering (liberal)
    pred_labels_liberal = predictions >= threshold1
    
    # Calculate metrics for conservative approach
    tp_cons = np.sum((pred_labels_conservative == True) & (labels == True))
    tn_cons = np.sum((pred_labels_conservative == False) & (labels == False))
    fp_cons = np.sum((pred_labels_conservative == True) & (labels == False))
    fn_cons = np.sum((pred_labels_conservative == False) & (labels == True))
    
    acc_cons = (tp_cons + tn_cons) / len(predictions)
    prec_cons = tp_cons / (tp_cons + fp_cons) if (tp_cons + fp_cons) > 0 else 0
    rec_cons = tp_cons / (tp_cons + fn_cons) if (tp_cons + fn_cons) > 0 else 0
    f1_cons = 2 * prec_cons * rec_cons / (prec_cons + rec_cons) if (prec_cons + rec_cons) > 0 else 0
    
    # Calculate metrics for liberal approach
    tp_lib = np.sum((pred_labels_liberal == True) & (labels == True))
    tn_lib = np.sum((pred_labels_liberal == False) & (labels == False))
    fp_lib = np.sum((pred_labels_liberal == True) & (labels == False))
    fn_lib = np.sum((pred_labels_liberal == False) & (labels == True))
    
    acc_lib = (tp_lib + tn_lib) / len(predictions)
    prec_lib = tp_lib / (tp_lib + fp_lib) if (tp_lib + fp_lib) > 0 else 0
    rec_lib = tp_lib / (tp_lib + fn_lib) if (tp_lib + fn_lib) > 0 else 0
    f1_lib = 2 * prec_lib * rec_lib / (prec_lib + rec_lib) if (prec_lib + rec_lib) > 0 else 0
    
    # Count class sizes
    n_evading = np.sum(pred_class == 0)
    n_intermediate = np.sum(pred_class == 1)
    n_triggering = np.sum(pred_class == 2)
    
    return {
        'threshold1': threshold1,
        'threshold2': threshold2,
        'n_evading': n_evading,
        'n_intermediate': n_intermediate,
        'n_triggering': n_triggering,
        'accuracy_conservative': acc_cons,
        'precision_conservative': prec_cons,
        'recall_conservative': rec_cons,
        'f1_conservative': f1_cons,
        'accuracy_liberal': acc_lib,
        'precision_liberal': prec_lib,
        'recall_liberal': rec_lib,
        'f1_liberal': f1_lib,
    }


def plot_distribution_with_mixture(
    predictions: np.ndarray,
    gmm: GaussianMixture,
    threshold1: float,
    threshold2: float,
    rule_labels: np.ndarray = None,
    var_type: str = "common",
    output_dir: Path = None
):
    """
    Plot prediction distribution with fitted 3-component mixture model and thresholds.
    
    Args:
        predictions: Array of predictions
        gmm: Fitted 3-component Gaussian mixture model
        threshold1: First threshold (evading/intermediate)
        threshold2: Second threshold (intermediate/triggering)
        rule_labels: Rule based labels (optional)
        var_type: Type of variants
        output_dir: Directory to save figures
    """
    if output_dir is None:
        output_dir = FIGURES_DIR / "gnomad_analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram with mixture components
    ax = axes[0, 0]
    ax.hist(predictions, bins=100, density=True, alpha=0.6, color='gray', label='Data')
    
    # Plot mixture components
    x = np.linspace(predictions.min(), predictions.max(), 1000)
    x_2d = x.reshape(-1, 1)
    
    # Overall density
    logprob = gmm.score_samples(x_2d)
    responsibilities = gmm.predict_proba(x_2d)
    pdf = np.exp(logprob)
    ax.plot(x, pdf, 'k-', linewidth=2, label='Mixture')
    
    # Individual components - 3 colors
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    colors = [COLOURS[0], COLOURS[2], COLOURS[1]]  # evading, intermediate, triggering
    labels = ['NMD-evading', 'Intermediate', 'NMD-triggering']
    
    # Sort components by mean to assign labels
    means = gmm.means_.flatten()
    idx_sorted = np.argsort(means)
    
    for i, idx in enumerate(idx_sorted):
        ax.plot(x, pdf_individual[:, idx], '--', color=colors[i], 
                linewidth=2, label=f'{labels[i]} component')
    
    ax.axvline(threshold1, color='orange', linestyle='--', linewidth=2, label=f'Threshold 1={threshold1:.3f}')
    ax.axvline(threshold2, color='red', linestyle='--', linewidth=2, label=f'Threshold 2={threshold2:.3f}')
    ax.set_xlabel('NMDetectiveAI Prediction', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Prediction Distribution ({var_type} variants)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # 2. Cumulative distribution
    ax = axes[0, 1]
    sorted_preds = np.sort(predictions)
    cumulative = np.arange(1, len(sorted_preds) + 1) / len(sorted_preds)
    ax.plot(sorted_preds, cumulative, linewidth=2, color='navy')
    ax.axvline(threshold1, color='orange', linestyle='--', linewidth=2, label=f'Threshold 1={threshold1:.3f}')
    ax.axvline(threshold2, color='red', linestyle='--', linewidth=2, label=f'Threshold 2={threshold2:.3f}')
    ax.set_xlabel('NMDetectiveAI Prediction', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 3. Comparison with rules (if available)
    ax = axes[1, 0]
    if rule_labels is not None:
        nmd_evading_preds = predictions[~rule_labels]
        nmd_triggering_preds = predictions[rule_labels]
        
        ax.hist(nmd_evading_preds, bins=50, alpha=0.6, density=True, 
                color=COLOURS[0], label='Rule based: NMD-evading')
        ax.hist(nmd_triggering_preds, bins=50, alpha=0.6, density=True,
                color=COLOURS[1], label='Rule based: NMD-triggering')
        ax.axvline(threshold1, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(threshold2, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('NMDetectiveAI Prediction', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distribution by Rule', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No rule labels available', 
                ha='center', va='center', fontsize=12)
        ax.set_title('Distribution by Rule', fontsize=14, fontweight='bold')
    
    # 4. Box plot by predicted class (3 classes)
    ax = axes[1, 1]
    pred_class = np.zeros(len(predictions), dtype=int)
    pred_class[predictions >= threshold1] = 1
    pred_class[predictions >= threshold2] = 2
    
    data_to_plot = [
        predictions[pred_class == 0],
        predictions[pred_class == 1],
        predictions[pred_class == 2]
    ]
    bp = ax.boxplot(data_to_plot, 
                    labels=['NMD-evading', 'Intermediate', 'NMD-triggering'],
                    patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(COLOURS[0])
    bp['boxes'][1].set_facecolor(COLOURS[2])
    bp['boxes'][2].set_facecolor(COLOURS[1])
    ax.axhline(threshold1, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(threshold2, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel('NMDetectiveAI Prediction', fontsize=12)
    ax.set_title('Predictions by Predicted Class', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"prediction_distribution_{var_type}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure to {output_file}")
    
    output_file_pdf = output_dir / f"prediction_distribution_{var_type}.pdf"
    fig.savefig(output_file_pdf, bbox_inches='tight')
    logger.info(f"Saved figure to {output_file_pdf}")
    
    plt.close()


@app.command()
def analyze_prediction_distribution(
    var_type: str = "common",
    output_dir: str = None
):
    """
    Analyze distribution of NMDetectiveAI predictions and find optimal thresholds.
    
    Args:
        var_type: Type of variants to analyze ("common" or "rare")
        output_dir: Output directory for figures (default: FIGURES_DIR/gnomad_analysis)
    """
    logger.info(f"Analyzing prediction distribution for {var_type} variants")
    
    # Set output directory
    if output_dir is None:
        output_dir = FIGURES_DIR / "gnomad_analysis"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_gnomad_predictions(var_type)
    predictions = df['NMDetectiveAI_prediction'].values
    
    # Get rule-based labels if available
    rule_labels = None
    if 'predicted_nmd_status' in df.columns:
        # Convert rule-based predictions to binary
        rule_labels = df['predicted_nmd_status'].str.contains('triggering', na=False).values
        logger.info(f"Found {rule_labels.sum()} NMD-triggering variants by evading rules")
    
    # Fit 3-component GMM
    logger.info("Fitting 3-component Gaussian mixture model...")
    gmm, bic, aic = fit_gaussian_mixture(predictions)
    
    # Find thresholds from GMM intersections
    threshold1, threshold2, gmm_params = find_thresholds_from_gmm(gmm)
    
    # Plot distribution with mixture
    plot_distribution_with_mixture(predictions, gmm, threshold1, threshold2, rule_labels, var_type, output_dir)
    
    # Calculate agreement with rules if available
    if rule_labels is not None:
        metrics = calculate_metrics_at_thresholds(predictions, rule_labels, threshold1, threshold2)
        
        logger.info("\nClassification metrics:")
        logger.info(f"  Threshold 1 (evading/intermediate): {metrics['threshold1']:.3f}")
        logger.info(f"  Threshold 2 (intermediate/triggering): {metrics['threshold2']:.3f}")
        logger.info(f"  Class sizes:")
        logger.info(f"    NMD-evading: {metrics['n_evading']} ({100*metrics['n_evading']/len(predictions):.1f}%)")
        logger.info(f"    Intermediate: {metrics['n_intermediate']} ({100*metrics['n_intermediate']/len(predictions):.1f}%)")
        logger.info(f"    NMD-triggering: {metrics['n_triggering']} ({100*metrics['n_triggering']/len(predictions):.1f}%)")
        logger.info(f"\n  Conservative metrics (intermediate=evading):")
        logger.info(f"    Accuracy: {metrics['accuracy_conservative']:.3f}")
        logger.info(f"    Precision: {metrics['precision_conservative']:.3f}")
        logger.info(f"    Recall: {metrics['recall_conservative']:.3f}")
        logger.info(f"    F1 Score: {metrics['f1_conservative']:.3f}")
        logger.info(f"\n  Liberal metrics (intermediate=triggering):")
        logger.info(f"    Accuracy: {metrics['accuracy_liberal']:.3f}")
        logger.info(f"    Precision: {metrics['precision_liberal']:.3f}")
        logger.info(f"    Recall: {metrics['recall_liberal']:.3f}")
        logger.info(f"    F1 Score: {metrics['f1_liberal']:.3f}")
        
        # Save metrics to file
        metrics_df = pd.DataFrame([metrics])
        metrics_file = output_dir / f"threshold_metrics_{var_type}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"\nSaved metrics to {metrics_file}")
    
    # Summary statistics
    logger.info("\nSummary statistics:")
    logger.info(f"  N variants: {len(predictions)}")
    logger.info(f"  Mean prediction: {predictions.mean():.3f}")
    logger.info(f"  Median prediction: {np.median(predictions):.3f}")
    logger.info(f"  Std prediction: {predictions.std():.3f}")
    logger.info(f"  Min prediction: {predictions.min():.3f}")
    logger.info(f"  Max prediction: {predictions.max():.3f}")
    
    if gmm_params is not None:
        mu1, mu2, mu3, std1, std2, std3, w1, w2, w3 = gmm_params
        logger.info(f"\nGMM parameters:")
        logger.info(f"  Component 1 (NMD-evading): μ={mu1:.3f}, σ={std1:.3f}, weight={w1:.3f}")
        logger.info(f"  Component 2 (Intermediate): μ={mu2:.3f}, σ={std2:.3f}, weight={w2:.3f}")
        logger.info(f"  Component 3 (NMD-triggering): μ={mu3:.3f}, σ={std3:.3f}, weight={w3:.3f}")
    
    logger.success("Analysis complete!")
    
    return threshold1, threshold2, gmm


@app.command()
def analyze_both_datasets():
    """
    Analyze both common and rare variants and compare thresholds.
    """
    logger.info("Analyzing both common and rare variant datasets")
    
    output_dir = FIGURES_DIR / "gnomad_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for var_type in ["common", "rare"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {var_type} variants")
        logger.info(f"{'='*60}\n")
        
        threshold1, threshold2, gmm = analyze_prediction_distribution(
            var_type=var_type,
            output_dir=str(output_dir)
        )
        
        results[var_type] = {
            'threshold1': threshold1,
            'threshold2': threshold2,
            'gmm': gmm
        }
    
    # Compare thresholds
    logger.info("\n" + "="*60)
    logger.info("COMPARISON")
    logger.info("="*60)
    logger.info(f"Common variants:")
    logger.info(f"  Threshold 1 (evading/intermediate): {results['common']['threshold1']:.3f}")
    logger.info(f"  Threshold 2 (intermediate/triggering): {results['common']['threshold2']:.3f}")
    logger.info(f"Rare variants:")
    logger.info(f"  Threshold 1 (evading/intermediate): {results['rare']['threshold1']:.3f}")
    logger.info(f"  Threshold 2 (intermediate/triggering): {results['rare']['threshold2']:.3f}")
    logger.info(f"Differences:")
    logger.info(f"  Threshold 1: {abs(results['common']['threshold1'] - results['rare']['threshold1']):.3f}")
    logger.info(f"  Threshold 2: {abs(results['common']['threshold2'] - results['rare']['threshold2']):.3f}")
    
    logger.success("All analyses complete!")


if __name__ == "__main__":
    app()
