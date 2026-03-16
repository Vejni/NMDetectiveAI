"""
Analyze curve fitting results for long exons.

This script analyzes the curve fitting results from extract_long_exon_predictions.py:
- Plots example exons with fitted curves
- Analyzes best-fitting models by exon length
- Performs PCA on fit parameters and R² values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

from NMD.config import TABLES_DIR, FIGURES_DIR, COLOURS


# ========== Curve Functions (same as in extract script) ==========

def linear_model(x, a, b):
    """Linear model: y = a*x + b"""
    return a * x + b


def logistic_4pl(x, A, B, C, D):
    """4-parameter logistic model (sigmoid)"""
    return D + (A - D) / (1 + (x / C) ** B)


def piecewise_linear_2(x, x1, a1, b1, a2, b2):
    """Piecewise linear with 2 segments"""
    return np.where(x <= x1, a1 * x + b1, a2 * x + b2)


def piecewise_linear_3(x, x1, x2, a1, b1, a2, b2, a3, b3):
    """Piecewise linear with 3 segments"""
    return np.where(x <= x1, a1 * x + b1,
                   np.where(x <= x2, a2 * x + b2, a3 * x + b3))


def polynomial_2(x, a, b, c):
    """Polynomial degree 2"""
    return a * x**2 + b * x + c


def polynomial_3(x, a, b, c, d):
    """Polynomial degree 3"""
    return a * x**3 + b * x**2 + c * x + d


def polynomial_4(x, a, b, c, d, e):
    """Polynomial degree 4"""
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def polynomial_5(x, a, b, c, d, e, f):
    """Polynomial degree 5"""
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f


# ========== Analysis Functions ==========

def load_data():
    """Load curve fitting results and individual exon predictions."""
    fits_path = TABLES_DIR / "exon_analysis" / "long_exon_curve_fits.csv"
    logger.info(f"Loading curve fitting results from {fits_path}")
    
    if not fits_path.exists():
        raise FileNotFoundError(f"Curve fitting results not found at {fits_path}")
    
    fits_df = pd.read_csv(fits_path)
    logger.success(f"Loaded {len(fits_df)} exon curve fits")
    
    return fits_df


def get_exon_data(gene_name, transcript_id, exon_idx):
    """Load the actual data for a specific exon."""
    le_dir = TABLES_DIR / "LE"
    exon_file = le_dir / f"{gene_name}_{transcript_id}_exon{exon_idx}.csv"
    
    if not exon_file.exists():
        raise FileNotFoundError(f"Exon data not found: {exon_file}")
    
    return pd.read_csv(exon_file)


def plot_exon_with_fits(exon_row, ax, fits_df):
    """Plot a single exon with all fitted curves."""
    gene_name = exon_row['gene_name']
    transcript_id = exon_row['transcript_id']
    exon_idx = int(exon_row['exon_idx'])
    exon_length = exon_row['exon_length']
    
    # Load actual data
    try:
        exon_data = get_exon_data(gene_name, transcript_id, exon_idx)
    except FileNotFoundError:
        logger.warning(f"Could not load data for {gene_name} exon {exon_idx}")
        return False
    
    x = exon_data['ptc_position'].values
    y = exon_data['prediction'].values
    
    # Normalize x for fitting (same as in extract script)
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    
    # Plot actual data
    ax.scatter(x, y, color='lightgray', s=15, alpha=0.6, label='Data', zorder=1)
    
    # Define models and their colors
    models = {
        'linear': (linear_model, COLOURS[0], 'Linear'),
        'logistic_4pl': (logistic_4pl, COLOURS[1], 'Logistic-4PL'),
        'poly_5': (polynomial_5, COLOURS[2], 'Poly-5'),
    }
    
    # Find best model by AIC
    aic_cols = [f'{model}_aic' for model in models.keys()]
    available_aics = exon_row[aic_cols].dropna()
    if len(available_aics) > 0:
        best_model = available_aics.idxmin().replace('_aic', '')
    else:
        best_model = None
    
    # Plot each fitted curve
    for model_name, (func, color, label) in models.items():
        # Get parameters
        param_cols = [col for col in exon_row.index if col.startswith(f'{model_name}_p')]
        if not param_cols or pd.isna(exon_row[f'{model_name}_r2']):
            continue
        
        params = [exon_row[col] for col in sorted(param_cols)]
        
        try:
            y_pred = func(x_norm, *params)
            
            # Highlight best model
            is_best = (model_name == best_model)
            linewidth = 2.5 if is_best else 1.2
            alpha = 1.0 if is_best else 0.6
            zorder = 3 if is_best else 2
            
            label_text = f"{label}" + (" ★" if is_best else "")
            ax.plot(x, y_pred, color=color, linewidth=linewidth, 
                   alpha=alpha, label=label_text, zorder=zorder)
        except Exception as e:
            logger.debug(f"Could not plot {model_name} for {gene_name}: {e}")
            continue
    
    # Formatting
    ax.set_xlabel('PTC Position (nt)', fontsize=10)
    ax.set_ylabel('NMD Efficiency', fontsize=10)
    ax.set_title(f'{gene_name} exon {exon_idx}\n({exon_length:.0f} bp, n={len(x)})', 
                fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return True


def plot_example_exons(fits_df, n_examples=16):
    """Plot randomly selected example exons with fitted curves."""
    
    # Filter to exons with successful fits
    valid_exons = fits_df[fits_df['n_points'] >= 10].copy()
    
    if len(valid_exons) < n_examples:
        n_examples = len(valid_exons)
        logger.warning(f"Only {n_examples} valid exons found")
    
    # Sample random exons
    sample_exons = valid_exons.sample(n=n_examples)
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(18, 15))
    axes = axes.flatten()
    
    plotted_exons = []
    plot_idx = 0
    
    for idx, (_, exon_row) in enumerate(sample_exons.iterrows()):
        if plot_idx >= n_examples:
            break
        
        ax = axes[plot_idx]
        success = plot_exon_with_fits(exon_row, ax, fits_df)
        
        if success:
            plotted_exons.append(exon_row)
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = FIGURES_DIR / "LE"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "example_exons_with_fits.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    logger.success(f"Saved example exons plot to {output_path}")
    plt.close()
    
    return pd.DataFrame(plotted_exons)


def analyze_best_models_by_length(fits_df):
    """Analyze which models perform best at different exon lengths."""
    model_names = ['linear', 'logistic_4pl', 'poly_5']
    
    # Determine best model by AIC for each exon
    aic_cols = [f'{model}_aic' for model in model_names]
    fits_df['best_model'] = fits_df[aic_cols].idxmin(axis=1).str.replace('_aic', '')
    
    # Create length bins
    length_bins = [0, 500, 750, 1000, 1500, 2000, 5000, 50000]
    fits_df['length_bin'] = pd.cut(fits_df['exon_length'], bins=length_bins)
    
    # Create figure with multiple panels
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. R² comparison across models
    ax = axes[0, 0]
    r2_data = [fits_df[f'{model}_r2'].dropna() for model in model_names]
    model_labels = ['Linear', 'Logistic-4PL', 'Poly-5']
    bp = ax.boxplot(r2_data, labels=model_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], COLOURS[:len(model_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('R² Comparison Across Models', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Best model distribution
    ax = axes[0, 1]
    best_counts = fits_df['best_model'].value_counts()
    colors_map = {model: COLOURS[i] for i, model in enumerate(model_names)}
    bar_colors = [colors_map.get(model, 'gray') for model in best_counts.index]
    best_counts.plot(kind='bar', ax=ax, color=bar_colors, alpha=0.7)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Best Model Distribution (by AIC)', fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Best model by length bin
    ax = axes[1, 0]
    bin_model_counts = fits_df.groupby(['length_bin', 'best_model']).size().unstack(fill_value=0)
    bin_model_counts.plot(kind='bar', ax=ax, stacked=False, 
                          color=[colors_map.get(col, 'gray') for col in bin_model_counts.columns],
                          alpha=0.7)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Exon Length Range (bp)', fontsize=12)
    ax.set_title('Best Model by Exon Length', fontsize=13, fontweight='bold')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Scatter: exon length vs best model R²
    ax = axes[1, 1]
    for model in model_names:
        model_data = fits_df[fits_df['best_model'] == model]
        if len(model_data) > 0:
            ax.scatter(model_data['exon_length'], model_data[f'{model}_r2'],
                      alpha=0.5, s=30, color=colors_map[model], label=model.upper())
    ax.set_xlabel('Exon Length (bp)', fontsize=12)
    ax.set_ylabel('R² (Best Model)', fontsize=12)
    ax.set_title('Best Model R² vs Exon Length', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = FIGURES_DIR / "LE"
    output_path = output_dir / "model_performance_by_length.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    logger.success(f"Saved model performance analysis to {output_path}")
    plt.close()
    
    # Print summary statistics
    logger.info("\nBest model counts:")
    logger.info(f"\n{best_counts}")
    
    logger.info("\nMean R² by model:")
    for model in model_names:
        mean_r2 = fits_df[f'{model}_r2'].mean()
        logger.info(f"  {model}: {mean_r2:.4f}")


def perform_pca_analysis(fits_df, example_exons_df):
    """Perform PCA on interpolated curve data, colored by exon length."""
    
    # Parameters for interpolation
    n_points = 50  # Number of points to interpolate each curve to
    
    # Build feature matrix from interpolated curves
    interpolated_curves = []
    valid_indices = []
    
    logger.info("Interpolating curves to uniform length...")
    
    for idx, row in fits_df.iterrows():
        gene_name = row['gene_name']
        transcript_id = row['transcript_id']
        exon_idx = int(row['exon_idx'])
        
        try:
            # Load actual exon data
            exon_data = get_exon_data(gene_name, transcript_id, exon_idx)
            
            # Get positions and predictions
            x = exon_data['ptc_position'].values
            y = exon_data['prediction'].values
            
            if len(x) < 5:  # Skip if too few points
                continue
            
            # Normalize x to 0-1 (relative position within exon)
            x_norm = (x - x.min()) / (x.max() - x.min())
            
            # Sort by normalized position
            sort_idx = np.argsort(x_norm)
            x_norm = x_norm[sort_idx]
            y = y[sort_idx]
            
            # Interpolate to fixed number of points
            x_uniform = np.linspace(0, 1, n_points)
            interpolator = interp1d(x_norm, y, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
            y_uniform = interpolator(x_uniform)
            
            # Check for invalid values
            if np.any(np.isnan(y_uniform)) or np.any(np.isinf(y_uniform)):
                continue
            
            interpolated_curves.append(y_uniform)
            valid_indices.append(idx)
            
        except Exception as e:
            logger.debug(f"Could not interpolate {gene_name} exon {exon_idx}: {e}")
            continue
    
    # Convert to numpy array
    X = np.array(interpolated_curves)
    fits_df_clean = fits_df.loc[valid_indices].copy()
    
    logger.info(f"PCA on {len(X)} exons with {n_points} interpolated points per curve")
    
    # Standardize features (each position across all curves)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=min(10, n_points))
    X_pca = pca.fit_transform(X_scaled)
    
    # Create length bins
    length_bins = [0, 1000, 2500, 5000, 10000, np.inf]
    length_labels = ['<1000', '1000-2500', '2500-5000', '5000-10000', '>10000']
    fits_df_clean['length_category'] = pd.cut(fits_df_clean['exon_length'], 
                                               bins=length_bins, 
                                               labels=length_labels,
                                               right=False)
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: PCA colored by exon length bins
    ax = axes[0]
    
    # Define colors for each bin
    bin_colors = [COLOURS[0], COLOURS[2], COLOURS[4], COLOURS[6], COLOURS[8]]
    color_map = {label: color for label, color in zip(length_labels, bin_colors)}
    
    # Plot each length category separately for proper legend
    for length_cat in length_labels:
        mask = fits_df_clean['length_category'] == length_cat
        if mask.any():
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[color_map[length_cat]], 
                      s=30, alpha=0.6, label=f'{length_cat} bp')
    
    # Annotate example exons
    if example_exons_df is not None and len(example_exons_df) > 0:
        for _, exon_row in example_exons_df.iterrows():
            # Find this exon in the clean dataframe
            mask = ((fits_df_clean['gene_name'] == exon_row['gene_name']) & 
                   (fits_df_clean['transcript_id'] == exon_row['transcript_id']) &
                   (fits_df_clean['exon_idx'] == exon_row['exon_idx']))
            
            if mask.any():
                idx = np.where(mask)[0][0]
                ax.annotate(f"{exon_row['gene_name']}", 
                          xy=(X_pca[idx, 0], X_pca[idx, 1]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='red', alpha=0.7))
    
    ax.legend(title='Exon Length', fontsize=10, title_fontsize=11, loc='best')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('PCA of Interpolated Curve Data', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Visualize principal components as curves
    ax = axes[1]
    x_uniform = np.linspace(0, 1, n_points)
    
    # Plot first 3 PCs as curves
    for i in range(min(3, len(pca.components_))):
        # Inverse transform to get curve in original space
        pc_curve = pca.components_[i] * np.sqrt(pca.explained_variance_[i])
        ax.plot(x_uniform, pc_curve, linewidth=2, 
               color=COLOURS[i], label=f'PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)',
               alpha=0.8)
    
    ax.set_xlabel('Relative Position in Exon', fontsize=12)
    ax.set_ylabel('Component Loading', fontsize=12)
    ax.set_title('Principal Component Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = FIGURES_DIR / "LE"
    output_path = output_dir / "pca_curve_parameters.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    logger.success(f"Saved PCA plot to {output_path}")
    plt.close()
    
    # Print PCA summary
    logger.info(f"\nTotal variance explained by first 2 PCs: "
               f"{np.sum(pca.explained_variance_ratio_[:2])*100:.1f}%")
    logger.info(f"Total variance explained by first 5 PCs: "
               f"{np.sum(pca.explained_variance_ratio_[:5])*100:.1f}%")
    
    # Return PCA results for plotting extreme examples
    return X_pca, fits_df_clean, pca, X, scaler


def plot_extreme_pc_examples(X_pca, fits_df_clean, pca):
    """Plot example exons with extreme (lowest/highest) PC scores for first 3 PCs."""
    
    # Identify extreme exons for first 3 PCs
    extreme_exons = []
    
    for pc_idx in range(min(3, X_pca.shape[1])):
        pc_scores = X_pca[:, pc_idx]
        
        # Get indices of lowest and highest scores
        lowest_idx = np.argmin(pc_scores)
        highest_idx = np.argmax(pc_scores)
        
        extreme_exons.append({
            'idx': lowest_idx,
            'pc': pc_idx + 1,
            'type': 'Lowest',
            'score': pc_scores[lowest_idx]
        })
        extreme_exons.append({
            'idx': highest_idx,
            'pc': pc_idx + 1,
            'type': 'Highest',
            'score': pc_scores[highest_idx]
        })
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    
    for plot_idx, extreme in enumerate(extreme_exons):
        ax = axes[plot_idx // 2, plot_idx % 2]
        
        # Get exon info
        exon_row = fits_df_clean.iloc[extreme['idx']]
        gene_name = exon_row['gene_name']
        transcript_id = exon_row['transcript_id']
        exon_idx = int(exon_row['exon_idx'])
        exon_length = exon_row['exon_length']
        
        try:
            # Load actual exon data
            exon_data = get_exon_data(gene_name, transcript_id, exon_idx)
            
            x = exon_data['ptc_position'].values
            y = exon_data['prediction'].values
            
            # Plot data
            ax.scatter(x, y, color=COLOURS[extreme['pc']-1], s=20, alpha=0.7, zorder=3)
            ax.plot(x, y, color=COLOURS[extreme['pc']-1], linewidth=2, alpha=0.8, zorder=2)
            
            # Formatting
            ax.set_xlabel('PTC Position (nt)', fontsize=11)
            ax.set_ylabel('NMD Efficiency', fontsize=11)
            ax.set_title(f"PC{extreme['pc']} {extreme['type']} (score={extreme['score']:.2f})\n"
                        f"{gene_name} exon {exon_idx} ({exon_length:.0f} bp)",
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        except Exception as e:
            logger.warning(f"Could not plot extreme example for PC{extreme['pc']} {extreme['type']}: {e}")
            ax.set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = FIGURES_DIR / "LE"
    output_path = output_dir / "extreme_pc_examples.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    logger.success(f"Saved extreme PC examples to {output_path}")
    plt.close()


def perform_polynomial_features_pca(X, fits_df_clean, scaler, example_exons_df):
    """Perform PCA on fitted polynomial coefficients (degree 5)."""
    
    logger.info("Extracting polynomial-5 fitted coefficients...")
    
    # Extract polynomial coefficients from fits_df_clean
    poly_param_cols = [col for col in fits_df_clean.columns if col.startswith('poly_5_p')]
    
    if not poly_param_cols:
        logger.error("No polynomial-5 coefficients found in fits_df_clean")
        return
    
    # Get coefficients matrix (each row is an exon, each column is a coefficient)
    X_poly = fits_df_clean[sorted(poly_param_cols)].values
    
    # Remove rows with NaN values
    valid_mask = ~np.isnan(X_poly).any(axis=1)
    X_poly = X_poly[valid_mask]
    fits_df_poly = fits_df_clean[valid_mask].copy()
    
    logger.info(f"PCA on {len(X_poly)} exons with {X_poly.shape[1]} polynomial coefficients")
    
    # Standardize polynomial coefficients
    scaler_poly = StandardScaler()
    X_poly_scaled = scaler_poly.fit_transform(X_poly)
    
    # Perform PCA
    pca_poly = PCA(n_components=min(X_poly.shape[1], len(X_poly)))
    X_pca_poly = pca_poly.fit_transform(X_poly_scaled)
    
    # Create length bins
    length_bins = [0, 1000, 2500, 5000, 10000, np.inf]
    length_labels = ['<1000', '1000-2500', '2500-5000', '5000-10000', '>10000']
    fits_df_poly['length_category'] = pd.cut(fits_df_poly['exon_length'], 
                                               bins=length_bins, 
                                               labels=length_labels,
                                               right=False)
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: PCA colored by exon length bins
    ax = axes[0]
    
    # Define colors for each bin
    bin_colors = [COLOURS[0], COLOURS[2], COLOURS[4], COLOURS[6], COLOURS[8]]
    color_map = {label: color for label, color in zip(length_labels, bin_colors)}
    
    # Plot each length category separately
    for length_cat in length_labels:
        mask = fits_df_poly['length_category'] == length_cat
        if mask.any():
            ax.scatter(X_pca_poly[mask, 0], X_pca_poly[mask, 1],
                      c=[color_map[length_cat]], 
                      s=30, alpha=0.6, label=f'{length_cat} bp')
    
    # Annotate example exons
    if example_exons_df is not None and len(example_exons_df) > 0:
        for _, exon_row in example_exons_df.iterrows():
            mask = ((fits_df_poly['gene_name'] == exon_row['gene_name']) & 
                   (fits_df_poly['transcript_id'] == exon_row['transcript_id']) &
                   (fits_df_poly['exon_idx'] == exon_row['exon_idx']))
            
            if mask.any():
                idx = np.where(mask)[0][0]
                ax.annotate(f"{exon_row['gene_name']}", 
                          xy=(X_pca_poly[idx, 0], X_pca_poly[idx, 1]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='red', alpha=0.7))
    
    ax.legend(title='Exon Length', fontsize=10, title_fontsize=11, loc='best')
    ax.set_xlabel(f'PC1 ({pca_poly.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca_poly.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('PCA of Fitted Polynomial Coefficients (Degree 5)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Compare variance explained
    ax = axes[1]
    n_components = min(10, len(pca_poly.explained_variance_ratio_))
    
    x_pos = np.arange(1, n_components + 1)
    width = 0.35
    
    # Plot bars
    ax.bar(x_pos, pca_poly.explained_variance_ratio_[:n_components] * 100,
           width, color=COLOURS[0], alpha=0.7, label='Individual')
    
    # Plot cumulative line
    ax.plot(x_pos, np.cumsum(pca_poly.explained_variance_ratio_[:n_components]) * 100,
           'o-', color=COLOURS[6], linewidth=2, markersize=8, label='Cumulative')
    
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variance Explained (%)', fontsize=12)
    ax.set_title('Variance Explained (Polynomial Coefficients)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x_pos)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = FIGURES_DIR / "LE"
    output_path = output_dir / "pca_polynomial_features.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    logger.success(f"Saved polynomial features PCA plot to {output_path}")
    plt.close()
    
    # Print summary
    logger.info(f"\nPolynomial coefficients PCA - Total variance explained by first 2 PCs: "
               f"{np.sum(pca_poly.explained_variance_ratio_[:2])*100:.1f}%")
    logger.info(f"Polynomial coefficients PCA - Total variance explained by first 5 PCs: "
               f"{np.sum(pca_poly.explained_variance_ratio_[:min(5, len(pca_poly.explained_variance_ratio_))])*100:.1f}%")


def main():
    """Main analysis pipeline."""
    logger.info("Starting long exon curve analysis")
    
    # Load data
    fits_df = load_data()
    
    # Filter out exons with decreasing NMD efficiency (negative slope)
    # Use the linear model slope parameter (linear_p0 is the 'a' in y = a*x + b)
    initial_count = len(fits_df)
    fits_df = fits_df[fits_df['linear_p0'] >= 0].copy()
    dropped_count = initial_count - len(fits_df)
    logger.info(f"\nFiltered out {dropped_count} exons with decreasing NMD efficiency (negative slope)")
    logger.info(f"Retained {len(fits_df)} exons for analysis")
    
    # Plot example exons with fitted curves
    logger.info("\nPlotting example exons with fitted curves...")
    example_exons_df = plot_example_exons(fits_df, n_examples=16)
    
    # Analyze best models by exon length
    logger.info("\nAnalyzing model performance by exon length...")
    analyze_best_models_by_length(fits_df)
    
    # Perform PCA analysis
    logger.info("\nPerforming PCA analysis...")
    X_pca, fits_df_clean, pca, X, scaler = perform_pca_analysis(fits_df, example_exons_df)
    
    # Plot extreme PC examples
    logger.info("\nPlotting extreme PC score examples...")
    plot_extreme_pc_examples(X_pca, fits_df_clean, pca)
    
    # Perform polynomial features PCA
    logger.info("\nPerforming polynomial features PCA...")
    perform_polynomial_features_pca(X, fits_df_clean, scaler, example_exons_df)
    
    logger.success("\nAnalysis complete! All figures saved to FIGURES_DIR/LE")


if __name__ == "__main__":
    main()
