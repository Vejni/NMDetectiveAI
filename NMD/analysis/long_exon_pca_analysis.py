"""
PCA analysis of long exon NMD efficiency curves.

This script performs PCA on interpolated NMD efficiency curves from long exons
and saves the results for use by manuscript figure scripts.

Outputs:
- long_exon_pca_scores.csv: PC scores for each exon with metadata
- long_exon_pca_components.csv: PC loadings and variance explained
"""

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

from NMD.config import TABLES_DIR


# ============================================================================
# CONFIGURATION
# ============================================================================

CURVE_FITS_FILE = TABLES_DIR / "exon_analysis" / "long_exon_curve_fits.csv"
LE_DIR = TABLES_DIR / "LE"
OUTPUT_DIR = TABLES_DIR / "exon_analysis"

# PCA parameters
N_INTERPOLATION_POINTS = 50
N_PCS = 3  # Number of principal components to compute


# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_exon_data(gene_name, transcript_id, exon_idx):
    """Load the actual data for a specific exon."""
    exon_file = LE_DIR / f"{gene_name}_{transcript_id}_exon{exon_idx}.csv"
    if not exon_file.exists():
        raise FileNotFoundError(f"Exon data not found: {exon_file}")
    return pd.read_csv(exon_file)


def run_pca_analysis():
    """Load curve fitting results and perform PCA on interpolated curves."""
    logger.info(f"Loading curve fitting results from {CURVE_FITS_FILE}")
    
    if not CURVE_FITS_FILE.exists():
        raise FileNotFoundError(f"Curve fitting results not found at {CURVE_FITS_FILE}")
    
    fits_df = pd.read_csv(CURVE_FITS_FILE)
    logger.info(f"Loaded {len(fits_df)} exon curve fits")
    
    # Filter out exons with decreasing NMD efficiency (negative slope)
    initial_count = len(fits_df)
    fits_df = fits_df[fits_df['linear_p0'] >= 0].copy()
    dropped_count = initial_count - len(fits_df)
    logger.info(f"Filtered out {dropped_count} exons with decreasing NMD efficiency")
    logger.info(f"Retained {len(fits_df)} exons for analysis")
    
    # Build feature matrix from interpolated curves
    interpolated_curves = []
    valid_indices = []
    
    logger.info(f"Interpolating curves to {N_INTERPOLATION_POINTS} uniform points...")
    
    for idx, row in fits_df.iterrows():
        gene_name = row['gene_name']
        transcript_id = row['transcript_id']
        exon_idx = int(row['exon_idx'])
        
        try:
            exon_data = load_exon_data(gene_name, transcript_id, exon_idx)
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
            x_uniform = np.linspace(0, 1, N_INTERPOLATION_POINTS)
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
    
    logger.info(f"Running PCA on {len(X)} exons with {N_INTERPOLATION_POINTS} interpolated points per curve")
    
    # Standardize features (each position across all curves)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=N_PCS)
    X_pca = pca.fit_transform(X_scaled)
    
    logger.info(f"Total variance explained by {N_PCS} PCs: "
               f"{np.sum(pca.explained_variance_ratio_)*100:.1f}%")
    
    for i in range(N_PCS):
        logger.info(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}%")
    
    # Prepare PC scores dataframe
    scores_data = fits_df_clean[['gene_name', 'transcript_id', 'exon_idx', 
                                  'exon_length', 'linear_r2', 'linear_mse']].copy()
    for i in range(N_PCS):
        scores_data[f'PC{i+1}'] = X_pca[:, i]
    
    # Prepare PC components dataframe
    x_uniform = np.linspace(0, 1, N_INTERPOLATION_POINTS)
    components_data = pd.DataFrame({'relative_position': x_uniform})
    
    for i in range(N_PCS):
        # Component loading scaled by variance
        pc_curve = pca.components_[i] * np.sqrt(pca.explained_variance_[i])
        components_data[f'PC{i+1}_loading'] = pc_curve
    
    # Add variance explained as additional rows (metadata)
    variance_row = pd.DataFrame({
        'relative_position': [-1],  # Use -1 as sentinel value for metadata
        **{f'PC{i+1}_loading': [pca.explained_variance_ratio_[i]] for i in range(N_PCS)}
    })
    components_data = pd.concat([components_data, variance_row], ignore_index=True)
    
    return scores_data, components_data


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run PCA analysis and save results."""
    logger.info("Starting PCA analysis of long exon NMD efficiency curves")
    
    # Run PCA
    scores_data, components_data = run_pca_analysis()
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    scores_path = OUTPUT_DIR / "long_exon_pca_scores.csv"
    scores_data.to_csv(scores_path, index=False)
    logger.info(f"Saved PC scores to {scores_path}")
    
    components_path = OUTPUT_DIR / "long_exon_pca_components.csv"
    components_data.to_csv(components_path, index=False)
    logger.info(f"Saved PC components to {components_path}")
    
    logger.success("PCA analysis complete!")


if __name__ == "__main__":
    main()
