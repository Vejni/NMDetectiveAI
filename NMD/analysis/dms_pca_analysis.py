"""
DMS PCA analysis with LOESS interpolation.

Provides functions for performing PCA on DMS fitness data with LOESS smoothing.
This module is used across multiple manuscript figures for consistent PCA analysis.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from loguru import logger
from pathlib import Path

from NMD.config import PROCESSED_DATA_DIR, TABLES_DIR


# Configuration
PTC_POSITIONS = np.arange(1, 84)  # 1-83
LOESS_FRAC = 0.3  # Smoothing parameter
RANDOM_SEED = 42  # For reproducibility


def fit_loess_interpolate(gene_df: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    """
    Fit LOESS model to gene data and interpolate NMDeff_Norm values at all positions.
    
    Args:
        gene_df: DataFrame with PTCposition and NMDeff_Norm columns for a single gene
        positions: Array of positions to interpolate at (1-83)
    
    Returns:
        Array of interpolated NMDeff_Norm values
    """
    # Sort by position
    gene_df = gene_df.sort_values('PTCposition')
    
    x = gene_df['PTCposition'].values
    y = gene_df['NMDeff_Norm'].values
    
    # Check if we have enough points
    if len(x) < 3:
        logger.warning(f"Gene has only {len(x)} points, using linear interpolation")
        # Use linear interpolation for genes with few points
        interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        return interp_func(positions)
    
    # Fit LOESS
    smoothed = lowess(y, x, frac=LOESS_FRAC, return_sorted=True)
    
    # Create interpolation function from smoothed data
    interp_func = interp1d(
        smoothed[:, 0], 
        smoothed[:, 1], 
        kind='linear', 
        bounds_error=False, 
        fill_value='extrapolate'
    )
    
    return interp_func(positions)


def create_interpolated_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create matrix of interpolated NMDeff_Norm values for all genes.
    
    Args:
        df: DMS SP fitness DataFrame with 'gene', 'PTCposition', 'NMDeff_Norm' columns
    
    Returns:
        DataFrame with genes as rows and PTCpositions as columns
    """
    genes = df['gene'].unique()
    logger.info(f"Processing {len(genes)} genes for LOESS interpolation")
    
    # Initialize matrix
    matrix = np.zeros((len(genes), len(PTC_POSITIONS)))
    
    for i, gene in enumerate(genes):
        gene_df = df[df['gene'] == gene][['PTCposition', 'NMDeff_Norm']]
        
        try:
            matrix[i, :] = fit_loess_interpolate(gene_df, PTC_POSITIONS)
        except Exception as e:
            logger.warning(f"Error interpolating {gene}: {e}")
            matrix[i, :] = np.nan
    
    # Create DataFrame
    matrix_df = pd.DataFrame(
        matrix, 
        index=genes, 
        columns=[f'PTC_{pos}' for pos in PTC_POSITIONS]
    )
    
    # Remove genes with NaN values
    matrix_df = matrix_df.dropna()
    logger.info(f"Matrix shape after removing NaN: {matrix_df.shape}")
    
    return matrix_df


def perform_pca(
    matrix_df: pd.DataFrame,
    n_components: int = None,
    random_seed: int = RANDOM_SEED
) -> tuple:
    """
    Perform PCA on interpolated matrix.
    
    Args:
        matrix_df: Matrix of interpolated values (genes x positions)
        n_components: Number of PCA components to compute (None = all)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (pca_result_array, pca_object, scaler_object)
    """
    # Standardize features
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix_df)
    
    # Perform PCA
    pca = PCA(n_components=n_components, random_state=random_seed)
    pca_result = pca.fit_transform(matrix_scaled)
    
    # Log explained variance
    logger.info(f"PCA computed with {pca.n_components_} components")
    logger.info(f"Explained variance (first 5): {pca.explained_variance_ratio_[:5]}")
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    logger.info(f"Cumulative variance (first 5): {cumsum_var[:5]}")
    
    return pca_result, pca, scaler


def compute_and_save_dms_pca(
    fitness_df: pd.DataFrame,
    output_matrix_file: Path = None,
    n_components: int = None
) -> tuple:
    """
    Compute PCA on DMS data and save interpolated matrix.
    
    Args:
        fitness_df: DMS fitness DataFrame
        output_matrix_file: Path to save interpolated matrix
        n_components: Number of PCA components (None = all)
    
    Returns:
        Tuple of (pca_result, pca_object, scaler_object, matrix_df)
    """
    logger.info("Creating interpolated matrix with LOESS...")
    matrix_df = create_interpolated_matrix(fitness_df)
    
    # Save matrix if output file provided
    if output_matrix_file is not None:
        output_matrix_file.parent.mkdir(parents=True, exist_ok=True)
        matrix_df.to_csv(output_matrix_file)
        logger.info(f"Saved interpolated matrix to {output_matrix_file}")
    
    # Perform PCA
    logger.info("Performing PCA...")
    pca_result, pca, scaler = perform_pca(matrix_df, n_components=n_components)
    
    return pca_result, pca, scaler, matrix_df


def load_or_compute_dms_pca(
    fitness_file: Path = None,
    matrix_file: Path = None,
    n_components: int = None,
    force_recompute: bool = False
) -> tuple:
    """
    Load interpolated matrix from file or compute it, then perform PCA.
    
    Args:
        fitness_file: Path to DMS fitness data
        matrix_file: Path to save/load interpolated matrix
        n_components: Number of PCA components (None = all)
        force_recompute: If True, recompute even if file exists
    
    Returns:
        Tuple of (pca_result, pca_object, scaler_object, matrix_df)
    """
    # Set defaults
    if fitness_file is None:
        fitness_file = PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv"
    if matrix_file is None:
        matrix_file = TABLES_DIR / "SP" / "loess_interpolated_matrix.csv"
    
    # Check if matrix already exists
    if matrix_file.exists() and not force_recompute:
        logger.info(f"Loading existing interpolated matrix from {matrix_file}")
        matrix_df = pd.read_csv(matrix_file, index_col=0)
        
        logger.info("Performing PCA on loaded matrix...")
        pca_result, pca, scaler = perform_pca(matrix_df, n_components=n_components)
        
        return pca_result, pca, scaler, matrix_df
    
    # Compute from scratch
    logger.info(f"Computing interpolated matrix from {fitness_file}")
    fitness_df = pd.read_csv(fitness_file)
    
    return compute_and_save_dms_pca(
        fitness_df,
        output_matrix_file=matrix_file,
        n_components=n_components
    )


def main(fitness_file: Path = None, force_recompute: bool = False):
    """
    Compute LOESS-interpolated matrix and PCA, save results to TABLES_DIR / SP.

    Saves:
    - TABLES_DIR / SP / loess_interpolated_matrix.csv
    - TABLES_DIR / SP / pca_matrix.csv
    """
    if fitness_file is None:
        fitness_file = PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv"

    # Ensure output directory
    loess_file = TABLES_DIR / "SP" / "loess_interpolated_matrix.csv"
    pca_file = TABLES_DIR / "SP" / "pca_matrix.csv"

    # Compute interpolated matrix and PCA (this function saves the loess matrix)
    pca_result, pca, scaler, matrix_df = compute_and_save_dms_pca(
        pd.read_csv(fitness_file), output_matrix_file=loess_file, n_components=None
    )

    # Build PCA DataFrame and save
    pca_df = pd.DataFrame(
        pca_result,
        index=matrix_df.index,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
    )
    pca_file.parent.mkdir(parents=True, exist_ok=True)
    pca_df.to_csv(pca_file)
    logger.info(f"Saved PCA matrix to {pca_file}")

    return pca_df
