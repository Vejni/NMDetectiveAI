"""
DMS sigmoid fitting utilities.

Provides functions for fitting 4-parameter logistic curves to DMS data.
This module is used across multiple manuscript figures for consistent sigmoid fitting.
"""

import numpy as np
import pandas as pd
import torch
import typer
from scipy.optimize import least_squares
from loguru import logger
from pathlib import Path
from torch.utils.data import DataLoader

from NMD.config import PROCESSED_DATA_DIR, OUT_DIR, TABLES_DIR, MODELS_DIR

# Create typer app for CLI
app = typer.Typer()


# ============================================================================
# SIGMOID FITTING FUNCTIONS
# ============================================================================

def logistic4(x, A, K, B, M):
    """
    4-parameter logistic function.
    
    Args:
        x: Input positions
        A: Lower asymptote
        K: Upper asymptote
        B: Growth rate
        M: Midpoint (inflection point)
    
    Returns:
        Sigmoid values
    """
    return A + (K - A) / (1.0 + np.exp(-B * (x - M)))


def detect_orientation(y):
    """
    Detect if data is increasing or decreasing.
    
    Args:
        y: Y-values
    
    Returns:
        1 if increasing, -1 if decreasing
    """
    return 1 if y[-1] >= y[0] else -1


def logistic_initials(x_scaled, y):
    """
    Generate initial parameter guesses for logistic fit.
    
    Args:
        x_scaled: Scaled x-values (0-1)
        y: Y-values
    
    Returns:
        Tuple of (A0, K0, B0, M0) initial parameter guesses
    """
    A0 = np.min(y)
    K0 = np.max(y)
    mid = A0 + 0.5 * (K0 - A0)
    mid_index = np.argmin(np.abs(y - mid))
    M0 = x_scaled[mid_index]
    B0 = 5.0
    return A0, K0, B0, M0


def bounded_least_squares(func, x, y, p0, bounds):
    """
    Perform bounded least squares fitting.
    
    Args:
        func: Function to fit
        x: X-values
        y: Y-values
        p0: Initial parameter guesses
        bounds: Parameter bounds (lower, upper)
    
    Returns:
        Optimal parameters
    """
    def resid(p):
        return func(x, *p) - y
    res = least_squares(resid, p0, bounds=bounds, loss="soft_l1", f_scale=0.5, max_nfev=40000)
    return res.x


def r2_calc(y_true, y_pred):
    """
    Calculate R-squared.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        R-squared value
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan


def fit_logistic(
    x,
    y,
    min_slope: float = 0.00,
    max_slope: float = 15.0,
    return_full: bool = True
):
    """
    Fit the 4-parameter logistic model to (x, y).
    
    Args:
        x: x-coordinates (positions)
        y: y-coordinates (values)
        min_slope: Lower bound for slope (B)
        max_slope: Upper bound for slope (B)
        return_full: If True returns dict with diagnostics
        
    Returns:
        dict with params, y_pred, r2, etc. or tuple (params, y_pred)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    notes = []
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of equal length.")
    
    if np.allclose(y, y[0]):
        raise ValueError("y is (near) constant; logistic fit is not identifiable.")
    
    # Scale x to [0,1]
    x_min = float(x.min())
    x_max = float(x.max())
    span = x_max - x_min
    if span == 0:
        raise ValueError("All x values identical; cannot scale.")
    x_scaled = (x - x_min) / span
    
    # No orientation flip logic: always fit as written
    orientation = detect_orientation(y)
    x_fit = x_scaled
    
    # Initial parameter guesses
    A0, K0, B0, M0 = logistic_initials(x_fit, y)
    
    # Build bounds
    lower_A = A0 - abs(K0) * 2.0
    upper_K = K0 + abs(K0) * 2.0
    bounds = (
        [lower_A, A0, min_slope, 0.0],
        [K0, upper_K, max_slope, 1.0]
    )

    p0 = [A0, K0, min(max(B0, min_slope), max_slope * 0.5), M0]
    
    # Fit
    try:
        popt = bounded_least_squares(
            lambda xs, A, K, B, M: logistic4(xs, A, K, B, M),
            x_fit,
            y,
            p0,
            bounds
        )
    except Exception as e:
        raise RuntimeError(f"logistic4 fit failed: {e}") from e
    
    # Predictions
    y_pred_fit = logistic4(x_fit, *popt)
    
    y_pred = y_pred_fit
    
    # Goodness of fit
    r2_val = r2_calc(y, y_pred)
    
    result = {
        "params": [float(p) for p in popt],
        "y_pred": y_pred,
        "r2": float(r2_val),
        "orientation": int(orientation),
        "x_min": x_min,
        "x_max": x_max,
        "scaled_midpoint": float(popt[3]),
        "notes": ""
    }
    
    return result if return_full else (result["params"], result["y_pred"])


def fit_sigmoids_to_dms_genes(
    fitness_df: pd.DataFrame,
    output_file: Path = None
) -> pd.DataFrame:
    """
    Fit 4-parameter sigmoids to all DMS SP genes and save results.
    
    Args:
        fitness_df: DMS fitness DataFrame with 'gene', 'PTCposition', 'NMDeff_Norm' columns
        output_file: Optional path to save sigmoid parameters
    
    Returns:
        DataFrame with sigmoid parameters (A, K, B, M, r2) for each gene
    """
    logger.info(f"Fitting sigmoids to {fitness_df['gene'].nunique()} genes")
    
    genes = fitness_df['gene'].unique()
    results = []
    failed_genes = []
    
    for gene in genes:
        gene_df = fitness_df[fitness_df['gene'] == gene][['PTCposition', 'NMDeff_Norm']]
        
        x = gene_df['PTCposition'].values
        y = gene_df['NMDeff_Norm'].values
        
        if len(x) < 4:
            failed_genes.append(gene)
            continue
        
        try:
            result = fit_logistic(x, y)
            results.append({
                'gene': gene,
                'A': result['params'][0],
                'K': result['params'][1],
                'B': result['params'][2],
                'M': result['params'][3],
                'r2': result['r2']
            })
        except Exception:
            failed_genes.append(gene)
    
    params_df = pd.DataFrame(results)
    
    if len(params_df) == 0:
        raise ValueError(f"Failed to fit sigmoid to any genes! Failed: {failed_genes[:10]}")
    
    logger.info(f"Successfully fit {len(params_df)}/{len(genes)} genes")
    
    if failed_genes:
        logger.warning(f"Failed to fit {len(failed_genes)} genes: {failed_genes[:5]}...")
    
    # Save if output file provided
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        params_df.to_csv(output_file, index=False)
        logger.info(f"Saved sigmoid parameters to {output_file}")
    
    return params_df


def load_or_compute_sigmoid_params(
    fitness_file: Path = None,
    output_file: Path = None,
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Load sigmoid parameters from file or compute them.
    
    Args:
        fitness_file: Path to DMS fitness data
        output_file: Path to save/load sigmoid parameters
        force_recompute: If True, recompute even if file exists
    
    Returns:
        DataFrame with sigmoid parameters
    """
    # Set defaults
    if fitness_file is None:
        fitness_file = PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv"
    if output_file is None:
        output_file = TABLES_DIR / "SP" / "sigmoid_params_observations.csv"
    
    # Check if already computed
    if output_file.exists() and not force_recompute:
        logger.info(f"Loading sigmoid parameters from {output_file}")
        return pd.read_csv(output_file)
    
    # Compute sigmoid parameters
    logger.info(f"Computing sigmoid parameters from {fitness_file}")
    fitness_df = pd.read_csv(fitness_file)
    
    return fit_sigmoids_to_dms_genes(fitness_df, output_file)


@app.command()
def fit_sigmoids_to_observations(
    fitness_file: Path = None,
    output_file: Path = None
) -> pd.DataFrame:
    """
    Fit sigmoids to observed NMDeff values for each DMS gene.
    
    Args:
        fitness_file: Path to DMS fitness data with 'gene', 'PTCposition_nt', 'NMDeff' columns
        output_file: Path to save sigmoid parameters (default: TABLES_DIR/SP/sigmoid_params_observations.csv)
    
    Returns:
        DataFrame with sigmoid parameters (gene, A, K, B, M, r2) for each gene
    """
    # Set defaults
    if fitness_file is None:
        fitness_file = PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv"
    if output_file is None:
        output_file = TABLES_DIR / "SP" / "sigmoid_params_observations.csv"
    
    logger.info(f"Fitting sigmoids to observed NMDeff values from {fitness_file}")
    fitness_df = pd.read_csv(fitness_file)
    
    genes = fitness_df['gene'].unique()
    results = []
    failed_genes = []
    
    for gene in genes:
        gene_df = fitness_df[fitness_df['gene'] == gene].copy()
        
        positions = gene_df['PTCposition_nt'].values
        observations = gene_df['NMDeff_Norm'].values
        
        if len(positions) < 4:
            logger.warning(f"Skipping {gene}: insufficient data ({len(positions)} points)")
            failed_genes.append(gene)
            continue
        
        try:
            result = fit_logistic(positions, observations)
            results.append({
                'gene': gene,
                'A': result['params'][0],
                'K': result['params'][1],
                'B': result['params'][2],
                'M': result['params'][3],
                'r2': result['r2'],
                'n_observations': len(positions)
            })
            logger.info(f"  {gene}: R²={result['r2']:.3f}")
        except Exception as e:
            logger.warning(f"Failed to fit sigmoid for {gene}: {e}")
            failed_genes.append(gene)
    
    params_df = pd.DataFrame(results)
    
    if len(params_df) == 0:
        raise ValueError(f"Failed to fit sigmoid to any genes!")
    
    logger.info(f"Successfully fit {len(params_df)}/{len(genes)} genes")
    
    if failed_genes:
        logger.warning(f"Failed to fit {len(failed_genes)} genes: {failed_genes[:10]}")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    params_df.to_csv(output_file, index=False)
    logger.success(f"Saved sigmoid parameters to {output_file}")
    
    return params_df


@app.command()
def fit_sigmoids_to_predictions(
    fitness_file: Path = None,
    model_path: Path = None,
    sequences_file: Path = None,
    output_file: Path = None
) -> pd.DataFrame:
    """
    Predict with NMDetectiveAI and fit sigmoids to predictions for each DMS gene.
    
    Args:
        fitness_file: Path to DMS fitness data
        model_path: Path to NMDetectiveAI model (default: MODELS_DIR/NMDetectiveAI.pt)
        sequences_file: Path to pre-processed sequences pickle (default: PROCESSED_DATA_DIR/DMS_SP/processed_sequences.pkl)
        output_file: Path to save sigmoid parameters (default: TABLES_DIR/SP/sigmoid_params_predictions.csv)
    
    Returns:
        DataFrame with sigmoid parameters (gene, A, K, B, M, r2) for each gene
    """
    # Import here to avoid circular dependencies
    from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
    from NMD.modeling.SequenceDataset import SequenceDataset
    from NMD.modeling.TrainerConfig import TrainerConfig
    from NMD.utils import load_model
    import pickle
    
    # Set defaults
    if fitness_file is None:
        fitness_file = PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv"
    if model_path is None:
        model_path = MODELS_DIR / "NMDetectiveAI.pt"
    if sequences_file is None:
        sequences_file = PROCESSED_DATA_DIR / "DMS_SP" / "processed_sequences.pkl"
    if output_file is None:
        output_file = TABLES_DIR / "SP" / "sigmoid_params_predictions.csv"
    
    logger.info(f"Fitting sigmoids to NMDetectiveAI predictions")
    logger.info(f"Loading model from {model_path}")
    
    # Load model
    config = TrainerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NMDetectiveAI(
        hidden_dims=config.dnn_hidden_dims,
        dropout=config.dnn_dropout,
        random_init=config.random_init,
        use_mlm=config.Orthrus_MLM,
        activation_function=config.activation_function,
        use_layer_norm=config.use_layer_norm,
    ).to(device)
    load_model(model, model_path, device=device)
    model.eval()
    logger.info("Model loaded successfully")
    
    # Load data
    logger.info(f"Loading sequences from {sequences_file}")
    with open(sequences_file, 'rb') as f:
        dms_sequences_all = pickle.load(f)
    
    dms_metadata_all = pd.read_csv(fitness_file)
    logger.info(f"Loaded {len(dms_sequences_all)} sequences")
    
    # Process each gene
    genes = dms_metadata_all['gene'].unique()
    results = []
    failed_genes = []
    
    for gene in genes:
        logger.info(f"Processing {gene}...")
        gene_data = dms_metadata_all[dms_metadata_all['gene'] == gene].copy()
        
        if len(gene_data) < 4:
            logger.warning(f"Skipping {gene}: insufficient data ({len(gene_data)} points)")
            failed_genes.append(gene)
            continue
        
        # Get sequences for this gene
        gene_sequences = [dms_sequences_all[i] for i in gene_data.index]
        positions = gene_data['PTCposition_nt'].values
        
        # Generate predictions
        eval_dataset = SequenceDataset(gene_data, gene_sequences, label_col="NMDeff")
        eval_loader = DataLoader(eval_dataset, batch_size=1)
        
        predictions = []
        with torch.no_grad():
            for batch_sequences, batch_lengths, _ in eval_loader:
                batch_sequences, batch_lengths = [x.to(device) for x in (batch_sequences, batch_lengths)]
                batch_preds = model(batch_sequences, batch_lengths).squeeze()
                if batch_preds.dim() == 0:
                    predictions.append(float(batch_preds.cpu().numpy()))
                else:
                    predictions.extend(batch_preds.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Fit sigmoid to predictions
        try:
            result = fit_logistic(positions, predictions)
            results.append({
                'gene': gene,
                'A': result['params'][0],
                'K': result['params'][1],
                'B': result['params'][2],
                'M': result['params'][3],
                'r2': result['r2'],
                'rho': np.corrcoef(predictions, gene_data['NMDeff'].values)[0,1],
                'n_observations': len(positions)
            })
            logger.info(f"  {gene}: R²={result['r2']:.3f}")
        except Exception as e:
            logger.warning(f"Failed to fit sigmoid for {gene}: {e}")
            failed_genes.append(gene)
    
    params_df = pd.DataFrame(results)
    
    if len(params_df) == 0:
        raise ValueError(f"Failed to fit sigmoid to any genes!")
    
    logger.info(f"Successfully fit {len(params_df)}/{len(genes)} genes")
    
    if failed_genes:
        logger.warning(f"Failed to fit {len(failed_genes)} genes: {failed_genes[:10]}")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    params_df.to_csv(output_file, index=False)
    logger.success(f"Saved sigmoid parameters to {output_file}")
    
    return params_df


if __name__ == "__main__":
    #fit_sigmoids_to_observations()
    fit_sigmoids_to_predictions()