"""
Extract PTC predictions for long exons from genome-wide MANE transcript predictions.

A long exon is defined as:
- Length > 400 nucleotides
- Not the first, last, or penultimate coding exon
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from genome_kit import Genome
from scipy.optimize import curve_fit

from NMD.config import TABLES_DIR, GENCODE_VERSION


# ========== Curve Fitting Functions ==========

def linear_model(x, a, b):
    """Linear model: y = a*x + b"""
    return a * x + b


def logistic_4pl(x, A, B, C, D):
    """
    4-parameter logistic model (sigmoid):
    y = D + (A - D) / (1 + (x/C)^B)
    
    A: minimum asymptote
    D: maximum asymptote
    C: inflection point
    B: slope factor (Hill slope)
    """
    return D + (A - D) / (1 + (x / C) ** B)


def piecewise_linear_2(x, x1, a1, b1, a2, b2):
    """
    Piecewise linear with 2 segments:
    - Segment 1: y = a1*x + b1 for x <= x1
    - Segment 2: y = a2*x + b2 for x > x1
    """
    return np.where(x <= x1, a1 * x + b1, a2 * x + b2)


def piecewise_linear_3(x, x1, x2, a1, b1, a2, b2, a3, b3):
    """
    Piecewise linear with 3 segments:
    - Segment 1: y = a1*x + b1 for x <= x1
    - Segment 2: y = a2*x + b2 for x1 < x <= x2
    - Segment 3: y = a3*x + b3 for x > x2
    """
    return np.where(x <= x1, a1 * x + b1,
                   np.where(x <= x2, a2 * x + b2, a3 * x + b3))


def polynomial_2(x, a, b, c):
    """Polynomial degree 2: y = a*x^2 + b*x + c"""
    return a * x**2 + b * x + c


def polynomial_3(x, a, b, c, d):
    """Polynomial degree 3: y = a*x^3 + b*x^2 + c*x + d"""
    return a * x**3 + b * x**2 + c * x + d


def polynomial_4(x, a, b, c, d, e):
    """Polynomial degree 4: y = a*x^4 + b*x^3 + c*x^2 + d*x + e"""
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def polynomial_5(x, a, b, c, d, e, f):
    """Polynomial degree 5: y = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f"""
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f


def calculate_r2(y_true, y_pred):
    """Calculate R-squared value"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def calculate_aic(n, mse, k):
    """
    Calculate Akaike Information Criterion (AIC)
    n: number of observations
    mse: mean squared error
    k: number of parameters
    """
    return n * np.log(mse) + 2 * k


def calculate_bic(n, mse, k):
    """
    Calculate Bayesian Information Criterion (BIC)
    n: number of observations
    mse: mean squared error
    k: number of parameters
    """
    return n * np.log(mse) + k * np.log(n)


def get_cds_exon_boundaries(transcript):
    """
    Get the cumulative boundaries of CDS exons in coding nucleotide coordinates.
    
    Args:
        transcript: genome_kit Transcript object
        
    Returns:
        list of tuples: [(start_nt, end_nt, exon_idx, exon_length), ...]
                       where positions are 1-indexed in CDS coordinates
    """
    if not hasattr(transcript, 'cdss') or transcript.cdss is None or len(transcript.cdss) == 0:
        return []
    
    exon_boundaries = []
    cumulative_position = 1  # 1-indexed CDS position
    
    for exon_idx, cds_exon in enumerate(transcript.cdss, start=1):
        exon_length = len(cds_exon)
        exon_start = cumulative_position
        exon_end = cumulative_position + exon_length - 1
        
        exon_boundaries.append((exon_start, exon_end, exon_idx, exon_length))
        cumulative_position += exon_length
    
    return exon_boundaries


def identify_long_exons(transcript, min_length=400):
    """
    Identify long exons in a transcript that are not first, last, or penultimate coding exons.
    
    Args:
        transcript: genome_kit Transcript object
        min_length: Minimum length in nucleotides to be considered a long exon
        
    Returns:
        list of tuples: [(exon_idx, exon_start, exon_end, exon_length), ...]
    """
    exon_boundaries = get_cds_exon_boundaries(transcript)
    
    if len(exon_boundaries) < 4:  # Need at least 4 exons to have non-edge long exons
        return []
    
    long_exons = []
    num_exons = len(exon_boundaries)
    
    for exon_start, exon_end, exon_idx, exon_length in exon_boundaries:
        # Skip first, last, and penultimate exons
        if exon_idx == 1 or exon_idx == num_exons or exon_idx == num_exons - 1:
            continue
        
        # Check if long enough
        if exon_length > min_length:
            long_exons.append((exon_idx, exon_start, exon_end, exon_length))
    
    return long_exons


def extract_long_exon_predictions(predictions_df, long_exons, utr5_offset=0):
    """
    Extract predictions for PTCs that fall within long exons.
    
    Args:
        predictions_df: DataFrame with ptc_position (transcript coords) and prediction columns
        long_exons: List of (exon_idx, exon_start, exon_end, exon_length) tuples in CDS coords
        utr5_offset: Length of 5'UTR to convert CDS coords to transcript coords.
            ptc_position in the predictions file is in transcript coordinates
            (1-indexed, includes 5'UTR), while exon boundaries from
            get_cds_exon_boundaries() are in CDS-only coordinates.
        
    Returns:
        dict: {exon_idx: DataFrame with ptc_position and prediction for that exon}
    """
    exon_predictions = {}
    
    for exon_idx, exon_start, exon_end, exon_length in long_exons:
        # Convert CDS coordinates to transcript coordinates for filtering
        tx_exon_start = exon_start + utr5_offset
        tx_exon_end = exon_end + utr5_offset
        
        # Filter predictions within this exon's boundaries (transcript coords)
        mask = (predictions_df['ptc_position'] >= tx_exon_start) & (predictions_df['ptc_position'] <= tx_exon_end)
        exon_df = predictions_df[mask].copy()
        
        if len(exon_df) > 0:
            # Add exon metadata (CDS coordinates)
            exon_df['exon_idx'] = exon_idx
            exon_df['exon_start'] = exon_start
            exon_df['exon_end'] = exon_end
            exon_df['exon_length'] = exon_length
            
            exon_predictions[exon_idx] = exon_df
    
    return exon_predictions


def fit_all_models_to_exon(x, y):
    """
    Fit all models to a single exon's data.
    Returns a dictionary with fit results for each model.
    
    Args:
        x: array of positions
        y: array of predictions
        
    Returns:
        dict: Results for each model with parameters and metrics
    """
    # Normalize positions to 0-1 for better fitting
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    
    results = {}
    
    # Define models with their functions, initial parameters, and number of params
    models = {
        'linear': {
            'func': linear_model,
            'p0': [0, y.mean()],
            'n_params': 2
        },
        'logistic_4pl': {
            'func': logistic_4pl,
            'p0': [y.min(), 1.0, 0.5, y.max()],
            'n_params': 4
        },
        'piecewise_2': {
            'func': piecewise_linear_2,
            'p0': [0.5, 0, y.mean(), 0, y.mean()],
            'n_params': 5
        },
        'piecewise_3': {
            'func': piecewise_linear_3,
            'p0': [0.33, 0.67, 0, y.mean(), 0, y.mean(), 0, y.mean()],
            'n_params': 8
        },
        'poly_2': {
            'func': polynomial_2,
            'p0': [0, 0, y.mean()],
            'n_params': 3
        },
        'poly_3': {
            'func': polynomial_3,
            'p0': [0, 0, 0, y.mean()],
            'n_params': 4
        },
        'poly_4': {
            'func': polynomial_4,
            'p0': [0, 0, 0, 0, y.mean()],
            'n_params': 5
        },
        'poly_5': {
            'func': polynomial_5,
            'p0': [0, 0, 0, 0, 0, y.mean()],
            'n_params': 6
        }
    }
    
    for model_name, model_info in models.items():
        try:
            popt, _ = curve_fit(
                model_info['func'], 
                x_norm, 
                y,
                p0=model_info['p0'],
                maxfev=5000
            )
            y_pred = model_info['func'](x_norm, *popt)
            r2 = calculate_r2(y, y_pred)
            mse = np.mean((y - y_pred) ** 2)
            
            results[model_name] = {
                'params': popt,
                'r2': r2,
                'mse': mse,
                'aic': calculate_aic(len(y), mse, model_info['n_params']),
                'bic': calculate_bic(len(y), mse, model_info['n_params']),
                'success': True
            }
        except Exception as e:
            results[model_name] = {
                'params': None,
                'r2': np.nan,
                'mse': np.nan,
                'aic': np.nan,
                'bic': np.nan,
                'success': False,
                'error': str(e)
            }
    
    return results


def process_all_transcripts(gw_dir, output_dir, min_length=400):
    """
    Process all transcript prediction files, extract long exon data, and fit curves.
    
    Args:
        gw_dir: Directory containing genome-wide prediction files
        output_dir: Output directory for long exon predictions
        min_length: Minimum length for long exons
    """
    gw_dir = Path(gw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load genome
    logger.info(f"Loading genome: {GENCODE_VERSION}")
    genome = Genome(GENCODE_VERSION)
    
    # Get all prediction files
    prediction_files = list(gw_dir.glob("*_ptc_predictions.csv"))
    logger.info(f"Found {len(prediction_files)} prediction files")
    
    transcripts_with_long_exons = 0
    total_long_exons = 0
    all_fit_results = []
    
    for pred_file in tqdm(prediction_files, desc="Processing transcripts"):
        # Extract transcript ID from filename (e.g., "A1BG_ENST00000263100.7_ptc_predictions.csv")
        filename_parts = pred_file.stem.split('_')
        gene_name = filename_parts[0]
        transcript_id = filename_parts[1]  # Full versioned ID
        
        try:
            # Load predictions
            df = pd.read_csv(pred_file)
            
            # Get transcript from genome
            transcript = genome.transcripts[transcript_id]
            
            # Identify long exons
            long_exons = identify_long_exons(transcript, min_length=min_length)
            
            if len(long_exons) == 0:
                continue
            
            transcripts_with_long_exons += 1
            
            # Calculate 5'UTR offset for coordinate conversion
            # ptc_position is in transcript coords (includes 5'UTR),
            # but exon boundaries are in CDS-only coords
            utr5_offset = sum(len(exon) for exon in transcript.utr5s) if transcript.utr5s is not None else 0
            
            # Extract predictions for each long exon
            exon_predictions = extract_long_exon_predictions(df, long_exons, utr5_offset=utr5_offset)
            
            # Save each long exon to a separate file and fit curves
            for exon_idx, exon_df in exon_predictions.items():
                # Save individual exon file
                output_filename = f"{gene_name}_{transcript_id}_exon{exon_idx}.csv"
                output_path = output_dir / output_filename
                exon_df.to_csv(output_path, index=False)
                total_long_exons += 1
                
                # Fit all curve models
                x = exon_df['ptc_position'].values
                y = exon_df['prediction'].values
                
                if len(x) < 10:  # Skip if too few points
                    continue
                
                fit_results = fit_all_models_to_exon(x, y)
                
                # Compile results for this exon
                exon_result = {
                    'gene_name': gene_name,
                    'transcript_id': transcript_id,
                    'exon_idx': int(exon_idx),
                    'exon_length': int(exon_df['exon_length'].iloc[0]),
                    'exon_start': int(exon_df['exon_start'].iloc[0]),
                    'exon_end': int(exon_df['exon_end'].iloc[0]),
                    'n_points': len(x)
                }
                
                # Add results for each model
                for model_name, model_result in fit_results.items():
                    if model_result['success']:
                        # Store parameters as separate columns
                        params = model_result['params']
                        for i, param in enumerate(params):
                            exon_result[f'{model_name}_p{i}'] = param
                        
                        # Store performance metrics
                        exon_result[f'{model_name}_r2'] = model_result['r2']
                        exon_result[f'{model_name}_mse'] = model_result['mse']
                        exon_result[f'{model_name}_aic'] = model_result['aic']
                        exon_result[f'{model_name}_bic'] = model_result['bic']
                    else:
                        # Mark as failed
                        exon_result[f'{model_name}_r2'] = np.nan
                        exon_result[f'{model_name}_mse'] = np.nan
                        exon_result[f'{model_name}_aic'] = np.nan
                        exon_result[f'{model_name}_bic'] = np.nan
                
                all_fit_results.append(exon_result)
        
        except KeyError:
            logger.warning(f"Transcript {transcript_id} not found in genome")
            continue
        except Exception as e:
            logger.error(f"Error processing {pred_file.name}: {e}")
            continue
    
    # Save comprehensive fit results table
    if all_fit_results:
        fit_results_df = pd.DataFrame(all_fit_results)
        exon_analysis_dir = TABLES_DIR / "exon_analysis"
        exon_analysis_dir.mkdir(parents=True, exist_ok=True)
        fit_results_path = exon_analysis_dir / "long_exon_curve_fits.csv"
        fit_results_df.to_csv(fit_results_path, index=False)
        logger.success(f"Saved curve fitting results to {fit_results_path}")
    
    logger.success(f"Processed {transcripts_with_long_exons} transcripts with long exons")
    logger.success(f"Extracted {total_long_exons} long exon prediction files to {output_dir}")
    logger.success(f"Fit curves for {len(all_fit_results)} long exons")


def main():
    """Main entry point."""
    gw_dir = TABLES_DIR / "GW"
    output_dir = TABLES_DIR / "LE"
    
    logger.info("Starting long exon extraction from MANE transcript predictions")
    logger.info(f"Input directory: {gw_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    process_all_transcripts(gw_dir, output_dir, min_length=400)


if __name__ == "__main__":
    main()
