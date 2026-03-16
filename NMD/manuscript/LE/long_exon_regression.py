"""
Radar plots comparing DMS and TCGA regression parameters across exon lengths.

Creates two radar plots showing:
1. Intercept values for DMS and TCGA across different exon lengths
2. Slope values (per 100nt) for DMS and TCGA across different exon lengths
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

import pickle
import torch
from torch.utils.data import DataLoader

from NMD.modeling.SequenceDataset import SequenceDataset
from NMD.modeling.TrainerConfig import TrainerConfig
from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
from NMD.utils import load_model, collate_fn

from NMD.config import (
    COLOURS,
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR
)
from NMD.manuscript.output import get_paths


# ============================================================================
# CONFIGURATION
# ============================================================================

# Input paths
SCRIPT_NAME = "long_exon_regression"
DMS_LE_FILE = PROCESSED_DATA_DIR / "DMS_LE/fitness.csv"
PTC_FILE = INTERIM_DATA_DIR / "PTC" / "somatic_TCGA.csv"

PLOT_TITLE = "Linear regression\nparameters by exon length"

# Exon lengths to include
EXON_LENGTHS = ['500bps', '750bps', '1000bps', '1500bps', '2500bps', '3000bps', '3426bps']

# Sublibrary ranges for PTC data matching
SUBLIB_RANGES = {
    '500bps': (450, 550),
    '750bps': (650, 850),
    '1000bps': (850, 1150),
    '1500bps': (1250, 1750),
    '2500bps': (2250, 2750),
    '3000bps': (2750, 3250),
    '3426bps': (3250, 5000),
}

# Colors
DMS_COLOR = COLOURS[1]
TCGA_COLOR = 'darkgray'

# Figure settings
FIGURE_SIZE = (6, 5)
DPI = 300
MODEL_PATH = MODELS_DIR / "NMDetectiveAI.pt"
PTC_SEQUENCES = PROCESSED_DATA_DIR / "PTC" / "somatic_TCGA.pkl"
PREDICTION_COLOR = COLOURS[6] if len(COLOURS) > 6 else "#333333"
# Plot styling
AXIS_LINEWIDTH = 1.8
TICK_WIDTH = 1.4
BASE_FONTSIZE = 14
TITLE_FONTSIZE = 18


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_data():
    """Process DMS and TCGA data and calculate regression parameters."""
    logger.info("Starting data processing...")
    
    # Load DMS Long Exon data
    logger.info(f"Loading DMS Long Exon data from {DMS_LE_FILE}")
    dms = pd.read_csv(DMS_LE_FILE)
    logger.info(f"Loaded {len(dms)} DMS Long Exon variants")
    
    # Calculate stop codon position
    dms['stop_down_123nt'] = (dms.stop_type + dms.down_123nt).str.replace("U", "T")
    dms['stop_pos'] = dms.apply(lambda row: row.nt_seq.find(row.stop_down_123nt), axis=1)
    dms["PTC_EJC_dist"] = dms["exon_length"].str.replace("bps", "").astype(int) - dms["stop_pos"]
    
    # Load PTC TCGA data
    logger.info(f"Loading PTC TCGA data from {PTC_FILE}")
    ptc_full = pd.read_csv(PTC_FILE)
    logger.info(f"Loaded {len(ptc_full)} PTC variants (full)")

    # Define filter mask (apply after attaching predictions to full table)
    mask = (
        (ptc_full.Last_Exon == False) &
        (ptc_full.Penultimate_Exon == False) &
        (ptc_full.Start_Prox == False) &
        (ptc_full.Ref != "-") &
        (ptc_full.Alt != "-")
    )
    ptc_df = ptc_full[mask].copy()
    logger.info(f"After filtering PTC data: {len(ptc_df)} variants")
    
    # Calculate regression parameters for each exon length
    results = []

    # Try to load PTC sequences and run NMDetective-AI predictions (optional)
    ptc_predictions = None
    if PTC_SEQUENCES.exists() and MODEL_PATH.exists():
        try:
            logger.info(f"Loading PTC sequences from {PTC_SEQUENCES}")
            with open(PTC_SEQUENCES, "rb") as f:
                sequences = pickle.load(f)
            # Ensure sequences length matches the full PTC dataframe
            if len(sequences) == len(ptc_full):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                cfg = TrainerConfig()
                model = NMDetectiveAI(
                    hidden_dims=cfg.dnn_hidden_dims,
                    dropout=cfg.dnn_dropout,
                    random_init=cfg.random_init,
                    use_mlm=cfg.Orthrus_MLM,
                    activation_function=cfg.activation_function,
                    use_layer_norm=cfg.use_layer_norm,
                ).to(device)

                load_model(model, MODEL_PATH, device=device)
                model.eval()

                # Create dataset and dataloader to predict on all PTCs
                dummy_df = pd.DataFrame({"y": [0] * len(sequences)})
                eval_dataset = SequenceDataset(dummy_df, sequences, label_col="y")
                eval_loader = DataLoader(eval_dataset, batch_size=8, collate_fn=collate_fn)

                preds = []
                with torch.no_grad():
                    for batch_sequences, batch_lengths, _ in eval_loader:
                        batch_sequences = batch_sequences.to(device)
                        batch_lengths = batch_lengths.to(device)
                        outputs = model(batch_sequences, batch_lengths)
                        preds.extend(outputs.cpu().numpy().flatten())

                ptc_predictions = np.array(preds)
                # attach predictions to the full table, then re-derive filtered table
                ptc_full['NMDeff_Predicted'] = ptc_predictions
                ptc_df = ptc_full[mask].copy()
                logger.info(f"Generated NMDetective-AI predictions for {len(preds)} PTCs")
            else:
                logger.warning("PTC sequences length does not match full PTC dataframe; skipping AI predictions")
        except Exception as e:
            logger.warning(f"Failed to generate PTC predictions: {e}")

    
    for exon_len in EXON_LENGTHS:
        # DMS data - handle merged category
        if exon_len == '125-250bps':
            exon_data = dms[dms['exon_length'].isin(['125bps', '250bps'])]
        else:
            exon_data = dms[dms['exon_length'] == exon_len]
        
        if len(exon_data) > 1:
            x = -exon_data['PTC_EJC_dist']
            y = exon_data['NMDeff']
            try:
                (z, cov) = np.polyfit(x, y, 1, cov=True)
                dms_slope = z[0] * 100  # Convert to per 100nt
                dms_intercept = z[1]
                # standard errors: sqrt of diagonal of covariance
                dms_slope_se = np.sqrt(cov[0, 0]) * 100
                dms_intercept_se = np.sqrt(cov[1, 1])
            except Exception:
                dms_slope = np.nan
                dms_intercept = np.nan
                dms_slope_se = np.nan
                dms_intercept_se = np.nan

            dms_r = np.corrcoef(x, y)[0, 1]
            dms_n = len(exon_data)
        else:
            dms_slope = np.nan
            dms_intercept = np.nan
            dms_r = np.nan
            dms_n = 0
            dms_slope_se = np.nan
            dms_intercept_se = np.nan
        
        # TCGA data
        if exon_len in SUBLIB_RANGES:
            min_len, max_len = SUBLIB_RANGES[exon_len]
            ptc_sublib = ptc_df[
                (ptc_df['PTC_CDS_exon_length'] >= min_len) & 
                (ptc_df['PTC_CDS_exon_length'] <= max_len)
            ]
            ptc_sublib = ptc_sublib[ptc_sublib['PTC_EJC_dist'] <= ptc_sublib['PTC_CDS_exon_length']]
            
            if len(ptc_sublib) > 1:
                x_ptc = -ptc_sublib['PTC_EJC_dist']
                y_ptc = ptc_sublib['NMDeff_Norm']
                try:
                    (z_ptc, cov_ptc) = np.polyfit(x_ptc, y_ptc, 1, cov=True)
                    tcga_slope = z_ptc[0] * 100  # Convert to per 100nt
                    tcga_intercept = z_ptc[1]
                    tcga_slope_se = np.sqrt(cov_ptc[0, 0]) * 100
                    tcga_intercept_se = np.sqrt(cov_ptc[1, 1])
                except Exception:
                    tcga_slope = np.nan
                    tcga_intercept = np.nan
                    tcga_slope_se = np.nan
                    tcga_intercept_se = np.nan

                tcga_r = np.corrcoef(x_ptc, y_ptc)[0, 1]
                tcga_n = len(ptc_sublib)

                # If AI predictions available, fit regression to predictions as well
                if 'NMDeff_Predicted' in ptc_sublib.columns and ptc_predictions is not None:
                    x_pred = -ptc_sublib['PTC_EJC_dist']
                    y_pred = ptc_sublib['NMDeff_Predicted']
                    try:
                        (z_pred, cov_pred) = np.polyfit(x_pred, y_pred, 1, cov=True)
                        pred_slope = z_pred[0] * 100
                        pred_intercept = z_pred[1]
                        pred_slope_se = np.sqrt(cov_pred[0, 0]) * 100
                        pred_intercept_se = np.sqrt(cov_pred[1, 1])
                    except Exception:
                        pred_slope = np.nan
                        pred_intercept = np.nan
                        pred_slope_se = np.nan
                        pred_intercept_se = np.nan
                else:
                    pred_slope = np.nan
                    pred_intercept = np.nan
                    pred_slope_se = np.nan
                    pred_intercept_se = np.nan
            else:
                tcga_slope = np.nan
                tcga_intercept = np.nan
                tcga_r = np.nan
                tcga_n = 0
                tcga_slope_se = np.nan
                tcga_intercept_se = np.nan
                pred_slope = np.nan
                pred_intercept = np.nan
                pred_slope_se = np.nan
                pred_intercept_se = np.nan
        else:
            tcga_slope = np.nan
            tcga_intercept = np.nan
            tcga_r = np.nan
            tcga_n = 0
        
        results.append({
            'exon_length': exon_len,
            'dms_slope': dms_slope,
            'dms_intercept': dms_intercept,
            'dms_r': dms_r,
            'dms_n': dms_n,
            'dms_slope_se': dms_slope_se,
            'dms_intercept_se': dms_intercept_se,
            'tcga_slope': tcga_slope,
            'tcga_intercept': tcga_intercept,
            'tcga_r': tcga_r,
            'tcga_n': tcga_n,
            'tcga_slope_se': tcga_slope_se,
            'tcga_intercept_se': tcga_intercept_se
            , 'pred_slope': pred_slope,
            'pred_intercept': pred_intercept,
            'pred_slope_se': pred_slope_se,
            'pred_intercept_se': pred_intercept_se
        })
    
    results_df = pd.DataFrame(results)
    logger.info(f"Calculated regression parameters for {len(results_df)} exon lengths")
    
    return results_df


# ============================================================================
# PLOTTING
# ============================================================================

def plot_scatter_comparison(results_df):
    """Create line plot with exon length on y-axis and dual x-axes for intercept and slope."""
    logger.info("Creating line plot...")
    
    # Filter out rows with NaN values in DMS data
    valid_rows = results_df[results_df['dms_n'] > 0].dropna(subset=['dms_slope', 'dms_intercept'])
    
    if len(valid_rows) == 0:
        logger.error("No valid DMS data to plot")
        return None
    
    # Convert exon lengths to integers
    exon_lengths_int = []
    for exon_len in valid_rows['exon_length']:
        if '-' in exon_len:
            # For merged category, take the average
            parts = exon_len.replace('bps', '').split('-')
            exon_lengths_int.append((int(parts[0]) + int(parts[1])) / 2)
        else:
            exon_lengths_int.append(int(exon_len.replace('bps', '')))
    
    intercepts = valid_rows['dms_intercept'].tolist()
    slopes = valid_rows['dms_slope'].tolist()
    # standard errors for error bars (may contain NaN)
    intercepts_se = valid_rows.get('dms_intercept_se', pd.Series([np.nan]*len(valid_rows))).tolist()
    slopes_se = valid_rows.get('dms_slope_se', pd.Series([np.nan]*len(valid_rows))).tolist()
    
    # Faceted side-by-side plots (intercept | slope) sharing y-axis
    plot_color = '#FFB300'  # consistent yellow for both panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=True)
    fig.suptitle(PLOT_TITLE, fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Top: Intercept with error bars
    ax1.errorbar(exon_lengths_int, intercepts, yerr=intercepts_se, fmt='o-', linewidth=2.5,
                 color=plot_color, markersize=8, capsize=4, label='DMS')
    ax1.set_ylabel('Intercept', fontsize=BASE_FONTSIZE, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=BASE_FONTSIZE, width=TICK_WIDTH)
    ax1.tick_params(axis='y', labelsize=BASE_FONTSIZE, width=TICK_WIDTH)
    for spine in ax1.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)
    ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))

    # Overlay AI prediction intercepts if available
    pred_intercepts = valid_rows.get('pred_intercept', pd.Series([np.nan]*len(valid_rows))).tolist()
    pred_intercepts_se = valid_rows.get('pred_intercept_se', pd.Series([np.nan]*len(valid_rows))).tolist()
    if any(~pd.isna(pred_intercepts)):
        ax1.errorbar(exon_lengths_int, pred_intercepts, yerr=pred_intercepts_se, fmt='^',
                     color=PREDICTION_COLOR, markersize=8, capsize=4, label='NMDetective-AI')
        # connect AI prediction points
        ax1.plot(exon_lengths_int, pred_intercepts, color=PREDICTION_COLOR, linestyle='--', linewidth=1)

    # Plot TCGA intercepts if available
    tcga_intercepts = valid_rows.get('tcga_intercept', pd.Series([np.nan]*len(valid_rows))).tolist()
    tcga_intercepts_se = valid_rows.get('tcga_intercept_se', pd.Series([np.nan]*len(valid_rows))).tolist()
    if any(~pd.isna(tcga_intercepts)):
        ax1.errorbar(exon_lengths_int, tcga_intercepts, yerr=tcga_intercepts_se, fmt='s',
                     color=TCGA_COLOR, markersize=7, capsize=4, label='TCGA')
        ax1.plot(exon_lengths_int, tcga_intercepts, color=TCGA_COLOR, linestyle=':', linewidth=1)

    # Bottom: Slope per 100nt with error bars
    ax2.errorbar(exon_lengths_int, slopes, yerr=slopes_se, fmt='s-', linewidth=2.5,
                 color=plot_color, markersize=8, capsize=4, label='DMS')
    ax2.set_ylabel('Slope per 100nt', fontsize=BASE_FONTSIZE, fontweight='bold')
    ax2.set_xlabel('Exon length (nt)', fontsize=BASE_FONTSIZE, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=BASE_FONTSIZE, width=TICK_WIDTH)
    ax2.tick_params(axis='y', labelsize=BASE_FONTSIZE, width=TICK_WIDTH)
    for spine in ax2.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)
    ax2.grid(True, alpha=0.25)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(4))

    # Overlay AI prediction slopes if available
    pred_slopes = valid_rows.get('pred_slope', pd.Series([np.nan]*len(valid_rows))).tolist()
    pred_slopes_se = valid_rows.get('pred_slope_se', pd.Series([np.nan]*len(valid_rows))).tolist()
    if any(~pd.isna(pred_slopes)):
        ax2.errorbar(exon_lengths_int, pred_slopes, yerr=pred_slopes_se, fmt='D',
                     color=PREDICTION_COLOR, markersize=8, capsize=4, label='AI prediction')
        # connect AI prediction points
        ax2.plot(exon_lengths_int, pred_slopes, color=PREDICTION_COLOR, linestyle='--', linewidth=1)

    # Plot TCGA slopes if available
    tcga_slopes = valid_rows.get('tcga_slope', pd.Series([np.nan]*len(valid_rows))).tolist()
    tcga_slopes_se = valid_rows.get('tcga_slope_se', pd.Series([np.nan]*len(valid_rows))).tolist()
    if any(~pd.isna(tcga_slopes)):
        ax2.errorbar(exon_lengths_int, tcga_slopes, yerr=tcga_slopes_se, fmt='s',
                     color=TCGA_COLOR, markersize=7, capsize=4, label='TCGA')
        ax2.plot(exon_lengths_int, tcga_slopes, color=TCGA_COLOR, linestyle=':', linewidth=1)

    # Add legends to subplots
    ax1.legend(loc='best', fontsize=12)
    ax2.legend(loc='best', fontsize=12)

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = True,
):
    """Generate long exon regression parameters figure.

    Args:
        figure_label: Panel label when called from the manuscript app.
        figure_number: Figure number when called from the manuscript app.
        regenerate: If False and source data exists, skip processing.
    """
    paths = get_paths(
        script_name=SCRIPT_NAME,
        figure_label=figure_label,
        figure_number=figure_number,
    )
    
    # Check if source data already exists
    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        results_df = pd.read_csv(paths.source_data)
    else:
        logger.info("Processing data and calculating regression parameters...")
        results_df = process_data()
        
        # Save source data
        plot_columns = [
            'exon_length', 'dms_intercept', 'dms_slope', 'dms_r', 'dms_n',
            'dms_slope_se', 'dms_intercept_se',
            'tcga_slope', 'tcga_intercept', 'tcga_r', 'tcga_n',
            'tcga_slope_se', 'tcga_intercept_se',
            'pred_slope', 'pred_intercept', 'pred_slope_se', 'pred_intercept_se'
        ]
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        results_df[plot_columns].to_csv(paths.source_data, index=False)
        logger.info(f"Source data saved to {paths.source_data}")
    
    # Create and save figure
    fig = plot_scatter_comparison(results_df)
    if fig is not None:
        fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
        fig.savefig(paths.figure_pdf, bbox_inches='tight')
        logger.info(f"Figure saved to {paths.figure_png}")
        plt.close(fig)
    
    logger.success("Long exon regression complete!")


if __name__ == "__main__":
    main()
