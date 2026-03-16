"""
Plot comparison of DMS Long Exon data vs PTC TCGA data by exon length.

This script creates two main visualizations:
1. Faceted scatter plots comparing DMS and PTC data across different exon lengths
   with linear regression fits and confidence intervals
2. Violin plots showing the distribution of NMDeff by exon length for both datasets

The script follows the analysis from notebook 02_DMS_normalise.ipynb for Long Exon data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from loguru import logger
import pickle
from sklearn.metrics import r2_score
import torch
from torch.utils.data import DataLoader

from NMD.config import (
    COLOURS,
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    MODELS_DIR
)
from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
from NMD.modeling.SequenceDataset import SequenceDataset
from NMD.utils import load_model, collate_fn
from NMD.modeling.TrainerConfig import TrainerConfig


# ============================================================================
# CONFIGURATION - Define all paths and parameters here
# ============================================================================

# Input paths
DMS_LE_FILE = PROCESSED_DATA_DIR / "DMS_LE/fitness.csv"
DMS_LE_ALL_POSITIONS_DIR = PROCESSED_DATA_DIR / "DMS_LE_all_positions"
PTC_FILE = INTERIM_DATA_DIR / "PTC" / "somatic_TCGA.csv"
MODEL_PATH = MODELS_DIR / "NMDetectiveAI.pt"

# Output paths
OUTPUT_TABLE = TABLES_DIR / 'manuscript' / 'supplementary' / 'long_exon_DMS_PTC_comparison.xlsx'
OUTPUT_FIGURE = FIGURES_DIR / 'manuscript' / 'supplementary' / 'long_exon_DMS_PTC_comparison.png'
OUTPUT_FIGURE_PDF = FIGURES_DIR / 'manuscript' / 'supplementary' / 'long_exon_DMS_PTC_comparison.pdf'

# Exon lengths to plot (in order)
SELECTED_EXON_LENGTHS = ["125-250bps", "500bps", "750bps", "1000bps", "1500bps", "2500bps", "3000bps", "3426bps"]
PLOT_TITLE = "DMS long exon experiment vs PTCs in TCGA comparison by exon length"

# Sublibrary ranges for PTC data matching
SUBLIB_RANGES = {
    '125bps': (115, 135),
    '125-250bps': (115, 275),
    '250bps': (225, 275),
    '500bps': (450, 550),
    '750bps': (650, 850),
    '1000bps': (850, 1150),
    '1500bps': (1250, 1750),
    '2500bps': (2250, 2750),
    '3000bps': (2750, 3250),
    '3426bps': (3250, 5000),
}

# Plot aesthetics
STOP_COLORS = {
    'UAA': 'red',
    'UAG': 'blue',
    'UGA': 'green'
}

DMS_COLOR = COLOURS[1]
TCGA_COLOR = 'lightgray'
PREDICTION_COLOR = COLOURS[6]  # Color for AI predictions

FIGURE_SIZE_COMBINED = (14, 10)
DPI = 300

# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_and_predict_all_positions():
    """
    Load all possible DMS LE positions and run NMDetectiveAI predictions.
    
    Returns:
        pd.DataFrame: DataFrame with predictions for all positions
    """
    logger.info("Loading DMS LE all positions data...")
    
    # Check if the data exists
    sequences_file = DMS_LE_ALL_POSITIONS_DIR / "processed_sequences.pkl"
    fitness_file = DMS_LE_ALL_POSITIONS_DIR / "fitness.csv"
    
    if not sequences_file.exists() or not fitness_file.exists():
        logger.warning("DMS LE all positions data not found. Please run: python -m NMD.data.DMS generate-dms-le-all-positions")
        return None
    
    # Load sequences and metadata
    with open(sequences_file, "rb") as f:
        sequences = pickle.load(f)
    metadata = pd.read_csv(fitness_file)
    
    logger.info(f"Loaded {len(sequences)} sequences for prediction")
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainerConfig()
    model = NMDetectiveAI(
        hidden_dims=config.dnn_hidden_dims,
        dropout=config.dnn_dropout,
        random_init=config.random_init,
        use_mlm=config.Orthrus_MLM,
        activation_function=config.activation_function,
        use_layer_norm=config.use_layer_norm,
    ).to(device)
    
    # Load trained weights
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        return None
    
    load_model(model, MODEL_PATH, device=device)
    model.eval()
    logger.info(f"Loaded model from {MODEL_PATH}")
    
    # Create dataset and dataloader
    dummy_df = pd.DataFrame({"y": [0] * len(sequences)})
    eval_dataset = SequenceDataset(dummy_df, sequences, label_col="y")
    eval_loader = DataLoader(eval_dataset, batch_size=8, collate_fn=collate_fn)
    
    # Run predictions
    predictions = []
    with torch.no_grad():
        for batch_sequences, batch_lengths, _ in eval_loader:
            batch_sequences = batch_sequences.to(device)
            batch_lengths = batch_lengths.to(device)
            outputs = model(batch_sequences, batch_lengths)
            predictions.extend(outputs.cpu().numpy().flatten())
    
    # Add predictions to metadata
    metadata['NMDeff_Predicted'] = predictions
    
    logger.info(f"Generated {len(predictions)} predictions")
    logger.info(f"Prediction range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
    
    return metadata


def process_data():
    """
    Process DMS Long Exon and PTC TCGA data for comparison.
    
    Returns:
        tuple: (dms_df, ptc_df) - processed DataFrames ready for plotting
    """
    logger.info("Starting data processing...")
    
    # -------------------------------------------------------------------------
    # 1. Load DMS Long Exon data
    # -------------------------------------------------------------------------
    logger.info(f"Loading DMS Long Exon data from {DMS_LE_FILE}")
    dms = pd.read_csv(DMS_LE_FILE)
    logger.info(f"Loaded {len(dms)} DMS Long Exon variants")
    
    # -------------------------------------------------------------------------
    # 2. Load PTC TCGA data
    # -------------------------------------------------------------------------
    logger.info(f"Loading PTC TCGA data from {PTC_FILE}")
    ptc_df = pd.read_csv(PTC_FILE)
    logger.info(f"Loaded {len(ptc_df)} PTC variants")
    
    # Filter PTC data
    ptc_df = ptc_df[ptc_df.Last_Exon == False]
    ptc_df = ptc_df[ptc_df.Penultimate_Exon == False]
    ptc_df = ptc_df[ptc_df.Start_Prox == False]
    ptc_df = ptc_df[ptc_df.Ref != "-"]
    ptc_df = ptc_df[ptc_df.Alt != "-"]
    logger.info(f"After filtering PTC data: {len(ptc_df)} variants")
    
    # -------------------------------------------------------------------------
    # 3. Process DMS data
    # -------------------------------------------------------------------------
    logger.info("Processing DMS Long Exon data...")
    
    # Calculate stop codon position
    dms['stop_down_123nt'] = (dms.stop_type + dms.down_123nt).str.replace("U", "T")
    dms['stop_pos'] = dms.apply(lambda row: row.nt_seq.find(row.stop_down_123nt), axis=1)
    
    # Calculate PTC-EJC distance
    dms["PTC_EJC_dist"] = dms["exon_length"].str.replace("bps", "").astype(int) - dms["stop_pos"]
    
    # Calculate relative position
    dms["rel_pos"] = dms["PTC_EJC_dist"] / dms["exon_length"].str.replace("bps", "").astype(int)
    
    # -------------------------------------------------------------------------
    # 4. Process PTC data for comparison
    # -------------------------------------------------------------------------
    logger.info("Processing PTC data for comparison...")
    
    # Create temporary dataframe with relevant columns
    ptc_comparison = ptc_df[["PTC_CDS_exon_length", "PTC_EJC_dist", "NMDeff", "NMDeff_Norm"]].copy()
    
    # Calculate relative position
    ptc_comparison["rel_pos"] = ptc_comparison["PTC_EJC_dist"] / ptc_comparison["PTC_CDS_exon_length"]
    
    logger.info(f"PTC data ready for comparison: {len(ptc_comparison)} variants")
    
    return dms, ptc_comparison


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_combined_figure(dms, ptc_comparison, predictions_df=None, col="NMDeff"):
    """
    Create figure with faceted scatter plots comparing DMS and TCGA data.
    
    Args:
        dms: Processed DMS DataFrame
        ptc_comparison: Processed PTC DataFrame
        predictions_df: DataFrame with NMDetectiveAI predictions for all positions (optional)
        col: Column name to use for TCGA data
    
    Returns:
        matplotlib Figure object
    """
    logger.info("Creating figure with scatter plots...")
    
    # Create figure with gridspec for flexible layout (3x3 grid)
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(1, 1)

    # -------------------------------------------------------------------------
    # Faceted scatter plots in a 3x3 grid for selected exon lengths
    # -------------------------------------------------------------------------
    n_selected = len(SELECTED_EXON_LENGTHS)
    n_rows = 3
    n_cols = 3
    gs_scatter = gs[0].subgridspec(n_rows, n_cols, hspace=0.3, wspace=0.15)
    
    # First pass: collect all y values to determine common y-axis limits
    all_y_values = []
    for exon_len in SELECTED_EXON_LENGTHS:
        # Handle merged category for 125-250bps
        if exon_len == '125-250bps':
            exon_data = dms[dms['exon_length'].isin(['125bps', '250bps'])]
        else:
            exon_data = dms[dms['exon_length'] == exon_len]
        
        if len(exon_data) > 0:
            all_y_values.extend(exon_data['NMDeff'].values)
        
        # Add corresponding TCGA data if available
        if exon_len in SUBLIB_RANGES:
            min_len, max_len = SUBLIB_RANGES[exon_len]
            ptc_sublib = ptc_comparison[(ptc_comparison['PTC_CDS_exon_length'] >= min_len) & 
                                        (ptc_comparison['PTC_CDS_exon_length'] <= max_len)]
            ptc_sublib = ptc_sublib[ptc_sublib['PTC_EJC_dist'] <= ptc_sublib['PTC_CDS_exon_length']]
            
            if len(ptc_sublib) > 0:
                all_y_values.extend(ptc_sublib[col].values)
        
        # Add predictions if available
        if predictions_df is not None:
            if exon_len == '125-250bps':
                pred_data = predictions_df[predictions_df['exon_length'].isin(['125bps', '250bps'])]
            else:
                pred_data = predictions_df[predictions_df['exon_length'] == exon_len]
            
            if len(pred_data) > 0:
                all_y_values.extend(pred_data['NMDeff_Predicted'].values)
    
    # Calculate common y-axis limits with some padding
    y_min = np.min(all_y_values)
    y_max = np.max(all_y_values)
    y_padding = (y_max - y_min) * 0.05
    common_ylim = (y_min - y_padding, y_max + y_padding)
    
    for i, exon_len in enumerate(SELECTED_EXON_LENGTHS):
        row = i // n_cols
        col_idx = i % n_cols
        ax = fig.add_subplot(gs_scatter[row, col_idx])
        
        # Handle merged category for 125-250bps
        if exon_len == '125-250bps':
            exon_data = dms[dms['exon_length'].isin(['125bps', '250bps'])]
        else:
            exon_data = dms[dms['exon_length'] == exon_len]
        
        # Plot all DMS points in yellow
        if len(exon_data) > 0:
            ax.scatter(-exon_data['PTC_EJC_dist'], exon_data['NMDeff'], 
                      alpha=0.6, s=30, color=DMS_COLOR, 
                      label=f'DMS (n={len(exon_data)})', zorder=3)
        
        # Add AI predictions if available
        if predictions_df is not None:
            if exon_len == '125-250bps':
                pred_data = predictions_df[predictions_df['exon_length'].isin(['125bps', '250bps'])]
            else:
                pred_data = predictions_df[predictions_df['exon_length'] == exon_len]
            
            if len(pred_data) > 0:
                # Invert x-axis for predictions to match DMS/TCGA orientation
                ax.scatter(pred_data['PTC_EJC_dist'] - (pred_data['PTC_EJC_dist'].max()+1), pred_data['NMDeff_Predicted'], 
                          alpha=0.4, s=15, color=PREDICTION_COLOR, marker='x',
                          label=f'AI predictions (n={len(pred_data)})', zorder=2)
                
                # Calculate R² and Spearman correlation with DMS observations
                if len(exon_data) > 0:
                    # Merge predictions with DMS data on PTC_EJC_dist
                    merged = exon_data.merge(pred_data, on='PTC_EJC_dist', suffixes=('_dms', '_pred'))
                    
                    if len(merged) > 0:
                        # Calculate R² 
                        r_squared = r2_score(merged['NMDeff_dms'], merged['NMDeff_Predicted'])
                        
                        # Calculate Spearman correlation
                        spearman_corr, spearman_p = stats.spearmanr(merged['NMDeff_dms'], merged['NMDeff_Predicted'])
                        
                        # Add text annotation with metrics
                        ax.text(0.98, 0.95, f'AI: R²={r_squared:.2f}, ρ={spearman_corr:.2f}', 
                               transform=ax.transAxes, 
                               bbox=dict(boxstyle="round,pad=0.2", facecolor=PREDICTION_COLOR, alpha=0.3),
                               verticalalignment='top', horizontalalignment='right', fontsize=10, fontweight='bold')
        
        # Add corresponding TCGA data if available
        if exon_len in SUBLIB_RANGES:
            min_len, max_len = SUBLIB_RANGES[exon_len]
            ptc_sublib = ptc_comparison[(ptc_comparison['PTC_CDS_exon_length'] >= min_len) & 
                                        (ptc_comparison['PTC_CDS_exon_length'] <= max_len)]
            
            # Filter out PTC data points where PTC-EJC distance is larger than the exon length
            # (these are impossible and indicate data inconsistencies)
            ptc_sublib = ptc_sublib[ptc_sublib['PTC_EJC_dist'] <= ptc_sublib['PTC_CDS_exon_length']]
            
            #if len(ptc_sublib) > 0:
            #    ax.scatter(-ptc_sublib['PTC_EJC_dist'], ptc_sublib[col], 
            #              alpha=0.5, s=20, color=TCGA_COLOR, marker='s', 
            #              label=f'TCGA (n={len(ptc_sublib)})')
            if len(ptc_sublib) > 0:
                ax.scatter(-ptc_sublib['PTC_EJC_dist'], ptc_sublib[col], 
                          alpha=0.5, s=20, color=TCGA_COLOR, marker='s', 
                          label=f'TCGA (n={len(ptc_sublib)})')
        
        # Fit and plot linear regression for DMS data
        if len(exon_data) > 1:
            z = np.polyfit(-exon_data['PTC_EJC_dist'], exon_data['NMDeff'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(-exon_data['PTC_EJC_dist'].max(), 
                                 -exon_data['PTC_EJC_dist'].min(), 100)
            # Show slope per 100nt for better readability
            slope_per_100nt = z[0] * 100
            intercept = z[1]
            ax.plot(x_range, p(x_range), color='darkgoldenrod', alpha=0.8, linewidth=2, 
                    label=f'DMS fit: {slope_per_100nt:.3f} per 100nt + {intercept:.3f}')
            
            # Calculate R for DMS data
            correlation = np.corrcoef(-exon_data['PTC_EJC_dist'], exon_data['NMDeff'])[0, 1]
            ax.text(0.02, 0.98, f'R={correlation:.2f}', transform=ax.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    verticalalignment='top', fontsize=10, fontweight='bold')
        
        # Add linear fit for TCGA data if available
        if exon_len in SUBLIB_RANGES and len(ptc_sublib) > 1:
            z_ptc = np.polyfit(-ptc_sublib['PTC_EJC_dist'], ptc_sublib[col], 1)
            p_ptc = np.poly1d(z_ptc)
            x_range_ptc = np.linspace(-ptc_sublib['PTC_EJC_dist'].max(), 
                                     -ptc_sublib['PTC_EJC_dist'].min(), 100)
            # Show slope per 100nt for better readability
            slope_per_100nt_tcga = z_ptc[0] * 100
            intercept_tcga = z_ptc[1]
            ax.plot(x_range_ptc, p_ptc(x_range_ptc), color='darkgray', alpha=0.8, linewidth=2, 
                    linestyle='--', label=f'TCGA fit: {slope_per_100nt_tcga:.3f} per 100nt + {intercept_tcga:.3f}')
            
            # Calculate and plot CI for PTC fit
            slope_ptc, intercept_ptc, r_value_ptc, p_value, std_err = stats.linregress(
                -ptc_sublib['PTC_EJC_dist'], ptc_sublib[col])
            y_pred_ptc = slope_ptc * x_range_ptc + intercept_ptc
            
            # Calculate confidence interval
            n = len(ptc_sublib)
            mse_ptc = np.sum((ptc_sublib[col] - (slope_ptc * -ptc_sublib['PTC_EJC_dist'] + intercept_ptc))**2) / (n - 2)
            x_mean_ptc = -ptc_sublib['PTC_EJC_dist'].mean()
            sxx_ptc = np.sum((-ptc_sublib['PTC_EJC_dist'] - x_mean_ptc)**2)
            
            t_val = stats.t.ppf(0.975, n-2)  # 95% CI
            ci_ptc = t_val * np.sqrt(mse_ptc * (1/n + (x_range_ptc - x_mean_ptc)**2 / sxx_ptc))
            
            # CI
            ci_label = f'TCGA 95% CI (n={n})'
            ax.fill_between(x_range_ptc, y_pred_ptc - ci_ptc, y_pred_ptc + ci_ptc, 
                           color='darkgray', alpha=0.2, label=ci_label)
            
            # Calculate R for TCGA data
            ptc_correlation = np.corrcoef(-ptc_sublib['PTC_EJC_dist'], ptc_sublib[col])[0, 1]
            ax.text(0.02, 0.90, f'R={ptc_correlation:.2f}', transform=ax.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='darkgray', alpha=0.6),
                    verticalalignment='top', fontsize=10, fontweight='bold')
        
            # Calculate spearman correlation between DMS data and TCGA fit
            if len(exon_data) > 1 and len(ptc_sublib) > 1:
                # Get predicted values from TCGA fit for DMS x values
                tcga_fit_for_dms = slope_ptc * -exon_data['PTC_EJC_dist'] + intercept_ptc
                spearman_corr, spearman_p = stats.spearmanr(exon_data['NMDeff'], tcga_fit_for_dms)
                ax.text(0.02, 0.82, f'Spearman R={spearman_corr:.2f}\np={spearman_p:.3e}', transform=ax.transAxes, 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.6),
                        verticalalignment='top', fontsize=10)

        # Only show x-axis label on bottom row
        if row == n_rows - 1:
            ax.set_xlabel('PTC-EJC Distance (nt)', fontsize=14)
        
        # Only show y-axis label and ticks on first column
        if col_idx == 0:
            ax.set_ylabel('NMD efficiency', fontsize=14)
            ax.tick_params(labelsize=12)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', labelsize=0)
            ax.tick_params(axis='x', labelsize=12)
        
        ax.set_ylim(common_ylim)  # Set common y-axis limits
        ax.set_title(f'Exon Length: {exon_len}', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(PLOT_TITLE, fontsize=20, fontweight='bold')
    return fig

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Ensure output directories exist
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    
    dms, ptc_comparison = process_data()
    
    # Load and run predictions on all positions
    predictions_df = load_and_predict_all_positions()
    if predictions_df is not None:
        logger.info(f"Loaded {len(predictions_df)} predictions for all positions")
    else:
        logger.warning("No predictions loaded - plotting only observed data")
    
    # Save table data for each exon length panel
    logger.info("Saving source data tables...")
    with pd.ExcelWriter(OUTPUT_TABLE, engine='openpyxl') as writer:
        for i, exon_len in enumerate(SELECTED_EXON_LENGTHS):
            # Handle merged category for 125-250bps
            if exon_len == '125-250bps':
                exon_data = dms[dms['exon_length'].isin(['125bps', '250bps'])].copy()
            else:
                exon_data = dms[dms['exon_length'] == exon_len].copy()
            
            # Prepare DMS data
            dms_data = exon_data[['PTC_EJC_dist', 'NMDeff']].copy()
            dms_data['data_type'] = 'DMS'
            dms_data.rename(columns={'NMDeff': 'value'}, inplace=True)
            
            # Prepare TCGA data if available
            tcga_data = pd.DataFrame()
            if exon_len in SUBLIB_RANGES:
                min_len, max_len = SUBLIB_RANGES[exon_len]
                ptc_sublib = ptc_comparison[
                    (ptc_comparison['PTC_CDS_exon_length'] >= min_len) & 
                    (ptc_comparison['PTC_CDS_exon_length'] <= max_len)
                ].copy()
                ptc_sublib = ptc_sublib[ptc_sublib['PTC_EJC_dist'] <= ptc_sublib['PTC_CDS_exon_length']]
                
                if len(ptc_sublib) > 0:
                    tcga_data = ptc_sublib[['PTC_EJC_dist', 'NMDeff_Norm']].copy()
                    tcga_data['data_type'] = 'TCGA'
                    tcga_data.rename(columns={'NMDeff_Norm': 'value'}, inplace=True)
            
            # Prepare AI predictions if available
            pred_data = pd.DataFrame()
            if predictions_df is not None:
                if exon_len == '125-250bps':
                    pred_subset = predictions_df[predictions_df['exon_length'].isin(['125bps', '250bps'])].copy()
                else:
                    pred_subset = predictions_df[predictions_df['exon_length'] == exon_len].copy()
                
                if len(pred_subset) > 0:
                    pred_data = pred_subset[['PTC_EJC_dist', 'NMDeff_Predicted']].copy()
                    pred_data['data_type'] = 'AI_prediction'
                    pred_data.rename(columns={'NMDeff_Predicted': 'value'}, inplace=True)
            
            # Combine all data for this panel
            panel_data = pd.concat([dms_data, tcga_data, pred_data], ignore_index=True)
            
            # Create sheet name
            sheet_name = f'exon_{exon_len}'.replace('-', '_')
            panel_data.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"  Saved {len(panel_data)} rows for {sheet_name}")
    
    logger.info(f"Source data table saved to {OUTPUT_TABLE}")
    
    # Create and save figure
    logger.info("Generating figure with scatter plots...")
    fig = plot_combined_figure(dms, ptc_comparison, predictions_df, col="NMDeff_Norm")
    fig.savefig(OUTPUT_FIGURE, dpi=DPI, bbox_inches='tight')
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches='tight')
    logger.info(f"Combined figure saved to {OUTPUT_FIGURE}")
    logger.info(f"Combined figure PDF saved to {OUTPUT_FIGURE_PDF}")
    plt.close(fig)    
    logger.success("\nPlotting complete!")


if __name__ == "__main__":
    main()
