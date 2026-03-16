"""
Compare NMDetective-A, NMDetective-B, and NMDetective-AI across multiple datasets.

This script evaluates:
- NMDetective-A: Random Forest Regressor
- NMDetective-B: Simple decision tree
- NMDetective-AI: Deep learning model with Orthrus encoder

on somatic TCGA, germline TCGA, and GTEx datasets.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid libstdc++ issues

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
import pickle

from NMD.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    CONTRASTING_3_COLOURS,
    VAL_CHRS
)
from NMD.manuscript.output import get_paths
from NMD.modeling.TrainerConfig import TrainerConfig
from NMD.modeling.SequenceDataset import SequenceDataset
from NMD.modeling.models.NMDetectiveA import NMDetectiveA
from NMD.modeling.models.NMDetectiveB import NMDetectiveB
from NMD.modeling.models.NMDetectiveB_original import NMDetectiveB_original
from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
from NMD.utils import load_model, collate_fn


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_NAME = "NMDetective_comparison"

# Target column selection
NMD_COLUMN = "NMDeff_Norm" 

# Input data paths
SOMATIC_TCGA_CSV = PROCESSED_DATA_DIR / "PTC" / "somatic_TCGA.csv"
SOMATIC_TCGA_PKL = PROCESSED_DATA_DIR / "PTC" / "somatic_TCGA.pkl"
GERMLINE_TCGA_CSV = PROCESSED_DATA_DIR / "PTC" / "germline_TCGA.csv"
GERMLINE_TCGA_PKL = PROCESSED_DATA_DIR / "PTC" / "germline_TCGA.pkl"
GTEX_CSV = PROCESSED_DATA_DIR / "PTC" / "GTEx.csv"
GTEX_PKL = PROCESSED_DATA_DIR / "PTC" / "GTEx.pkl"

# Model path
MODEL_PATH = MODELS_DIR / "NMDetectiveAI.pt"

# Plot aesthetics
NMDETECTIVE_A_COLOR = CONTRASTING_3_COLOURS[1]  # Model A color
NMDETECTIVE_B_COLOR = CONTRASTING_3_COLOURS[0]  # Model B color
NMDETECTIVE_B_ORIGINAL_COLOR = '#9b59b6'  # Model B_original color (purple)
NMDETECTIVE_AI_COLOR = CONTRASTING_3_COLOURS[2]  # Model AI color
FIGURE_SIZE = (10, 9)
PLOT_TITLE = "Comparison of NMDetective models across datasets"
PLOT_TITLE_FONTSIZE = 18
DPI = 300


# ============================================================================
# HELPER FUNCTIONS - NMDEtective-AI Model
# ============================================================================

def setup_model(config: TrainerConfig, device: torch.device):
    """Setup and return the NMDetectiveAI model."""
    model = NMDetectiveAI(
        hidden_dims=config.dnn_hidden_dims,
        dropout=config.dnn_dropout,
        random_init=config.random_init,
        use_mlm=config.Orthrus_MLM,
        activation_function=config.activation_function,
        use_layer_norm=config.use_layer_norm,
    ).to(device)
    return model


def predict_nmdetective_ai(df, sequences, model, device):
    """
    Generate NMDEtective-AI predictions using the finetuned Orthrus model.
    
    Args:
        df: DataFrame with metadata
        sequences: List of sequence tensors
        model: Trained NMDOrthrus model
        device: torch device
        
    Returns:
        Array of predictions
    """
    eval_dataset = SequenceDataset(df, sequences, label_col="NMD")
    eval_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=collate_fn)
    
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for batch_sequences, batch_lengths, _ in tqdm(eval_loader):
            batch_sequences, batch_lengths = [
                x.to(device) for x in (batch_sequences, batch_lengths)
            ]
            outputs = model(batch_sequences, batch_lengths)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions)


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data(csv_path, pkl_path, dataset_name, col):
    """
    Load and prepare a dataset for evaluation.
    
    Args:
        csv_path: Path to CSV file with metadata
        pkl_path: Path to pickle file with sequences
        dataset_name: Name of the dataset for logging
        col: Name of the target column (e.g., 'NMDeff' or 'NMDeff_Norm')
        
    Returns:
        Tuple of (df, sequences) where df has standardized column names
    """
    logger.info(f"Loading {dataset_name} from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Rename target column to NMD for consistency
    df.rename(columns={col: "NMD"}, inplace=True)
    
    # Load sequences
    logger.info(f"Loading sequences from {pkl_path}")

    with open(pkl_path, "rb") as f:
        sequences = pickle.load(f)
    
    logger.info(f"Loaded {len(df)} samples and {len(sequences)} sequences from {dataset_name}")

    
    return df, sequences


# ============================================================================
# DATA PROCESSING
def process_data():
    """
    Process all datasets and compute metrics for all three methods.
    
    Returns:
        DataFrame with metrics for each dataset and method
    """
    logger.info("Starting data processing...")
    logger.info(f"Using NMD column: {NMD_COLUMN}")
    
    # Load somatic TCGA data (used for training NMDetective-B)
    df_somatic, seq_somatic = load_and_prepare_data(
        SOMATIC_TCGA_CSV, SOMATIC_TCGA_PKL, "somatic TCGA", col=NMD_COLUMN
    )
    
    # Split somatic TCGA into train and validation
    train_mask = ~df_somatic['chr'].isin(VAL_CHRS)
    df_train = df_somatic[train_mask].copy().reset_index(drop=True)
    df_val = df_somatic[~train_mask].copy().reset_index(drop=True)
    seq_val = [seq_somatic[i] for i, mask in enumerate(train_mask) if not mask]
    
    logger.info(f"Split somatic TCGA: {len(df_train)} train, {len(df_val)} validation")
    
    # Load other datasets
    df_germline, seq_germline = load_and_prepare_data(
        GERMLINE_TCGA_CSV, GERMLINE_TCGA_PKL, "germline TCGA", col=NMD_COLUMN
    )
    df_gtex, seq_gtex = load_and_prepare_data(
        GTEX_CSV, GTEX_PKL, "GTEx", col=NMD_COLUMN
    )
    
    # Train NMDetective-A (Random Forest)
    logger.info("Training NMDetective-A (Random Forest)...")
    nmdetective_a = NMDetectiveA(n_estimators=100, random_state=42)
    nmdetective_a.fit(df_train, label_col="NMD")
    
    # Train NMDetective-B (Decision Tree)
    logger.info("Training NMDetective-B (Decision Tree)...")
    nmdetective_b = NMDetectiveB()
    nmdetective_b.fit(df_train, label_col="NMD")
    
    # Setup NMDetective-C (Fixed Decision Tree)
    logger.info("Setting up NMDetective-C (Fixed Decision Tree)...")
    nmdetective_b_original = NMDetectiveB_original()
    nmdetective_b_original.fit(df_train, label_col="NMD")  # No-op fit
    
    # Setup NMDetective-AI model
    logger.info("Loading NMDetective-AI model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    config = TrainerConfig()
    model_ai = setup_model(config, device)
    load_model(model_ai, MODEL_PATH, device=device)
    logger.info(f"Model loaded from {MODEL_PATH}")
    
    # Evaluate on all datasets (excluding train set)
    results = []
    datasets = [
        ("Somatic TCGA\n(validation set)", df_val, seq_val),
        ("Germline TCGA", df_germline, seq_germline),
        ("GTEx", df_gtex, seq_gtex),
    ]
    
    for dataset_name, df, sequences in datasets:
        logger.info(f"Evaluating on {dataset_name}...")
        
        # NMDetective-A predictions
        pred_a = nmdetective_a.predict(df)
        r2_a = r2_score(df['NMD'], pred_a)
        spearman_a = spearmanr(df['NMD'], pred_a)[0]
        pearson_a = pearsonr(df['NMD'], pred_a)[0]
        
        # NMDetective-B predictions
        pred_b = nmdetective_b.predict(df)
        r2_b = r2_score(df['NMD'], pred_b)
        spearman_b = spearmanr(df['NMD'], pred_b)[0]
        pearson_b = pearsonr(df['NMD'], pred_b)[0]
        
        # NMDetective-C predictions
        pred_b_original = nmdetective_b_original.predict(df)
        r2_b_original = r2_score(df['NMD'], pred_b_original)
        spearman_b_original = spearmanr(df['NMD'], pred_b_original)[0]
        pearson_b_original = pearsonr(df['NMD'], pred_b_original)[0]
        
        # NMDetective-AI predictions
        pred_ai = predict_nmdetective_ai(df, sequences, model_ai, device).flatten()
        r2_ai = r2_score(df['NMD'], pred_ai)
        spearman_ai = spearmanr(df['NMD'], pred_ai)[0]
        pearson_ai = pearsonr(df['NMD'], pred_ai)[0]
        
        logger.info(f"  NMDetective-A: R²={r2_a:.3f}, Spearman={spearman_a:.3f}, Pearson={pearson_a:.3f}")
        logger.info(f"  NMDetective-B: R²={r2_b:.3f}, Spearman={spearman_b:.3f}, Pearson={pearson_b:.3f}")
        logger.info(f"  NMDetective-B_original: R²={r2_b_original:.3f}, Spearman={spearman_b_original:.3f}, Pearson={pearson_b_original:.3f}")
        logger.info(f"  NMDetective-AI: R²={r2_ai:.3f}, Spearman={spearman_ai:.3f}, Pearson={pearson_ai:.3f}")
        
        results.append({
            'Dataset': dataset_name,
            'NMDetective-A R²': r2_a,
            'NMDetective-A Spearman': spearman_a,
            'NMDetective-A Pearson': pearson_a,
            'NMDetective-B R²': r2_b,
            'NMDetective-B Spearman': spearman_b,
            'NMDetective-B Pearson': pearson_b,
            'NMDetective-B_original R²': r2_b_original,
            'NMDetective-B_original Spearman': spearman_b_original,
            'NMDetective-B_original Pearson': pearson_b_original,
            'NMDetective-AI R²': r2_ai,
            'NMDetective-AI Spearman': spearman_ai,
            'NMDetective-AI Pearson': pearson_ai,
        })
    
    logger.info("Data processing complete!")
    return pd.DataFrame(results)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_from_table(results_df):
    """
    Create stacked bar plot comparing all three models with Spearman and R² metrics.
    R² is stacked on top since it's always lower than Spearman correlation.
    
    Args:
        results_df: DataFrame with correlation metrics for each dataset and method
    """
    logger.info("Creating comparison stacked bar plot...")
    
    # Set up the plot with single subplot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    
    # Prepare data for plotting
    datasets = results_df['Dataset'].values
    
    # Extract R² and Spearman values for each model
    r2_a = results_df['NMDetective-A R²'].values
    r2_b = results_df['NMDetective-B R²'].values
    r2_b_original = results_df['NMDetective-B_original R²'].values
    r2_ai = results_df['NMDetective-AI R²'].values
    
    spearman_a = results_df['NMDetective-A Spearman'].values
    spearman_b = results_df['NMDetective-B Spearman'].values
    spearman_b_original = results_df['NMDetective-B_original Spearman'].values
    spearman_ai = results_df['NMDetective-AI Spearman'].values
    
    # Calculate the difference (Spearman - R²) for stacking
    # Bottom bar will be R², top will be the difference to reach Spearman
    diff_a = spearman_a - r2_a
    diff_b = spearman_b - r2_b
    diff_b_original = spearman_b_original - r2_b_original
    diff_ai = spearman_ai - r2_ai
    
    x = np.arange(len(datasets))
    width = 0.2
    
    # Create stacked bars
    # Bottom: R²
    bars_r2_a = ax.bar(x - 1.5*width, r2_a, width, 
                       label='NMDetective-A (R²)', color=NMDETECTIVE_A_COLOR, alpha=0.6)
    bars_r2_b = ax.bar(x - 0.5*width, r2_b, width,
                       label='NMDetective-B (R²)', color=NMDETECTIVE_B_COLOR, alpha=0.6)
    bars_r2_b_original = ax.bar(x + 0.5*width, r2_b_original, width,
                       label='NMDetective-B_original (R²)', color=NMDETECTIVE_B_ORIGINAL_COLOR, alpha=0.6)
    bars_r2_ai = ax.bar(x + 1.5*width, r2_ai, width,
                        label='NMDetective-AI (R²)', color=NMDETECTIVE_AI_COLOR, alpha=0.6)
    
    # Top: Difference to reach Spearman correlation
    bars_diff_a = ax.bar(x - 1.5*width, diff_a, width, bottom=r2_a,
                         label='NMDetective-A (Spearman)', color=NMDETECTIVE_A_COLOR, alpha=1.0)
    bars_diff_b = ax.bar(x - 0.5*width, diff_b, width, bottom=r2_b,
                         label='NMDetective-B (Spearman)', color=NMDETECTIVE_B_COLOR, alpha=1.0)
    bars_diff_b_original = ax.bar(x + 0.5*width, diff_b_original, width, bottom=r2_b_original,
                         label='NMDetective-B_original (Spearman)', color=NMDETECTIVE_B_ORIGINAL_COLOR, alpha=1.0)
    bars_diff_ai = ax.bar(x + 1.5*width, diff_ai, width, bottom=r2_ai,
                          label='NMDetective-AI (Spearman)', color=NMDETECTIVE_AI_COLOR, alpha=1.0)
    
    # Add value labels on bars (Spearman values at top)
    for i, (sp_a, sp_b, sp_c, sp_ai) in enumerate(zip(spearman_a, spearman_b, spearman_b_original, spearman_ai)):
        ax.text(i - 1.5*width, sp_a + 0.02, f'{sp_a:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(i - 0.5*width, sp_b + 0.02, f'{sp_b:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(i + 0.5*width, sp_c + 0.02, f'{sp_c:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(i + 1.5*width, sp_ai + 0.02, f'{sp_ai:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add R² value labels (in middle of bottom bars)
    for i, (r2a, r2b, r2b_original_val, r2ai_val) in enumerate(zip(r2_a, r2_b, r2_b_original, r2_ai)):
        if r2a > 0.08:  # Only show if bar is tall enough
            ax.text(i - 1.5*width, r2a / 2, f'{r2a:.2f}', ha='center', va='center', fontsize=12, fontweight='bold')
        if r2b > 0.08:
            ax.text(i - 0.5*width, r2b / 2, f'{r2b:.2f}', ha='center', va='center', fontsize=12, fontweight='bold')
        if r2b_original_val > 0.08:
            ax.text(i + 0.5*width, r2b_original_val / 2, f'{r2b_original_val:.2f}', ha='center', va='center', fontsize=12, fontweight='bold')
        if r2ai_val > 0.08:
            ax.text(i + 1.5*width, r2ai_val / 2, f'{r2ai_val:.2f}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Customize subplot
    ax.set_ylabel('Spearman ρ; R²', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=14, ha='center')
    
    # Create custom legend with clearer grouping
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=NMDETECTIVE_A_COLOR, alpha=1.0, label='NMDetective-A (reimplemented in this study, refit to training data)'),
        Patch(facecolor=NMDETECTIVE_B_COLOR, alpha=1.0, label='NMDetective-B (refit to training data)'),
        Patch(facecolor=NMDETECTIVE_B_ORIGINAL_COLOR, alpha=1.0, label='NMDetective-B (original predictions from published model)'),
        Patch(facecolor=NMDETECTIVE_AI_COLOR, alpha=1.0, label='NMDetective-AI'),
        Patch(facecolor='gray', alpha=1.0, label='Spearman ρ (top)'),
        Patch(facecolor='gray', alpha=0.6, label='R² (bottom)'),
    ]
    ax.legend(handles=legend_elements, fontsize=14, frameon=True, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2)
    
    # Set y-axis limits
    max_val = max(spearman_a.max(), spearman_b.max(), spearman_b_original.max(), spearman_ai.max())
    ax.set_ylim(0, max_val * 1.3)
    
    # Add grid for readability
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, 0.8)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(PLOT_TITLE, fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = True,
):
    """Generate NMDetective model comparison bar plot.

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
    logger.info("Starting NMDEtective comparison analysis")

    # Check if source data already exists
    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        results_df = pd.read_csv(paths.source_data)
    else:
        results_df = process_data()
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(paths.source_data, index=False)
        logger.info(f"Source data saved to {paths.source_data}")

    # Create and save figure
    fig = plot_from_table(results_df)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("NMDetective comparison complete!")


if __name__ == "__main__":
    main()
