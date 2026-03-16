#!/usr/bin/env python3
"""
Manuscript figure: Start Proximal predictions vs DMS observations

This script generates predictions from NMDetective-AI, A, and B models
for start proximal PTCs and compares them with DMS observations.
Each gene is shown in a separate subplot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
from loguru import logger
from genome_kit import Genome

from NMD.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    MODELS_DIR,
    TABLES_DIR,
    COLOURS,
    GENCODE_VERSION,
    VAL_CHRS,
)
from NMD.manuscript.output import get_paths
from NMD.modeling.predict import predict_transcript_ptcs
from NMD.modeling.models.NMDetectiveA import NMDetectiveA
from NMD.modeling.models.NMDetectiveB import NMDetectiveB
from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
from NMD.modeling.SequenceDataset import SequenceDataset
from NMD.modeling.features import setup_data
from NMD.utils import load_model
from NMD.modeling.TrainerConfig import TrainerConfig
from torch.utils.data import DataLoader


# ============================================================================
# CONFIGURATION - All paths and parameters defined here
# ============================================================================

# Model and data paths
SCRIPT_NAME = "start_prox_predictions"

MODEL_PATH = MODELS_DIR / "NMDetectiveAI.pt"
DMS_SP_FILE = PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv"
PTC_FILE = PROCESSED_DATA_DIR / "PTC" / "somatic_TCGA.csv"

# Gene configurations - using MANE transcripts
GENE_NAMES = ["MTOR", "CDKL5"]  # Genes to plot
ANNOTATION_FILE = RAW_DATA_DIR / "annotations" / "ensembl_v88_gene_transcript_genesymbol.txt"

# Plot parameters
MAX_PTC_POSITION = 250  # Maximum position in CDS to plot (nucleotides)
FIGURE_SIZE = (16, 8)  # Overall figure size (width, height)
DPI = 300
LOESS_FRAC = 0.3  # LOESS smoothing fraction

# Prediction mode for NMDetective-AI
USE_DMS_SEQUENCES = False  # If True, use DMS SP sequences for AI predictions; if False, generate PTC sequences

# Colors (using config COLOURS scheme)
MODEL_AI_COLOR = COLOURS[6]  # '#022778' - dark blue
MODEL_A_COLOR = COLOURS[0]   # '#fb731d' - orange
MODEL_B_COLOR = COLOURS[5]   # '#2778ff' - blue
DMS_COLOR = COLOURS[1]       # '#ff9e9d' - pink/red
DMS_SMOOTH_COLOR = COLOURS[3]  # '#ffdfcb' - light pink


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def load_gene_transcript_mapping():
    """
    Load gene name to Ensembl ID mapping from annotation file.
    
    Returns:
        Dictionary mapping gene names to Ensembl gene IDs
    """
    df = pd.read_csv(ANNOTATION_FILE, sep='\t')
    # Keep only rows with non-empty gene_id and gene_name
    df = df.dropna(subset=['gene_id', 'gene_name'])
    # Create mapping from gene_name to gene_id (take first occurrence)
    mapping = df.drop_duplicates('gene_name').set_index('gene_name')['gene_id'].to_dict()
    return mapping


def get_transcript_for_gene(genome, gene_name, gene_id_mapping):
    """
    Get the first valid transcript for a gene that has CDS.
    
    Args:
        genome: Genome object
        gene_name: Gene name
        gene_id_mapping: Dictionary mapping gene names to Ensembl IDs
        
    Returns:
        Tuple of (gene_object, transcript_object) or (None, None) if not found
    """
    if gene_name not in gene_id_mapping:
        logger.warning(f"Gene {gene_name} not found in annotation file")
        return None, None
    
    gene_id = gene_id_mapping[gene_name]
    
    try:
        gene_obj = genome.genes[gene_id]
    except KeyError:
        logger.warning(f"Gene ID {gene_id} for {gene_name} not found in genome")
        return None, None
    
    # Find first transcript with CDS
    for transcript in gene_obj.transcripts:
        if transcript.cdss and len(transcript.cdss) > 0:
            return gene_obj, transcript
    
    logger.warning(f"No transcripts with CDS found for {gene_name}")
    return None, None


def compute_features_for_ptc_positions(transcript, ptc_positions):
    """
    Compute NMDetective-A/B features for given PTC positions.
    
    Args:
        transcript: genome_kit Transcript object
        ptc_positions: List of PTC positions in transcript coordinates
        
    Returns:
        DataFrame with features needed for NMDetective-A and B
    """
    features = []
    
    # Get CDS intervals
    if not transcript.cdss or len(transcript.cdss) == 0:
        logger.warning(f"No CDS found for transcript {transcript.id}")
        return pd.DataFrame()
    
    cds_intervals = transcript.cdss
    cds_length = sum(len(interval) for interval in cds_intervals)
    normal_stop_pos = cds_length
    
    # Get 5'UTR length
    utr5_length = sum(len(exon) for exon in transcript.utr5s) if transcript.utr5s else 0
    
    # Determine last exon position in transcript coordinates
    last_exon_start = sum(len(exon) for exon in transcript.exons[:-1])
    
    # Get EJC positions (50nt downstream of each exon junction)
    ejc_positions = []
    cumulative = 0
    for i, exon in enumerate(transcript.exons[:-1]):
        cumulative += len(exon)
        if cumulative > utr5_length:  # Only count exons in CDS
            ejc_positions.append(cumulative + 50)
    
    for ptc_pos in ptc_positions:
        # Convert to CDS coordinates
        ptc_pos_cds = ptc_pos - utr5_length
        
        # InLastExon
        in_last_exon = ptc_pos >= last_exon_start
        
        # DistanceToStart (in CDS)
        distance_to_start = min(ptc_pos_cds, 1000)  # Capped at 1000nt
        
        # Find which exon contains this PTC and get ExonLength
        cumulative = 0
        exon_length = 0
        for i, exon in enumerate(transcript.exons):
            exon_start = cumulative
            cumulative += len(exon)
            if exon_start <= ptc_pos < cumulative:
                # Found the exon
                if cumulative <= utr5_length:
                    # PTC is in 5'UTR (shouldn't happen)
                    exon_length = 0
                elif exon_start < utr5_length:
                    # Exon spans 5'UTR and CDS
                    exon_length = cumulative - utr5_length
                else:
                    # Exon is fully in CDS
                    exon_length = len(exon)
                break
        
        # 50ntToLastEJ - within 50nt of last exon junction (start of last exon)
        within_50nt = (ptc_pos < last_exon_start) and ((last_exon_start - ptc_pos) <= 55)
        
        # PTC_EJC_dist - distance to nearest downstream EJC
        downstream_ejcs = [ejc for ejc in ejc_positions if ejc > ptc_pos]
        if downstream_ejcs:
            ptc_ejc_dist = downstream_ejcs[0] - ptc_pos
        else:
            ptc_ejc_dist = 0
        
        # DistanceToWTStop
        distance_to_wt_stop = (utr5_length + normal_stop_pos) - ptc_pos
        
        # RNAHalfLife - use default value
        rna_halflife = 5.0
        
        features.append({
            'ptc_position': ptc_pos,
            'InLastExon': in_last_exon,
            '50ntToLastEJ': within_50nt,
            'DistanceToStart': distance_to_start,
            'ExonLength': exon_length,
            'PTC_EJC_dist': ptc_ejc_dist,
            'DistanceToWTStop': distance_to_wt_stop,
            'RNAHalfLife': rna_halflife
        })
    
    return pd.DataFrame(features)


def load_dms_data(gene_name):
    """
    Load and process DMS data for start proximal region.
    
    Args:
        gene_name: Name of the gene
    
    Returns:
        DataFrame with DMS observations
    """
    dms = pd.read_csv(DMS_SP_FILE)
    dms = dms[dms['gene'] == gene_name].copy()
    logger.info(f"  Loaded {len(dms)} DMS observations for {gene_name}")
    return dms


def process_data(source_data_path=None, regenerate=True):
    """Process data and generate predictions. Returns DataFrame for plotting."""
    
    # Check if output table already exists
    if source_data_path and source_data_path.exists() and not regenerate:
        logger.info(f"Loading existing data from {source_data_path}")
        return pd.read_csv(source_data_path)
    
    logger.info("Starting data processing...")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Using DMS sequences for AI predictions: {USE_DMS_SEQUENCES}")
    
    # Train NMDetective-A and B on somatic TCGA training data
    logger.info("Training NMDetective-A and NMDetective-B...")
    df_somatic = pd.read_csv(PTC_FILE)
    train_mask = ~df_somatic['chr'].isin(VAL_CHRS)
    df_train = df_somatic[train_mask].copy()
    df_train['NMD'] = df_train['NMDeff']
    
    nmdetective_a = NMDetectiveA(n_estimators=100, random_state=42)
    nmdetective_a.fit(df_train, label_col="NMD")
    
    nmdetective_b = NMDetectiveB()
    nmdetective_b.fit(df_train, label_col="NMD")
    logger.info("Models trained successfully")
    
    # Load NMDetective-AI model and DMS sequences if using DMS sequences
    if USE_DMS_SEQUENCES:
        logger.info("Loading NMDetective-AI model...")
        config = TrainerConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_ai = NMDetectiveAI(
            hidden_dims=config.dnn_hidden_dims,
            dropout=config.dnn_dropout,
            random_init=config.random_init,
            use_mlm=config.Orthrus_MLM,
            activation_function=config.activation_function,
            use_layer_norm=config.use_layer_norm,
        ).to(device)
        load_model(model_ai, MODEL_PATH, device=device)
        model_ai.eval()
        logger.info("NMDetective-AI model loaded successfully")
        
        # Load pre-encoded DMS sequences and metadata
        logger.info("Loading pre-encoded DMS sequences...")
        dms_sequences_all, dms_metadata_all = setup_data(
            DMS_SP_FILE,
            batch_size=config.batch_size,
            data_type="DMS"
        )
        logger.info(f"Loaded {len(dms_sequences_all)} DMS sequences")
    
    # Initialize genome
    genome = Genome(GENCODE_VERSION)
    
    # Load gene to transcript mapping
    gene_id_mapping = load_gene_transcript_mapping()
    logger.info(f"Loaded gene ID mapping for {len(gene_id_mapping)} genes")
    
    all_data = []

    for gene_name in GENE_NAMES:
        logger.info(f"\nProcessing {gene_name}...")

        # Get gene and transcript objects
        gene_obj, transcript = get_transcript_for_gene(genome, gene_name, gene_id_mapping)
        if transcript is None:
            logger.warning(f"Skipping {gene_name} - no valid transcript found")
            continue

        logger.info(f"  Using transcript {transcript.id} for {gene_name}")

        # Get 5'UTR length for coordinate conversion
        utr5_length = sum(len(exon) for exon in transcript.utr5s) if transcript.utr5s else 0

        # Load DMS data
        dms_data = load_dms_data(gene_name)

        if USE_DMS_SEQUENCES:
            # Use DMS sequences for AI predictions
            logger.info(f"  Using DMS sequences for NMDetective-AI predictions...")

            # Get sequences from pre-loaded data by matching indices
            dms_sequences = [dms_sequences_all[i] for i in dms_data.index]
            dms_positions_nt = dms_data['PTCposition_nt'].tolist()

            # Predict with NMDetective-AI on DMS sequences
            label_col = "NMDeff_Norm"
            eval_dataset = SequenceDataset(dms_data, dms_sequences, label_col=label_col)
            eval_loader = DataLoader(eval_dataset, batch_size=1)

            predictions_ai_dms = []
            with torch.no_grad():
                for batch_sequences, batch_lengths, _ in eval_loader:
                    batch_sequences, batch_lengths = [
                        x.to(device) for x in (batch_sequences, batch_lengths)
                    ]
                    batch_preds = model_ai(batch_sequences, batch_lengths).squeeze()

                    # Handle single prediction case
                    if batch_preds.dim() == 0:
                        predictions_ai_dms.append(batch_preds.item())
                    else:
                        predictions_ai_dms.extend(batch_preds.tolist())

            logger.info(f"  Generated {len(predictions_ai_dms)} AI predictions on DMS sequences")

            # Compute features for NMDetective-A/B using DMS positions
            ptc_positions_transcript = [pos + utr5_length for pos in dms_positions_nt]
            features_df = compute_features_for_ptc_positions(transcript, ptc_positions_transcript)
            predictions_a = nmdetective_a.predict(features_df)
            predictions_b = nmdetective_b.predict(features_df)
            logger.info(f"  Generated NMDetective-A and B predictions")

            # Store model predictions at DMS positions
            for cds_pos, pred_ai, pred_a, pred_b in zip(dms_positions_nt, predictions_ai_dms, predictions_a, predictions_b):
                all_data.append({
                    'gene': gene_name,
                    'transcript_id': transcript.id,
                    'ptc_position_cds': cds_pos,
                    'prediction_ai': pred_ai,
                    'prediction_a': pred_a,
                    'prediction_b': pred_b,
                    'data_type': 'prediction'
                })
        else:
            # Generate all PTC sequences for AI predictions
            logger.info(f"  Generating PTC sequences for NMDetective-AI predictions...")
            results = predict_transcript_ptcs(
                gene_name=gene_name,
                transcript_id=transcript.id,
                transcript_idx=0,
                model_path=str(MODEL_PATH),
                max_positions=None
            )

            ptc_positions = results['ptc_positions']
            predictions_ai = results['predictions']

            logger.info(f"  Generated {len(predictions_ai)} AI predictions")

            # Compute features and get NMDetective-A/B predictions
            features_df = compute_features_for_ptc_positions(transcript, ptc_positions)
            predictions_a = nmdetective_a.predict(features_df)
            predictions_b = nmdetective_b.predict(features_df)
            logger.info(f"  Generated NMDetective-A and B predictions")

            # Convert positions to CDS coordinates
            cds_positions = [pos - utr5_length for pos in ptc_positions]

            # Store model predictions
            for cds_pos, pred_ai, pred_a, pred_b in zip(cds_positions, predictions_ai, predictions_a, predictions_b):
                all_data.append({
                    'gene': gene_name,
                    'transcript_id': transcript.id,
                    'ptc_position_cds': cds_pos,
                    'prediction_ai': pred_ai,
                    'prediction_a': pred_a,
                    'prediction_b': pred_b,
                    'data_type': 'prediction'
                })

        # Store DMS observations without scaling
        for idx, row in dms_data.iterrows():
            all_data.append({
                'gene': gene_name,
                'transcript_id': transcript.id,
                'ptc_position_cds': row['PTCposition_nt'],
                'dms_value': row['NMDeff_Norm'],  # Use raw DMS value
                'data_type': 'dms'
            })

    # Save to table
    df = pd.DataFrame(all_data)
    if source_data_path:
        source_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(source_data_path, index=False)
        logger.info(f"Data saved to {source_data_path}")

    return df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_from_table(df):
    """
    Create plots from processed data table.
    
    Args:
        df: DataFrame with predictions and DMS observations
    """
    logger.info("Creating plots...")
    
    # Create figure with subplots (one per gene)
    n_genes = len(GENE_NAMES)
    fig, axes = plt.subplots(n_genes, 1, figsize=FIGURE_SIZE, sharex=True)
    
    # If only one gene, axes is not an array
    if n_genes == 1:
        axes = [axes]
    
    for idx, gene_name in enumerate(GENE_NAMES):
        ax = axes[idx]
        
        # Filter data for this gene
        gene_data = df[df['gene'] == gene_name]
        
        # Get predictions
        pred_data = gene_data[gene_data['data_type'] == 'prediction'].copy()
        pred_data = pred_data.sort_values('ptc_position_cds')
        
        # Filter to start proximal region
        pred_data = pred_data[pred_data['ptc_position_cds'] <= MAX_PTC_POSITION]
        
        # Get DMS observations
        dms_data = gene_data[gene_data['data_type'] == 'dms'].dropna(subset=['dms_value'])
        dms_data = dms_data[dms_data['ptc_position_cds'] <= MAX_PTC_POSITION]
        dms_data = dms_data.sort_values('ptc_position_cds')
        
        # Calculate Spearman correlations if we have overlapping data
        spearman_ai, spearman_a, spearman_b = None, None, None
        
        if len(pred_data) > 0 and len(dms_data) > 0:
            # For USE_DMS_SEQUENCES mode, predictions and DMS are at same positions
            # For generated PTC mode, need to match by position
            if USE_DMS_SEQUENCES:
                # Direct comparison - predictions were made on DMS sequences
                if 'dms_value' in pred_data.columns:
                    # DMS values are stored with predictions
                    dms_vals = pred_data['dms_value'].values
                    ai_vals = pred_data['prediction_ai'].values
                    a_vals = pred_data['prediction_a'].values
                    b_vals = pred_data['prediction_b'].values
                    
                    spearman_ai, _ = spearmanr(ai_vals, dms_vals)
                    spearman_a, _ = spearmanr(a_vals, dms_vals)
                    spearman_b, _ = spearmanr(b_vals, dms_vals)
            else:
                # Need to interpolate predictions at DMS positions
                dms_positions = dms_data['ptc_position_cds'].values
                dms_vals = dms_data['dms_value'].values
                pred_positions = pred_data['ptc_position_cds'].values
                
                # Interpolate predictions at DMS positions
                ai_interp = np.interp(dms_positions, pred_positions, pred_data['prediction_ai'].values)
                a_interp = np.interp(dms_positions, pred_positions, pred_data['prediction_a'].values)
                b_interp = np.interp(dms_positions, pred_positions, pred_data['prediction_b'].values)
                
                spearman_ai, _ = spearmanr(ai_interp, dms_vals)
                spearman_a, _ = spearmanr(a_interp, dms_vals)
                spearman_b, _ = spearmanr(b_interp, dms_vals)
        
        # Plot predictions (no smoothing)
        if len(pred_data) > 0:
            positions = pred_data['ptc_position_cds'].values
            
            # Create legend labels with Spearman R if available
            label_ai = 'NMDetective-AI'
            label_a = 'NMDetective-A'
            label_b = 'NMDetective-B'
            
            if spearman_ai is not None:
                label_ai += f' (ρ={spearman_ai:.2f})'
                label_a += f' (ρ={spearman_a:.2f})'
                label_b += f' (ρ={spearman_b:.2f})'
            
            # Plot NMDetective-AI predictions
            ax.plot(positions, pred_data['prediction_ai'].values, 
                   color=MODEL_AI_COLOR, linewidth=2.5, 
                   label=label_ai, zorder=3)
            
            # Plot NMDetective-A predictions
            ax.plot(positions, pred_data['prediction_a'].values, 
                   color=MODEL_A_COLOR, linewidth=2.5, linestyle='--',
                   label=label_a, zorder=2)
            
            # Plot NMDetective-B predictions
            ax.plot(positions, pred_data['prediction_b'].values, 
                   color=MODEL_B_COLOR, linewidth=2.5, linestyle=':',
                   label=label_b, zorder=2)
        
        # Plot DMS observations with LOESS smoothing
        if len(dms_data) > 0:
            dms_positions = dms_data['ptc_position_cds'].values
            dms_values = dms_data['dms_value'].values
            
            # Only add labels for the first subplot to avoid repetition
            dms_obs_label = 'DMS observations' if idx == 0 else None
            dms_loess_label = 'DMS (LOESS)' if idx == 0 else None
            
            # Plot raw DMS observations
            ax.scatter(dms_positions, dms_values,
                      color=DMS_COLOR, alpha=0.4, s=30,
                      label=dms_obs_label, zorder=1)
            
            # Plot LOESS smoothed DMS trend
            if len(dms_values) > 10:
                dms_loess = lowess(dms_values, dms_positions, frac=LOESS_FRAC)
                ax.plot(dms_loess[:, 0], dms_loess[:, 1],
                       color=DMS_SMOOTH_COLOR, linewidth=2.5,
                       label=dms_loess_label, zorder=1)
        
        # Customize subplot
        ax.set_ylabel('NMD efficiency', fontsize=18, fontweight='bold')
        ax.legend(fontsize=13, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
        
        # Add gene name as text in top center
        ax.text(0.5, 0.98, gene_name, 
               transform=ax.transAxes,
               fontsize=18, fontweight='bold',
               verticalalignment='top',
               horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set x-label only on bottom subplot
    axes[-1].set_xlabel('PTC Position (nt)', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(figure_label=None, figure_number=None, regenerate=False):
    """Main execution function."""
    paths = get_paths(SCRIPT_NAME, figure_label=figure_label, figure_number=figure_number)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    if paths.source_data:
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
    
    # Process data (or load if already exists)
    df = process_data(source_data_path=paths.source_data, regenerate=regenerate)
    
    # Create and save figure
    logger.info("Generating figure...")
    fig = plot_from_table(df)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    logger.info(f"Figure PDF saved to {paths.figure_pdf}")
    plt.close(fig)
    
    logger.success("\nPlotting complete!")


if __name__ == "__main__":
    main()
