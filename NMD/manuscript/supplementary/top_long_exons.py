"""
Visualize NMDetective-AI predictions across the top 25 longest exons with many PTCs.

This script identifies the longest exons in the dataset that contain multiple PTCs
and generates a 5x5 grid showing model predictions along each exon.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from genome_kit import Genome
from scipy.ndimage import gaussian_filter1d

from NMD.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    MODELS_DIR,
    COLOURS,
    GENCODE_VERSION,
)
from NMD.modeling.predict import predict_transcript_ptcs

# Output paths
OUTPUT_TABLE = TABLES_DIR / "manuscript/supplementary" / "top_long_exons.xlsx"
OUTPUT_FIG_PNG = FIGURES_DIR / "manuscript/supplementary" / "top_long_exons.png"
OUTPUT_FIG_PDF = FIGURES_DIR / "manuscript/supplementary" / "top_long_exons.pdf"
PLOT_TITLE = "Top 25 longest exons with multiple PTCs: NMDetective-AI predictions vs observations"

# Constants
MODEL_PATH = MODELS_DIR / "NMDetectiveAI.pt"
N_EXONS = 25  # Top 25 exons
GRID_SIZE = 5  # 5x5 grid
MIN_PTCS_PER_EXON = 5  # Minimum number of PTCs to consider
SIGMA = 3  # Smoothing parameter for predictions

# Colors
MODEL_COLOR = COLOURS[6]  # Dark blue for AI model predictions
OBS_COLOR = "gray"
EXON_BOUNDARY_COLOR = '#BDC3C7'  # Gray for exon boundaries


def identify_top_long_exons():
    """
    Identify the top longest exons with many PTCs from the dataset.
    
    Returns:
        DataFrame with columns: gene_name, transcript_id, exon_idx, exon_length, n_ptcs
    """
    # Load somatic TCGA data
    df = pd.read_csv(INTERIM_DATA_DIR / "PTC" / "somatic_TCGA.csv")
    
    # Load gene mapping
    gene_map_file = INTERIM_DATA_DIR.parent / "raw" / "annotations" / "ensembl_v88_gene_transcript_genesymbol.txt"
    gene_map = pd.read_csv(gene_map_file, sep="\t", header=0)[["gene_id", "gene_name"]].drop_duplicates()
    
    # Merge with gene names
    df = df.merge(gene_map, on='gene_id', how='left')
    
    # Filter for nonsense variants only
    df = df[df['stopgain'] == 'nonsense']
    
    # Group by gene, transcript, and exon to get counts
    exon_stats = df.groupby(['gene_name', 'transcript_id', 'PTC_CDS_exon_num']).agg({
        'PTC_CDS_pos': 'count'
    }).reset_index()
    
    exon_stats.columns = ['gene_name', 'transcript_id', 'exon_idx', 'n_ptcs']
    
    # Calculate actual exon lengths from genome_kit
    genome = Genome(GENCODE_VERSION)
    exon_lengths = []
    
    for _, row in exon_stats.iterrows():
        try:
            transcript = genome.transcripts[row['transcript_id']]
            exon_idx = int(row['exon_idx'])
            
            # Get CDS exon length (PTC_CDS_exon_num refers to CDS exons)
            if transcript and hasattr(transcript, 'cdss') and transcript.cdss and 0 < exon_idx <= len(transcript.cdss):
                cds_exon = transcript.cdss[exon_idx - 1]  # Convert to 0-indexed
                exon_lengths.append(len(cds_exon))
            else:
                exon_lengths.append(0)
        except:
            exon_lengths.append(0)
    
    exon_stats['exon_length'] = exon_lengths
    
    # Filter for exons with minimum number of PTCs and valid length
    exon_stats = exon_stats[(exon_stats['n_ptcs'] >= MIN_PTCS_PER_EXON) & (exon_stats['exon_length'] > 0)]
    
    # Sort by exon length (descending) and take top N
    exon_stats = exon_stats.sort_values('exon_length', ascending=False).head(N_EXONS)
    
    logger.info(f"Identified {len(exon_stats)} top long exons")
    logger.info(f"Exon length range: {exon_stats['exon_length'].min():.0f} - {exon_stats['exon_length'].max():.0f} bp")
    logger.info(f"PTC count range: {exon_stats['n_ptcs'].min():.0f} - {exon_stats['n_ptcs'].max():.0f}")
    
    return exon_stats


def get_exon_boundaries_in_cds(transcript, exon_idx):
    """
    Get the start and end positions of an exon within the CDS.
    
    Args:
        transcript: genome_kit Transcript object
        exon_idx: 1-indexed CDS exon number (as stored in PTC_CDS_exon_num)
        
    Returns:
        tuple: (exon_start_nt, exon_end_nt) in CDS coordinates, or (None, None) if not found
    """
    if transcript is None or not hasattr(transcript, 'cdss') or transcript.cdss is None:
        return None, None
    
    # PTC_CDS_exon_num refers to CDS exons, not all transcript exons
    # So we should use transcript.cdss instead of transcript.exons
    if exon_idx < 1 or exon_idx > len(transcript.cdss):
        return None, None
    
    # Calculate cumulative CDS length up to this exon
    cumulative_length = 0
    for i, cds_exon in enumerate(transcript.cdss):
        exon_start = cumulative_length
        cumulative_length += len(cds_exon)
        exon_end = cumulative_length
        
        # Check if this is our target exon (1-indexed)
        if i + 1 == exon_idx:
            return exon_start, exon_end
    
    return None, None


def load_observations_for_exon(gene_name, transcript_id, exon_idx):
    """
    Load observed PTC data for a specific exon.
    
    Args:
        gene_name: Gene name
        transcript_id: Transcript ID
        exon_idx: 1-indexed exon number
        
    Returns:
        DataFrame with PTC positions and NMDeff_Norm values, or None if no data
    """
    somatic_file = INTERIM_DATA_DIR / "PTC" / "somatic_TCGA.csv"
    if not somatic_file.exists():
        return None
    
    df = pd.read_csv(somatic_file)
    
    # Load gene mapping
    gene_map_file = INTERIM_DATA_DIR.parent / "raw" / "annotations" / "ensembl_v88_gene_transcript_genesymbol.txt"
    gene_map = pd.read_csv(gene_map_file, sep="\t", header=0)[["gene_id", "gene_name"]].drop_duplicates()
    df = df.merge(gene_map, on='gene_id', how='left')
    
    # Filter for specific gene, transcript, and exon
    df = df[
        (df['gene_name'] == gene_name) & 
        (df['transcript_id'] == transcript_id) &
        (df['PTC_CDS_exon_num'] == exon_idx) &
        (df['stopgain'] == 'nonsense')
    ]
    
    if df.empty:
        return None
    
    return df[['PTC_CDS_pos', 'NMDeff_Norm']]


def process_data():
    """
    Process all top exons and generate prediction data.
    Saves results to OUTPUT_TABLE.
    """
    genome = Genome(GENCODE_VERSION)
    
    # Identify top exons
    exon_stats = identify_top_long_exons()
    
    all_data = []
    
    for idx, row in exon_stats.iterrows():
        gene_name = row['gene_name']
        transcript_id = row['transcript_id']
        exon_idx = int(row['exon_idx'])
        
        logger.info(f"Processing {gene_name} exon {exon_idx} ({row['exon_length']:.0f} bp, {row['n_ptcs']:.0f} PTCs)...")
        
        try:
            # Get model predictions for entire transcript
            results = predict_transcript_ptcs(
                gene_name=gene_name,
                transcript_id=transcript_id,
                model_path=str(MODEL_PATH),
                max_positions=None
            )
            
            ptc_positions = results['ptc_positions']
            predictions = results['predictions']
            
            # Get transcript and exon boundaries
            transcript = genome.transcripts[transcript_id]
            exon_start, exon_end = get_exon_boundaries_in_cds(transcript, exon_idx)
            
            if exon_start is None or exon_end is None:
                logger.warning(f"Could not determine exon boundaries for {gene_name} exon {exon_idx}")
                continue
            
            # Convert transcript positions to CDS positions
            # ptc_positions from predict_transcript_ptcs are in transcript coordinates (including UTR5)
            # We need to convert them to CDS coordinates (starting from 0)
            utr5_length = sum(len(utr) for utr in transcript.utr5s) if transcript.utr5s else 0
            ptc_positions_cds = [pos - utr5_length for pos in ptc_positions]
            
            # Load observations for this exon
            obs_df = load_observations_for_exon(gene_name, transcript_id, exon_idx)
            
            # Filter predictions for this exon
            for pos_cds, pred in zip(ptc_positions_cds, predictions):
                if exon_start <= pos_cds <= exon_end:
                    data_entry = {
                        'gene_name': gene_name,
                        'transcript_id': transcript_id,
                        'exon_idx': exon_idx,
                        'exon_length': row['exon_length'],
                        'exon_start': exon_start,
                        'exon_end': exon_end,
                        'ptc_position': pos_cds,
                        'prediction': pred,
                        'data_type': 'prediction'
                    }
                    all_data.append(data_entry)
            
            # Add observations
            if obs_df is not None:
                for _, obs_row in obs_df.iterrows():
                    data_entry = {
                        'gene_name': gene_name,
                        'transcript_id': transcript_id,
                        'exon_idx': exon_idx,
                        'exon_length': row['exon_length'],
                        'exon_start': exon_start,
                        'exon_end': exon_end,
                        'ptc_position': obs_row['PTC_CDS_pos'],
                        'nmd_eff': obs_row['NMDeff_Norm'],
                        'data_type': 'observation'
                    }
                    all_data.append(data_entry)
            
        except Exception as e:
            logger.warning(f"Failed to process {gene_name} exon {exon_idx}: {e}")
            continue
    
    # Convert to DataFrame and save as XLSX with sheets per exon
    df = pd.DataFrame(all_data)
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    
    # Delete existing file if it exists to avoid corruption issues
    if OUTPUT_TABLE.exists():
        OUTPUT_TABLE.unlink()
        logger.info(f"Deleted existing table file: {OUTPUT_TABLE}")
    
    # Get unique exons and save each to a separate sheet
    with pd.ExcelWriter(OUTPUT_TABLE, engine='openpyxl') as writer:
        exons = df[['gene_name', 'exon_idx']].drop_duplicates()
        for i, (_, exon_info) in enumerate(exons.iterrows()):
            if i >= N_EXONS:
                break
            
            gene_name = exon_info['gene_name']
            exon_idx = exon_info['exon_idx']
            
            # Filter data for this exon
            exon_df = df[
                (df['gene_name'] == gene_name) & 
                (df['exon_idx'] == exon_idx)
            ].copy()
            
            # Keep only columns needed for plotting
            plot_columns = ['ptc_position', 'prediction', 'nmd_eff', 'data_type']
            exon_df = exon_df[plot_columns]
            
            # Create sheet name: Exon_1_GENE, Exon_2_GENE, etc.
            sheet_name = f'Exon_{i+1:02d}_{gene_name}'[:31]  # Excel sheet name limit
            exon_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"  Saved {len(exon_df)} rows for {sheet_name}")
    
    logger.info(f"Saved data to {OUTPUT_TABLE}")
    
    return df


def plot_from_table(df):
    """
    Create 5x5 grid plot from the processed data table.
    """
    # Get unique exons (top 25)
    exons = df[['gene_name', 'transcript_id', 'exon_idx', 'exon_length', 'exon_start', 'exon_end']].drop_duplicates()
    exons = exons.sort_values('exon_length', ascending=False).head(N_EXONS)
    
    # Create 5x5 subplot grid
    fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, (_, exon_info) in enumerate(exons.iterrows()):
        if i >= N_EXONS:
            break
            
        ax = axes[i]
        gene_name = exon_info['gene_name']
        transcript_id = exon_info['transcript_id']
        exon_idx = exon_info['exon_idx']
        exon_start = exon_info['exon_start']
        exon_end = exon_info['exon_end']
        exon_length = exon_info['exon_length']
        
        # Get data for this exon
        exon_data = df[
            (df['gene_name'] == gene_name) & 
            (df['transcript_id'] == transcript_id) &
            (df['exon_idx'] == exon_idx)
        ]
        
        # Plot predictions
        pred_data = exon_data[exon_data['data_type'] == 'prediction'].sort_values('ptc_position')
        if len(pred_data) > 0:
            # Apply smoothing
            smoothed_preds = gaussian_filter1d(pred_data['prediction'].values, sigma=SIGMA)
            ax.plot(pred_data['ptc_position'], smoothed_preds, 
                   color=MODEL_COLOR, linewidth=1.5, alpha=0.8, zorder=2)
        
        # Plot observations
        obs_data = exon_data[exon_data['data_type'] == 'observation']
        if len(obs_data) > 0:
            ax.scatter(obs_data['ptc_position'], obs_data['nmd_eff'],
                      color=OBS_COLOR, s=15, alpha=0.6, zorder=3, edgecolors='none')
        
        # Mark exon boundaries
        ax.axvline(exon_start, color=EXON_BOUNDARY_COLOR, linestyle='--', 
                  linewidth=1, alpha=0.7, zorder=1)
        ax.axvline(exon_end, color=EXON_BOUNDARY_COLOR, linestyle='--', 
                  linewidth=1, alpha=0.7, zorder=1)
        
        # Customize subplot
        ax.set_xlim(exon_start - 50, exon_end + 50)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('CDS Position (nt)', fontsize=8)
        ax.set_ylabel('NMD Efficiency', fontsize=8)
        ax.set_title(f'{gene_name} exon {int(exon_idx)}\n({exon_length:.0f} bp)', 
                    fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        
        # Add spines styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Add legend to first subplot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=MODEL_COLOR, linewidth=2, label='NMDetective-AI'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=OBS_COLOR, 
               markersize=6, label='Observed', linestyle='None'),
        Line2D([0], [0], color=EXON_BOUNDARY_COLOR, linewidth=1, 
               linestyle='--', label='Exon boundary')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=7)
    
    plt.title(PLOT_TITLE, fontsize=18, y=0.92)
    plt.tight_layout()
    
    # Save figures
    OUTPUT_FIG_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_FIG_PDF, bbox_inches='tight')
    logger.info(f"Saved figures to {OUTPUT_FIG_PNG} and {OUTPUT_FIG_PDF}")
    
    plt.close()


def main():
    """Main execution function."""
    if OUTPUT_TABLE.exists():
        try:
            logger.info(f"Loading existing data from {OUTPUT_TABLE}")
            # Read all sheets and combine
            all_sheets = pd.read_excel(OUTPUT_TABLE, sheet_name=None, engine='openpyxl')
            df = pd.concat(all_sheets.values(), ignore_index=True)
        except Exception as e:
            logger.warning(f"Failed to load existing table ({e}), regenerating...")
            OUTPUT_TABLE.unlink()
            df = process_data()
    else:
        logger.info("Processing data...")
        df = process_data()
    
    logger.info("Generating plots...")
    plot_from_table(df)
    logger.info("Done!")


if __name__ == "__main__":
    main()
