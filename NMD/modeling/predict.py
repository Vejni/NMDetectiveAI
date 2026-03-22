from time import time

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import typer
from genome_kit import Genome
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import re
import subprocess
import urllib.request
import shutil
import tempfile

from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
from NMD.config import RAW_DATA_DIR, TABLES_DIR, MODELS_DIR, GENCODE_VERSION, PROCESSED_DATA_DIR, FIGURES_DIR, PROJ_ROOT
from NMD.data.transcripts import find_transcript_by_gene_name, generate_penultimate_exon_ptc_sequences, generate_all_ptc_sequences, create_six_track_encoding_with_variant, create_six_track_encoding, get_exon_boundaries_in_cds
from NMD.modeling.TrainerConfig import TrainerConfig
from NMD.modeling.SequenceDataset import SequenceDataset
from NMD.utils import load_model, collate_fn
from NMD.plots import plot_transcript_ptc_predictions

app = typer.Typer()


def _setup_model(config: TrainerConfig) -> tuple:
    """Setup model and return model, criterion, optimizer, scheduler, device."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create NMDetectiveAI model
    model = NMDetectiveAI(
        hidden_dims=config.dnn_hidden_dims,
        dropout=config.dnn_dropout,
        random_init=config.random_init,
        use_mlm=config.Orthrus_MLM,
        activation_function=config.activation_function,
        use_layer_norm=config.use_layer_norm,
    ).to(device)
    if config.loss_type == "MSE":
        criterion = nn.MSELoss()
    else:
        criterion = nn.HuberLoss(config.huber_delta)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_gamma)
    
    return model, criterion, optimizer, scheduler, device


def _predict_batch(model, sequences, device):
    """Helper function to predict a batch of sequences."""
    if len(sequences) == 0:
        return []
    
    # Create dummy metadata for SequenceDataset
    dummy_df = pd.DataFrame({"y": [0] * len(sequences)})
    
    # Create dataset and dataloader
    eval_dataset = SequenceDataset(dummy_df, sequences, label_col="y")
    eval_loader = DataLoader(eval_dataset, batch_size=1)
    
    # Run predictions
    predictions = []
    with torch.no_grad():
        for batch_sequences, batch_lengths, _ in eval_loader:
            batch_sequences, batch_lengths = [
                x.to(device) for x in (batch_sequences, batch_lengths)
            ]
            batch_preds = model(batch_sequences, batch_lengths).squeeze()
            
            # Handle single prediction case
            if batch_preds.dim() == 0:
                predictions.append(float(batch_preds.cpu().numpy()))
            else:
                predictions.extend(batch_preds.cpu().numpy())
    
    return predictions


def _process_transcript_penultimate_exon(transcript, gene_name, model, device, all_predictions):
    """
    Process a single transcript and generate predictions for penultimate exon PTCs.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Generate PTC sequences for penultimate exon
        ptc_sequences = generate_penultimate_exon_ptc_sequences(transcript)
        
        if len(ptc_sequences) == 0:
            logger.debug(f"No penultimate exon PTC sequences generated for {gene_name}:{transcript.id}")
            return False
        
        # Create dummy metadata for SequenceDataset (predictions don't need labels)
        dummy_df = pd.DataFrame({"y": [0] * len(ptc_sequences)})
        
        # Create dataset and dataloader
        eval_dataset = SequenceDataset(
            dummy_df,
            ptc_sequences,
            label_col="y",
        )
        eval_loader = DataLoader(eval_dataset, batch_size=4, collate_fn=collate_fn)
        
        # Run predictions
        predictions = []
        with torch.no_grad():
            for batch_sequences, batch_lengths, _ in eval_loader:
                batch_sequences, batch_lengths = [
                    x.to(device) for x in (batch_sequences, batch_lengths)
                ]
                batch_preds = model(batch_sequences, batch_lengths).squeeze()
                predictions.extend(batch_preds.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Store predictions with metadata
        for i, pred in enumerate(predictions):
            prediction_data = {
                'gene_name': gene_name,
                'transcript_id': transcript.id,
                'ptc_position': i + 1,  # 1-based PTC position relative to start of region
                'predicted_fitness': float(pred),
                'region': 'penultimate_exon'
            }
            all_predictions.append(prediction_data)
        
        logger.debug(f"Generated {len(predictions)} penultimate exon predictions for {gene_name}:{transcript.id}")
        return True
        
    except Exception as e:
        logger.debug(f"Failed to generate penultimate exon predictions for transcript {transcript.id} in gene {gene_name}: {e}")
        return False


def predict_transcript_ptcs(gene_name=None, transcript_id=None, transcript_idx=None, model_path=None, max_positions=None, stop_codon="TAG", disable=True, gencode_version=GENCODE_VERSION):
    """
    Generate all PTCs for a specific transcript and predict NMD efficiency.
    
    Args:
        gene_name (str, optional): Name of the gene (required if transcript_id not provided)
        transcript_id (str, optional): Transcript ID (e.g., ENST00000012345 or ENST00000012345.7)
                                     Will match based on base ID before the dot
        transcript_idx (int): Index of transcript to use when using gene_name (default: 0 for first transcript)
        model_path (str): Path to the trained model (if None, uses default setup)
        max_positions (int): Maximum number of PTC positions to generate
        stop_codon (str): Stop codon to insert (default: "TAG")
        gencode_version (str): GENCODE version to use for genome initialization
    Returns:
        dict: Dictionary containing predictions, positions, and transcript info
    """
    # Validate input parameters
    if gene_name is None and transcript_id is None:
        raise ValueError("Either gene_name or transcript_id must be provided")
    
    # Initialize genome and find transcript
    genome = Genome(gencode_version)
    
    if transcript_id is not None:
        # Find transcript by ID (match base ID before the dot)
        transcript = genome.transcripts[transcript_id]
        
        # Extract gene name from transcript for logging
        if gene_name is None:
            gene_name = transcript.gene.name if transcript.gene else "Unknown"
            
    else:
        transcripts = find_transcript_by_gene_name(genome, gene_name)
        if len(transcripts) <= transcript_idx:
            raise ValueError(f"Transcript index {transcript_idx} not found for gene {gene_name}. Available: {len(transcripts)}")
        transcript = transcripts[transcript_idx]
    
    # Setup model
    config = TrainerConfig()
    model, _, _, _, device = _setup_model(config)
    
    if model_path:
        load_model(model, model_path, device=device)
    else:
        logger.warning("No model path provided, predicting with default weights")
    
    model.eval()
    
    # Generate all PTC sequences
    ptc_sequences, ptc_positions = generate_all_ptc_sequences(transcript, stop_codon, max_positions, gencode_version)
    
    if len(ptc_sequences) == 0:
        raise ValueError(f"No PTC sequences generated for {gene_name}:{transcript.id}")
    
    # Create dataset and run predictions
    dummy_df = pd.DataFrame({"y": [0] * len(ptc_sequences)})
    eval_dataset = SequenceDataset(dummy_df, ptc_sequences, label_col="y")
    eval_loader = DataLoader(eval_dataset, batch_size=4, collate_fn=collate_fn)
    
    predictions = []
    with torch.no_grad():
        for batch_sequences, batch_lengths, _ in tqdm(eval_loader, desc=f"Predicting PTCs for {gene_name}", disable=disable):
            batch_sequences, batch_lengths = [
                x.to(device) for x in (batch_sequences, batch_lengths)
            ]
            batch_preds = model(batch_sequences, batch_lengths).squeeze()
            if batch_preds.dim() == 0:
                predictions.append(float(batch_preds.cpu().numpy()))
            else:
                predictions.extend(batch_preds.cpu().numpy())
    
    # Create results dictionary
    results = {
        'gene_name': gene_name,
        'transcript_id': transcript.id,
        'ptc_positions': ptc_positions,
        'predictions': predictions,
        'stop_codon': stop_codon,
        'num_ptcs': len(predictions),
        'transcript_info': {
            'cds_length': sum(len(exon) for exon in transcript.cdss) if transcript.cdss else 0,
            'num_exons': len(transcript.exons),
            'strand': transcript.strand
        }
    }
    
    return results


@app.command()
def plot_transcript_predictions(
    gene_name: str = None,
    transcript_id: str = None,
    transcript_idx: int = 0,
    model_path: str = None,
    sigma: float = 1.0,
    output_dir: Path = None,
    save_predictions: bool = False,
    show_exon_boundaries: bool = True,
    show_55nt_rule: bool = True,
    gencode_version: str = GENCODE_VERSION
):
    """
    Generate PTC predictions for a transcript, optionally plot and/or save results.
    """

    # Generate predictions
    logger.info("Generating PTC predictions...")
    results = predict_transcript_ptcs(
        gene_name=gene_name,
        transcript_id=transcript_id,
        transcript_idx=transcript_idx,
        model_path=model_path if model_path else MODELS_DIR / "NMDetectiveAI.pt",
        max_positions=None,
        gencode_version=gencode_version
    )

    transcript_id_full = results['transcript_id']
    gene_name_result = results['gene_name']
    ptc_positions = results['ptc_positions']
    predictions = results['predictions']

    logger.info(f"Generated {len(predictions)} predictions for {gene_name_result} ({transcript_id_full})")

    # Save predictions as CSV if requested
    if save_predictions:
        if output_dir is None:
            output_dir = TABLES_DIR / "transcripts"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "gene_name": gene_name_result,
            "transcript_id": transcript_id_full,
            "ptc_position": ptc_positions,
            "prediction": predictions,
        })
        csv_path = output_dir / f"{transcript_id_full}_ptc_predictions.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions as CSV to {csv_path}")

    # Plot and save figure if requested
    fig = plot_transcript_ptc_predictions(
        transcript_id=transcript_id_full,
        gene_name=gene_name_result,
        ptc_positions=ptc_positions,
        predictions=predictions,
        sigma=sigma,
        show_exon_boundaries=show_exon_boundaries,
        show_55nt_rule=show_55nt_rule,
        gencode_version=gencode_version
    )
    if output_dir is None:
        output_dir = FIGURES_DIR / "transcripts"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{transcript_id_full}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    output_file_pdf = output_dir / f"{transcript_id_full}.pdf"
    fig.savefig(output_file_pdf, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file_pdf}")
    logger.success(f"Successfully generated and saved transcript plot for {gene_name_result}")
    return 


@app.command()
def generate_all_mane_predictions(gencode_version: str = GENCODE_VERSION):
    """
    Generate genome-wide PTC predictions for all genes and save predictions as CSV.
    Selects the best transcript per gene (prefers transcripts with complete CDS annotations).
    """
    logger.info("Starting genome-wide PTC prediction generation")

    # Initialize genome
    genome = Genome(gencode_version)
    output_dir = TABLES_DIR / "GW_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for protein-coding genes only
    gene_names = [
        gene.name for gene in genome.genes 
        if gene.name is not None and gene.type == 'protein_coding'
    ]
    logger.info(f"Found {len(gene_names)} protein-coding genes in the genome")

    processed_genes = 0
    skipped_genes = 0
    saved_files = 0
    already_exists = 0

    for gene_name in tqdm(gene_names, desc="Processing genes"):
        # Check if prediction file already exists for this gene
        # We need to check all possible transcript files for this gene
        # Use a pattern to find any existing file for this gene
        existing_files = list(output_dir.glob(f"{gene_name}_*_ptc_predictions.csv"))
        if existing_files:
            already_exists += 1
            logger.debug(f"Skipping gene {gene_name}: prediction file already exists")
            continue
        
        # Find the best transcript for this gene
        try:
            logger.info(f"Processing gene {gene_name}...")
            transcripts = find_transcript_by_gene_name(genome, gene_name)
            
            # Prefer transcripts with CDS information
            transcripts_with_cds = [
                tr for tr in transcripts 
                if tr.cdss is not None and len(tr.cdss) > 0
            ]
            
            if not transcripts_with_cds:
                logger.warning(f"Skipping gene {gene_name}: no transcripts with CDS found")
                skipped_genes += 1
                continue
            
            # Select the transcript with the longest CDS
            best_transcript = max(
                transcripts_with_cds,
                key=lambda tr: sum(len(interval) for interval in tr.cdss)
            )
            
            # Calculate CDS length to check if it's reasonable
            cds_length = sum(len(interval) for interval in best_transcript.cdss)
            logger.debug(f"  Transcript {best_transcript.id}: CDS length = {cds_length} nt")
            
            # Skip extremely long transcripts to prevent memory issues (>30kb CDS = >10k codons)
            if cds_length > 30000:
                logger.warning(f"Skipping gene {gene_name}: CDS too long ({cds_length} nt)")
                skipped_genes += 1
                continue
            
            results = predict_transcript_ptcs(
                gene_name=gene_name,
                transcript_id=best_transcript.id.split('.')[0],
                transcript_idx=None,
                model_path=MODELS_DIR / "NMDetectiveAI.pt",
                max_positions=None,
                gencode_version=gencode_version
            )
            processed_genes += 1
            logger.success(f"  Successfully processed {gene_name}: {results['num_ptcs']} PTCs")
        except Exception as e:
            logger.error(f"Skipping gene {gene_name} due to error: {e}")
            skipped_genes += 1
            continue

        # For memory efficiency: write a per-transcript CSV and don't accumulate in memory
        transcript_id_full = results['transcript_id']
        gene_name_result = results['gene_name']
        ptc_positions = results['ptc_positions']
        predictions = results['predictions']
        stop_codon = results.get('stop_codon', None)
        transcript_info = results.get('transcript_info', {})

        # Build dataframe for this transcript
        df_trans = pd.DataFrame({
            'ptc_position': ptc_positions,
            'prediction': list(predictions),
        })
        df_trans['gene_name'] = gene_name_result
        df_trans['transcript_id'] = transcript_id_full
        df_trans['stop_codon'] = stop_codon
        df_trans['cds_length'] = transcript_info.get('cds_length', None)
        df_trans['num_exons'] = transcript_info.get('num_exons', None)
        df_trans['strand'] = transcript_info.get('strand', None)

        # Sanitize filename: use gene_transcript and replace non-alphanum with underscore
        safe_name = f"{gene_name_result}_{transcript_id_full}"
        safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', safe_name)
        csv_path = output_dir / f"{safe_name}_ptc_predictions.csv"
        try:
            df_trans.to_csv(csv_path, index=False)
            saved_files += 1
            logger.success(f"Saved predictions for {gene_name_result}:{transcript_id_full} to {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to save predictions for {gene_name_result}:{transcript_id_full}: {e}")
            # continue processing other genes
            continue

    logger.info(f"Processed {processed_genes} genes, skipped {skipped_genes} genes (errors), {already_exists} genes (already exist)")
    logger.info(f"Saved {saved_files} per-transcript prediction files to {output_dir}")
    return


@app.command()
def annotate_vcf_with_predictions(
    vcf_file: str,
    output_dir: str = None,
    batch_size: int = 8
):
    """
    Annotate a VCF/TSV file with NMDetectiveAI predictions for stop-gained variants.
    
    Args:
        vcf_file (str): Path to the VCF/TSV file containing variants (e.g., from selection pipeline)
        model_name (str): Name of the model to use (default: "PTC")
        output_dir (str): Output directory path (default: PROCESSED_DATA_DIR / "selection")
        batch_size (int): Batch size for prediction (default: 8)
    """
    from pathlib import Path
    
    # Set up paths
    vcf_path = Path(vcf_file)
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "selection"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting VCF annotation with NMDetectiveAI predictions")
    logger.info(f"Input file: {vcf_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load the trained model
    config = TrainerConfig()
    model, _, _, _, device = _setup_model(config)
    
    # Load model weights
    model_path = MODELS_DIR / f"NMDetectiveAI.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    load_model(model, model_path, device=device)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    
    # Read the VCF/TSV file
    logger.info(f"Reading variants from {vcf_path}")
    if vcf_path.suffix.lower() == '.vcf':
        # Handle VCF format (though this is likely TSV from our pipeline)
        df = pd.read_csv(vcf_path, sep='\t', comment='#')
    else:
        # Handle TSV format
        df = pd.read_csv(vcf_path, sep='\t')
    
    logger.info(f"Found {len(df)} variants in input file")
    
    # Filter for stop-gained variants if not already filtered
    if 'Consequence' in df.columns:
        original_count = len(df)
        df = df[df['Consequence'].str.contains('stop_gained', na=False)]
        logger.info(f"Filtered to {len(df)} stop-gained variants (from {original_count} total)")
    
    # Add prediction columns
    df['NMDetectiveAI_prediction'] = np.nan
    df['NMDetectiveAI_status'] = 'not_processed'
    df['NMDetectiveAI_error'] = ''
    
    logger.info("Processing variants and running predictions...")
    
    # Process variants in smaller batches to handle memory efficiently
    batch_sequences = []
    batch_indices = []
    predictions_made = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing variants"):
        try:
            # Parse variant information from Uploaded_variation column
            # Format: chr_pos_ref/alt (e.g., "8_138153038_A/T")
            uploaded_var = row['Uploaded_variation']
            feature = row['Feature']  # Transcript ID
            
            # Parse uploaded_variation: chr_pos_ref/alt
            parts = uploaded_var.split('_')
            chrom = parts[0]
            pos = int(parts[1])
            ref, alt = parts[2].split('/') 
            
            # Create variant string in format chr:pos:ref:alt
            var_str = f"{chrom}:{pos}:{ref}:{alt}"
            
            # Create 6-track encoding using the existing function
            # Use gencode.v47 (closest to VEP's GENCODE v45 available in GenomeKit)
            six_track = create_six_track_encoding_with_variant(feature, var_str, gencode_version="gencode.v47")
            
            # Add to batch
            batch_sequences.append(six_track)
            batch_indices.append(idx)
            
            # Process batch when it reaches the specified size
            if len(batch_sequences) >= batch_size:
                predictions = _predict_batch(model, batch_sequences, device)
                
                # Store predictions
                for batch_idx, pred in zip(batch_indices, predictions):
                    df.at[batch_idx, 'NMDetectiveAI_prediction'] = float(pred)
                    df.at[batch_idx, 'NMDetectiveAI_status'] = 'processed'
                
                predictions_made += len(predictions)
                
                # Reset batch
                batch_sequences = []
                batch_indices = []
            
        except Exception as e:
            df.at[idx, 'NMDetectiveAI_status'] = 'failed'
            df.at[idx, 'NMDetectiveAI_error'] = str(e)
            logger.debug(f"Failed to process variant at row {idx}: {e}")
            continue
    
    # Process any remaining sequences in the final batch
    if len(batch_sequences) > 0:
        try:
            predictions = _predict_batch(model, batch_sequences, device)
            
            # Store predictions
            for batch_idx, pred in zip(batch_indices, predictions):
                df.at[batch_idx, 'NMDetectiveAI_prediction'] = float(pred)
                df.at[batch_idx, 'NMDetectiveAI_status'] = 'processed'
            
            predictions_made += len(predictions)
            logger.debug(f"Processed final batch: {len(predictions)} predictions made")
            
        except Exception as e:
            # Mark all remaining variants as failed
            for batch_idx in batch_indices:
                df.at[batch_idx, 'NMDetectiveAI_status'] = 'failed'
                df.at[batch_idx, 'NMDetectiveAI_error'] = f"Batch prediction failed: {str(e)}"
    
    logger.info(f"Successfully processed {predictions_made} variants")
    
    # Generate output files
    output_base = output_dir / f"{vcf_path.stem}_annotated"
    df.to_csv(output_base.with_suffix('.tsv'), sep='\t', index=False)
    return 


@app.command()
def annotate_gnomad_with_predictions(
    var_type: str = "rare",
    output_dir: str = None,
    batch_size: int = 4
):
    """
    Annotate gnomAD annotated file with NMDetectiveAI predictions.
    
    Args:
        var_type (str): Type of variants to annotate (default: "rare").
        output_dir (str): Output directory path. If None, saves to same location as input.
        batch_size (int): Batch size for prediction (default: 16)
    """
    from pathlib import Path
    
    # Set default input file if not provided
    gnomad_file = PROCESSED_DATA_DIR / "gnomad_v4.1" / f"annotated_{var_type}" / f"gnomad.v4.1.all_chromosomes.{var_type}_stopgain_snv.mane.annotated.tsv"
    if not gnomad_file.exists():
        raise FileNotFoundError(f"gnomAD file not found: {gnomad_file}")
    
    # Set output directory
    if output_dir is None:
        output_dir = gnomad_file.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting gnomAD annotation with NMDetectiveAI predictions")
    logger.info(f"Input file: {gnomad_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load the trained model
    config = TrainerConfig()
    model, _, _, _, device = _setup_model(config)
    
    # Load model weights
    model_path = MODELS_DIR / f"NMDetectiveAI.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    load_model(model, model_path, device=device)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
      
    # Read the gnomAD annotated file
    logger.info(f"Reading gnomAD variants from {gnomad_file}")
    df = pd.read_csv(gnomad_file, sep='\t')
    logger.info(f"Found {len(df)} variants in input file")
    
    # Add prediction columns
    df['NMDetectiveAI_prediction'] = np.nan
    df['NMDetectiveAI_status'] = 'not_processed'
    df['NMDetectiveAI_error'] = ''
    
    logger.info("Processing variants and running predictions...")
    
    # Process variants in batches
    batch_sequences = []
    batch_indices = []
    predictions_made = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing variants"):
        try:
            # Extract variant information
            chrom = row['chr']
            pos = int(row['pos'])
            ref = row['ref']
            alt = row['alt']
            transcript_id = row['transcript_id']
            
            # Create variant string in format chr:pos:ref:alt
            var_str = f"{chrom}:{pos}:{ref}:{alt}"
            
            # Create 6-track encoding using the full versioned transcript ID
            six_track = create_six_track_encoding_with_variant(transcript_id, var_str, gencode_version="gencode.v41")
            
            # Add to batch
            batch_sequences.append(six_track)
            batch_indices.append(idx)
            
            # Process batch when it reaches the specified size
            if len(batch_sequences) >= batch_size:
                predictions = _predict_batch(model, batch_sequences, device)
                
                # Store predictions
                for batch_idx, pred in zip(batch_indices, predictions):
                    df.at[batch_idx, 'NMDetectiveAI_prediction'] = float(pred)
                    df.at[batch_idx, 'NMDetectiveAI_status'] = 'processed'
                
                predictions_made += len(predictions)
                
                # Reset batch
                batch_sequences = []
                batch_indices = []
            
        except Exception as e:
            df.at[idx, 'NMDetectiveAI_status'] = 'failed'
            df.at[idx, 'NMDetectiveAI_error'] = str(e)
            logger.debug(f"Failed to process variant at row {idx}: {e}")
            continue
    
    # Process any remaining sequences in the final batch
    if len(batch_sequences) > 0:
        try:
            predictions = _predict_batch(model, batch_sequences, device)
            
            # Store predictions
            for batch_idx, pred in zip(batch_indices, predictions):
                df.at[batch_idx, 'NMDetectiveAI_prediction'] = float(pred)
                df.at[batch_idx, 'NMDetectiveAI_status'] = 'processed'
            
            predictions_made += len(predictions)
            logger.debug(f"Processed final batch: {len(predictions)} predictions made")
            
        except Exception as e:
            # Mark all remaining variants as failed
            for batch_idx in batch_indices:
                df.at[batch_idx, 'NMDetectiveAI_status'] = 'failed'
                df.at[batch_idx, 'NMDetectiveAI_error'] = f"Batch prediction failed: {str(e)}"
    
    logger.info(f"Successfully processed {predictions_made} variants")
    
    # Generate output file
    output_file = output_dir / f"{gnomad_file.stem}_with_predictions.tsv"
    df.to_csv(output_file, sep='\t', index=False)

    return 

@app.command()
def generate_penultimate_exon_mutations(all_transcripts: bool = False, model_name: str = "SP"):
    """
    Generate penultimate exon PTC predictions for all genes, inserting PTCs from 
    -100th codon to the last exon junction.
    
    Args:
        all_transcripts (bool): If True, process all transcripts per gene.
                              If False, process only the first transcript per gene.
        model_name (str): Name of the model to use ("PTC" or "DMS").
    """
    logger.info("Starting penultimate exon PTC prediction generation")
    
    # Load the trained model
    config = TrainerConfig()
    model, _, _, _, device = _setup_model(config)
    
    # Load model weights
    model_path = MODELS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    load_model(model, model_path, device=device)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    
    # Initialize genome
    genome = Genome(GENCODE_VERSION)
    
    # Create output directory
    output_dir = TABLES_DIR / "GW"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all gene names
    gene_names = [gene.name for gene in genome.genes if gene.name is not None]
    logger.info(f"Found {len(gene_names)} genes in the genome")
    
    all_predictions = []
    processed_genes = 0
    skipped_genes = 0

        # read MANE 
    mane_map = pd.read_csv(RAW_DATA_DIR / "DMS/MANE_transcripts.tsv", sep="\t", header=0)
    mane_dict = {
        row.gene_name: row.transcript_id
        for _, row in mane_map.iterrows()
    }
        
    # Process each gene
    for gene_name in tqdm(gene_names, desc="Processing genes"):
        # Select transcripts based on parameter
        if all_transcripts:
            transcripts = find_transcript_by_gene_name(genome, gene_name)
            # Process each transcript
            for transcript in transcripts:
                _process_transcript_penultimate_exon(transcript, gene_name, model, device, all_predictions)
        else:
            # Process MANE transcript if available, otherwise try transcripts one by one
            processed_successfully = False
            
            if gene_name in mane_dict:
                # Try MANE transcript first
                mane_transcript_id = mane_dict[gene_name]
                try:
                    transcript = genome.transcripts[mane_transcript_id]
                    if _process_transcript_penultimate_exon(transcript, gene_name, model, device, all_predictions):
                        processed_successfully = True
                except KeyError:
                    logger.debug(f"MANE transcript {mane_transcript_id} not found in genome for {gene_name}")
            
            # If MANE transcript failed or not available, try other transcripts
            if not processed_successfully:
                transcripts = find_transcript_by_gene_name(genome, gene_name)
                for transcript in transcripts:
                    if _process_transcript_penultimate_exon(transcript, gene_name, model, device, all_predictions):
                        processed_successfully = True
                        break  # Successfully processed one transcript, move to next gene
            
            if not processed_successfully:
                logger.warning(f"Failed to process any transcript for gene {gene_name}")
                skipped_genes += 1
                continue
    
    logger.info(f"Processed {processed_genes} genes, skipped {skipped_genes} genes")
    logger.info(f"Generated {len(all_predictions)} total penultimate exon PTC predictions")
    
    # Convert to DataFrame for easier analysis
    predictions_df = pd.DataFrame(all_predictions)
    
    # Save results as both CSV and pickle
    output_filename_base = f"penultimate_exon_ptc_predictions_{model_name.lower()}_{'all' if all_transcripts else 'first'}_transcripts"
    
    # Save as CSV
    csv_path = output_dir / f"{output_filename_base}.csv"
    predictions_df.to_csv(csv_path, index=False)
    logger.info(f"Saved penultimate exon predictions as CSV to {csv_path}")
    
    # Print summary statistics
    logger.info(f"Summary statistics:")
    logger.info(f"  Unique genes: {predictions_df['gene_name'].nunique()}")
    logger.info(f"  Unique transcripts: {predictions_df['transcript_id'].nunique()}")
    
    return all_predictions


# ── Constants for genome-wide bigBed generation ──────────────────────────────
_STOP_OHE = {
    "TAG": np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]]),
    "TAA": np.array([[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]]),
    "TGA": np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]),
}
_STOP_SET = frozenset(["TAG", "TAA", "TGA"])
_REVCOMP = {"A": "T", "T": "A", "C": "G", "G": "C"}

_AUTOSQL = """\
table nmdetectiveAIPredictions
"NMDetective-AI stop-gain SNV NMD efficiency predictions"
    (
    string chrom;        "Chromosome"
    uint chromStart;     "Start position (0-based)"
    uint chromEnd;       "End position (exclusive)"
    string name;         "Gene|Ref>Alt|StopCodon"
    uint score;          "Prediction scaled to 0-1000"
    char[1] strand;      "Strand"
    uint thickStart;     "Thick draw start"
    uint thickEnd;       "Thick draw end"
    uint reserved;       "Item RGB color"
    float prediction;    "NMD efficiency prediction (0=triggered, 1=evading)"
    string geneName;     "Gene name"
    string transcriptId; "Transcript ID"
    uint aaPosition;     "Amino acid position (1-based)"
    string refCodon;     "Reference codon (mRNA)"
    )
"""


def _cds_to_genomic_positions(transcript):
    """Map each CDS nucleotide (in 5'->3' mRNA order) to its 0-based genomic coordinate."""
    strand = transcript.strand
    positions = []
    for cds in transcript.cdss:
        if strand == "+":
            positions.extend(range(cds.start, cds.end))
        else:
            positions.extend(range(cds.end - 1, cds.start - 1, -1))
    return positions


def _enumerate_stopgain_snvs(codon_seq):
    """Return all single-nucleotide changes in *codon_seq* that produce a stop codon.

    Args:
        codon_seq: Three-letter codon in mRNA orientation (e.g. "CAG").

    Returns:
        List of (pos_in_codon, ref_base, alt_base, resulting_stop_codon).
    """
    codon = list(codon_seq.upper())
    if "".join(codon) in _STOP_SET:
        return []
    snvs = []
    for pos in range(3):
        ref = codon[pos]
        for alt in "ACGT":
            if alt == ref:
                continue
            mutated = codon.copy()
            mutated[pos] = alt
            if "".join(mutated) in _STOP_SET:
                snvs.append((pos, ref, alt, "".join(mutated)))
    return snvs


def _prediction_to_rgb(pred):
    """Map prediction to an RGB string for the BED itemRgb field.

    Blue (#022778) for NMD-triggered (pred ~ 0) to red (#ff9e9d) for NMD-evading (pred ~ 1).
    """
    blue, red = (2, 39, 120), (255, 158, 157)
    p = max(0.0, min(1.0, pred))
    r = int(blue[0] + (red[0] - blue[0]) * p)
    g = int(blue[1] + (red[1] - blue[1]) * p)
    b = int(blue[2] + (red[2] - blue[2]) * p)
    return f"{r},{g},{b}"


def _predict_sequences_batched(model, sequences, device, batch_size=16):
    """Run batched model inference on a list of encoded sequences.

    Args:
        model: NMDetectiveAI model in eval mode.
        sequences: List of numpy arrays, each shape (seq_len, 6).
        device: torch device.
        batch_size: Batch size for DataLoader.

    Returns:
        List of float predictions, one per input sequence.
    """
    if not sequences:
        return []
    dummy_df = pd.DataFrame({"y": [0.0] * len(sequences)})
    dataset = SequenceDataset(dummy_df, sequences, label_col="y")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    predictions = []
    with torch.no_grad():
        for batch_seqs, batch_lens, _ in loader:
            batch_seqs = batch_seqs.to(device)
            batch_lens = batch_lens.to(device)
            preds = model(batch_seqs, batch_lens).squeeze(-1)
            if preds.dim() == 0:
                predictions.append(float(preds.cpu()))
            else:
                predictions.extend(preds.cpu().tolist())
    return predictions


def _process_gene_for_bigbed(gene, model, device, genome, gencode_version, batch_size):
    """Predict NMD efficiency for all stop-gain SNVs in all coding transcripts of *gene*.

    Returns:
        List of tab-delimited BED9+5 lines (without trailing newline).
    """
    transcripts = [
        tr for tr in gene.transcripts
        if tr.cdss and len(tr.cdss) > 0
    ]
    if not transcripts:
        logger.debug(f"Skipping gene {gene.name}: no transcripts with CDS found")
        return []

    all_lines = []
    for transcript in tqdm(transcripts, desc=f"Processing transcripts of {gene.name}", leave=False):
        cds_length = sum(len(c) for c in transcript.cdss)
        if cds_length > 30000:
            logger.warning(f"Skipping transcript {transcript.id} of gene {gene.name}: CDS too long ({cds_length} nt)")
            continue

        effective_batch_size = 1 if cds_length > 10_000 else batch_size

        tid = transcript.id.split(".")[0]
        try:
            wt_sequence = create_six_track_encoding(tid, gencode_version=gencode_version)
        except Exception:
            logger.debug(f"Skipping transcript {transcript.id} of gene {gene.name}: failed to create sequence encoding")
            time.sleep(5)
            # need to wait after Cuda OOM to prevent immediate repeated failures; in practice this should be rare and only affect a few very long transcripts
            continue

        utr5_length = sum(len(u) for u in transcript.utr5s)
        n_codons = cds_length // 3
        cds_seq = "".join(genome.dna(cds) for cds in transcript.cdss)
        genomic_pos_map = _cds_to_genomic_positions(transcript)
        strand = transcript.strand
        chrom = transcript.chrom
        transcript_id = transcript.id

        # Predict for all 3 stop codons at every codon position
        pred_map = {}
        for stop_codon, stop_ohe in _STOP_OHE.items():
            sequences = []
            for ci in range(n_codons):
                ptc_pos = utr5_length + ci * 3
                if ptc_pos + 3 > len(wt_sequence):
                    break
                mutated = wt_sequence.copy()
                mutated[ptc_pos : ptc_pos + 3, :4] = stop_ohe
                sequences.append(mutated)
            if not sequences:
                logger.debug(f"No valid sequences for stop codon {stop_codon} in transcript {transcript.id}")
                continue
            preds = _predict_sequences_batched(model, sequences, device, effective_batch_size)
            for ci, p in enumerate(preds):
                pred_map[(ci, stop_codon)] = p

        # Enumerate stop-gain SNVs and build BED records
        for ci in range(n_codons):
            codon = cds_seq[ci * 3 : ci * 3 + 3].upper()
            if len(codon) < 3:
                continue
            for pos_in_codon, ref_mrna, alt_mrna, result_stop in _enumerate_stopgain_snvs(codon):
                pred = pred_map.get((ci, result_stop))
                if pred is None:
                    logger.debug(f"No prediction for stop codon {result_stop} at codon index {ci} in transcript {transcript.id}")
                    continue
                cds_nt = ci * 3 + pos_in_codon
                if cds_nt >= len(genomic_pos_map):
                    logger.debug(f"CDS nucleotide index {cds_nt} out of range for transcript {transcript.id}")
                    continue
                gpos = genomic_pos_map[cds_nt]
                ref_fwd = _REVCOMP[ref_mrna] if strand == "-" else ref_mrna
                alt_fwd = _REVCOMP[alt_mrna] if strand == "-" else alt_mrna
                score = max(0, min(1000, int(pred * 1000)))
                rgb = _prediction_to_rgb(pred)
                name = f"{gene.name}|{ref_fwd}>{alt_fwd}|{result_stop}"
                line = (
                    f"{chrom}\t{gpos}\t{gpos + 1}\t{name}\t{score}\t{strand}\t"
                    f"{gpos}\t{gpos + 1}\t{rgb}\t{pred:.4f}\t{gene.name}\t"
                    f"{transcript_id}\t{ci + 1}\t{codon}"
                )
                all_lines.append(line)

    return all_lines


@app.command()
def generate_genome_wide_bigbed(
    gencode_version: str = GENCODE_VERSION,
    batch_size: int = 16,
):
    """Generate genome-wide NMD predictions for all stop-gain SNVs as a bigBed file.

    For each protein-coding gene, all coding transcripts are processed.
    NMD efficiency is predicted for every codon position with all three stop codons
    (TAG, TAA, TGA).  Only positions reachable by a single-nucleotide variant are
    kept.  Output is saved to manuscript/supplementary/files/.
    """
    supp_dir = PROJ_ROOT / "manuscript" / "supplementary" / "files"
    supp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        # ── 1. Setup model ────────────────────────────────────────────────
        config = TrainerConfig()
        model, _, _, _, device = _setup_model(config)
        load_model(model, MODELS_DIR / "NMDetectiveAI.pt", device=device)
        model.eval()
        logger.info(f"Model loaded on {device}")

        # ── 2. Genome ─────────────────────────────────────────────────────
        genome = Genome(gencode_version)
        genes = [g for g in genome.genes if g.name and g.type == "protein_coding"]
        logger.info(f"Found {len(genes)} protein-coding genes")

        # ── 3. Predict all genes → unsorted BED ──────────────────────────
        unsorted_bed = tmp_dir / "stopgain_predictions.unsorted.bed"
        processed = 0
        skipped = 0
        total_records = 0

        with open(unsorted_bed, "w") as fh:
            for gene in tqdm(genes, desc="Predicting stop-gain SNVs"):
                try:
                    lines = _process_gene_for_bigbed(
                        gene, model, device, genome, gencode_version, batch_size
                    )
                    for line in lines:
                        fh.write(line + "\n")
                    total_records += len(lines)
                    if lines:
                        processed += 1
                except Exception as e:
                    logger.debug(f"Skipping {gene.name}: {e}")
                    skipped += 1

        logger.info(
            f"Processed {processed} genes, skipped {skipped}, "
            f"wrote {total_records} BED records"
        )

        # ── 4. Sort BED (no header — required by bedToBigBed) ─────────────
        sorted_noheader = tmp_dir / "stopgain_predictions.sorted.bed"
        subprocess.run(
            ["sort", "-k1,1", "-k2,2n", str(unsorted_bed), "-o", str(sorted_noheader)],
            check=True,
        )
        logger.info("BED sorted")

        # ── 5. Fetch chrom sizes (temp) ───────────────────────────────────
        chrom_sizes = tmp_dir / "hg38.chrom.sizes"
        logger.info("Downloading hg38 chrom.sizes from UCSC …")
        urllib.request.urlretrieve(
            "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes",
            str(chrom_sizes),
        )

        # ── 6. AutoSql schema (temp) ──────────────────────────────────────
        as_path = tmp_dir / "NMDetectiveAI.as"
        as_path.write_text(_AUTOSQL)

        # ── 7. Convert to bigBed ──────────────────────────────────────────
        bb_path = supp_dir / "NMDetectiveAI_stopgain_predictions.bb"
        subprocess.run(
            [
                "bedToBigBed",
                "-type=bed9+5",
                f"-as={as_path}",
                str(sorted_noheader),
                str(chrom_sizes),
                str(bb_path),
            ],
            check=True,
        )
        logger.success(f"bigBed written to {bb_path}")

        # ── 8. Prepend track + column headers to the final .bed ───────────
        bed_path = supp_dir / "NMDetectiveAI_stopgain_predictions.bed"
        track_header = (
            'track type=bed name="NMDetective-AI" '
            'description="NMDetective-AI stop-gain SNV NMD efficiency predictions '
            '(prediction: 0=NMD triggered, 1=NMD evading)" '
            'itemRgb=On visibility=pack\n'
        )
        col_header = (
            "#chrom\tchromStart\tchromEnd\tname\tscore\tstrand\t"
            "thickStart\tthickEnd\titemRgb\tprediction\tgeneName\t"
            "transcriptId\taaPosition\trefCodon\n"
        )
        with open(bed_path, "w") as out_fh, open(sorted_noheader) as in_fh:
            out_fh.write(track_header)
            out_fh.write(col_header)
            shutil.copyfileobj(in_fh, out_fh)
        logger.success(f"BED written to {bed_path}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    app()