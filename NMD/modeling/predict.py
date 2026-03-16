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

from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
from NMD.config import RAW_DATA_DIR, TABLES_DIR, MODELS_DIR, GENCODE_VERSION, PROCESSED_DATA_DIR, FIGURES_DIR
from NMD.data.transcripts import find_transcript_by_gene_name, generate_penultimate_exon_ptc_sequences, generate_all_ptc_sequences, create_six_track_encoding_with_variant, get_exon_boundaries_in_cds
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


if __name__ == "__main__":
    app()