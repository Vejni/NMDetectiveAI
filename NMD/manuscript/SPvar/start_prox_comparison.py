#!/usr/bin/env python3
"""
Manuscript figure: DMS Model Performance Comparison

This script evaluates NMDetective-AI, A, and B models on all DMS genes
and creates boxplots comparing their performance using a selected metric
(Spearman R, R2, or MSE).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from loguru import logger
from genome_kit import Genome
from torch.utils.data import DataLoader

from NMD.config import (
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    COLOURS,
    GENCODE_VERSION,
    VAL_CHRS,
    MODELS_DIR,
    OUT_DIR
)
from NMD.manuscript.output import get_paths
from NMD.modeling.models.NMDetectiveA import NMDetectiveA
from NMD.modeling.models.NMDetectiveB import NMDetectiveB
from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
from NMD.modeling.SequenceDataset import SequenceDataset
from NMD.modeling.features import setup_data
from NMD.utils import load_model
from NMD.modeling.TrainerConfig import TrainerConfig


# ============================================================================
# CONFIGURATION - All paths and parameters defined here
# ============================================================================

# Model and data paths
MODEL_PATH = MODELS_DIR / "NMDetectiveAI.pt"
DMS_SP_FILE = PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv"
PTC_FILE = PROCESSED_DATA_DIR / "PTC" / "somatic_TCGA.csv"
CV_METRICS_FILE = OUT_DIR / "DMS_SP" / "CV_normalize" / "all_gene_metrics.csv"
DMS_SEQUENCES_FILE = PROCESSED_DATA_DIR / "DMS_SP" / "processed_sequences.pkl"
CLUSTER_TABLE = TABLES_DIR / "SP" / "cluster_assignments.csv"

# Prediction mode for NMDetective-AI
USE_DMS_SEQUENCES = True  # If True, use pre-processed DMS sequences; if False, use setup_data encoding

# Metric selection: 'r2', 'rse', 'spearman'
METRIC_TO_PLOT = 'pearson'  # Options: 'r2', 'rse', 'spearman'

# Cross-validation settings
USE_SIGMOID_CV = True  # If True, use 10-fold CV for sigmoid methods (comparable to AI-SP)
                       # When True, sigmoid parameters are computed only from training genes in each fold,
                       # making the evaluation fair and comparable to NMDetective-AI-SP's CV approach
N_FOLDS = 10  # Number of CV folds for sigmoid methods
SIGMOID_CV_FILE = TABLES_DIR / "SP" / "sigmoid_cv_results.csv"  # CV results for sigmoid methods

SCRIPT_NAME = "start_prox_comparison"

# Plot parameters
FIGURE_SIZE = (12, 5)
DPI = 300

# Colors (using config COLOURS scheme)
MODEL_AI_SP_COLOR = COLOURS[4]  # '#fab387' - light orange for NMDetective-AI-SP
MODEL_AI_COLOR = COLOURS[6]  # '#022778' - dark blue
MODEL_A_COLOR = COLOURS[0]   # '#fb731d' - orange
MODEL_B_COLOR = COLOURS[5]   # '#2778ff' - blue
DMS_SIGMOID_COLOR = COLOURS[3]  # '#ffdfcb' - light pink
CLUSTER_SIGMOID_COLOR = COLOURS[1]  # Color for cluster-based sigmoids


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

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
        if cumulative > utr5_length:
            ejc_positions.append(cumulative + 50)
    
    for ptc_pos in ptc_positions:
        # Convert to CDS coordinates
        ptc_pos_cds = ptc_pos - utr5_length
        
        # InLastExon
        in_last_exon = ptc_pos >= last_exon_start
        
        # DistanceToStart (in CDS)
        distance_to_start = min(ptc_pos_cds, 1000)
        
        # Find which exon contains this PTC and get ExonLength
        cumulative = 0
        exon_length = 0
        for i, exon in enumerate(transcript.exons):
            exon_start = cumulative
            cumulative += len(exon)
            if exon_start <= ptc_pos < cumulative:
                if cumulative <= utr5_length:
                    exon_length = 0
                elif exon_start < utr5_length:
                    exon_length = cumulative - utr5_length
                else:
                    exon_length = len(exon)
                break
        
        # 50ntToLastEJ
        within_50nt = (ptc_pos < last_exon_start) and ((last_exon_start - ptc_pos) <= 55)
        
        # PTC_EJC_dist
        downstream_ejcs = [ejc for ejc in ejc_positions if ejc > ptc_pos]
        if downstream_ejcs:
            ptc_ejc_dist = downstream_ejcs[0] - ptc_pos
        else:
            ptc_ejc_dist = 0
        
        # DistanceToWTStop
        distance_to_wt_stop = (utr5_length + normal_stop_pos) - ptc_pos
        
        # RNAHalfLife
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


def calculate_metric(predictions, observations, metric_type=None):
    """
    Calculate evaluation metric between predictions and observations.
    
    Args:
        predictions: Model predictions
        observations: True observations
        metric_type: Type of metric ('r2', 'rse', 'pearson', 'spearman'). 
                    If None, uses METRIC_TO_PLOT
        
    Returns:
        R², RSE, Spearman ρ, or Pearson r depending on metric_type
    """
    if metric_type is None:
        metric_type = METRIC_TO_PLOT
        
    if metric_type == 'r2':
        ss_res = np.sum((observations - predictions)**2)
        ss_tot = np.sum((observations - observations.mean())**2)
        return 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    elif metric_type == 'rse':
        # Relative Squared Error: RMSE / sd(observations)
        mse = np.mean((observations - predictions)**2)
        rmse = np.sqrt(mse)
        std_obs = np.std(observations, ddof=1)
        return rmse / std_obs if std_obs > 0 else np.nan
    elif metric_type == 'pearson':
        return pearsonr(predictions, observations)[0]
    else:  # 'spearman'
        return spearmanr(predictions, observations)[0]


def load_sigmoid_params():
    """
    Load pre-computed sigmoid parameters from observations.
    
    Returns:
        DataFrame with gene names and their sigmoid parameters (A, K, B, M)
    """
    sigmoid_file = TABLES_DIR / "SP" / "sigmoid_params_observations.csv"
    
    if not sigmoid_file.exists():
        raise FileNotFoundError(
            f"Sigmoid parameters not found at {sigmoid_file}. "
            f"Run fit_sigmoids_to_observations() from dms_sigmoid_fitting.py first."
        )
    
    logger.info(f"Loading sigmoid parameters from {sigmoid_file}")
    params_df = pd.read_csv(sigmoid_file)
    
    # Keep only needed columns
    return params_df[['gene', 'A', 'K', 'B', 'M']].copy()


def get_median_sigmoid_params(gene_params_df):
    """
    Get median sigmoid parameters from a set of gene fits.
    
    Args:
        gene_params_df: DataFrame with gene sigmoid parameters
        
    Returns:
        Array of median parameters [A, K, B, M]
    """
    median_params = [
        gene_params_df['A'].median(),
        gene_params_df['K'].median(),
        gene_params_df['B'].median(),
        gene_params_df['M'].median()
    ]
    
    return median_params


def evaluate_gene_with_global_sigmoid(gene_name, gene_data, all_gene_params):
    """
    Evaluate a gene using global median sigmoid parameters.
    
    Args:
        gene_name: Name of the gene
        gene_data: DataFrame with gene's DMS data
        all_gene_params: DataFrame with all genes' sigmoid parameters
        
    Returns:
        Spearman ρ or R² between global sigmoid and observations
    """
    # Get median parameters from all genes
    median_params = get_median_sigmoid_params(all_gene_params)
    logger.debug(f"Evaluating {gene_name} with global median sigmoid parameters: {median_params}")

    positions = gene_data['PTCposition_nt'].values
    observations = gene_data['NMDeff'].values

    logger.debug(f"Evaluating {gene_name} with global sigmoid parameters: {median_params}")
    
    # Scale positions to [0,1] to match the space where sigmoid was fitted
    x_min = positions.min()
    x_max = positions.max()
    positions_scaled = (positions - x_min) / (x_max - x_min)
    
    # Get predictions from median sigmoid
    from NMD.analysis.dms_sigmoid_fitting import logistic4
    predictions = logistic4(positions_scaled, *median_params)
    
    # Debug: check predictions
    logger.debug(f"Global predictions for {gene_name}: min={predictions.min():.4f}, "
                 f"max={predictions.max():.4f}, mean={predictions.mean():.4f}, "
                 f"first3={predictions[:3].tolist()}")
    
    return calculate_metric(predictions, observations)


def evaluate_gene_with_cluster_sigmoid(gene_name, gene_data, cluster_num, all_gene_params, cluster_assignments):
    """
    Evaluate a gene using cluster median sigmoid parameters.
    
    Args:
        gene_name: Name of the gene
        gene_data: DataFrame with gene's DMS data
        cluster_num: Cluster assignment for this gene
        all_gene_params: DataFrame with all genes' sigmoid parameters
        cluster_assignments: DataFrame with gene-cluster mapping
        
    Returns:
        Spearman ρ or R² between cluster sigmoid and observations
    """
    # Get genes in this cluster
    cluster_genes = cluster_assignments[cluster_assignments['cluster'] == cluster_num]['gene'].tolist()
    
    # Debug: check what we're filtering
    logger.debug(f"Cluster {cluster_num} has {len(cluster_genes)} genes: {cluster_genes[:5]}...")
    logger.debug(f"all_gene_params has {len(all_gene_params)} rows with genes: {all_gene_params['gene'].head().tolist()}")
    
    # Filter parameters for genes in this cluster
    cluster_params = all_gene_params[all_gene_params['gene'].isin(cluster_genes)].copy()
    
    logger.debug(f"After filtering for cluster {cluster_num}: found {len(cluster_params)} parameter sets")
    logger.debug(f"Cluster genes in params: {cluster_params['gene'].tolist()}")
    
    if len(cluster_params) == 0:
        logger.error(f"No sigmoid parameters found for any genes in cluster {cluster_num}!")
        logger.error(f"Genes in cluster: {cluster_genes}")
        logger.error(f"Genes in params: {all_gene_params['gene'].unique().tolist()}")
        raise ValueError(f"No parameters found for cluster {cluster_num}")
    
    # Get median parameters from cluster
    median_params = get_median_sigmoid_params(cluster_params)
    
    # Debug logging
    logger.info(f"{gene_name} in cluster {cluster_num}: {len(cluster_genes)} cluster genes, "
                 f"{len(cluster_params)} params found, median params: {median_params}")
    
    
    positions = gene_data['PTCposition_nt'].values
    observations = gene_data['NMDeff'].values
    
    # Scale positions to [0,1] to match the space where sigmoid was fitted
    x_min = positions.min()
    x_max = positions.max()
    positions_scaled = (positions - x_min) / (x_max - x_min)
    
    # Get predictions from median sigmoid
    from NMD.analysis.dms_sigmoid_fitting import logistic4
    predictions = logistic4(positions_scaled, *median_params)
    
    # Debug: check predictions
    logger.debug(f"Cluster {cluster_num} predictions for {gene_name}: min={predictions.min():.4f}, "
                 f"max={predictions.max():.4f}, mean={predictions.mean():.4f}, "
                 f"first3={predictions[:3].tolist()}")
    
    return calculate_metric(predictions, observations)


def perform_sigmoid_cv(force_recompute: bool = False):
    """
    Perform 10-fold cross-validation for sigmoid methods.
    Evaluates Global and Clustered logistic models using CV to match NMDetective-AI-SP evaluation.
    
    Returns:
        DataFrame with per-gene mean CV metrics (one row per gene)
    """
    logger.info("Performing 10-fold CV for sigmoid methods...")
    
    # Check if CV results already exist
    if SIGMOID_CV_FILE.exists() and not force_recompute:
        logger.info(f"Loading existing sigmoid CV results from {SIGMOID_CV_FILE}")
        cv_df = pd.read_csv(SIGMOID_CV_FILE)
        if 'Global logistic' not in cv_df.columns or 'Clustered logistic' not in cv_df.columns:
            logger.warning("Existing CV file has unexpected columns — recomputing.")
        else:
            return cv_df
    
    # Load DMS data and sigmoid parameters
    dms_metadata = pd.read_csv(DMS_SP_FILE)
    all_gene_params = load_sigmoid_params()
    cluster_assignments = pd.read_csv(CLUSTER_TABLE)
    
    # Get unique genes
    unique_genes = dms_metadata['gene'].unique()
    n_genes = len(unique_genes)
    logger.info(f"Running CV on {n_genes} genes with {N_FOLDS} folds")
    
    # Create folds - simple split by gene index
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    cv_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_genes)):
        logger.info(f"Processing fold {fold_idx + 1}/{N_FOLDS}")
        
        # Split genes into train and test
        train_genes = unique_genes[train_idx]
        test_genes = unique_genes[test_idx]
        
        logger.info(f"  Train genes: {len(train_genes)}, Test genes: {len(test_genes)}")
        
        # Get sigmoid parameters for training genes only
        train_gene_params = all_gene_params[all_gene_params['gene'].isin(train_genes)].copy()
        
        # Evaluate each test gene
        for gene_name in test_genes:
            gene_data = dms_metadata[dms_metadata['gene'] == gene_name].copy()
            
            if len(gene_data) == 0:
                logger.warning(f"No data for {gene_name}")
                continue
            
            # Get cluster assignment for this gene
            gene_cluster_row = cluster_assignments[cluster_assignments['gene'] == gene_name]
            if len(gene_cluster_row) == 0:
                logger.warning(f"No cluster assignment for {gene_name}")
                continue
            
            cluster_num = gene_cluster_row['cluster'].iloc[0]
            
            # Evaluate with global sigmoid (using training genes only)
            positions = gene_data['PTCposition_nt'].values
            observations = gene_data['NMDeff'].values
            
            # Global sigmoid: median of all training genes
            global_median_params = get_median_sigmoid_params(train_gene_params)
            positions_scaled = (positions - positions.min()) / (positions.max() - positions.min())
            
            from NMD.analysis.dms_sigmoid_fitting import logistic4
            global_predictions = logistic4(positions_scaled, *global_median_params)
            
            # Evaluate with cluster sigmoid (using training genes in same cluster only)
            # Get training genes in the same cluster as test gene
            cluster_train_genes = cluster_assignments[
                (cluster_assignments['cluster'] == cluster_num) & 
                (cluster_assignments['gene'].isin(train_genes))
            ]['gene'].tolist()
            
            if len(cluster_train_genes) == 0:
                logger.warning(f"No training genes in cluster {cluster_num} for {gene_name}")
                # Fall back to global sigmoid
                cluster_predictions = global_predictions
            else:
                cluster_train_params = train_gene_params[
                    train_gene_params['gene'].isin(cluster_train_genes)
                ].copy()
                
                cluster_median_params = get_median_sigmoid_params(cluster_train_params)
                cluster_predictions = logistic4(positions_scaled, *cluster_median_params)
            
            # Calculate metrics
            global_metric = calculate_metric(global_predictions, observations)
            cluster_metric = calculate_metric(cluster_predictions, observations)

            cv_results.append({
                'gene': gene_name,
                'fold': fold_idx,
                'Global logistic': global_metric,
                'Clustered logistic': cluster_metric
            })
    
    # Aggregate: mean metric across folds per gene
    cv_df = pd.DataFrame(cv_results)
    cv_df = cv_df.groupby('gene')[['Global logistic', 'Clustered logistic']].mean().reset_index()
    SIGMOID_CV_FILE.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(SIGMOID_CV_FILE, index=False)
    logger.info(f"Saved sigmoid CV results to {SIGMOID_CV_FILE}")
    
    return cv_df


def evaluate_gene(gene_name, gene_data, dms_sequences_all, nmdetective_a, nmdetective_b, 
                  model_ai, device, genome):
    """
    Evaluate all three models on a single gene.
    
    Args:
        gene_name: Name of the gene
        gene_data: DataFrame with gene's DMS data
        dms_sequences_all: All pre-encoded DMS sequences
        nmdetective_a: Trained NMDetective-A model
        nmdetective_b: Trained NMDetective-B model
        model_ai: Trained NMDetective-AI model
        device: Torch device
        genome: Genome object
        
    Returns:
        Dictionary with gene name and Spearman correlation for each model
    """
    # Get transcript ID for this gene (use first available)
    gene_chromosomes = gene_data['chr'].unique()
    if len(gene_chromosomes) == 0:
        logger.warning(f"No chromosome info for {gene_name}")
        return None
    
    # Find transcript for this gene
    matching_genes = [g for g in genome.genes if g.name == gene_name]
    if not matching_genes:
        logger.warning(f"Gene {gene_name} not found in genome")
        return None
    
    gene_obj = matching_genes[0]
    if not gene_obj.transcripts:
        logger.warning(f"No transcripts found for {gene_name}")
        return None
    
    # Use first transcript with CDS
    transcripts_with_cds = [
        tr for tr in gene_obj.transcripts 
        if tr.cdss is not None and len(tr.cdss) > 0
    ]
    
    if not transcripts_with_cds:
        logger.warning(f"No transcripts with CDS found for {gene_name}")
        return None
    
    transcript = transcripts_with_cds[0]
    utr5_length = sum(len(exon) for exon in transcript.utr5s) if transcript.utr5s else 0
    
    # Get sequences for this gene
    gene_sequences = [dms_sequences_all[i] for i in gene_data.index]
    dms_positions_nt = gene_data['PTCposition_nt'].tolist()
    dms_nmdeff = gene_data['NMDeff'].values
    
    # Predict with NMDetective-AI
    label_col = "NMDeff"
    eval_dataset = SequenceDataset(gene_data, gene_sequences, label_col=label_col)
    eval_loader = DataLoader(eval_dataset, batch_size=1)
    
    predictions_ai = []
    with torch.no_grad():
        for batch_sequences, batch_lengths, _ in eval_loader:
            batch_sequences, batch_lengths = [
                x.to(device) for x in (batch_sequences, batch_lengths)
            ]
            batch_preds = model_ai(batch_sequences, batch_lengths).squeeze()
            
            if batch_preds.dim() == 0:
                predictions_ai.append(float(batch_preds.cpu().numpy()))
            else:
                predictions_ai.extend(batch_preds.cpu().numpy())
    
    predictions_ai = np.array(predictions_ai)
    
    # Compute features and predict with NMDetective-A/B
    ptc_positions_transcript = [pos + utr5_length for pos in dms_positions_nt]
    features_df = compute_features_for_ptc_positions(transcript, ptc_positions_transcript)
    
    if len(features_df) == 0:
        logger.warning(f"Could not compute features for {gene_name}")
        return None
    
    predictions_a = nmdetective_a.predict(features_df)
    predictions_b = nmdetective_b.predict(features_df)
    
    return {
        'gene': gene_name,
        'n_observations': len(dms_nmdeff),
        'NMDetective-AI': calculate_metric(predictions_ai, dms_nmdeff),
        'NMDetective-A': calculate_metric(predictions_a, dms_nmdeff),
        'NMDetective-B': calculate_metric(predictions_b, dms_nmdeff),
    }


def process_data(source_data_path=None, regenerate=True):
    """Process data and evaluate all models on all genes. Returns DataFrame with results."""
    
    # Check if output table already exists
    if source_data_path and source_data_path.exists() and not regenerate:
        logger.info(f"Loading existing data from {source_data_path}")
        return pd.read_csv(source_data_path)
    
    logger.info("Starting model evaluation...")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Metric: {METRIC_TO_PLOT}")
    
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
    
    # Load NMDetective-AI model
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
    
    # Load DMS sequences and metadata
    if USE_DMS_SEQUENCES:
        logger.info("Loading pre-processed DMS sequences from pickle...")
        import pickle
        with open(DMS_SEQUENCES_FILE, 'rb') as f:
            dms_sequences_all = pickle.load(f)
        dms_metadata_all = pd.read_csv(DMS_SP_FILE)
        logger.info(f"Loaded {len(dms_sequences_all)} pre-processed sequences")
    else:
        logger.info("Encoding DMS sequences with setup_data...")
        dms_sequences_all, dms_metadata_all = setup_data(
            DMS_SP_FILE,
            batch_size=config.batch_size,
            data_type="DMS"
        )
        logger.info(f"Loaded {len(dms_sequences_all)} DMS sequences")
    
    # Load cluster assignments
    cluster_assignments_df = None
    if CLUSTER_TABLE.exists() and not USE_SIGMOID_CV:
        logger.info("Loading cluster assignments...")
        cluster_assignments_df = pd.read_csv(CLUSTER_TABLE)
        logger.info(f"Loaded {len(cluster_assignments_df)} cluster assignments")
    elif not USE_SIGMOID_CV:
        logger.warning(f"Cluster assignments not found: {CLUSTER_TABLE}")
    
    # Initialize genome
    genome = Genome(GENCODE_VERSION)
    
    unique_genes = dms_metadata_all['gene'].unique()
    logger.info(f"Evaluating on {len(unique_genes)} genes")
    
    # Evaluate sigmoid methods with CV if requested
    if USE_SIGMOID_CV:
        logger.info("Using 10-fold CV for sigmoid methods...")
        sigmoid_cv_results = perform_sigmoid_cv(force_recompute=regenerate)
    else:
        logger.info("Using all genes for sigmoid parameter estimation (no CV)...")
        # Load pre-computed sigmoid parameters
        logger.info("Loading sigmoid parameters...")
        all_gene_params = load_sigmoid_params()
        logger.info(f"Loaded sigmoid parameters for {len(all_gene_params)} genes")
    
    # Evaluate each gene
    results = []
    for gene_name in unique_genes:
        logger.info(f"Evaluating {gene_name}...")
        gene_data = dms_metadata_all[dms_metadata_all['gene'] == gene_name].copy()
        
        result = evaluate_gene(
            gene_name, gene_data, dms_sequences_all,
            nmdetective_a, nmdetective_b, model_ai, device,
            genome
        )
        
        if result is None:
            logger.warning(f"Skipping {gene_name}: evaluate_gene returned None")
            continue

        # Add sigmoid evaluations
        if USE_SIGMOID_CV:
            gene_cv_row = sigmoid_cv_results[sigmoid_cv_results['gene'] == gene_name]
            if len(gene_cv_row) > 0:
                result['Global logistic'] = gene_cv_row['Global logistic'].iloc[0]
                result['Clustered logistic'] = gene_cv_row['Clustered logistic'].iloc[0]
                logger.info(f"  Using CV sigmoid results for {gene_name}")
            else:
                logger.warning(f"No CV results found for {gene_name}")
                result['Global logistic'] = np.nan
                result['Clustered logistic'] = np.nan
        else:
            result['Global logistic'] = evaluate_gene_with_global_sigmoid(gene_name, gene_data, all_gene_params)
            cluster_num = cluster_assignments_df[cluster_assignments_df['gene'] == gene_name]['cluster'].iloc[0]
            result['Clustered logistic'] = evaluate_gene_with_cluster_sigmoid(
                gene_name, gene_data, cluster_num, all_gene_params, cluster_assignments_df
            )

        logger.info(f"  AI: {result['NMDetective-AI']:.4f}, "
                    f"A: {result['NMDetective-A']:.4f}, "
                    f"B: {result['NMDetective-B']:.4f}")

        results.append(result)
    
    # Save results
    df = pd.DataFrame(results)
    
    # Load CV metrics and merge
    logger.info("Loading NMDetective-AI-SP CV metrics...")
    cv_metrics = pd.read_csv(CV_METRICS_FILE)
    
    # Map METRIC_TO_PLOT to the column name in the CV file
    metric_col_map = {'r2': 'r2', 'rse': 'rse', 'pearson': 'corr', 'spearman': 'corr'}
    metric_col = metric_col_map.get(METRIC_TO_PLOT, 'corr')
    if METRIC_TO_PLOT == 'spearman':
        logger.warning("Spearman not available for NMDetective-AI-SP CV metrics; using Pearson instead.")
    cv_metrics_subset = cv_metrics[['gene', metric_col]].rename(columns={metric_col: 'NMDetective-AI-SP'})
    logger.info(f"Using CV metric column '{metric_col}' for NMDetective-AI-SP")

    # Merge with results
    df = df.merge(cv_metrics_subset, on='gene', how='left')
    n_merged = df['NMDetective-AI-SP'].notna().sum()
    logger.info(f"Merged CV metrics for {n_merged} genes")

    if source_data_path:
        source_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(source_data_path, index=False)
        logger.info(f"Results saved to {source_data_path}")
    
    return df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_from_table(df):
    """
    Create boxplot comparing model performance.

    Args:
        df: DataFrame with evaluation results
    """
    logger.info("Creating boxplot...")

    models = ['NMDetective-AI-SP', 'NMDetective-AI', 'NMDetective-A', 'NMDetective-B', 'Global logistic', 'Clustered logistic']
    colors = [MODEL_AI_SP_COLOR, MODEL_AI_COLOR, MODEL_A_COLOR, MODEL_B_COLOR, DMS_SIGMOID_COLOR, CLUSTER_SIGMOID_COLOR]
    ylabel = {
        'r2': 'R²',
        'rse': 'Relative Squared Error',
        'spearman': 'Spearman ρ',
        'pearson': 'Pearson r',
    }.get(METRIC_TO_PLOT, METRIC_TO_PLOT.upper())

    data_for_plot = [df[model].dropna().values for model in models]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bp = ax.boxplot(data_for_plot, labels=models, patch_artist=True,
                    showmeans=True, meanline=False,
                    meanprops=dict(marker='D', markerfacecolor='red',
                                  markeredgecolor='red', markersize=6))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.set_xlabel('Model', fontsize=18, fontweight='bold')
    ax.tick_params(axis='x', labelsize=16, rotation=15)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, alpha=0.3, axis='y')

    if METRIC_TO_PLOT in ['r2', 'pearson', 'spearman']:
        ax.set_ylim(-0.25, 1.0)
    elif METRIC_TO_PLOT == 'rse':
        ax.set_ylim(0, 2.0)

    summary_text = f"n={len(df)} genes\n"
    for model in models:
        values = df[model].dropna()
        summary_text += f"{model}: μ={values.mean():.3f}, m={values.median():.3f} (n={len(values)})\n"

    ax.text(0.98, 0.02, summary_text.strip(),
            transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1))

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def compute_summary_stats(df):
    """Compute summary statistics for all models."""
    models = ['NMDetective-AI-SP', 'NMDetective-AI', 'NMDetective-A', 'NMDetective-B', 'Global logistic', 'Clustered logistic']
    summary_rows = []
    for model in models:
        if model in df.columns:
            values = df[model].dropna()
            if len(values) > 0:
                q1, q3 = values.quantile(0.25), values.quantile(0.75)
                summary_rows.append({
                    'model': model,
                    'metric': METRIC_TO_PLOT,
                    'n': len(values),
                    'mean': values.mean(),
                    'median': values.median(),
                    'sd': values.std(),
                    'q1': q1,
                    'q3': q3,
                    'iqr': q3 - q1,
                })
    return pd.DataFrame(summary_rows)


def main(figure_label=None, figure_number=None, regenerate=False):
    """Main execution function."""
    paths = get_paths(SCRIPT_NAME, figure_label=figure_label, figure_number=figure_number)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    if paths.source_data:
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
    
    # Process data (or load if already exists)
    df = process_data(source_data_path=paths.source_data, regenerate=regenerate)
    
    # Compute and save summary statistics
    logger.info("Computing summary statistics...")
    summary_df = compute_summary_stats(df)
    summary_file = TABLES_DIR / "SP" / "dms_model_comparison_summary.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary statistics saved to {summary_file}")
    
    # Create and save figure
    logger.info("Generating figure...")
    fig = plot_from_table(df)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    logger.info(f"Figure PDF saved to {paths.figure_pdf}")
    plt.close(fig)
    
    logger.success("\nModel comparison complete!")


if __name__ == "__main__":
    main()
