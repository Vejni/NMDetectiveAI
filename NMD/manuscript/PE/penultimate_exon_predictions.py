#!/usr/bin/env python3
"""
Manuscript figure: Penultimate + Last Exon predictions vs DMS observations
Compare NMDetective-AI predictions with DMS data for BRCA1 and ATP7A genes.
Shows penultimate exon (PE) and last exon regions with PE DMS observations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import torch
from loguru import logger
from genome_kit import Genome
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.utils import resample
from scipy.optimize import curve_fit

from NMD.config import (
    PROCESSED_DATA_DIR, 
    MODELS_DIR, 
    INTERIM_DATA_DIR,
    COLOURS,
    GENCODE_VERSION,
    CONTRASTING_3_COLOURS,
    VAL_CHRS
)
from NMD.manuscript.output import get_paths
from NMD.modeling.predict import predict_transcript_ptcs
from NMD.modeling.models.NMDetectiveA import NMDetectiveA
from NMD.modeling.models.NMDetectiveB import NMDetectiveB


# ============================================================================
# CONFIGURATION - All paths and parameters defined here
# ============================================================================

# Model and data paths
SCRIPT_NAME = "penultimate_exon_predictions"
MODEL_PATH = MODELS_DIR / "NMDetectiveAI.pt"
PE_DATA_FILE = PROCESSED_DATA_DIR / "DMS_PE" / "fitness.csv"
PTC_DATA_FILE = INTERIM_DATA_DIR / "PTC" / "somatic_TCGA.csv"

# Gene configurations
GENES = {
    "BRCA1": {
        "transcript_id": "ENST00000357654.7"
    },
    "ATP7A": {
        "transcript_id": "ENST00000341514.10"
    }
}
PLOT_TITLE = "NMDetective-AI predictions vs DMS observations in penultimate and last exon regions"

# Plot parameters
SMOOTHING_SIGMA = 0  # Gaussian smoothing sigma
FIGURE_SIZE = (14, 7)  # Overall figure size
DPI = 300
MAX_NTS_LAST_EXON = 100  # Maximum nucleotides to show after last exon junction

# Marker size for NMDetective-AI prediction scatter
PRED_MARKER_SIZE = 40

# Metric calculation mode
COMPARE_AGAINST_DMS_LOESS = True  # If True, compare models against DMS LOESS curve; if False, against raw DMS observations
INCLUDE_NMDETECTIVE_A = False  # If True, include NMDetective-A predictions in the plot

# Colors (using config COLOURS scheme)
MODEL_AI_COLOR = CONTRASTING_3_COLOURS[2]  # '#022778' - dark blue
MODEL_A_COLOR = COLOURS[0]   # '#fb731d' - orange
MODEL_B_COLOR = CONTRASTING_3_COLOURS[0]   # '#2778ff' - blue
PE_COLOR = COLOURS[1]     # '#fcbb01' - yellow/gold
JUNCTION_COLOR = COLOURS[2]  # '#ff9e9d' - pink/red
RULE_55NT_COLOR = COLOURS[2]  # Different from MODEL_A
LOESS_COLOR = COLOURS[4]  # '#d6f3ff' - light blue
TCGA_LOESS_COLOR = 'darkgray'  # TCGA PTC LOESS curve
TCGA_CI_COLOR = 'lightgray'  # TCGA PTC confidence interval

# Stop codon markers
STOP_MARKERS = {
    'UAG': 'o',  # circle
    'UAA': 's',  # square
    'UGA': '^'   # triangle
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def compute_loess_with_ci(x, y, frac=0.3, n_bootstrap=100, ci=0.95):
    """
    Compute LOESS smoothing with bootstrap confidence intervals.
    
    Args:
        x: x-coordinates (positions)
        y: y-coordinates (values)
        frac: LOESS smoothing fraction
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval (default 0.95 for 95% CI)
    
    Returns:
        x_smooth: Smoothed x positions
        y_smooth: LOESS smoothed values
        y_lower: Lower confidence bound
        y_upper: Upper confidence bound
    """
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Compute main LOESS curve
    loess_result = lowess(y_sorted, x_sorted, frac=frac)
    x_smooth = loess_result[:, 0]
    y_smooth = loess_result[:, 1]
    
    # Bootstrap for confidence intervals
    bootstrap_curves = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = resample(np.arange(len(x_sorted)), replace=True, n_samples=len(x_sorted))
        x_boot = x_sorted[indices]
        y_boot = y_sorted[indices]
        
        # Compute LOESS on bootstrap sample
        try:
            loess_boot = lowess(y_boot, x_boot, frac=frac)
            # Interpolate to match x_smooth positions
            y_boot_interp = np.interp(x_smooth, loess_boot[:, 0], loess_boot[:, 1])
            bootstrap_curves.append(y_boot_interp)
        except:
            # Skip failed bootstrap samples
            continue
    
    # Compute confidence intervals
    if len(bootstrap_curves) > 0:
        bootstrap_curves = np.array(bootstrap_curves)
        alpha = (1 - ci) / 2
        y_lower = np.percentile(bootstrap_curves, alpha * 100, axis=0)
        y_upper = np.percentile(bootstrap_curves, (1 - alpha) * 100, axis=0)
    else:
        # Fallback if bootstrap fails
        y_lower = y_smooth
        y_upper = y_smooth
    
    return x_smooth, y_smooth, y_lower, y_upper


def four_param_logistic(x, a, d, c, b):
    """
    4-parameter logistic (4PL) function.
    y = d + (a - d) / (1 + (x / c)**b)
    Args:
        x: independent variable (positions)
        a: minimum asymptote
        d: maximum asymptote
        c: inflection point (EC50)
        b: slope
    Returns:
        y values for x
    """
    x = np.array(x, dtype=float)
    # Prevent division by zero for c near zero by adding small epsilon
    c = float(c) if c != 0 else 1e-8
    return d + (a - d) / (1.0 + (x / c) ** b)


def load_dms_data(gene_name):
    """
    Load and process DMS data for penultimate exon region.
    
    Args:
        gene_name: Name of the gene (BRCA1 or ATP7A)
    
    Returns:
        pe_data: DataFrame with PE observations (keeps stop_type column)
    """
    # Load PE data
    pe = pd.read_csv(PE_DATA_FILE)
    pe = pe[pe.gene == gene_name][["stop_type", "PTC_pos_rev", "NMDeff"]]
    
    # Group by stop codon type and position, take median
    pe_grouped = pe.groupby(['stop_type', 'PTC_pos_rev'], as_index=False)['NMDeff'].median()
    
    return pe_grouped


def load_genomic_ptc_data():
    """
    Load genomic PTC data from ALL genes in penultimate and last exon regions.
    This provides background context similar to the notebook analysis.
    
    Returns:
        DataFrame with PTC_EJC_dist (distance from EJC), NMDeff values, and region type
    """
    # Load PTC data
    ptc_df = pd.read_csv(PTC_DATA_FILE)
    
    # Filter for no indels (common to both regions)
    ptc_df = ptc_df[ptc_df.Ref != "-"]
    ptc_df = ptc_df[ptc_df.Alt != "-"]
    
    # Get penultimate exon PTCs
    ptc_pe = ptc_df[
        (ptc_df.Last_Exon == False) &
        (ptc_df.CDS_num_exons_downstream == 1) &
        (ptc_df.Long_Exon == False)
    ].copy()
    ptc_pe['region'] = 'penultimate'
    # For penultimate exon: PTC_EJC_dist is negative (upstream), invert sign
    ptc_pe['PTC_EJC_dist'] = ptc_pe['PTC_EJC_dist'] * (-1)
    
    # Get last exon PTCs
    ptc_le = ptc_df[ptc_df.Last_Exon == True].copy()
    ptc_le['region'] = 'last'
    # For last exon: calculate distance from last exon junction
    # The last exon junction is at the start of the last exon
    # Distance from junction = position within the last exon
    # PTC_CDS_pos is the position in CDS, PTC_CDS_exon_length is the exon length
    # Position within last exon = PTC_CDS_pos - (CDS_mut_length - PTC_CDS_exon_length)
    # Simplified: Position from start of last exon
    ptc_le['PTC_EJC_dist'] = ptc_le['PTC_CDS_exon_length'] - (ptc_le['CDS_mut_length'] - ptc_le['PTC_CDS_pos'])
    
    # Combine both regions
    ptc_combined = pd.concat([ptc_pe, ptc_le], ignore_index=True)
    
    if len(ptc_combined) == 0:
        logger.warning("No genomic PTC data found for penultimate or last exon regions")
        return pd.DataFrame()
    
    # Keep PTC_EJC_dist (distance from exon junction) and NMDeff
    ptc_combined = ptc_combined[['PTC_EJC_dist', 'NMDeff_Norm', 'region']].copy()
    ptc_combined = ptc_combined.dropna()
    
    logger.info(f"Loaded {len(ptc_pe)} penultimate exon and {len(ptc_le)} last exon genomic PTC variants from all genes")
    
    return ptc_combined


def get_exon_boundaries(transcript):
    """
    Calculate exon boundaries in transcript coordinates (1-based).
    Returns the position of the last nucleotide of each exon that contains or follows the CDS.
    
    Returns:
        exon_boundaries: List of exon boundary positions (1-based, last nt of each exon)
    """
    # Get exon boundaries (in nucleotides)
    utr5_length = sum(len(exon) for exon in transcript.utr5s) if transcript.utr5s else 0
    cumulative_length = 0
    exon_boundaries = []
    
    for exon in transcript.exons:
        cumulative_length += len(exon)
        if cumulative_length > utr5_length:
            # Convert to 1-based coordinates to match ptc_positions
            # cumulative_length is 0-based end position, which equals the 1-based position of the last nt
            exon_boundaries.append(cumulative_length)
    return exon_boundaries


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
        # The last exon junction is where the last exon starts
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


def process_data():
    """Process data and generate predictions. Returns DataFrame for plotting."""
    
    logger.info("Starting data processing...")
    logger.info(f"Model: {MODEL_PATH}")
    
    # Train NMDetective models on somatic TCGA training data
    somatic_csv = PROCESSED_DATA_DIR / "PTC" / "somatic_TCGA.csv"
    df_somatic = pd.read_csv(somatic_csv)
    train_mask = ~df_somatic['chr'].isin(VAL_CHRS)
    df_train = df_somatic[train_mask].copy()
    df_train['NMD'] = df_train['NMDeff_Norm']
    
    if INCLUDE_NMDETECTIVE_A:
        logger.info("Training NMDetective-A and NMDetective-B...")
        nmdetective_a = NMDetectiveA(n_estimators=100, random_state=42)
        nmdetective_a.fit(df_train, label_col="NMD")
    else:
        logger.info("Training NMDetective-B (NMDetective-A skipped)...")
        nmdetective_a = None
    
    nmdetective_b = NMDetectiveB()
    nmdetective_b.fit(df_train, label_col="NMD")
    logger.info("Models trained successfully")
    
    # Load genomic PTC data (from ALL genes) - load once, use for both genes
    genomic_ptc = load_genomic_ptc_data()
    
    # Initialize genome
    genome = Genome(GENCODE_VERSION)
    
    all_data = []
    
    for gene_name, config in GENES.items():
        logger.info(f"\nProcessing {gene_name}...")
        
        # Get model predictions for this gene
        results = predict_transcript_ptcs(
            gene_name=gene_name,
            transcript_id=config["transcript_id"],
            transcript_idx=0,
            model_path=str(MODEL_PATH),
            max_positions=None
        )
        
        ptc_positions = results['ptc_positions']
        predictions_ai = results['predictions']
        
        logger.info(f"  Generated {len(predictions_ai)} AI predictions")
        
        # Get transcript for this gene
        transcript = genome.transcripts[config["transcript_id"]]
        
        # Compute features and get NMDetective-B predictions
        features_df = compute_features_for_ptc_positions(transcript, ptc_positions)
        predictions_b = nmdetective_b.predict(features_df)
        
        if INCLUDE_NMDETECTIVE_A:
            predictions_a = nmdetective_a.predict(features_df)
            logger.info(f"  Generated NMDetective-A and B predictions")
        else:
            predictions_a = np.full(len(predictions_b), np.nan)
            logger.info(f"  Generated NMDetective-B predictions (NMDetective-A skipped)")
        
        # Load DMS data
        pe_data = load_dms_data(gene_name)
        logger.info(f"  Loaded {len(pe_data)} PE observations")
        
        # Get exon boundaries for this gene
        exon_boundaries = get_exon_boundaries(transcript)        
        # Junction is the position BETWEEN exons (start of last exon)
        penultimate_junction = exon_boundaries[-2] + 1
        
        # Diagnostic: check coordinate alignment
        utr5_length = sum(len(exon) for exon in transcript.utr5s) if transcript.utr5s else 0
        logger.info(f"  {gene_name} coordinate system check:")
        logger.info(f"    5' UTR length: {utr5_length}")
        logger.info(f"    Penultimate junction (start of last exon): {penultimate_junction}")
        logger.info(f"    Last nt of penultimate exon: {exon_boundaries[-2]}")
        logger.info(f"    First few model prediction positions: {ptc_positions[:5]}")
        logger.info(f"    Exon boundaries: {exon_boundaries}")
        
        # Store predictions
        for pos, pred_ai, pred_a, pred_b in zip(ptc_positions, predictions_ai, predictions_a, predictions_b):
                data_entry = {
                    'gene': gene_name,
                    'transcript_id': config["transcript_id"],
                    'ptc_position': pos,
                    'prediction_ai': pred_ai,
                    'prediction_b': pred_b,
                    'penultimate_junction': penultimate_junction,
                    'data_type': 'prediction'
                }
                if INCLUDE_NMDETECTIVE_A:
                    data_entry['prediction_a'] = pred_a
                all_data.append(data_entry)
        
        # Get DMS positions and values and align to CDS codon frame (match sigmoid script)
        utr5_length = sum(len(exon) for exon in transcript.utr5s) if transcript.utr5s else 0
        cds_start = utr5_length + 1  # 1-indexed position of first CDS nucleotide

        # PTC_pos_rev is in codon positions relative to junction (negative upstream)
        dms_positions_from_junction = pe_data['PTC_pos_rev'].values * 3
        dms_abs_positions_raw = penultimate_junction + dms_positions_from_junction

        # Align to CDS codon frame: convert to CDS-relative positions then to codon indices
        dms_cds_positions = dms_abs_positions_raw - cds_start
        dms_codon_indices = np.floor(dms_cds_positions / 3).astype(int)
        pe_nt_positions = (cds_start + (dms_codon_indices * 3)) + 3
        dms_values = pe_data['NMDeff'].values
        
        # Normalize DMS data to match prediction scale (same method as long_exon_examples.py)
        # Use only predictions from penultimate exon region (before junction) for normalization
        model_positions = np.array(ptc_positions)
        model_values_ai = np.array(predictions_ai)
        
        # Define penultimate exon region: from 200nt before junction to junction
        pe_region_start = penultimate_junction - 200
        pe_region_end = penultimate_junction  # Junction = start of last exon
        
        # Find predictions within the penultimate exon region
        prediction_mask = (model_positions >= pe_region_start) & (model_positions < pe_region_end)
        
        if prediction_mask.any():
            pred_subset = model_values_ai[prediction_mask]
            pred_mean = np.mean(pred_subset)
            pred_std = np.std(pred_subset)
            
            # Normalize DMS data using mean and std
            dms_current_mean = dms_values.mean()
            dms_current_std = dms_values.std()
            
            if dms_current_std > 0:
                dms_values_transformed = (
                    (dms_values - dms_current_mean) / dms_current_std * pred_std + pred_mean
                )
            else:
                dms_values_transformed = dms_values
            
            logger.info(f"  DMS normalization for {gene_name}:")
            logger.info(f"    DMS range: [{dms_values.min():.3f}, {dms_values.max():.3f}]")
            logger.info(f"    DMS mean/std: {dms_current_mean:.3f} / {dms_current_std:.3f}")
            logger.info(f"    Model mean/std (penultimate region): {pred_mean:.3f} / {pred_std:.3f}")
            logger.info(f"    Transformed DMS range: [{dms_values_transformed.min():.3f}, {dms_values_transformed.max():.3f}]")
        else:
            logger.warning(f"  No predictions found in penultimate exon region for {gene_name}")
            dms_values_transformed = dms_values

        
        # Store PE DMS observations (with transformed values)
        for pos, nmd_eff, stop_type in zip(pe_nt_positions, dms_values_transformed, pe_data['stop_type']):
            all_data.append({
                'gene': gene_name,
                'transcript_id': config["transcript_id"],
                'ptc_position': pos,
                'prediction': nmd_eff,
                'penultimate_junction': penultimate_junction,
                'data_type': 'observation',
                'stop_type': stop_type
            })
        # Store genomic PTC data (converted to this gene's coordinate system)
        for ejc_dist, nmd_eff, region in zip(genomic_ptc['PTC_EJC_dist'], genomic_ptc['NMDeff_Norm'], genomic_ptc['region']):
            ptc_abs_pos = penultimate_junction + ejc_dist
            all_data.append({
                'gene': gene_name,
                'transcript_id': config["transcript_id"],
                'ptc_position': ptc_abs_pos,
                'prediction': nmd_eff,
                'penultimate_junction': penultimate_junction,
                'data_type': 'genomic_ptc',
                'region': region
            })
    
    df = pd.DataFrame(all_data)
    logger.info(f"Processed {len(df)} total data points")
    
    return df


def plot_from_table(df):
    """Generate plot from processed data table."""
    
    logger.info("Generating plot from processed data...")
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=False)
    for idx, gene_name in enumerate(["BRCA1", "ATP7A"]):
        gene_df = df[df['gene'] == gene_name]
        
        # Get predictions, observations, and genomic PTC data
        pred_df = gene_df[gene_df['data_type'] == 'prediction'].sort_values('ptc_position')
        obs_df = gene_df[gene_df['data_type'] == 'observation'].sort_values('ptc_position')
        genomic_ptc_df = gene_df[gene_df['data_type'] == 'genomic_ptc'].sort_values('ptc_position')
        penultimate_junction = pred_df['penultimate_junction'].iloc[0]
        
        # Determine plotting range (note: penultimate_junction is start of last exon)
        pe_start = max(penultimate_junction - 200, pred_df['ptc_position'].min())
        pe_end = min(penultimate_junction + MAX_NTS_LAST_EXON, pred_df['ptc_position'].max())
        
        # Filter to plotting range
        pred_df_plot = pred_df[(pred_df['ptc_position'] >= pe_start) & 
                               (pred_df['ptc_position'] <= pe_end)]
        obs_df_plot = obs_df[(obs_df['ptc_position'] >= pe_start) & 
                             (obs_df['ptc_position'] <= pe_end)]
        genomic_ptc_plot = genomic_ptc_df[(genomic_ptc_df['ptc_position'] >= pe_start) & 
                                          (genomic_ptc_df['ptc_position'] <= pe_end)]
        
        # Apply smoothing to predictions (but not NMDetective-B which is a decision tree)
        if SMOOTHING_SIGMA > 0:
            predictions_ai_smooth = gaussian_filter1d(pred_df_plot['prediction_ai'].values, sigma=SMOOTHING_SIGMA)
            if INCLUDE_NMDETECTIVE_A:
                predictions_a_smooth = gaussian_filter1d(pred_df_plot['prediction_a'].values, sigma=SMOOTHING_SIGMA)
        else:
            predictions_ai_smooth = pred_df_plot['prediction_ai'].values
            if INCLUDE_NMDETECTIVE_A:
                predictions_a_smooth = pred_df_plot['prediction_a'].values
        predictions_b_smooth = pred_df_plot['prediction_b'].values

        ax = axes[idx]
        
        # TCGA genomic PTC points removed from plot (kept LOESS curve and CI)
        
        # Fit and plot LOESS with CI for genomic PTCs
        tcga_positions = genomic_ptc_plot['ptc_position'].values
        tcga_values = genomic_ptc_plot['prediction'].values
        
        x_smooth, y_smooth, y_lower, y_upper = compute_loess_with_ci(
            tcga_positions, tcga_values, frac=0.2, n_bootstrap=100
        )
        
        # Plot confidence interval as filled area
        ax.fill_between(x_smooth, y_lower, y_upper, 
                        color=TCGA_CI_COLOR, alpha=0.5, 
                        label='TCGA LOESS 95% CI', zorder=2)
        
        # Plot LOESS curve
        ax.plot(x_smooth, y_smooth, 
                color=TCGA_LOESS_COLOR, linewidth=2.5, linestyle='-',
                alpha=0.8, zorder=2)
        
        # Plot NMDetective-AI predictions as scatter (lower alpha for clarity)
        ax.scatter(pred_df_plot['ptc_position'], predictions_ai_smooth,
               color=MODEL_AI_COLOR, s=PRED_MARKER_SIZE, alpha=0.6,
               marker='x', zorder=5)
        
        if INCLUDE_NMDETECTIVE_A:
            ax.plot(pred_df_plot['ptc_position'], predictions_a_smooth, color=MODEL_A_COLOR, 
                    linewidth=2.5, alpha=0.9, linestyle='--', zorder=4)
        
        ax.plot(pred_df_plot['ptc_position'], predictions_b_smooth, color=MODEL_B_COLOR, 
                linewidth=2.5, alpha=0.9, linestyle=':', zorder=3)
        
        # Plot PE observations (all stop types together)
        ax.scatter(obs_df_plot['ptc_position'], obs_df_plot['prediction'], 
                    c=PE_COLOR, s=50, alpha=0.8, 
                    label='DMS observations', 
                    marker='o', 
                    edgecolors='white', linewidths=0.5, zorder=4)
        
        # Group by position and take median to handle multiple stop types at same position
        obs_grouped = obs_df_plot.groupby('ptc_position')['prediction'].median().reset_index()
        obs_positions_unique = obs_grouped['ptc_position'].values
        obs_values_unique = obs_grouped['prediction'].values
        
        # Sort by position for LOESS
        sort_idx = np.argsort(obs_positions_unique)
        obs_positions_sorted = obs_positions_unique[sort_idx]
        obs_values_sorted = obs_values_unique[sort_idx]
        
        # Apply LOESS smoothing (frac controls smoothing, smaller = less smooth)
        # Reduced window size for DMS LOESS to tighten the fit
        loess_result = lowess(obs_values_sorted, obs_positions_sorted, frac=0.2)
        dms_loess_positions = loess_result[:, 0]
        dms_loess_values = loess_result[:, 1]
        ax.plot(dms_loess_positions, dms_loess_values, 
                color=PE_COLOR, linewidth=3, linestyle='-',
                alpha=0.9, zorder=6)
        
        # Calculate correlations for models (using unique observations)
        pred_ai_at_obs = np.interp(obs_positions_unique, pred_df_plot['ptc_position'].values, 
                                    pred_df_plot['prediction_ai'].values)
        pred_b_at_obs = np.interp(obs_positions_unique, pred_df_plot['ptc_position'].values, 
                                   pred_df_plot['prediction_b'].values)
        
        if INCLUDE_NMDETECTIVE_A:
            pred_a_at_obs = np.interp(obs_positions_unique, pred_df_plot['ptc_position'].values, 
                                       pred_df_plot['prediction_a'].values)
        
        # Choose comparison target: DMS LOESS or raw DMS observations
        if COMPARE_AGAINST_DMS_LOESS:
            # Interpolate DMS LOESS at observation positions
            dms_target = np.interp(obs_positions_unique, dms_loess_positions, dms_loess_values)
        else:
            # Use raw DMS observations (median by position)
            dms_target = obs_values_unique
        
        corr_ai = stats.spearmanr(pred_ai_at_obs, dms_target)[0]
        corr_b = stats.spearmanr(pred_b_at_obs, dms_target)[0]
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2_ai = r2_score(dms_target, pred_ai_at_obs)
        r2_b = r2_score(dms_target, pred_b_at_obs)
        
        if INCLUDE_NMDETECTIVE_A:
            corr_a = stats.spearmanr(pred_a_at_obs, dms_target)[0]
            r2_a = r2_score(dms_target, pred_a_at_obs)
        
        # Calculate correlation and R² for DMS LOESS vs DMS observations (always against raw obs)
        dms_loess_at_obs = np.interp(obs_positions_unique, dms_loess_positions, dms_loess_values)
        corr_dms_loess = stats.spearmanr(dms_loess_at_obs, obs_values_unique)[0]
        r2_dms_loess = r2_score(obs_values_unique, dms_loess_at_obs)
        
        # Calculate correlation between TCGA LOESS and DMS
        # Interpolate TCGA LOESS at DMS positions
        tcga_loess_at_dms = np.interp(obs_positions_unique, x_smooth, y_smooth)
        corr_tcga_dms = stats.spearmanr(tcga_loess_at_dms, dms_target)[0]
        r2_tcga_dms = r2_score(dms_target, tcga_loess_at_dms)

        # Fit 4-parameter logistic (4PL) to NMDetective-AI predictions and evaluate vs DMS
        sigmoid_label = None
        popt_sig = None
        try:
            x_fit_input = pred_df_plot['ptc_position'].values
            y_fit_input = pred_df_plot['prediction_ai'].values
            # Require enough points to fit
            if len(x_fit_input) >= 5 and np.isfinite(y_fit_input).all():
                # Normalize x to [0,1] for numerically stable fitting
                x_range = float(pe_end - pe_start) if (pe_end - pe_start) != 0 else 1.0
                x_fit_norm = (x_fit_input - pe_start) / x_range

                a0 = float(np.min(y_fit_input))
                d0 = float(np.max(y_fit_input))
                c0 = 0.5  # inflection roughly in middle of normalized range
                b0 = 1.0

                # Constrain parameters for stability: c in (0,1], b positive
                a_lower = float(np.min(y_fit_input) - np.abs(np.min(y_fit_input)))
                a_upper = float(np.max(y_fit_input) + np.abs(np.max(y_fit_input)))
                d_lower = a_lower
                d_upper = a_upper
                bounds_lower = [a_lower, d_lower, 1e-6, 0.01]
                bounds_upper = [a_upper, d_upper, 1.0, 20.0]
                popt, _ = curve_fit(
                    four_param_logistic, x_fit_norm, y_fit_input,
                    p0=[a0, d0, c0, b0], bounds=(bounds_lower, bounds_upper), maxfev=20000
                )
                popt_sig = popt

                # Generate sigmoid on normalized x and map back to absolute positions for plotting
                x_sig_norm = np.linspace(0.0, 1.0, 300)
                y_sig = four_param_logistic(x_sig_norm, *popt)
                x_sig = pe_start + x_sig_norm * x_range
                ax.plot(x_sig, y_sig, color=MODEL_AI_COLOR, linewidth=2.5,
                        linestyle='-', alpha=0.95, zorder=4)

                # Interpolate sigmoid at observation positions (use normalized obs positions)
                obs_norm = (obs_positions_unique - pe_start) / x_range
                sig_at_obs = np.interp(obs_norm, x_sig_norm, y_sig)
                corr_sig = stats.spearmanr(sig_at_obs, dms_target)[0]
                r2_sig = r2_score(dms_target, sig_at_obs)
                sigmoid_label = f'NMDetective-4PL (ρ={corr_sig:.3f}, R²={r2_sig:.3f})'
        except Exception as e:
            logger.warning(f"Sigmoid fit failed for {gene_name}: {e}")
            sigmoid_label = None

        # If sigmoid fit succeeded, show formula and parameters in a textbox
        if popt_sig is not None:
            a_val, d_val, c_val, b_val = popt_sig
            # Build LaTeX-rendered equation with parameter values embedded (values reflect normalized x)
            numerator = f"{a_val:.3g} {d_val:.3g}"
            denominator = f"1 + (x / {c_val:.3g})^{{{b_val:.3g}}}"
            param_text = f"$NMDeff \\approx \\frac{{{{{numerator}}}}}{{{{{denominator}}}}} {d_val:.3g}$"
            # Add title above the equation box
            ax.text(
                0.02, 0.17, "4PL Fit Equation", transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                fontweight='bold'
            )
            ax.text(
                0.02, 0.02, param_text, transform=ax.transAxes,
                fontsize=13, verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=MODEL_AI_COLOR)
            )
        
        # Add reference lines
        ax.axvline(x=penultimate_junction, color=JUNCTION_COLOR, linestyle='-', 
                  alpha=0.8, linewidth=2, label='Last exon junction', zorder=1)
        ax.axvline(x=penultimate_junction - 55, color=RULE_55NT_COLOR, linestyle='--',
                  alpha=0.8, linewidth=1.5, label='55nt threshold', zorder=1)
        
        # Add text annotations for reference lines
        ax.text(penultimate_junction + 3, 1, 'Last exon\njunction', 
                fontsize=8, color=JUNCTION_COLOR, ha='left', va='top', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
        ax.text(penultimate_junction - 33, 1, '55nt threshold', 
                fontsize=8, color=RULE_55NT_COLOR, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Styling
        ax.set_xlabel('PTC position in coding region nucleotides', fontsize=12, fontweight='bold')
        ax.set_ylabel('NMD efficiency', fontsize=12, fontweight='bold')
        ax.set_title(f'{PLOT_TITLE}: {gene_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Create custom legend with correlations
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='x', color=MODEL_AI_COLOR, markerfacecolor=MODEL_AI_COLOR, markersize=8, linestyle='None',
                   label=f'NMDetective-AI (ρ={corr_ai:.3f}, R²={r2_ai:.3f})'),
        ]
        
        if INCLUDE_NMDETECTIVE_A:
            legend_elements.append(
                Line2D([0], [0], color=MODEL_A_COLOR, linewidth=2.5, linestyle='--', 
                       label=f'NMDetective-A (ρ={corr_a:.3f}, R²={r2_a:.3f})')
            )
        
        legend_elements.extend([
            Line2D([0], [0], color=MODEL_B_COLOR, linewidth=2.5, linestyle=':', 
                   label=f'NMDetective-B (ρ={corr_b:.3f}, R²={r2_b:.3f})'),
            Line2D([0], [0], color=TCGA_LOESS_COLOR, linewidth=2.5, 
                   label=f'TCGA LOESS fit (ρ={corr_tcga_dms:.3f}, R²={r2_tcga_dms:.3f})'),
            Line2D([0], [0], color=PE_COLOR, linewidth=3, 
                   label=f'DMS LOESS fit (ρ={corr_dms_loess:.3f}, R²={r2_dms_loess:.3f})'),
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=PE_COLOR, markersize=8, label='DMS observations'),
        ])
        # Insert sigmoid legend entry next to AI if fitted
        if sigmoid_label is not None:
            legend_elements.insert(1, Line2D([0], [0], color=MODEL_AI_COLOR, linewidth=2.5,
                                             linestyle='-', label=sigmoid_label))
        
        ax.legend(handles=legend_elements, fontsize=9, loc='upper right', framealpha=0.9)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(pe_start, pe_end)
    
    plt.tight_layout()
    
    return fig


def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = True,
):
    """Generate penultimate exon predictions figure.

    Args:
        figure_label: Panel label when called from the manuscript app.
        figure_number: Figure number when called from the manuscript app.
        regenerate: If False and source data exists, skip processing.
    """
    paths = get_paths(
        script_name=SCRIPT_NAME,
        figure_label=figure_label,
        figure_number=figure_number,
        source_data_ext=".xlsx",
    )
    
    # Check if source data already exists
    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        df_dict = pd.read_excel(paths.source_data, sheet_name=None)
        df = pd.concat(df_dict.values(), ignore_index=True)
    else:
        logger.info("Processing data...")
        df = process_data()
        
        # Save source data as XLSX with separate sheets for each panel (gene)
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(paths.source_data, engine='openpyxl') as writer:
            for gene_name in ["BRCA1", "ATP7A"]:
                gene_df = df[df['gene'] == gene_name].copy()
                panel_name = "Panel_A_BRCA1" if gene_name == "BRCA1" else "Panel_B_ATP7A"
                gene_df.to_excel(writer, sheet_name=panel_name, index=False)
        logger.info(f"Source data saved to {paths.source_data}")
    
    # Generate and save figure
    fig = plot_from_table(df)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)
    
    logger.success("Penultimate exon predictions complete!")


if __name__ == "__main__":
    main()
