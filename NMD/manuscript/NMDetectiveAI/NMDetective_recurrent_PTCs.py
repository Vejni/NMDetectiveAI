import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from loguru import logger
from torch.utils.data import DataLoader

from NMD.config import (
    RAW_DATA_DIR, 
    MODELS_DIR, 
    INTERIM_DATA_DIR, 
    PROCESSED_DATA_DIR,
    VAL_CHRS,
    CONTRASTING_3_COLOURS
)
from NMD.manuscript.output import get_paths
from NMD.modeling.TrainerConfig import TrainerConfig
from NMD.modeling.SequenceDataset import SequenceDataset
from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
from NMD.modeling.models.NMDetectiveA import NMDetectiveA
from NMD.modeling.models.NMDetectiveB import NMDetectiveB
from NMD.utils import load_model, collate_fn


# ============================================================================
# CONFIGURATION - Define all paths and parameters here
# ============================================================================

# Genes to annotate with error bars - format: {gene_name: index}
# Index refers to which occurrence of the gene to annotate (0-indexed)
GENES_TO_ANNOTATE = {
    'PTEN': 1,
    'CASP8': 1,
    'ARHGAP35': 0,
    'CTCF': 0,
    'PBRM1': 1,
}

# Paths
SCRIPT_NAME = "NMDetective_recurrent_PTCs"
CANCER_GENES_FILE = RAW_DATA_DIR / "annotations/cancer_genes.csv"
GENE_MAPPING_FILE = RAW_DATA_DIR / "annotations/ensembl_v88_gene_transcript_genesymbol.txt"
DATA_FILE = INTERIM_DATA_DIR / "PTC" / "somatic_TCGA.csv"
SEQUENCES_FILE = PROCESSED_DATA_DIR / "PTC" / "somatic_TCGA.pkl"
MODEL_PATH = MODELS_DIR / "NMDetectiveAI.pt"

# Filtering parameters
MIN_PTC_COUNT = 3  # Minimum number of observations for a PTC to be included
CGC = True  # Filter to cancer genes only (TSGs + OCGs)
VAL_ONLY = False  # If True, only use validation chromosomes for analysis
SNVs_ONLY = True
PLOT_TITLE = f"Recurrent PTCs (n≥{MIN_PTC_COUNT}) in cancer genes"
PLOT_TITLE_FONTSIZE = 18

# Analysis parameters
USE_NORMALIZED = True  # If True, use NMDeff_Norm; if False, use ASE_NMD_efficiency_TPM
INCLUDE_NMDETECTIVE_A = False  # If True, include NMDetective-A predictions in the plot

# Plot aesthetics (using CONTRASTING_3_COLOURS from config)
COLORS = {
    'observations': CONTRASTING_3_COLOURS[1],  # '#2d8b4d' - green
    'nmdetective_b': CONTRASTING_3_COLOURS[0],   # '#ff9e9d' - pink/red  
    'nmdetective_ai': CONTRASTING_3_COLOURS[2], # '#022778' - dark blue
    'nmdetective_a': '#9b59b6',  # Purple for NMDetective-A
    'error_bars': '#555555'  # Dark gray
}

ALPHA = {
    'observations': 0.7,
    'nmdetective_b': 0.6,
    'nmdetective_a': 0.6,
    'nmdetective_ai': 0.6
}

MARKER_SIZE = {
    'observations': 80,
    'nmdetective_b': 60,
    'nmdetective_a': 60,
    'nmdetective_ai': 60
}

FIGURE_SIZE = (6, 10)
DPI = 300

# Standard genetic code codon table (DNA codons to single-letter amino acid codes)
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def make_annotation(row):
    ptc_pos = int(row['PTC_CDS_pos'])
    gene_name = row['gene_name']
    ref_aa = str(row["fasta_sequence_wt"][(ptc_pos-1):(ptc_pos+2)]).upper()
    ref_aa = CODON_TABLE[ref_aa]
    aa_change = f"{ref_aa}{(ptc_pos-1) // 3}X"
    label = f"{gene_name} ({aa_change})"
    return label

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
    Generate NMDetective-AI predictions using the finetuned Orthrus model.
    
    Args:
        df: DataFrame with metadata
        sequences: List of sequence tensors
        model: Trained NMDetectiveAI model
        device: torch device
        
    Returns:
        Array of predictions
    """
    # Create dataset and dataloader
    eval_dataset = SequenceDataset(df, sequences, label_col="NMDeff_Norm")
    eval_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=collate_fn)
    
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for batch_sequences, batch_lengths, _ in eval_loader:
            batch_sequences, batch_lengths = [
                x.to(device) for x in (batch_sequences, batch_lengths)
            ]
            batch_preds = model(batch_sequences, batch_lengths).squeeze()
            
            if batch_preds.dim() == 0:
                predictions.append(float(batch_preds.cpu().numpy()))
            else:
                predictions.extend(batch_preds.cpu().numpy())
    
    return np.array(predictions)


# ============================================================================
# DATA PROCESSING
# ============================================================================

def process_data():
    """Process data and generate predictions. Returns DataFrame for plotting."""
    
    logger.info("Starting data processing...")
    
    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    logger.info(f"Loading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    logger.info(f"Loaded {len(df)} PTCs from TCGA dataset")
    
    logger.info(f"Loading gene name mapping from {GENE_MAPPING_FILE}")
    gene_mapping = pd.read_csv(GENE_MAPPING_FILE, sep="\t")
    # Create a dictionary mapping gene_id (without version) to gene_name
    gene_mapping['gene_id_base'] = gene_mapping['gene_id'].str.split('.').str[0]
    gene_id_to_name = gene_mapping[['gene_id_base', 'gene_name']].drop_duplicates().set_index('gene_id_base')['gene_name'].to_dict()
    logger.info(f"Loaded {len(gene_id_to_name)} gene ID to name mappings")
    
    logger.info(f"Loading cancer genes from {CANCER_GENES_FILE}")
    cancer_genes = pd.read_csv(CANCER_GENES_FILE)
    logger.info(f"Loaded {len(cancer_genes)} cancer genes")
    
    logger.info(f"Loading sequences from {SEQUENCES_FILE}")
    import pickle
    with open(SEQUENCES_FILE, "rb") as f:
        sequences = pickle.load(f)
    logger.info(f"Loaded {len(sequences)} sequences")
    
    # Add gene names to the dataframe
    df['gene_id_base'] = df['gene_id'].str.split('.').str[0]
    df['gene_name'] = df['gene_id_base'].map(gene_id_to_name)
    logger.info(f"Mapped {df['gene_name'].notna().sum()} / {len(df)} gene IDs to gene names")
    
    # -------------------------------------------------------------------------
    # 2. Filter data
    # -------------------------------------------------------------------------
    logger.info("Filtering data...")
    
    # Filter for recurrent PTCs (count >= MIN_PTC_COUNT)
    df_filtered = df[df['count'] >= MIN_PTC_COUNT].copy()
    logger.info(f"After filtering for count >= {MIN_PTC_COUNT}: {len(df_filtered)} PTCs")
    
    # Filter for validation chromosomes if requested
    if VAL_ONLY:
        df_filtered = df_filtered[df_filtered['chr'].isin(VAL_CHRS)]
        logger.info(f"After filtering for validation chromosomes ({', '.join(VAL_CHRS)}): {len(df_filtered)} PTCs")
    
    # Filter for tumor suppressor genes if requested
    if CGC:
        tsg_genes = cancer_genes[cancer_genes['role'] == 'TSG']['Gene'].values
        ocg_genes = cancer_genes[cancer_genes['role'] == 'OCG']['Gene'].values
        tsg_genes = set(tsg_genes).union(set(ocg_genes))  # Include OCGs as well
        logger.info(f"Filtering for tumor suppressor genes (TSGs + OCGs), total {len(tsg_genes)} genes")
        # Filter based on gene names
        df_filtered = df_filtered[df_filtered['gene_name'].isin(tsg_genes)]
        logger.info(f"After filtering for tumor suppressor genes: {len(df_filtered)} PTCs")

    if SNVs_ONLY:
        df_filtered = df_filtered[df_filtered['stopgain'] == 'nonsense']
        logger.info(f"After filtering for SNVs only: {len(df_filtered)} PTCs")
    
    # Ensure sequences align with filtered data
    df_indices = df_filtered.index.tolist()
    sequences_filtered = [sequences[i] for i in df_indices]
    
    logger.info(f"Final dataset: {len(df_filtered)} recurrent PTCs")
    
    # -------------------------------------------------------------------------
    # 2b. Determine which column to use for observations and predictions
    # -------------------------------------------------------------------------
    if USE_NORMALIZED:
        obs_col = 'NMDeff_Norm'
        y_label = 'Normalized NMD Efficiency'
        logger.info("Using normalized values (NMDeff_Norm)")
    else:
        obs_col = 'ASE_NMD_efficiency_TPM'
        y_label = 'NMD Efficiency (ASE TPM)'
        logger.info("Using unnormalized values (ASE_NMD_efficiency_TPM)")

    # -------------------------------------------------------------------------
    # 3. Generate NMDetective-B predictions
    # -------------------------------------------------------------------------
    logger.info("Training NMDetective-B on training data...")
    # Use same train/val split as in the notebook: chr1 and chr20 for validation
    train = df[~df['chr'].isin(VAL_CHRS)].copy()
    train['NMD'] = train[obs_col]
    
    nmdetective_b = NMDetectiveB()
    nmdetective_b.fit(train, label_col="NMD")
    logger.info(f"NMDetective-B trained successfully using {obs_col}")
    
    logger.info("Generating NMDetective-B predictions...")
    df_filtered['pred_nmdetective_b'] = nmdetective_b.predict(df_filtered)

    # -------------------------------------------------------------------------
    # 3b. Generate NMDetective-A predictions
    # -------------------------------------------------------------------------
    if INCLUDE_NMDETECTIVE_A:
        logger.info("Training NMDetective-A on training data...")
        train_for_a = train.copy()
        train_for_a['NMD'] = train_for_a[obs_col]
        
        nmdetective_a = NMDetectiveA(n_estimators=100, random_state=42)
        nmdetective_a.fit(train_for_a, label_col="NMD")
        logger.info("NMDetective-A trained successfully")
        
        logger.info("Generating NMDetective-A predictions...")
        df_filtered['pred_nmdetective_a'] = nmdetective_a.predict(df_filtered)
    else:
        logger.info("Skipping NMDetective-A (INCLUDE_NMDETECTIVE_A=False)")
        df_filtered['pred_nmdetective_a'] = np.nan
    
    # -------------------------------------------------------------------------
    # 4. Generate NMDetective-AI predictions
    # -------------------------------------------------------------------------
    logger.info("Loading NMDetective-AI model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    config = TrainerConfig()
    model = setup_model(config, device)
    load_model(model, MODEL_PATH, device=device)
    logger.info(f"Model loaded from {MODEL_PATH}")
    
    logger.info("Generating NMDetective-AI predictions...")
    df_filtered['pred_nmdetective_ai'] = predict_nmdetective_ai(
        df_filtered, sequences_filtered, model, device
    )
    
    # Store normalization parameters for plotting (if using normalized values)
    if USE_NORMALIZED:
        triggering_mask = (df["NMD_Triggering"] == 1)
        last_exon_mask = (df["Last_Exon"] == 1)
        triggering_mean = df.loc[triggering_mask, "ASE_NMD_efficiency_TPM"].mean()
        last_exon_mean = df.loc[last_exon_mask, "ASE_NMD_efficiency_TPM"].mean()
        ase_center = (triggering_mean + last_exon_mean) / 2
        ase_scale = np.abs(triggering_mean - last_exon_mean)
        df_filtered['ase_center'] = ase_center
        df_filtered['ase_scale'] = ase_scale
    
    # Add obs_col as metadata column
    df_filtered['obs_col'] = obs_col
    
    logger.info(f"Data processing complete. Generated predictions for {len(df_filtered)} PTCs")
    
    return df_filtered


# ============================================================================
# PLOTTING
# ============================================================================

def plot_from_table(df_plot):
    """Generate plot from processed data table."""
    
    logger.info("Creating scatter plot from processed data...")
    
    # Determine obs_col from dataframe (for backwards compatibility with saved data)
    if 'NMDeff_Norm' in df_plot.columns:
        obs_col = 'NMDeff_Norm'
    else:
        obs_col = 'ASE_NMD_efficiency_TPM'
    
    # Determine y-axis label based on which column is used
    if obs_col == 'NMDeff_Norm':
        y_label = 'Observed normalized NMD efficiency'
    else:
        y_label = 'Observed NMD efficiency'
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    
    # Get observations and predictions
    observations = df_plot[obs_col].values
    pred_nmdetective_b = df_plot['pred_nmdetective_b'].values
    pred_nmdetective_a = df_plot['pred_nmdetective_a'].values
    pred_nmdetective_ai = df_plot['pred_nmdetective_ai'].values
    
    # Identify which points to annotate with error bars
    gene_names = df_plot['gene_name'].values
    annotation_mask = np.zeros(len(df_plot), dtype=bool)
    annotation_indices = []
    
    for gene_name, target_idx in GENES_TO_ANNOTATE.items():
        # Find all occurrences of this gene
        gene_mask = gene_names == gene_name
        gene_indices = np.where(gene_mask)[0]
        
        if len(gene_indices) > target_idx:
            annotation_idx = gene_indices[target_idx]
            annotation_mask[annotation_idx] = True
            annotation_indices.append(annotation_idx)
    
    # Prepare horizontal error bars (IQR of observations) for annotated genes only.
    # CI_lower_scaled / CI_upper_scaled are Q25/Q75 in the same absolute normalized
    # space as NMDeff_Norm.  Drawn as xerr centred on the AI prediction so the bars
    # show whether the observation (y) falls within pred ± IQR half-width.
    xerr_ai = None
    ci_lower = ci_upper = None
    if 'ASE_NMD_efficiency_CI_lower_scaled' in df_plot.columns and 'ASE_NMD_efficiency_CI_upper_scaled' in df_plot.columns:
        ci_lower = df_plot['ASE_NMD_efficiency_CI_lower_scaled'].values
        ci_upper = df_plot['ASE_NMD_efficiency_CI_upper_scaled'].values
    elif 'ASE_NMD_efficiency_CI_lower' in df_plot.columns and 'ASE_NMD_efficiency_CI_upper' in df_plot.columns:
        # Fallback: raw CI values present (normalize on the fly)
        if USE_NORMALIZED and 'ase_center' in df_plot.columns:
            ase_center = df_plot['ase_center'].iloc[0]
            ase_scale = df_plot['ase_scale'].iloc[0]
            ci_lower = ((df_plot['ASE_NMD_efficiency_CI_lower'] - ase_center) / ase_scale).clip(-2, 2).values
            ci_upper = ((df_plot['ASE_NMD_efficiency_CI_upper'] - ase_center) / ase_scale).clip(-2, 2).values
        else:
            ci_lower = df_plot['ASE_NMD_efficiency_CI_lower'].values
            ci_upper = df_plot['ASE_NMD_efficiency_CI_upper'].values

    if ci_lower is not None:
        xerr_lower = np.abs(observations - ci_lower)
        xerr_upper = np.abs(ci_upper - observations)
        xerr_ai = [xerr_lower, xerr_upper]

        # Print how many AI predictions fall within the observation CI (Q25–Q75)
        in_ci = (pred_nmdetective_ai >= ci_lower) & (pred_nmdetective_ai <= ci_upper)
        logger.info(
            f"AI predictions within observation CI (Q25–Q75): "
            f"{in_ci.sum()} / {len(in_ci)} "
            f"({100 * in_ci.mean():.1f}%)"
        )

    # Plot horizontal error bars (IQR) on AI-prediction axis for annotated genes only
    if xerr_ai is not None and annotation_mask.any():
        pred_ai_annotated = pred_nmdetective_ai[annotation_mask]
        observations_annotated = observations[annotation_mask]
        xerr_lower_filtered = xerr_ai[0][annotation_mask]
        xerr_upper_filtered = xerr_ai[1][annotation_mask]
        ax.errorbar(
            pred_ai_annotated,
            observations_annotated,
            xerr=[xerr_lower_filtered, xerr_upper_filtered],
            fmt='none',
            ecolor=COLORS['error_bars'],
            alpha=0.5,
            capsize=2,
            linewidth=0.8,
            zorder=4
        )
    
    # Calculate metrics for legend (need to compute before plotting)
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    
    rho_b, _ = spearmanr(observations, pred_nmdetective_b)
    r2_b = r2_score(observations, pred_nmdetective_b)
    
    rho_ai, _ = spearmanr(observations, pred_nmdetective_ai)
    r2_ai = r2_score(observations, pred_nmdetective_ai)
    
    # Plot NMDetective-AI predictions vs observations
    # Annotated points: filled markers; non-annotated: hollow markers
    facecolors_ai = np.where(annotation_mask, COLORS['nmdetective_ai'], 'white')
    ax.scatter(
        pred_nmdetective_ai,
        observations,
        facecolors=facecolors_ai,
        edgecolors=COLORS['nmdetective_ai'],
        alpha=ALPHA['nmdetective_ai'],
        s=MARKER_SIZE['nmdetective_ai'],
        marker='^',
        label=f'NMDetective-AI (ρ={rho_ai:.3f}, R²={r2_ai:.3f})',
        zorder=3
    )

    # Plot NMDetective-B predictions vs observations (hollow unless same y as annotated AI)
    annotated_observations = observations[annotation_mask] if annotation_mask.any() else np.array([])
    facecolors_b = np.where(np.isin(observations, annotated_observations), COLORS['nmdetective_b'], 'white')
    ax.scatter(
        pred_nmdetective_b,
        observations,
        facecolors=facecolors_b,
        edgecolors=COLORS['nmdetective_b'],
        alpha=ALPHA['nmdetective_b'],
        s=MARKER_SIZE['nmdetective_b'],
        marker='s',
        label=f'NMDetective-B (refit to training data; ρ={rho_b:.3f}, R²~0.0)',
        zorder=2
    )
    
    # Plot NMDetective-A predictions vs observations (all hollow)
    if INCLUDE_NMDETECTIVE_A:
        rho_a, _ = spearmanr(observations, pred_nmdetective_a)
        r2_a = r2_score(observations, pred_nmdetective_a)
        
        facecolors_a = np.where(np.isin(observations, annotated_observations), COLORS['nmdetective_a'], 'white')
        ax.scatter(
            pred_nmdetective_a,
            observations,
            facecolors=facecolors_a,
            edgecolors=COLORS['nmdetective_a'],
            alpha=ALPHA['nmdetective_a'],
            s=MARKER_SIZE['nmdetective_a'],
            marker='D',
            label=f'NMDetective-A (ρ={rho_a:.3f}, R²={r2_a:.3f})',
            zorder=2
        )
    
    # Add diagonal line (perfect prediction)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=1, linewidth=1, label='Perfect Prediction')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Add gene name annotations for specified genes
    for idx in annotation_indices:
        gene_name = gene_names[idx]
        obs_val = observations[idx]
        pred_val = pred_nmdetective_ai[idx]
        # Prefer PTC_stop_codon_type if available (should be single-letter AA code)
        label = make_annotation(df_plot.iloc[idx])
        # Annotate at NMDetectiveAI prediction point
        ax.annotate(
            label,
            xy=(pred_val, obs_val),
            xytext=(-80, 40),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', lw=1)
        )
    
    # Formatting
    ax.set_xlabel('Predicted normalized NMD efficiency', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{y_label}', fontsize=12, fontweight='bold')
    
    ax.legend(
        loc='best',
        frameon=True,
        framealpha=0.9,
        fontsize=10
    )
    
    ax.grid(True, alpha=0.3, linestyle=':', zorder=0)
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
    """Generate recurrent PTC scatter plot.

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
    logger.info("Starting recurrent PTC scatter plot generation")
    
    # Check if source data already exists
    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        df_plot = pd.read_csv(paths.source_data)
    else:
        logger.info("Processing data...")
        df_plot = process_data()
        # Compose annotation for all points (e.g., W26X format)
        df_plot['annotation'] = df_plot.apply(make_annotation, axis=1)

        # Compute scaled CI columns if using normalized values
        if 'ASE_NMD_efficiency_CI_lower' in df_plot.columns and 'ASE_NMD_efficiency_CI_upper' in df_plot.columns and 'ase_center' in df_plot.columns and 'ase_scale' in df_plot.columns:
            df_plot['ASE_NMD_efficiency_CI_lower_scaled'] = ((df_plot['ASE_NMD_efficiency_CI_lower'] - df_plot['ase_center']) / df_plot['ase_scale']).clip(-2, 2)
            df_plot['ASE_NMD_efficiency_CI_upper_scaled'] = ((df_plot['ASE_NMD_efficiency_CI_upper'] - df_plot['ase_center']) / df_plot['ase_scale']).clip(-2, 2)

        # Only save columns used in the plot
        plot_columns = [
            'gene_name', 'annotation',
            'NMDeff_Norm', 'pred_nmdetective_ai', 'pred_nmdetective_b',
            'ASE_NMD_efficiency_CI_lower_scaled', 'ASE_NMD_efficiency_CI_upper_scaled'
        ]
        plot_columns = [col for col in plot_columns if col in df_plot.columns]
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        df_plot[plot_columns].to_csv(paths.source_data, index=False)
        logger.info(f"Source data saved to {paths.source_data}")
    
    # Generate and save figure
    fig = plot_from_table(df_plot)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)
    
    logger.success("Recurrent PTC scatter plot complete!")


if __name__ == "__main__":
    main()