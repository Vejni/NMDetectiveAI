"""
TCGA cancer gene NMD triggering/evading ratio analysis.

Similar to gnomad_disease_genes.py but for TCGA somatic variants in cancer genes.
For each gene this script fits a 2x2 table:

    |            | NMD-triggering region | NMD-evading region |
    |------------|----------------------|-------------------|
    | PTCs       | n_trig                | n_evad            |
    | Synonymous | n_syn_trig           | n_syn_evad        |

The odds ratio (OR) = (n_trig / n_syn_trig) / (n_evad / n_syn_evad)
and its 95% CI are estimated via the Wald method on log(OR):
    SE = sqrt(1/a + 1/b + 1/c + 1/d)  (Haldane 0.5 pseudocount on each cell)
A two-sided Z-test yields the p-value.

Requires preprocessed data:
    - data/processed/selection/stopgain_variants_annotated_annotated.tsv (nonsense variants)
    - data/interim/selection/synonymous_variants_annotated.tsv (synonymous variants)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from scipy import stats as scipy_stats
from sklearn.mixture import GaussianMixture

from NMD.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR, TABLES_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "TCGA_cancer_genes"


# Configuration
USE_AI_PREDICTIONS = True  # If True, use AI predictions; if False, use rule-based

# Number of top genes to show for each category (selected by p-value)
TOP_N = 3

# CI level for Wald confidence intervals
CI_LEVEL = 0.95

# FDR threshold for gene selection
FDR_THRESHOLD = 0.05

# Paths
STOPGAIN_FILE = PROCESSED_DATA_DIR / "selection" / "stopgain_variants_annotated_annotated.tsv"
SYNONYMOUS_FILE = INTERIM_DATA_DIR / "selection" / "synonymous_variants_annotated.tsv"
GENE_MAP_FILE = RAW_DATA_DIR / "annotations" / "ensembl_v88_gene_transcript_genesymbol.txt"
CANCER_GENES_FILE = RAW_DATA_DIR / "annotations" / "cancer_genes.csv"
FULL_TABLE = TABLES_DIR / "selection" / "tcga_cancer_gene_nmd_ratio.csv"

# Minimum unique variants to include gene
MIN_PTC = 3      # Minimum unique PTC variants (lower for TCGA since fewer samples)
MIN_SYN = 3      # Minimum unique synonymous variants
MIN_PTC_GMM = 10  # Minimum PTCs per gene required to fit the per-gene GMM threshold
MAX_SYN_VARS = 500 # if a gene has more drop it

EVADING_NOMINAL_COLOR    = '#ffc8c8'   # nominal only – pale pink
EVADING_FDR_COLOR        = '#ff9e9d'   # FDR-passing – stronger pink/red

TRIGGERING_NOMINAL_COLOR  = '#d6f3ff'  # nominal only – pale blue
TRIGGERING_FDR_COLOR      = '#022778'  # FDR-passing – deep navy

NON_SIG_COLOR = '#d4d4d4'

def fit_per_gene_gmm_thresholds(df: pd.DataFrame, gene_col: str = 'gene_name',
                                min_ptcs: int = MIN_PTC_GMM) -> dict:
    """Fit a 2-component GMM to each gene's AI predictions to find a per-gene
    threshold separating NMD-triggering from NMD-evading PTCs.

    For each gene with at least ``min_ptcs`` valid predictions, a 2-component
    Gaussian mixture model is fitted. The threshold is the decision boundary
    where both components have equal posterior probability, found between the
    two component means.

    Args:
        df: DataFrame with nonsense variants including 'NMDetectiveAI_prediction'.
        gene_col: Column name for gene identifiers.
        min_ptcs: Minimum number of PTCs with valid predictions per gene.

    Returns:
        Dictionary mapping gene name to threshold value.
    """
    thresholds = {}
    valid_df = df[df['NMDetectiveAI_prediction'].notna()].copy()
    n_skipped = 0

    for gene, group in valid_df.groupby(gene_col):
        if len(group) < min_ptcs:
            continue

        preds = group['NMDetectiveAI_prediction'].values.reshape(-1, 1)

        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(preds)

            means = gmm.means_.flatten()
            low_idx = int(np.argmin(means))
            high_idx = int(np.argmax(means))

            # Skip genes where the two components are nearly identical
            if abs(means[high_idx] - means[low_idx]) < 0.05:
                n_skipped += 1
                continue

            # Decision boundary: where P(high | x) = 0.5
            x_range = np.linspace(means[low_idx], means[high_idx], 1000).reshape(-1, 1)
            posteriors = gmm.predict_proba(x_range)
            high_probs = posteriors[:, high_idx]
            cross_idx = int(np.argmin(np.abs(high_probs - 0.5)))
            threshold = float(x_range[cross_idx, 0])

            thresholds[gene] = threshold

        except Exception as e:
            logger.debug(f"GMM fitting failed for gene {gene}: {e}")
            continue

    logger.info(f"Fitted per-gene GMM thresholds for {len(thresholds)} genes "
                f"({n_skipped} skipped due to unimodal distributions)")
    return thresholds


def load_cancer_genes() -> tuple[list, list, pd.DataFrame]:
    """Load cancer gene annotations (oncogenes and tumor suppressors only)."""
    logger.info(f"Loading cancer genes from {CANCER_GENES_FILE}")
    
    # Load gene mapping
    gene_map = pd.read_csv(GENE_MAP_FILE, sep="\t", header=0)
    gene_map = gene_map[["gene_id", "gene_name"]].drop_duplicates()
    
    # Load cancer genes
    cgc = pd.read_csv(CANCER_GENES_FILE)
    cgc = cgc[(cgc.cancer_gene == "common mutational cancer driver") | 
              (cgc.cancer_gene == "rare mutational cancer driver")]
    
    oncogenes = cgc[cgc.role == "OG"].Gene.tolist()
    tumor_suppressors = cgc[cgc.role == "TSG"].Gene.tolist()
    
    logger.info(f"Loaded {len(oncogenes)} oncogenes, {len(tumor_suppressors)} tumor suppressors")
    
    return oncogenes, tumor_suppressors, gene_map


def load_stopgain_variants() -> pd.DataFrame:
    """Load stop-gained variants with NMD annotations."""
    logger.info(f"Loading stop-gained variants from {STOPGAIN_FILE}")
    df = pd.read_csv(STOPGAIN_FILE, sep='\t', low_memory=False)
    logger.info(f"Loaded {len(df)} stop-gained variants")
    return df


def load_synonymous_variants() -> pd.DataFrame:
    """Load synonymous variants."""
    logger.info(f"Loading synonymous variants from {SYNONYMOUS_FILE}")
    df = pd.read_csv(SYNONYMOUS_FILE, sep='\t', low_memory=False)
    logger.info(f"Loaded {len(df)} synonymous variants")
    return df


def classify_nmd_regions(df: pd.DataFrame, use_preds: bool = True,
                         gene_thresholds: dict = None,
                         gene_col: str = 'gene_name') -> pd.DataFrame:
    """Classify variants into NMD-triggering and NMD-evading regions.

    Always starts with rule-based classification using NMD_status. When
    ``use_preds=True`` and ``gene_thresholds`` are provided, variants where
    the AI prediction disagrees with the rule-based classification are marked
    as 'unknown' (concordance filter), using a per-gene decision boundary.

    Args:
        df: DataFrame with variant annotations.
        use_preds: If True, apply AI-based concordance filter.
        gene_thresholds: Per-gene thresholds from :func:`fit_per_gene_gmm_thresholds`.
        gene_col: Column name for gene identifiers.

    Returns:
        DataFrame with added 'nmd_region' column.
    """
    df = df.copy()

    # Rule-based classification
    df['nmd_region'] = df['NMD_status'].map({
        'NMD_triggering': 'NMD_triggering',
        'NMD_evading_last_exon': 'NMD_evading',
        'NMD_evading_long_exon': 'NMD_evading',
        'NMD_evading_150nt': 'NMD_evading',
        'NMD_evading_55nt': 'NMD_evading',
        'NMD_unknown': 'unknown'
    })

    if use_preds and gene_thresholds and 'NMDetectiveAI_prediction' in df.columns:
        # Map per-gene thresholds onto each variant
        df['_gene_thr'] = df[gene_col].map(gene_thresholds)
        eligible = df['_gene_thr'].notna() & df['NMDetectiveAI_prediction'].notna()

        # AI says triggering (>= threshold) but rule says evading → unknown
        ai_triggering = df['NMDetectiveAI_prediction'] >= df['_gene_thr']
        rule_evading = df['nmd_region'] == 'NMD_evading'
        df.loc[eligible & ai_triggering & rule_evading, 'nmd_region'] = 'unknown'

        # AI says evading (< threshold) but rule says triggering → unknown
        ai_evading = df['NMDetectiveAI_prediction'] < df['_gene_thr']
        rule_triggering = df['nmd_region'] == 'NMD_triggering'
        df.loc[eligible & ai_evading & rule_triggering, 'nmd_region'] = 'unknown'

        n_unknown = (df.loc[eligible, 'nmd_region'] == 'unknown').sum()
        logger.info(f"AI concordance filter: {n_unknown} variants marked unknown "
                    f"(out of {eligible.sum()} eligible)")
        df.drop(columns=['_gene_thr'], inplace=True)

    return df


def calculate_nmd_ratios(stopgain_df: pd.DataFrame,
                         synonymous_df: pd.DataFrame,
                         cancer_genes: set,
                         use_ai: bool = False,
                         gene_thresholds: dict = None,
                         min_ptc: int = 3,
                         min_syn: int = 3,
                         ci_level: float = 0.95) -> pd.DataFrame:
    """Calculate NMD triggering/evading log OR for each cancer gene using a 2x2 Wald approach.

    Args:
        stopgain_df: DataFrame with stop-gained variants.
        synonymous_df: DataFrame with synonymous variants.
        cancer_genes: Set of cancer gene symbols to analyze.
        use_ai: If True, apply AI-based concordance filter.
        gene_thresholds: Per-gene GMM thresholds from :func:`fit_per_gene_gmm_thresholds`.
            Only used when ``use_ai=True``.
        min_ptc: Minimum unique PTC variants to include gene.
        min_syn: Minimum unique synonymous variants in each region to include gene.
        ci_level: Confidence level for Wald CI.

    Returns:
        DataFrame with gene-level statistics, log OR, Wald CI, and p-value.
    """
    # Classify regions
    logger.info("Classifying NMD regions...")
    stopgain_df = classify_nmd_regions(stopgain_df, use_preds=use_ai,
                                       gene_thresholds=gene_thresholds)
    synonymous_df = classify_nmd_regions(synonymous_df, use_preds=False)
    
    logger.info(f"Stop-gained NMD region distribution:")
    logger.info(f"\n{stopgain_df['nmd_region'].value_counts()}")
    logger.info(f"Synonymous NMD region distribution:")
    logger.info(f"\n{synonymous_df['nmd_region'].value_counts()}")
    
    # Filter for cancer genes only
    stopgain_df = stopgain_df[stopgain_df['gene_name'].isin(cancer_genes)].copy()
    synonymous_df = synonymous_df[synonymous_df['gene_name'].isin(cancer_genes)].copy()
    
    logger.info(f"After filtering for cancer genes: {len(stopgain_df)} stop-gained, {len(synonymous_df)} synonymous variants")
    
    # Count stop-gained by gene and region
    ptc_counts = stopgain_df.groupby(['gene_name', 'nmd_region']).size().unstack(fill_value=0)
    ptc_counts = ptc_counts.reset_index()
    ptc_counts.columns.name = None
    
    # Count synonymous by gene and region
    syn_counts = synonymous_df.groupby(['gene_name', 'nmd_region']).size().unstack(fill_value=0)
    syn_counts = syn_counts.reset_index()
    syn_counts.columns.name = None
    
    # Ensure required columns exist
    for df_tmp in [ptc_counts, syn_counts]:
        for col in ['NMD_triggering', 'NMD_evading']:
            if col not in df_tmp.columns:
                df_tmp[col] = 0
    
    # Merge PTC and synonymous counts
    stats = ptc_counts[['gene_name', 'NMD_triggering', 'NMD_evading']].copy()
    stats.columns = ['gene_symbol', 'n_ptc_triggering', 'n_ptc_evading']
    
    syn_merge = syn_counts[['gene_name', 'NMD_triggering', 'NMD_evading']].copy()
    syn_merge.columns = ['gene_symbol', 'n_syn_triggering', 'n_syn_evading']
    
    stats = stats.merge(syn_merge, on='gene_symbol', how='outer').fillna(0)
    
    # Calculate total variants
    stats['n_variants'] = stats['n_ptc_triggering'] + stats['n_ptc_evading']
    
    logger.info(f"Calculated statistics for {len(stats)} genes")
    
    # Filter by minimum variant counts
    genes_before = len(stats)
    stats = stats[stats['n_variants'] >= min_ptc].copy()
    genes_after = len(stats)
    logger.info(f"Filtered genes with < {min_ptc} unique PTC variants: {genes_before - genes_after} removed, {genes_after} remaining")
    
    genes_before = len(stats)
    stats = stats[(stats['n_syn_triggering'] >= min_syn) &
                  (stats['n_syn_evading'] >= min_syn)].copy()
    genes_after = len(stats)
    logger.info(f"Filtered genes with < {min_syn} synonymous variants in either region: {genes_before - genes_after} removed, {genes_after} remaining")
    
    # Wald method on the 2x2 table per gene
    # Haldane-Anscombe pseudocount of 0.5 to handle zeros
    pseudocount = 0.5
    a = stats['n_ptc_triggering'] + pseudocount
    b = stats['n_ptc_evading'] + pseudocount
    c = stats['n_syn_triggering'] + pseudocount
    d = stats['n_syn_evading'] + pseudocount
    
    stats['log_odds'] = np.log(a / c) - np.log(b / d)
    stats['se_log_or'] = np.sqrt(1/a + 1/b + 1/c + 1/d)
    
    z_crit = scipy_stats.norm.ppf(1 - (1 - ci_level) / 2)
    stats['ci_lower'] = stats['log_odds'] - z_crit * stats['se_log_or']
    stats['ci_upper'] = stats['log_odds'] + z_crit * stats['se_log_or']
    
    stats['z_score'] = stats['log_odds'] / stats['se_log_or']
    stats['p_value'] = 2 * scipy_stats.norm.sf(np.abs(stats['z_score']))
    
    # Benjamini-Hochberg FDR correction
    p_vals = stats['p_value'].values
    n = len(p_vals)
    if n > 0:
        order = np.argsort(p_vals)
        ranks = np.argsort(np.argsort(p_vals)) + 1  # 1-based ranks
        fdr_sorted = p_vals * n / ranks
        # Enforce monotonicity
        fdr_ordered = fdr_sorted[order]
        for i in range(n - 2, -1, -1):
            fdr_ordered[i] = min(fdr_ordered[i], fdr_ordered[i + 1])
        fdr_corrected = np.empty(n)
        fdr_corrected[order] = fdr_ordered
        stats['p_value_fdr'] = np.minimum(fdr_corrected, 1.0)
    else:
        stats['p_value_fdr'] = np.nan
    
    logger.info(f"Calculated Wald log OR with {int(ci_level*100)}% CI and Z-test p-values")
    logger.info(f"  Median log OR: {stats['log_odds'].median():.4f}")
    logger.info(f"  Genes with p < 0.05: {(stats['p_value'] < 0.05).sum()}")
    logger.info(f"  Genes with FDR < {FDR_THRESHOLD}: {(stats['p_value_fdr'] < FDR_THRESHOLD).sum()}")
    
    return stats


def select_top_genes(stats: pd.DataFrame, top_n: int | None, fdr_threshold: float = 0.05) -> list:
    """
    Select genes to plot: from those passing FDR, take top N most significant
    evading-enriched and top N most significant triggering-enriched.
    
    Args:
        stats: DataFrame with gene statistics
        top_n: Number of top genes per direction
        fdr_threshold: FDR threshold for inclusion
        
    Returns:
        List of gene symbols to plot
    """
    sig = stats[stats['p_value_fdr'] < fdr_threshold]
    logger.info(f"Genes passing FDR < {fdr_threshold}: {len(sig)} "
                f"({(sig['log_odds'] > 0).sum()} triggering-enriched, "
                f"{(sig['log_odds'] < 0).sum()} evading-enriched)")

    # If top_n is None, return all genes that pass FDR (preserve ordering later in plotting)
    if top_n is None:
        selected_genes = sig['gene_symbol'].tolist()
        logger.info(f"Selecting all {len(selected_genes)} FDR-passing genes for plotting")
        return selected_genes

    # Otherwise keep previous behaviour: top N per direction by p-value
    evading_stats = sig[sig['log_odds'] < 0].sort_values('p_value')
    triggering_stats = sig[sig['log_odds'] > 0].sort_values('p_value')

    top_evading = evading_stats.head(top_n)['gene_symbol'].tolist()
    top_triggering = triggering_stats.head(top_n)['gene_symbol'].tolist()

    selected_genes = list(set(top_evading + top_triggering))

    logger.info(f"Selected {len(selected_genes)} genes to plot:")
    logger.info(f"  Top {top_n} evading-enriched (by p-value): {top_evading}")
    logger.info(f"  Top {top_n} triggering-enriched (by p-value): {top_triggering}")

    return selected_genes


def plot_barplot(stats: pd.DataFrame, genes_to_plot: list):
    """Horizontal forest plot of per-gene log OR with 95 % CI (TSGs only).

    All tested genes appear as background dots; FDR-significant genes are
    fully opaque and labelled. Bars are coloured by direction of effect.

    Args:
        stats: DataFrame with gene statistics including log_odds, ci_lower,
            ci_upper, p_value, and p_value_fdr columns.
        genes_to_plot: List of gene symbols to include.
    """
    plot_data = stats[stats['gene_symbol'].isin(genes_to_plot)].copy()
    if len(plot_data) == 0:
        logger.error(f"None of the selected genes found in data: {genes_to_plot}")
        return

    # Exclude MUC4: extreme synonymous variant count makes CI unusable
    plot_data = plot_data[plot_data['gene_symbol'] != 'MUC4'].copy()

    # Sort by log OR (lowest → highest)
    plot_data = plot_data.sort_values('log_odds', ascending=True).reset_index(drop=True)
    plot_data['plot_index'] = plot_data.index
    logger.info(f"Plotting {len(plot_data)} genes")

    plot_data['fdr_sig'] = plot_data['p_value_fdr'] < FDR_THRESHOLD

    # Color by direction of effect (NMD-evading enriched vs NMD-triggering enriched)
    def _point_color(row):
        if not row['fdr_sig']:
            return NON_SIG_COLOR
        return TRIGGERING_FDR_COLOR if row['log_odds'] > 0 else EVADING_FDR_COLOR

    fig_width = max(10, len(plot_data) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    # ---- Background: non-significant genes (dimmed) ----
    ns = plot_data[~plot_data['fdr_sig']]
    for _, row in ns.iterrows():
        ax.errorbar(
            row['plot_index'], row['log_odds'],
            yerr=[[row['log_odds'] - row['ci_lower']],
                  [row['ci_upper'] - row['log_odds']]],
            fmt='o', markersize=3, color=NON_SIG_COLOR, alpha=0.50,
            elinewidth=0.7, capsize=0, zorder=1,
        )

    # ---- Foreground: FDR-significant genes ----
    sig = plot_data[plot_data['fdr_sig']]
    for _, row in sig.iterrows():
        c = _point_color(row)
        ax.errorbar(
            row['plot_index'], row['log_odds'],
            yerr=[[row['log_odds'] - row['ci_lower']],
                  [row['ci_upper'] - row['log_odds']]],
            fmt='o', markersize=6, color=c, alpha=0.9,
            elinewidth=1.5, capsize=3, zorder=3,
        )

    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=0)

    # ---- X-axis: one tick per gene, * for FDR-significant ----
    xtick_labels = [
        f"{row['gene_symbol']}*" if row['fdr_sig'] else row['gene_symbol']
        for _, row in plot_data.iterrows()
    ]
    ax.set_xticks(plot_data['plot_index'])
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=12, ha='center')
    # Bold the FDR-significant tick labels
    for tick, (_, row) in zip(ax.get_xticklabels(), plot_data.iterrows()):
        if row['fdr_sig']:
            tick.set_fontweight('bold')
            tick.set_fontstyle('italic')

    ax.set_xlabel('')
    ax.set_ylabel('log OR NMD \n (triggering / evading)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('NMD triggering vs evading enrichment in TCGA tumor suppressor genes', fontsize=16, fontweight='bold')

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color=EVADING_FDR_COLOR, linestyle='none',
               markersize=8, label='NMD-evading enriched (FDR sig.)'),
        Line2D([0], [0], marker='o', color=TRIGGERING_FDR_COLOR, linestyle='none',
               markersize=8, label='NMD-triggering enriched (FDR sig.)'),
        Line2D([0], [0], marker='o', color=NON_SIG_COLOR, linestyle='none',
               markersize=8, label='Not significant'),
    ]
    ax.legend(handles=legend_elems, fontsize=13, frameon=False, loc='upper left')

    plt.tight_layout()
    return fig, plot_data


def main(figure_label='h', figure_number='7', regenerate=True):
    """Main analysis function."""
    paths = get_paths(SCRIPT_NAME, figure_label=figure_label, figure_number=figure_number)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    if paths.source_data:
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting TCGA cancer gene NMD ratio analysis")
    logger.info(f"Configuration:")
    logger.info(f"  USE_AI_PREDICTIONS: {USE_AI_PREDICTIONS}")
    if USE_AI_PREDICTIONS:
        logger.info(f"  MIN_PTC_GMM (for per-gene threshold fitting): {MIN_PTC_GMM}")
    logger.info(f"  TOP_N: {TOP_N}")
    logger.info(f"  MIN_PTC: {MIN_PTC}")
    logger.info(f"  MIN_SYN: {MIN_SYN}")
    logger.info(f"  FDR_THRESHOLD: {FDR_THRESHOLD}")
    
    # Check if full table exists
    if FULL_TABLE.exists() and not regenerate:
        logger.info(f"Loading existing full table from {FULL_TABLE}")
        stats = pd.read_csv(FULL_TABLE)
    else:
        # Load data
        oncogenes, tumor_suppressors, gene_map = load_cancer_genes()
        cancer_genes = set(tumor_suppressors)  # TSGs only
        
        stopgain_df = load_stopgain_variants()
        synonymous_df = load_synonymous_variants()
        
        # Map gene IDs to gene names
        stopgain_df = stopgain_df.merge(gene_map, left_on='Gene', right_on='gene_id', how='left')
        synonymous_df = synonymous_df.merge(gene_map, left_on='Gene', right_on='gene_id', how='left')
        
        # Fit per-gene GMM thresholds when using AI predictions
        gene_thresholds = None
        if USE_AI_PREDICTIONS:
            gene_thresholds = fit_per_gene_gmm_thresholds(
                stopgain_df, gene_col='gene_name', min_ptcs=MIN_PTC_GMM
            )

        # Calculate ratios
        stats = calculate_nmd_ratios(stopgain_df, synonymous_df,
                                     cancer_genes=cancer_genes,
                                     use_ai=USE_AI_PREDICTIONS,
                                     gene_thresholds=gene_thresholds,
                                     min_ptc=MIN_PTC,
                                     min_syn=MIN_SYN,
                                     ci_level=CI_LEVEL)
        
        # Save full table
        FULL_TABLE.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(FULL_TABLE, index=False)
        logger.info(f"Saved full statistics to {FULL_TABLE}")
    
    # Save source data
    if paths.source_data:
        stats.to_csv(paths.source_data, index=False)
        logger.info(f"Saved source data to {paths.source_data}")
    
    # Select genes to plot — all nominally significant genes (p < 0.05), excluding MUC4
    genes_to_plot = stats[stats['p_value'] < 0.05]['gene_symbol'].tolist()

    # Create plot
    result = plot_barplot(stats, genes_to_plot)
    if result is not None:
        fig, plot_data = result
        fig.savefig(paths.figure_png, dpi=300, bbox_inches='tight')
        fig.savefig(paths.figure_pdf, bbox_inches='tight')
        logger.info(f"Saved figure to {paths.figure_png}")
        logger.info(f"Saved PDF to {paths.figure_pdf}")
        plt.close(fig)
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
