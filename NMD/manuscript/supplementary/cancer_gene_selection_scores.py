"""
Gene-level NMD selection scores.

This script identifies genes with the strongest selection pressure for or against
NMD-triggering mutations by calculating Cohen's h effect sizes comparing the ratio
of nonsense to synonymous mutations in NMD-sensitive vs NMD-insensitive regions.

Displays top genes showing selection for NMD evasion (oncogenes) and NMD triggering
(tumor suppressors) in a single lollipop plot.

Requires preprocessed data:
    - data/processed/selection/nmd_variants_annotated_annotated.tsv (nonsense variants)
    - data/interim/selection/synonymous_variants_annotated.tsv (synonymous variants)
    - data/raw/ensembl_v88_gene_transcript_genesymbol.txt (gene mapping)
    - data/raw/cancer_genes.csv (cancer gene annotations)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from loguru import logger

from NMD.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR, FIGURES_DIR, TABLES_DIR


# Configuration
USE_PREDICTIONS = False  # If True, use AI predictions; if False, use rule-based NMD status
PREDICTION_THRESHOLD_TRIGGERING = 0.43  # AI predictions >= threshold = NMD-triggering
PREDICTION_THRESHOLD_EVADING = -0.17  # AI predictions <= threshold = NMD-evading

MIN_NONSENSE = 3  # Minimum nonsense mutations per gene
MIN_SYNONYMOUS = 5  # Minimum synonymous mutations per gene
TOP_N_GENES = 5  # Number of top genes to show per category

# Paths
NONSENSE_FILE = PROCESSED_DATA_DIR / "selection" / "nmd_variants_annotated_annotated.tsv"
SYNONYMOUS_FILE = INTERIM_DATA_DIR / "selection" / "synonymous_variants_annotated.tsv"
GENE_MAP_FILE = RAW_DATA_DIR / "annotations" / "ensembl_v88_gene_transcript_genesymbol.txt"
CANCER_GENES_FILE = RAW_DATA_DIR / "annotations" / "cancer_genes.csv"

OUTPUT_FIGURE = FIGURES_DIR / "manuscript" / "Fig5" / "gene_selection_scores.png"
OUTPUT_FIGURE_PDF = FIGURES_DIR / "manuscript" / "Fig5" / "gene_selection_scores.pdf"
OUTPUT_TABLE = TABLES_DIR / "manuscript" / "Fig5" / "gene_selection_scores.csv"


def load_gene_annotations() -> tuple[list, list, pd.DataFrame]:
    """Load gene category annotations."""
    logger.info("Loading gene annotations...")
    
    # Load gene mapping
    gene_map = pd.read_csv(GENE_MAP_FILE, sep="\t", header=0)
    gene_map = gene_map[["gene_id", "gene_name"]].drop_duplicates()
    
    # Load cancer genes
    cgc = pd.read_csv(CANCER_GENES_FILE)
    cgc = cgc[cgc.cancer_gene == "common mutational cancer driver"]
    
    oncogenes = cgc[cgc.role == "OG"].Gene.tolist()
    tumor_suppressors = cgc[cgc.role == "TSG"].Gene.tolist()
    
    logger.info(f"Loaded {len(oncogenes)} oncogenes, {len(tumor_suppressors)} tumor suppressors")
    
    return oncogenes, tumor_suppressors, gene_map


def classify_nmd_regions(df: pd.DataFrame, use_preds: bool = True, 
                        threshold_triggering: float = 0.43, threshold_evading: float = -0.17) -> pd.DataFrame:
    """
    Classify variants into NMD-sensitive and NMD-insensitive regions.
    
    Args:
        df: DataFrame with variant annotations
        use_preds: If True, use AI predictions; if False, use rule-based NMD status
        threshold_triggering: Threshold for AI predictions (>= threshold = NMD-triggering)
        threshold_evading: Threshold for AI predictions (<= threshold = NMD-evading)
        
    Returns:
        DataFrame with added 'nmd_region' column
    """
    df = df.copy()
    
    if use_preds and 'NMDOrthrus_prediction' in df.columns:
        df['nmd_region'] = "unknown"
        df.loc[df['NMDOrthrus_prediction'] >= threshold_triggering, 'nmd_region'] = 'NMD_sensitive'
        df.loc[df['NMDOrthrus_prediction'] <= threshold_evading, 'nmd_region'] = 'NMD_insensitive'
    else:
        # Use the existing NMD_status column
        df['nmd_region'] = df['NMD_status'].map({
            'NMD_triggering': 'NMD_sensitive',
            'NMD_evading_last': 'NMD_insensitive', 
            'NMD_evading_start': 'NMD_insensitive',
            'NMD_evading_55nt': 'NMD_insensitive',
            'NMD_unknown': 'unknown'
        })
    
    return df


def calculate_gene_nmd_enrichment(nonsense_df: pd.DataFrame, synonymous_df: pd.DataFrame,
                                  oncogenes: list, tumor_suppressors: list,
                                  min_nonsense: int = 3, min_synonymous: int = 5) -> pd.DataFrame:
    """
    Calculate Cohen's h effect size for each gene comparing NMD-sensitive vs NMD-insensitive regions.
    
    Args:
        nonsense_df: DataFrame with nonsense variants
        synonymous_df: DataFrame with synonymous variants
        oncogenes: List of oncogene names
        tumor_suppressors: List of tumor suppressor names
        min_nonsense: Minimum nonsense mutations required
        min_synonymous: Minimum synonymous mutations required
        
    Returns:
        DataFrame with gene-level enrichment scores
    """
    logger.info("Creating aggregated counts...")
    
    # Create pivot tables for nonsense mutations
    nonsense_pivot = nonsense_df.groupby(['gene_name', 'nmd_region']).size().unstack(fill_value=0)
    nonsense_pivot = nonsense_pivot.reindex(columns=['NMD_sensitive', 'NMD_insensitive'], fill_value=0)
    nonsense_pivot['total_nonsense'] = nonsense_pivot.sum(axis=1)
    
    # Create pivot tables for synonymous mutations  
    synonymous_pivot = synonymous_df.groupby(['gene_name', 'nmd_region']).size().unstack(fill_value=0)
    synonymous_pivot = synonymous_pivot.reindex(columns=['NMD_sensitive', 'NMD_insensitive'], fill_value=0)
    synonymous_pivot['total_synonymous'] = synonymous_pivot.sum(axis=1)
    
    # Merge the two dataframes
    combined = nonsense_pivot.join(synonymous_pivot, rsuffix='_synonymous')
    combined = combined.rename(columns={
        'NMD_sensitive': 'NMD_sensitive_nonsense',
        'NMD_insensitive': 'NMD_insensitive_nonsense'
    })
    
    # Filter genes with sufficient data
    mask = (combined['total_nonsense'] >= min_nonsense) & (combined['total_synonymous'] >= min_synonymous)
    combined = combined[mask]
    
    logger.info(f"Filtered to {len(combined)} genes with sufficient data")
    
    # Calculate proportions
    combined['prop_sensitive'] = combined['NMD_sensitive_nonsense'] / (
        combined['NMD_sensitive_nonsense'] + combined['NMD_sensitive_synonymous']
    )
    combined['prop_insensitive'] = combined['NMD_insensitive_nonsense'] / (
        combined['NMD_insensitive_nonsense'] + combined['NMD_insensitive_synonymous']
    )
    
    # Calculate Cohen's h effect size
    def cohens_h(p1, p2):
        """Calculate Cohen's h effect size for proportion differences"""
        return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    
    combined['cohens_h_score'] = cohens_h(combined['prop_sensitive'], combined['prop_insensitive'])
    
    logger.info(f"Cohen's h scores range: {combined['cohens_h_score'].min():.3f} to {combined['cohens_h_score'].max():.3f}")
    
    # Add gene categories
    combined['category'] = 'Other'
    combined.loc[combined.index.isin(tumor_suppressors), 'category'] = 'Tumor Suppressor'
    combined.loc[combined.index.isin(oncogenes), 'category'] = 'Oncogene'
    
    # Calculate p-values
    logger.info("Calculating p-values...")
    p_values = []
    
    for gene in tqdm(combined.index, desc="Computing Fisher's exact tests"):
        row = combined.loc[gene]
        
        contingency_table = np.array([
            [row['NMD_sensitive_nonsense'], row['NMD_insensitive_nonsense']],
            [row['NMD_sensitive_synonymous'], row['NMD_insensitive_synonymous']]
        ])
        
        try:
            _, p_value = stats.fisher_exact(contingency_table)
        except (ValueError, ZeroDivisionError):
            p_value = 1.0
            
        p_values.append(p_value)
    
    combined['p_value'] = p_values
    
    # Calculate FDR-corrected p-values
    _, fdr_pvals, _, _ = multipletests(combined['p_value'], method='fdr_bh')
    combined['fdr_p_value'] = fdr_pvals
    
    # Prepare final results
    results = combined.reset_index()
    results = results.rename(columns={
        'gene_name': 'gene',
        'NMD_sensitive_nonsense': 'nonsense_sensitive',
        'NMD_insensitive_nonsense': 'nonsense_insensitive',
        'NMD_sensitive_synonymous': 'synonymous_sensitive',
        'NMD_insensitive_synonymous': 'synonymous_insensitive'
    })
    
    return results[['gene', 'category', 'nonsense_sensitive', 'nonsense_insensitive', 
                   'total_nonsense', 'synonymous_sensitive', 'synonymous_insensitive', 
                   'total_synonymous', 'cohens_h_score', 'p_value', 'fdr_p_value']]


def plot_gene_scores(gene_enrichment_df: pd.DataFrame, top_n: int = 5):
    """
    Create single lollipop plot showing top NMD-triggering and NMD-evading genes.
    
    Args:
        gene_enrichment_df: DataFrame with gene enrichment scores
        top_n: Number of top genes to show per category
    """
    # Filter to cancer genes only
    gene_enrichment_df = gene_enrichment_df[gene_enrichment_df.category != 'Other'].copy()
    
    # Get top triggering (tumor suppressors with highest Cohen's h)
    top_triggering = gene_enrichment_df[
        gene_enrichment_df.category == 'Tumor Suppressor'
    ].nlargest(top_n, 'cohens_h_score')
    
    # Get top evading (oncogenes with lowest Cohen's h)
    top_evading = gene_enrichment_df[
        gene_enrichment_df.category == 'Oncogene'
    ].nsmallest(top_n, 'cohens_h_score')
    
    # Combine and sort by score
    combined_genes = pd.concat([top_evading, top_triggering])
    combined_genes = combined_genes.sort_values('cohens_h_score')
    
    logger.info(f"Plotting {len(combined_genes)} genes")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Define colors for categories
    category_colors = {
        'Tumor Suppressor': '#FF6B6B',
        'Oncogene': '#3d405b'
    }
    
    y_pos = np.arange(len(combined_genes))
    colors = [category_colors[cat] for cat in combined_genes['category']]
    
    # Create lollipop stems
    for i, (y, x) in enumerate(zip(y_pos, combined_genes['cohens_h_score'])):
        ax.plot([0, x], [y, y], color='gray', linewidth=1, alpha=0.7)
    
    # Create lollipop heads
    ax.scatter(combined_genes['cohens_h_score'], y_pos, 
              c=colors, s=120, alpha=0.8, edgecolors='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined_genes['gene'], fontsize=14)
    ax.set_xlabel("Cohen's h Effect Size", fontsize=16)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add effect size annotations
    for i, (y, score) in enumerate(zip(y_pos, combined_genes['cohens_h_score'])):
        x_offset = 0.05 if score > 0 else -0.05
        ha = 'left' if score > 0 else 'right'
        ax.text(score + x_offset, y, f'{score:.2f}', 
                va='center', ha=ha, fontsize=12, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=category_colors['Tumor Suppressor'], label='Tumor Suppressor'),
        Patch(facecolor=category_colors['Oncogene'], label='Oncogene')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches='tight')
    logger.info(f"Saved figure to {OUTPUT_FIGURE}")
    logger.info(f"Saved figure to {OUTPUT_FIGURE_PDF}")
    
    plt.close()


def main():
    """Main analysis function."""
    logger.info("Starting gene-level NMD selection score analysis")
    logger.info(f"Configuration:")
    logger.info(f"  USE_PREDICTIONS: {USE_PREDICTIONS}")
    if USE_PREDICTIONS:
        logger.info(f"  PREDICTION_THRESHOLD_TRIGGERING: {PREDICTION_THRESHOLD_TRIGGERING}")
        logger.info(f"  PREDICTION_THRESHOLD_EVADING: {PREDICTION_THRESHOLD_EVADING}")
    logger.info(f"  MIN_NONSENSE: {MIN_NONSENSE}")
    logger.info(f"  MIN_SYNONYMOUS: {MIN_SYNONYMOUS}")
    logger.info(f"  TOP_N_GENES: {TOP_N_GENES}")
    
    # Check if output table already exists
    if OUTPUT_TABLE.exists():
        logger.info(f"Loading existing results from {OUTPUT_TABLE}")
        gene_enrichment_df = pd.read_csv(OUTPUT_TABLE)
    else:
        # Load gene annotations
        oncogenes, tumor_suppressors, gene_map = load_gene_annotations()
        
        # Load variant data
        logger.info(f"Loading nonsense variants from {NONSENSE_FILE}")
        df = pd.read_csv(NONSENSE_FILE, sep="\t")
        logger.info(f"Loaded {len(df)} nonsense variants")
        
        logger.info(f"Loading synonymous variants from {SYNONYMOUS_FILE}")
        syn = pd.read_csv(SYNONYMOUS_FILE, sep="\t")
        logger.info(f"Loaded {len(syn)} synonymous variants")
        
        # Map gene IDs to gene names
        df = df.merge(gene_map, left_on="Gene", right_on="gene_id", how="left")
        syn = syn.merge(gene_map, left_on="Gene", right_on="gene_id", how="left")
        
        # Classify NMD regions
        logger.info("Classifying NMD regions...")
        df_classified = classify_nmd_regions(df, use_preds=USE_PREDICTIONS, 
                                           threshold_triggering=PREDICTION_THRESHOLD_TRIGGERING,
                                           threshold_evading=PREDICTION_THRESHOLD_EVADING)
        syn_classified = classify_nmd_regions(syn, use_preds=USE_PREDICTIONS,
                                            threshold_triggering=PREDICTION_THRESHOLD_TRIGGERING,
                                            threshold_evading=PREDICTION_THRESHOLD_EVADING)
        
        logger.info(f"NMD region distribution for nonsense variants:")
        logger.info(f"\n{df_classified['nmd_region'].value_counts()}")
        
        # Calculate gene-level enrichment
        gene_enrichment_df = calculate_gene_nmd_enrichment(
            df_classified, syn_classified, oncogenes, tumor_suppressors,
            min_nonsense=MIN_NONSENSE, min_synonymous=MIN_SYNONYMOUS
        )
        
        # Save results
        OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
        gene_enrichment_df.to_csv(OUTPUT_TABLE, index=False)
        logger.info(f"Saved results to {OUTPUT_TABLE}")
        
        # Print summary
        logger.info(f"\nAnalyzed {len(gene_enrichment_df)} genes with sufficient data")
        logger.info(f"Cohen's h score range: {gene_enrichment_df['cohens_h_score'].min():.3f} to {gene_enrichment_df['cohens_h_score'].max():.3f}")
        
        # Show distribution by category
        logger.info(f"\nEnrichment by gene category:")
        category_stats = gene_enrichment_df.groupby('category')['cohens_h_score'].agg(['count', 'mean', 'median'])
        logger.info(f"\n{category_stats}")
    
    # Create plot
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plot_gene_scores(gene_enrichment_df, top_n=TOP_N_GENES)
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
