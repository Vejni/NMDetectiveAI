"""
Disease gene NMD enrichment analysis.

This script analyzes enrichment/depletion of NMD-triggering variants in disease genes
from ClinVar compared to non-disease genes, using gnomAD rare variant data.

For each disease gene, we calculate:
    - Log odds ratio: log((AC_triggering + 1) / (AC_evading + 1))
    - Fisher's exact test p-value for enrichment/depletion
    - Comparison to genome-wide background

Outputs:
    - Table of disease genes with enrichment/depletion statistics
    - Summary statistics (N genes enriched, N genes depleted)
    - Volcano plot of log odds vs -log10(p-value)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from loguru import logger

from NMD.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, TABLES_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "gnomad_disease_gene_PTCs"


# Configuration
AI_THRESHOLD_TRIGGERING = 0.43  # AI predictions >= threshold = NMD-triggering
AI_THRESHOLD_EVADING = -0.17  # AI predictions <= threshold = NMD-evading
MIN_PTCS_PER_GENE = 10  # Minimum number of PTCs required to include gene
SIGNIFICANCE_THRESHOLD = 0.05  # P-value threshold (will be Bonferroni corrected)
LOG_ODDS_THRESHOLD = 0.5  # Absolute log odds threshold for calling enrichment/depletion

# Paths
RARE_PTC_FILE = PROCESSED_DATA_DIR / "gnomad_v4.1" / "annotated_rare" / "gnomad.v4.1.all_chromosomes.rare_stopgain_snv.mane.annotated_with_predictions.tsv"
CLINVAR_GENE_FILE = RAW_DATA_DIR / "annotations" / "gene_condition_source_id"
OUTPUT_SUMMARY = TABLES_DIR / "analysis" / "disease_gene_nmd_enrichment_summary.txt"


def load_disease_genes() -> set:
    """Load disease gene list from ClinVar."""
    logger.info(f"Loading disease genes from {CLINVAR_GENE_FILE}")
    
    # Read the file (tab-separated)
    df = pd.read_csv(CLINVAR_GENE_FILE, sep='\t', comment='#', 
                     names=['GeneID', 'AssociatedGenes', 'RelatedGenes', 'ConceptID', 
                            'DiseaseName', 'SourceName', 'SourceID', 'DiseaseMIM', 'LastUpdated'],
                     skiprows=1)
    
    # Get unique gene symbols from AssociatedGenes column
    disease_genes = set(df['AssociatedGenes'].dropna().unique())
    disease_genes.discard('')  # Remove empty strings if any
    
    logger.info(f"Loaded {len(disease_genes)} unique disease genes from ClinVar")
    return disease_genes


def load_and_classify_rare_ptcs() -> pd.DataFrame:
    """Load rare PTC data and classify by NMD status using AI predictions."""
    logger.info(f"Loading rare PTCs from {RARE_PTC_FILE}")
    df = pd.read_csv(RARE_PTC_FILE, sep='\t', low_memory=False)
    logger.info(f"Loaded {len(df)} rare PTCs")
    
    # Classify as NMD-triggering or evading based on AI predictions
    logger.info(f"Classifying variants with thresholds: triggering >= {AI_THRESHOLD_TRIGGERING}, evading <= {AI_THRESHOLD_EVADING}")
    df['is_nmd_triggering'] = df['NMDetectiveAI_prediction'].apply(
        lambda x: True if x >= AI_THRESHOLD_TRIGGERING else (False if x <= AI_THRESHOLD_EVADING else None)
    )
    
    # Filter out intermediate predictions
    intermediate_count = df['is_nmd_triggering'].isna().sum()
    logger.info(f"Filtering out {intermediate_count} variants with intermediate AI predictions")
    df = df.dropna(subset=['is_nmd_triggering']).copy()
    df['is_nmd_triggering'] = df['is_nmd_triggering'].astype(bool)
    
    logger.info(f"Retained {len(df)} variants after filtering ({df['is_nmd_triggering'].sum()} triggering, {(~df['is_nmd_triggering']).sum()} evading)")
    
    return df


def calculate_gene_statistics(rare_df: pd.DataFrame, disease_genes: set) -> pd.DataFrame:
    """
    Calculate NMD statistics for each gene.
    
    Args:
        rare_df: DataFrame with rare PTCs and NMD classifications
        disease_genes: Set of disease gene symbols
        
    Returns:
        DataFrame with gene-level statistics
    """
    logger.info("Calculating gene-level statistics...")
    
    # Aggregate by gene
    gene_stats = []
    
    for gene, gene_df in rare_df.groupby('gene_symbol'):
        # Count alleles for triggering and evading
        ac_triggering = gene_df[gene_df['is_nmd_triggering']]['AC'].sum()
        ac_evading = gene_df[~gene_df['is_nmd_triggering']]['AC'].sum()
        n_variants = len(gene_df)
        ac_total = gene_df['AC'].sum()
        
        # Calculate log odds with pseudocount
        log_odds = np.log((ac_triggering + 1) / (ac_evading + 1))
        
        # Mark if disease gene
        is_disease = gene in disease_genes
        
        gene_stats.append({
            'gene_symbol': gene,
            'is_disease_gene': is_disease,
            'n_variants': n_variants,
            'ac_total': ac_total,
            'ac_triggering': ac_triggering,
            'ac_evading': ac_evading,
            'log_odds': log_odds
        })
    
    stats_df = pd.DataFrame(gene_stats)
    
    logger.info(f"Calculated statistics for {len(stats_df)} genes")
    logger.info(f"  Disease genes: {stats_df['is_disease_gene'].sum()}")
    logger.info(f"  Non-disease genes: {(~stats_df['is_disease_gene']).sum()}")
    
    return stats_df


def perform_fisher_test(ac_triggering: float, ac_evading: float, 
                        bg_triggering: float, bg_evading: float) -> tuple:
    """
    Perform Fisher's exact test comparing gene to background.
    
    Args:
        ac_triggering: Allele count of triggering variants in gene
        ac_evading: Allele count of evading variants in gene
        bg_triggering: Total allele count of triggering variants in background
        bg_evading: Total allele count of evading variants in background
        
    Returns:
        Tuple of (odds_ratio, p_value)
    """
    # Create 2x2 contingency table
    # Rows: gene vs background
    # Columns: triggering vs evading
    contingency = np.array([
        [ac_triggering, ac_evading],  # Gene
        [bg_triggering - ac_triggering, bg_evading - ac_evading]  # Background (excluding gene)
    ])
    
    # Fisher's exact test (two-sided)
    odds_ratio, p_value = fisher_exact(contingency, alternative='two-sided')
    
    return odds_ratio, p_value


def add_statistical_tests(stats_df: pd.DataFrame, disease_only: bool = True) -> pd.DataFrame:
    """
    Add Fisher's exact test p-values for each gene.
    
    Args:
        stats_df: DataFrame with gene statistics
        disease_only: If True, only test disease genes; if False, test all genes
        
    Returns:
        DataFrame with added p-values
    """
    logger.info("Performing Fisher's exact tests...")
    
    # Calculate background (all genes)
    total_triggering = stats_df['ac_triggering'].sum()
    total_evading = stats_df['ac_evading'].sum()
    
    logger.info(f"Background: {total_triggering:.0f} triggering, {total_evading:.0f} evading alleles")
    
    # Perform test for each gene
    odds_ratios = []
    p_values = []
    
    for _, row in stats_df.iterrows():
        # Skip non-disease genes if disease_only
        if disease_only and not row['is_disease_gene']:
            odds_ratios.append(np.nan)
            p_values.append(np.nan)
            continue
        
        # Skip genes with too few variants
        if row['n_variants'] < MIN_PTCS_PER_GENE:
            odds_ratios.append(np.nan)
            p_values.append(np.nan)
            continue
        
        odds_ratio, p_value = perform_fisher_test(
            row['ac_triggering'], row['ac_evading'],
            total_triggering, total_evading
        )
        
        odds_ratios.append(odds_ratio)
        p_values.append(p_value)
    
    stats_df['fisher_odds_ratio'] = odds_ratios
    stats_df['fisher_p_value'] = p_values
    
    # Add Bonferroni-corrected p-values
    n_tests = (~stats_df['fisher_p_value'].isna()).sum()
    stats_df['bonferroni_p_value'] = stats_df['fisher_p_value'] * n_tests
    stats_df['bonferroni_p_value'] = stats_df['bonferroni_p_value'].clip(upper=1.0)
    
    logger.info(f"Performed {n_tests} statistical tests")
    
    return stats_df


def classify_enrichment(stats_df: pd.DataFrame, 
                        p_threshold: float = SIGNIFICANCE_THRESHOLD,
                        log_odds_threshold: float = LOG_ODDS_THRESHOLD) -> pd.DataFrame:
    """
    Classify genes as enriched, depleted, or neutral.
    
    Args:
        stats_df: DataFrame with statistics and p-values
        p_threshold: P-value threshold for significance (after Bonferroni correction)
        log_odds_threshold: Absolute log odds threshold
        
    Returns:
        DataFrame with classification column
    """
    logger.info(f"Classifying enrichment/depletion (p < {p_threshold}, |log_odds| > {log_odds_threshold})...")
    
    def classify(row):
        # Skip if missing p-value
        if pd.isna(row['bonferroni_p_value']):
            return 'Not tested'
        
        # Check significance
        is_significant = row['bonferroni_p_value'] < p_threshold
        
        # Check effect size
        has_large_effect = abs(row['log_odds']) > log_odds_threshold
        
        if not is_significant or not has_large_effect:
            return 'Neutral'
        elif row['log_odds'] > 0:
            return 'Enriched for triggering'
        else:
            return 'Enriched for evading'
    
    stats_df['enrichment_class'] = stats_df.apply(classify, axis=1)
    
    # Count classifications
    counts = stats_df['enrichment_class'].value_counts()
    logger.info("Classification results:")
    for cls, count in counts.items():
        logger.info(f"  {cls}: {count}")
    
    return stats_df


def plot_volcano(stats_df: pd.DataFrame):
    """
    Create volcano plot of log odds vs -log10(p-value).
    
    Args:
        stats_df: DataFrame with statistics and classifications
    """
    logger.info("Creating volcano plot...")
    
    # Filter to disease genes with valid p-values
    plot_df = stats_df[
        (stats_df['is_disease_gene']) & 
        (~stats_df['fisher_p_value'].isna())
    ].copy()
    
    # Calculate -log10(p-value)
    plot_df['-log10_p'] = -np.log10(plot_df['bonferroni_p_value'].clip(lower=1e-300))
    
    # Set up colors
    color_map = {
        'Enriched for triggering': '#fb731d',  # Orange
        'Enriched for evading': '#ff9e9d',  # Pink
        'Neutral': '#cccccc'  # Gray
    }
    
    plot_df['color'] = plot_df['enrichment_class'].map(color_map)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Plot points
    for cls in ['Neutral', 'Enriched for evading', 'Enriched for triggering']:
        mask = plot_df['enrichment_class'] == cls
        if mask.sum() > 0:
            ax.scatter(plot_df.loc[mask, 'log_odds'], 
                      plot_df.loc[mask, '-log10_p'],
                      c=color_map[cls], s=50, alpha=0.6, 
                      label=f"{cls} (n={mask.sum()})")
    
    # Add significance threshold line
    bonf_threshold = -np.log10(SIGNIFICANCE_THRESHOLD)
    ax.axhline(bonf_threshold, color='gray', linestyle='--', linewidth=1, alpha=0.5,
               label=f'p = {SIGNIFICANCE_THRESHOLD} (Bonferroni)')
    
    # Add log odds threshold lines
    ax.axvline(LOG_ODDS_THRESHOLD, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(-LOG_ODDS_THRESHOLD, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Label top genes
    top_enriched = plot_df.nlargest(5, 'log_odds')
    top_depleted = plot_df.nsmallest(5, 'log_odds')
    
    for _, row in pd.concat([top_enriched, top_depleted]).iterrows():
        if row['-log10_p'] > bonf_threshold:  # Only label significant genes
            ax.annotate(row['gene_symbol'], 
                       (row['log_odds'], row['-log10_p']),
                       fontsize=12, alpha=0.8,
                       xytext=(5, 5), textcoords='offset points')
    
    # Styling
    ax.set_xlabel('Log Odds (NMD-triggering / NMD-evading)', fontsize=16)
    ax.set_ylabel('-log₁₀(Bonferroni-corrected P-value)', fontsize=16)
    ax.set_title('NMD variant enrichment in disease genes', fontsize=18, weight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig


def write_summary(stats_df: pd.DataFrame):
    """Write summary statistics to file."""
    logger.info("Writing summary statistics...")
    
    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_SUMMARY, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DISEASE GENE NMD ENRICHMENT ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 70 + "\n")
        total_disease_genes = stats_df['is_disease_gene'].sum()
        tested_disease_genes = ((stats_df['is_disease_gene']) & 
                                (~stats_df['fisher_p_value'].isna())).sum()
        f.write(f"Total disease genes in gnomAD: {total_disease_genes}\n")
        f.write(f"Disease genes with ≥{MIN_PTCS_PER_GENE} PTCs (tested): {tested_disease_genes}\n\n")
        
        # Enrichment/depletion counts
        f.write("Classification Results:\n")
        f.write("-" * 70 + "\n")
        disease_tested = stats_df[
            (stats_df['is_disease_gene']) & 
            (~stats_df['fisher_p_value'].isna())
        ]
        
        counts = disease_tested['enrichment_class'].value_counts()
        for cls, count in counts.items():
            pct = 100 * count / len(disease_tested)
            f.write(f"{cls}: {count} ({pct:.1f}%)\n")
        
        # Key finding
        n_enriched = counts.get('Enriched for triggering', 0)
        n_depleted = counts.get('Enriched for evading', 0)
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("KEY FINDING:\n")
        f.write("=" * 70 + "\n")
        f.write(f"In {n_depleted} out of {tested_disease_genes} disease genes we see enrichment of NMD-evading variants\n")
        f.write(f"In {n_enriched} out of {tested_disease_genes} disease genes we see enrichment of NMD-triggering variants\n")
        f.write("=" * 70 + "\n\n")
        
        # Top enriched genes
        f.write("Top 10 genes enriched for NMD-TRIGGERING variants:\n")
        f.write("-" * 70 + "\n")
        top_trigger = disease_tested.nlargest(10, 'log_odds')
        for _, row in top_trigger.iterrows():
            f.write(f"  {row['gene_symbol']}: log_odds={row['log_odds']:.2f}, "
                   f"p={row['bonferroni_p_value']:.2e}, "
                   f"AC_trig={int(row['ac_triggering'])}, AC_evad={int(row['ac_evading'])}\n")
        
        f.write("\n")
        f.write("Top 10 genes enriched for NMD-EVADING variants:\n")
        f.write("-" * 70 + "\n")
        top_evad = disease_tested.nsmallest(10, 'log_odds')
        for _, row in top_evad.iterrows():
            f.write(f"  {row['gene_symbol']}: log_odds={row['log_odds']:.2f}, "
                   f"p={row['bonferroni_p_value']:.2e}, "
                   f"AC_trig={int(row['ac_triggering'])}, AC_evad={int(row['ac_evading'])}\n")
    
    logger.info(f"Saved summary to {OUTPUT_SUMMARY}")
    
    # Also print key finding to console
    logger.info("\n" + "=" * 70)
    logger.info("KEY FINDING:")
    logger.info("=" * 70)
    logger.info(f"In {n_depleted} out of {tested_disease_genes} disease genes we see enrichment of NMD-evading variants")
    logger.info(f"In {n_enriched} out of {tested_disease_genes} disease genes we see enrichment of NMD-triggering variants")
    logger.info("=" * 70 + "\n")


def main(figure_label=None, figure_number=None, regenerate=True):
    """Main analysis function."""
    paths = get_paths(SCRIPT_NAME, figure_label=figure_label, figure_number=figure_number)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    if paths.source_data:
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting disease gene NMD enrichment analysis")
    logger.info(f"Configuration:")
    logger.info(f"  AI_THRESHOLD_TRIGGERING: {AI_THRESHOLD_TRIGGERING}")
    logger.info(f"  AI_THRESHOLD_EVADING: {AI_THRESHOLD_EVADING}")
    logger.info(f"  MIN_PTCS_PER_GENE: {MIN_PTCS_PER_GENE}")
    logger.info(f"  SIGNIFICANCE_THRESHOLD: {SIGNIFICANCE_THRESHOLD}")
    logger.info(f"  LOG_ODDS_THRESHOLD: {LOG_ODDS_THRESHOLD}")
    
    # Load disease genes
    disease_genes = load_disease_genes()
    
    # Load and classify rare PTCs
    rare_df = load_and_classify_rare_ptcs()
    
    # Calculate gene-level statistics
    stats_df = calculate_gene_statistics(rare_df, disease_genes)
    
    # Add statistical tests (for disease genes with enough variants)
    stats_df = add_statistical_tests(stats_df, disease_only=True)
    
    # Classify enrichment/depletion
    stats_df = classify_enrichment(stats_df)
    
    # Save results
    if paths.source_data:
        stats_df.to_csv(paths.source_data, index=False)
        logger.info(f"Saved results to {paths.source_data}")
    
    # Write summary
    write_summary(stats_df)
    
    # Create volcano plot
    fig = plot_volcano(stats_df)
    fig.savefig(paths.figure_png, dpi=300, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Saved figure to {paths.figure_png}")
    logger.info(f"Saved PDF to {paths.figure_pdf}")
    plt.close(fig)
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
