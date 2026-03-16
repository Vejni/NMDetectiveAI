#!/usr/bin/env python3
"""
Manuscript figure: gnomAD Rare vs Common PTC NMD Categories

This script compares the distribution of NMD categories between rare (<0.1% AF)
and common (>=0.1% AF) PTCs from gnomAD v4.1.

Creates a stacked bar chart showing the proportion of:
- NMD triggering
- NMD evading:
  - Start-proximal (<150 nt from start)
  - Last exon
  - 55nt rule (<=55 nt from last EJC)
  - Long exon (>400 nt)
  
Can use either rule-based or AI-predicted NMD status.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from scipy.stats import fisher_exact

from NMD.config import (
    PROCESSED_DATA_DIR,
    TABLES_DIR,
    EVADING_2_TRIGGERING_COLOUR_GRAD,
    RAW_DATA_DIR
)
from NMD.manuscript.output import get_paths


# ============================================================================
# CONFIGURATION
# ============================================================================

# Use AI predictions or rule-based annotations
USE_AI_PREDICTIONS = False  # If True, use AI predictions; if False, use rule-based
AI_THRESHOLD_TRIGGERING = 0.43
AI_THRESHOLD_EVADING = -0.17

# Data paths
RARE_PTC_FILE = PROCESSED_DATA_DIR / "gnomad_v4.1" / "annotated_rare" / "gnomad.v4.1.all_chromosomes.rare_stopgain_snv.mane.annotated_with_predictions.tsv"
COMMON_PTC_FILE = PROCESSED_DATA_DIR / "gnomad_v4.1" / "annotated_common" / "gnomad.v4.1.all_chromosomes.common_stopgain_snv.mane.annotated_with_predictions.tsv"
CONSTRAINT_FILE = RAW_DATA_DIR / "annotations" / "supplementary_dataset_11_full_constraint_metrics.tsv"

SCRIPT_NAME = "gnomad_rarity_and_constrain"

# Analysis parameters
MAF_THRESHOLD = 0.001  # 0.1% threshold
LONG_EXON_THRESHOLD = 400  # nucleotides
START_PROX_THRESHOLD = 150  # nucleotides
LAST_EJC_THRESHOLD = 55  # nucleotides

# Plot parameters
DPI = 300

# Colors for categories (consistent with gnomad_nmd_categories.py)
CATEGORY_COLORS = {
    'NMD_triggering': EVADING_2_TRIGGERING_COLOUR_GRAD[4],  # '#fb731d' - orange
    'Last_exon': EVADING_2_TRIGGERING_COLOUR_GRAD[0],  # '#ff9e9d' - pink
    'Last_EJC_55nt': EVADING_2_TRIGGERING_COLOUR_GRAD[1],  # '#ffdfcb' - light pink
    'Start_proximal': EVADING_2_TRIGGERING_COLOUR_GRAD[2],  # '#fcbb01' - yellow
    'Long_exon': EVADING_2_TRIGGERING_COLOUR_GRAD[3],  # '#2778ff' - blue
}

# Category order for stacking
CATEGORY_ORDER = ['NMD_triggering', 'Last_exon', 'Last_EJC_55nt', 'Start_proximal', 'Long_exon']


# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_constraint_metrics() -> pd.DataFrame:
    """Load gnomAD gene constraint metrics."""
    logger.info(f"Loading constraint metrics from {CONSTRAINT_FILE}")
    df = pd.read_csv(CONSTRAINT_FILE, sep='\t')
    
    # Keep only relevant columns and deduplicate by gene (take canonical transcript)
    df = df[['gene', 'oe_lof_upper_bin', 'canonical']].copy()
    
    # Prefer canonical transcripts, otherwise take first entry
    df_canonical = df[df['canonical'] == True].drop_duplicates(subset=['gene'], keep='first')
    df_non_canonical = df[~df['gene'].isin(df_canonical['gene'])].drop_duplicates(subset=['gene'], keep='first')
    df_unique = pd.concat([df_canonical, df_non_canonical], ignore_index=True)
    
    # Rename for clarity
    df_unique = df_unique.rename(columns={
        'gene': 'gene_symbol',
        'oe_lof_upper_bin': 'loeuf_decile'  # 0-9, where 0 is most constrained
    })
    
    logger.info(f"Loaded constraint metrics for {len(df_unique)} genes")
    return df_unique[['gene_symbol', 'loeuf_decile']]

def categorize_ptc_rule_based(row):
    """
    Categorize a PTC by NMD evasion mechanism using rules.
    
    Args:
        row: DataFrame row with PTC annotations
        
    Returns:
        Category string
    """
    # Check evasion rules in priority order
    # 1. Last exon rule (highest priority)
    if row['is_in_last_exon']:
        return 'Last_exon'
    
    # 2. Penultimate/55nt rule (within 55 nt of last EJC, not in last exon)
    if not row['is_in_last_exon'] and row['distance_from_last_ejc'] <= LAST_EJC_THRESHOLD:
        return 'Last_EJC_55nt'
    
    # 3. Start-proximal rule (<150 nt from start)
    if row['ptc_cds_position'] <= START_PROX_THRESHOLD:
        return 'Start_proximal'
    
    # 4. Long exon rule (in exon > 400 nt)
    if row['ptc_exon_length'] > LONG_EXON_THRESHOLD:
        return 'Long_exon'
    
    # 5. NMD triggering (doesn't match any evasion rule)
    return 'NMD_triggering'


def categorize_ptc_ai(row):
    """
    Categorize a PTC using AI predictions and rule-based mechanisms.
    
    Args:
        row: DataFrame row with PTC annotations and AI predictions
        
    Returns:
        Category string or None if intermediate prediction
    """
    # Check if AI prediction is available and successful
    if row.get('NMDetectiveAI_status') != 'processed':
        return None
    
    pred = row['NMDetectiveAI_prediction']
    
    # Intermediate predictions - exclude
    if AI_THRESHOLD_EVADING < pred < AI_THRESHOLD_TRIGGERING:
        return None
    
    # Triggering
    if pred >= AI_THRESHOLD_TRIGGERING:
        return 'NMD_triggering'
    
    # Evading - determine mechanism by rules
    # Check evasion rules in priority order
    if row['is_in_last_exon']:
        return 'Last_exon'
    
    if not row['is_in_last_exon'] and row['distance_from_last_ejc'] <= LAST_EJC_THRESHOLD:
        return 'Last_EJC_55nt'
    
    if row['ptc_cds_position'] <= START_PROX_THRESHOLD:
        return 'Start_proximal'
    
    if row['ptc_exon_length'] > LONG_EXON_THRESHOLD:
        return 'Long_exon'
    
    # AI predicted evading but no rule match
    return 'NMD_triggering'  # Conservative - treat as triggering if no mechanism found


def process_data(source_data_path=None, regenerate=True):
    """
    Process gnomAD data and categorize PTCs for rare and common variants.
    Split rare variants by LOEUF decile (0-1 vs 2-9).
    
    Returns:
        DataFrame with category counts and percentages for each variant type
    """
    # Check if output table already exists
    if source_data_path and source_data_path.exists() and not regenerate:
        logger.info(f"Loading existing results from {source_data_path}")
        results_df = pd.read_csv(source_data_path)
        return results_df
    
    logger.info("Starting gnomAD rare vs common comparison...")
    
    # Load constraint metrics
    constraint_df = load_constraint_metrics()
    
    # Load rare PTCs
    logger.info(f"Loading rare PTCs from {RARE_PTC_FILE}")
    if not RARE_PTC_FILE.exists():
        raise FileNotFoundError(f"Rare PTC file not found: {RARE_PTC_FILE}")
    
    df_rare = pd.read_csv(RARE_PTC_FILE, sep='\t')
    logger.info(f"Loaded {len(df_rare)} rare PTCs (AF < {MAF_THRESHOLD})")
    
    # Merge with constraint data
    df_rare = df_rare.merge(constraint_df, on='gene_symbol', how='left')
    logger.info(f"Merged constraint data: {df_rare['loeuf_decile'].notna().sum()} variants have LOEUF info")
    
    # Split rare by LOEUF decile
    df_rare_constrained = df_rare[df_rare['loeuf_decile'].isin([0, 1])].copy()
    df_rare_other = df_rare[~df_rare['loeuf_decile'].isin([0, 1]) & df_rare['loeuf_decile'].notna()].copy()
    
    logger.info(f"Split rare PTCs: {len(df_rare_constrained)} in LOEUF 0-1, {len(df_rare_other)} in LOEUF 2-9")
    
    # Load common PTCs
    logger.info(f"Loading common PTCs from {COMMON_PTC_FILE}")
    if not COMMON_PTC_FILE.exists():
        raise FileNotFoundError(f"Common PTC file not found: {COMMON_PTC_FILE}")
    
    df_common = pd.read_csv(COMMON_PTC_FILE, sep='\t')
    logger.info(f"Loaded {len(df_common)} common PTCs (AF >= {MAF_THRESHOLD})")
    
    # Categorize PTCs
    if USE_AI_PREDICTIONS:
        logger.info("Using AI predictions for categorization")
        df_rare_constrained['category'] = df_rare_constrained.apply(categorize_ptc_ai, axis=1)
        df_rare_other['category'] = df_rare_other.apply(categorize_ptc_ai, axis=1)
        df_common['category'] = df_common.apply(categorize_ptc_ai, axis=1)
        
        # Remove intermediate predictions
        n_rare_constrained_before = len(df_rare_constrained)
        n_rare_other_before = len(df_rare_other)
        n_common_before = len(df_common)
        df_rare_constrained = df_rare_constrained.dropna(subset=['category']).copy()
        df_rare_other = df_rare_other.dropna(subset=['category']).copy()
        df_common = df_common.dropna(subset=['category']).copy()
        
        logger.info(f"Filtered out {n_rare_constrained_before - len(df_rare_constrained)} rare constrained variants with intermediate predictions")
        logger.info(f"Filtered out {n_rare_other_before - len(df_rare_other)} rare other variants with intermediate predictions")
        logger.info(f"Filtered out {n_common_before - len(df_common)} common variants with intermediate predictions")
    else:
        logger.info("Using rule-based categorization")
        df_rare_constrained['category'] = df_rare_constrained.apply(categorize_ptc_rule_based, axis=1)
        df_rare_other['category'] = df_rare_other.apply(categorize_ptc_rule_based, axis=1)
        df_common['category'] = df_common.apply(categorize_ptc_rule_based, axis=1)
    
    # Count categories for each variant type
    results = []
    
    # Combine all rare for the first set of bars
    df_rare_all = pd.concat([df_rare_constrained, df_rare_other], ignore_index=True)
    
    for variant_type, df in [('All_Rare', df_rare_all), ('Common', df_common), ('Rare_LOEUF_0-1', df_rare_constrained), ('Rare_LOEUF_2-9', df_rare_other)]:
        total = len(df)
        category_counts = df['category'].value_counts()
        
        logger.info(f"\n{variant_type} variants (n={total}):")
        for category in CATEGORY_ORDER:
            count = category_counts.get(category, 0)
            percentage = (count / total * 100) if total > 0 else 0
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
            
            results.append({
                'variant_type': variant_type,
                'category': category,
                'count': count,
                'percentage': percentage,
                'total': total
            })
    
    # Save results
    if source_data_path:
        source_data_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(source_data_path, index=False)
        logger.info(f"\nResults saved to {source_data_path}")
    
    # Compute statistical tests for each category
    logger.info("Computing statistical tests...")
    stats_results = []
    
    # Test 1: All Rare vs Common
    for category in CATEGORY_ORDER:
        rare_all_row = results_df[(results_df['variant_type'] == 'All_Rare') & (results_df['category'] == category)]
        common_row = results_df[(results_df['variant_type'] == 'Common') & (results_df['category'] == category)]
        
        if len(rare_all_row) > 0 and len(common_row) > 0:
            count_rare = rare_all_row['count'].values[0]
            total_rare = rare_all_row['total'].values[0]
            count_common = common_row['count'].values[0]
            total_common = common_row['total'].values[0]
            
            table = [[count_rare, total_rare - count_rare], [count_common, total_common - count_common]]
            odds_ratio, p_value = fisher_exact(table)
            
            stats_results.append({
                'variant_type': 'Common',
                'category': category,
                'comparison': 'All_Rare_vs_Common',
                'odds_ratio': odds_ratio,
                'p_value': p_value
            })
    
    # Test 2: Rare LOEUF 0-1 vs Rare LOEUF 2-9
    for category in CATEGORY_ORDER:
        rare_01_row = results_df[(results_df['variant_type'] == 'Rare_LOEUF_0-1') & (results_df['category'] == category)]
        rare_29_row = results_df[(results_df['variant_type'] == 'Rare_LOEUF_2-9') & (results_df['category'] == category)]
        
        if len(rare_01_row) > 0 and len(rare_29_row) > 0:
            count_01 = rare_01_row['count'].values[0]
            total_01 = rare_01_row['total'].values[0]
            count_29 = rare_29_row['count'].values[0]
            total_29 = rare_29_row['total'].values[0]
            
            table = [[count_01, total_01 - count_01], [count_29, total_29 - count_29]]
            odds_ratio, p_value = fisher_exact(table)
            
            stats_results.append({
                'variant_type': 'Rare_LOEUF_0-1',
                'category': category,
                'comparison': 'LOEUF_0-1_vs_2-9',
                'odds_ratio': odds_ratio,
                'p_value': p_value
            })
    
    stats_df = pd.DataFrame(stats_results)
    
    # Merge stats into results_df
    results_df = results_df.merge(stats_df, on=['variant_type', 'category'], how='left')
    
    return results_df


# ============================================================================
# PLOTTING
# ============================================================================

def plot_stacked_bar(results_df):
    """
    Create stacked bar chart comparing rare (split by LOEUF) and common variants.
    
    Args:
        results_df: DataFrame with category counts and percentages
    """
    logger.info("Creating stacked bar chart...")
    
    # Prepare data for plotting - order: All Rare, Common, (divider), Rare LOEUF 0-1, Rare LOEUF 2-9
    variant_types = ['All_Rare', 'Common', 'Rare_LOEUF_0-1', 'Rare_LOEUF_2-9']
    variant_labels = ['Gnomad\nRare', 'Gnomad\nCommon', 'Rare more\n constrained\n(LOEUF 0-1)', 'Rare less\nconstrained\n(LOEUF 2-9)']
    
    # Create figure with more width for 4 bars
    fig, ax = plt.subplots(figsize=(8, 12), dpi=DPI)
    
    # Calculate bar positions with gap for divider between bar 2 and 3
    x = np.array([0, 1, 2.5, 3.5])  # Add gap after Common
    bar_width = 0.4
    
    # Create stacked bars
    bottom = np.zeros(len(variant_types))
    
    for category in CATEGORY_ORDER:
        # Get percentages for this category
        percentages = []
        for variant_type in variant_types:
            row = results_df[(results_df['variant_type'] == variant_type) & 
                           (results_df['category'] == category)]
            if len(row) > 0:
                percentages.append(row['percentage'].values[0])
            else:
                percentages.append(0)
        
        # Plot bar segment
        color = CATEGORY_COLORS.get(category, '#808080')
        ax.bar(x, percentages, bar_width, bottom=bottom, 
               label=category.replace('_', ' '), color=color)
        
        # Annotate if significant between All Rare and Common (between bars 0 and 1)
        if percentages[0] > 0:
            common_row = results_df[(results_df['variant_type'] == 'Common') & 
                                    (results_df['category'] == category) & 
                                    (results_df['comparison'] == 'All_Rare_vs_Common')]
            if len(common_row) > 0 and pd.notna(common_row['p_value'].values[0]) and common_row['p_value'].values[0] < 0.05:
                or_val = common_row['odds_ratio'].values[0]
                p_val = common_row['p_value'].values[0]
                y_pos = bottom[0] + percentages[0] / 2
                # Text between All Rare and Common bars
                text_x = (x[0] + x[1]) / 2
                ax.text(text_x, y_pos, f'OR={or_val:.2f}\np={p_val:.2e}', 
                        ha='center', va='center', fontsize=12, color='black', 
                        fontweight='bold', bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8, edgecolor=color))
                # Arrows from text to both bars (point to middle of bars at category height)
                ax.annotate('', xy=(x[0], y_pos), xytext=(text_x - 0.08, y_pos),
                           arrowprops=dict(arrowstyle='->', color='black', linewidth=1.5, alpha=0.7))
                ax.annotate('', xy=(x[1], y_pos), xytext=(text_x + 0.08, y_pos),
                           arrowprops=dict(arrowstyle='->', color='black', linewidth=1.5, alpha=0.7))
        
        # Annotate if significant between Rare LOEUF 0-1 and 2-9 (between bars 2 and 3)
        if percentages[2] > 0:
            rare_01_row = results_df[(results_df['variant_type'] == 'Rare_LOEUF_0-1') & 
                                     (results_df['category'] == category) & 
                                     (results_df['comparison'] == 'LOEUF_0-1_vs_2-9')]
            if len(rare_01_row) > 0 and pd.notna(rare_01_row['p_value'].values[0]) and rare_01_row['p_value'].values[0] < 0.05:
                or_val = rare_01_row['odds_ratio'].values[0]
                p_val = rare_01_row['p_value'].values[0]
                y_pos = bottom[2] + percentages[2] / 2
                # Text between LOEUF 0-1 and LOEUF 2-9 bars
                text_x = (x[2] + x[3]) / 2
                ax.text(text_x, y_pos, f'OR={or_val:.2f}\np={p_val:.2e}', 
                        ha='center', va='center', fontsize=12, color='black', 
                        fontweight='bold', bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8, edgecolor=color))
                # Arrows from text to both bars (point to middle of bars at category height)
                ax.annotate('', xy=(x[2], y_pos), xytext=(text_x - 0.08, y_pos),
                           arrowprops=dict(arrowstyle='->', color='black', linewidth=1.5, alpha=0.7))
                ax.annotate('', xy=(x[3], y_pos), xytext=(text_x + 0.08, y_pos),
                           arrowprops=dict(arrowstyle='->', color='black', linewidth=1.5, alpha=0.7))
        
        bottom += percentages
    
    # Add vertical divider between Common and Rare LOEUF groups
    divider_x = (x[1] + x[2]) / 2
    ax.axvline(divider_x, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Customize plot
    ax.set_ylabel('Percentage of PTCs', fontsize=20)
    ax.set_xlabel('Variant type', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, fontsize=16)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', labelsize=18)
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=16, ncol=2)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.title('Distribution of NMD categories by \nvariant type and genetic constraint', fontsize=22, weight='bold')

    return fig


# ============================================================================
# MAIN
# ============================================================================

def main(figure_label=None, figure_number=None, regenerate=True):
    """Main execution function."""
    paths = get_paths(SCRIPT_NAME, figure_label=figure_label, figure_number=figure_number)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    if paths.source_data:
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("gnomAD Rare vs Common NMD Categories Analysis")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  USE_AI_PREDICTIONS: {USE_AI_PREDICTIONS}")
    if USE_AI_PREDICTIONS:
        logger.info(f"  AI_THRESHOLD_TRIGGERING: {AI_THRESHOLD_TRIGGERING}")
        logger.info(f"  AI_THRESHOLD_EVADING: {AI_THRESHOLD_EVADING}")
    
    # Process data
    results_df = process_data(source_data_path=paths.source_data, regenerate=regenerate)
    
    # Create plot
    fig = plot_stacked_bar(results_df)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    logger.info(f"Figure PDF saved to {paths.figure_pdf}")
    plt.close(fig)
    
    logger.info("="*60)
    logger.info("Analysis complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
