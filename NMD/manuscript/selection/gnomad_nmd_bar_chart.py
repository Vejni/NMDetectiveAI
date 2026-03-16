#!/usr/bin/env python3
"""
Manuscript figure: gnomAD NMD Evasion Mechanisms Bar Chart (Fig7c)

This script creates a bar chart showing the breakdown of NMD evasion
mechanisms among rare gnomAD PTCs, with shading based on AI predictions.

Analyzes rare (<0.1% MAF) PTCs from the gnomAD v4.1 database.
"""

import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from NMD.config import (
    PROCESSED_DATA_DIR,
    EVADING_2_TRIGGERING_COLOUR_GRAD,
)
from NMD.manuscript.output import get_paths


# ============================================================================
# CONFIGURATION
# ============================================================================

# Script identity (used for default standalone output filenames)
SCRIPT_NAME = "gnomad_nmd_bar_chart"

# Data paths
GNOMAD_ANNOTATED_FILE = PROCESSED_DATA_DIR / "gnomad_v4.1" / "annotated_rare" / "gnomad.v4.1.all_chromosomes.rare_stopgain_snv.mane.annotated.tsv"
GNOMAD_WITH_PREDICTIONS_FILE = PROCESSED_DATA_DIR / "gnomad_v4.1" / "annotated_rare" / "gnomad.v4.1.all_chromosomes.rare_stopgain_snv.mane.annotated_with_predictions.tsv"

# Analysis parameters
MAF_THRESHOLD = 0.001  # 0.1% = rare variants
LONG_EXON_THRESHOLD = 400  # nucleotides
START_PROX_THRESHOLD = 150  # nucleotides
LAST_EJC_THRESHOLD = 55  # nucleotides
PREDICTION_THRESHOLD_TRIGGERING = 0.43
PREDICTION_THRESHOLD_EVADING = -0.17

# Plot parameters
FIGURE_SIZE = (16, 6)
DPI = 300

# Colors for categories
COLORS = {
    'NMD_triggering': EVADING_2_TRIGGERING_COLOUR_GRAD[4],  # '#fb731d' - orange
    'Start_proximal': EVADING_2_TRIGGERING_COLOUR_GRAD[2],  # '#fcbb01' - yellow
    'Last_exon': EVADING_2_TRIGGERING_COLOUR_GRAD[0],  # '#ff9e9d' - pink
    'Last_EJC_50nt': EVADING_2_TRIGGERING_COLOUR_GRAD[1],  # '#ffdfcb' - light pink
    'Long_exon': EVADING_2_TRIGGERING_COLOUR_GRAD[3],  # '#2778ff' - blue
    'AI_predicted_evading': '#7f7f7f',  # gray
}


# ============================================================================
# DATA PROCESSING
# ============================================================================

def categorize_ptc(row):
    """
    Categorize a PTC by NMD evasion mechanism.
    
    Args:
        row: DataFrame row with PTC annotations
        
    Returns:
        Category string
    """
    # Check evasion rules in priority order (most to least important)
    # 1. Last exon rule (highest priority)
    if row['is_in_last_exon']:
        return 'Last_exon'
    
    # 2. Penultimate/50nt rule (within 50-55 nt of last EJC, not in last exon)
    if not row['is_in_last_exon'] and row['distance_from_last_ejc'] <= LAST_EJC_THRESHOLD:
        return 'Last_EJC_50nt'
    
    # 3. Start-proximal rule (<100 nt from start)
    if row['ptc_cds_position'] <= START_PROX_THRESHOLD:
        return 'Start_proximal'
    
    # 4. Long exon rule (in exon > 400 nt)
    if row['ptc_exon_length'] > LONG_EXON_THRESHOLD:
        return 'Long_exon'
    
    # 5. NMD triggering (doesn't match any evasion rule)
    return 'NMD_triggering'


def determine_nmd_status(category):
    """
    Determine if a category is NMD triggering or evading.
    
    Args:
        category: PTC category
        
    Returns:
        'Triggering' or 'Evading'
    """
    return 'Triggering' if category == 'NMD_triggering' else 'Evading'


def process_data(source_data_path=None, regenerate=True):
    """
    Process gnomAD data and categorize PTCs by evasion mechanism.

    Args:
        source_data_path: Path to the Excel source data file.
            Used for caching processed results.
        regenerate: If False and source_data_path exists, load
            from cache instead of reprocessing.

    Returns:
        DataFrame with bar chart data
    """
    # Check if output table already exists
    if not regenerate and source_data_path is not None and source_data_path.exists():
        logger.info(f"Loading existing results from {source_data_path}")
        bar_data = pd.read_excel(source_data_path, sheet_name='Evading_Categories', engine='openpyxl')
        return bar_data
    
    logger.info("Starting gnomAD data processing...")
    logger.info(f"Loading data from {GNOMAD_ANNOTATED_FILE}")
    
    # Load annotated gnomAD data
    if not GNOMAD_ANNOTATED_FILE.exists():
        raise FileNotFoundError(
            f"Annotated gnomAD file not found: {GNOMAD_ANNOTATED_FILE}\n"
            "Please run the annotation script first: python -m NMD.data.annotate_gnomad_stopgain --chr all"
        )
    
    df = pd.read_csv(GNOMAD_ANNOTATED_FILE, sep='\t')
    logger.info(f"Loaded {len(df)} total PTCs")
    
    # Filter for rare variants (AF < 0.1%)
    df_rare = df[df['AF'] < MAF_THRESHOLD].copy()
    logger.info(f"Filtered to {len(df_rare)} rare variants (AF < {MAF_THRESHOLD})")
    
    # Categorize each PTC
    logger.info("Categorizing PTCs by NMD evasion mechanism...")
    df_rare['category'] = df_rare.apply(categorize_ptc, axis=1)
    df_rare['nmd_status'] = df_rare['category'].apply(determine_nmd_status)
    
    # Count categories (only evading categories)
    category_counts = df_rare['category'].value_counts()
    total_ptcs = len(df_rare)
    
    # Calculate percentages for evading categories
    results = []
    for category in ['Start_proximal', 'Last_exon', 'Last_EJC_50nt', 'Long_exon']:
        count = category_counts.get(category, 0)
        percentage = (count / total_ptcs * 100) if total_ptcs > 0 else 0
        results.append({
            'category': category,
            'count': count,
            'percentage': percentage,
            'ai_predicted_evading': 0,  # Will be filled if predictions available
            'penultimate_exon_count': 0  # Will be filled for AI category if applicable
        })
    
    logger.info(f"\nEvasion mechanism breakdown:")
    for res in results:
        logger.info(f"  {res['category']}: {res['count']} ({res['percentage']:.1f}%)")
    
    # Load and process AI predictions if available
    if GNOMAD_WITH_PREDICTIONS_FILE.exists():
        logger.info(f"Loading predictions from {GNOMAD_WITH_PREDICTIONS_FILE}")
        df_full = pd.read_csv(GNOMAD_WITH_PREDICTIONS_FILE, sep='\t')
        df_with_preds = df_full[df_full['AF'] < MAF_THRESHOLD].copy()
        df_with_preds['category'] = df_with_preds.apply(categorize_ptc, axis=1)
        
        # Calculate how many in each category have predictions below threshold
        for res in results:
            cat = res['category']
            cat_variants = df_with_preds[df_with_preds['category'] == cat]
            cat_pred_success = cat_variants[cat_variants['NMDetectiveAI_status'] == 'processed']
            if len(cat_pred_success) > 0:
                below_threshold = (cat_pred_success['NMDetectiveAI_prediction'] <= PREDICTION_THRESHOLD_EVADING).sum()
                res['ai_predicted_evading'] = below_threshold
        
        # Calculate AI-predicted evading but not in any rule-based category
        # These are variants categorized as 'NMD_triggering' but predicted as evading
        ai_only_evading = df_with_preds[
            (df_with_preds['category'] == 'NMD_triggering') &
            (df_with_preds['NMDetectiveAI_status'] == 'processed') &
            (df_with_preds['NMDetectiveAI_prediction'] <= PREDICTION_THRESHOLD_EVADING)
        ]
        ai_only_count = len(ai_only_evading)
        
        # Count how many are in penultimate exon
        if ai_only_count > 0:
            ai_only_penultimate = ai_only_evading[
                ai_only_evading['ptc_exon_idx'] == ai_only_evading['num_cds_exons'] - 2
            ]
            ai_only_penultimate_count = len(ai_only_penultimate)
            
            results.append({
                'category': 'AI_predicted_evading',
                'count': ai_only_count,
                'percentage': (ai_only_count / total_ptcs) * 100,
                'ai_predicted_evading': ai_only_count,
                'penultimate_exon_count': ai_only_penultimate_count
            })
            logger.info(f"  NMDetective-AI-predicted evading (no rule match): {ai_only_count} ({ai_only_count/total_ptcs*100:.1f}%)")
            logger.info(f"    Of which {ai_only_penultimate_count} ({ai_only_penultimate_count/ai_only_count*100:.1f}%) are in penultimate exon")
        
        logger.info(f"\nAI predictions for evading categories:")
        for res in results:
            if res['ai_predicted_evading'] > 0:
                logger.info(f"  {res['category']}: {res['ai_predicted_evading']}/{res['count']} predicted evading")
    
    # Save results as XLSX
    bar_data = pd.DataFrame(results)
    if source_data_path is not None:
        source_data_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(source_data_path, engine='openpyxl') as writer:
            bar_data.to_excel(writer, sheet_name='Evading_Categories', index=False)
        logger.info(f"Results saved to {source_data_path}")
    
    return bar_data


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_from_table(df):
    """
    Create bar chart visualization of NMD evasion mechanisms.
    
    Args:
        df: DataFrame with evasion category counts
    """
    logger.info("Creating figure...")
    
    # Sort by count (descending)
    df = df.sort_values('count', ascending=False).reset_index(drop=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    
    # Check if we have AI predictions
    has_ai_predictions = df['ai_predicted_evading'].sum() > 0
    
    y_pos = range(len(df))
    total_counts = df['count'].values
    colors_list = [COLORS.get(cat, '#7f7f7f') for cat in df['category']]
    
    if has_ai_predictions:
        # Base bars (full counts in light color)
        ax.barh(y_pos, total_counts, color=colors_list, alpha=0.3, label='Total')
        
        # Shaded bars (predicted evading)
        shaded_counts = df['ai_predicted_evading'].values
        ax.barh(y_pos, shaded_counts, color=colors_list, alpha=1.0,
                label=f'Predicted evading (≤ {PREDICTION_THRESHOLD_EVADING})')
    else:
        # Simple bars without shading
        ax.barh(y_pos, total_counts, color=colors_list)
    
    # Category labels
    category_labels = {
        'Start_proximal': 'Start-proximal\n(<150 nt)',
        'Last_exon': 'Last exon',
        'Last_EJC_50nt': '55nt rule\n(≤55 nt from last EJC)',
        'Long_exon': 'Long exon\n(>400 nt)',
        'AI_predicted_evading': 'AI-predicted evading\n(no rule match)',
    }
    
    labels = [category_labels.get(cat, cat) for cat in df['category']]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=15)
    ax.set_xlabel('Number of PTCs', fontsize=16, fontweight='bold')
    ax.set_title('NMD evasion mechanisms', fontsize=16, fontweight='bold')

    # Add count labels on bars
    for i, row in df.iterrows():
        count = row['count']
        pct = row['percentage']
        
        # Check if this is the AI-predicted category and has penultimate exon count
        if row['category'] == 'AI_predicted_evading' and 'penultimate_exon_count' in row and pd.notna(row['penultimate_exon_count']):
            pe_count = int(row['penultimate_exon_count'])
            pe_pct = (pe_count / count * 100) if count > 0 else 0
            label = f"{count} ({pct:.1f}%)\n{pe_count} in penultimate exon ({pe_pct:.1f}%)"
        else:
            label = f'{count} ({pct:.1f}%)'
        
        ax.text(count + max(df['count']) * 0.02, i,
                label,
                va='center', fontsize=14)
    
    if has_ai_predictions:
        ax.legend(loc='upper right', fontsize=14)
    
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
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
    """Generate gnomAD NMD evasion mechanisms bar chart (Fig7c).

    Args:
        figure_label: Panel label (e.g. "Fig7c") when called from the
            manuscript app.  *None* → standalone mode.
        figure_number: Figure number (e.g. "Fig7") when called from the
            manuscript app.
        regenerate: If *False* and source data already exists, skip the
            data-processing step and plot directly from the saved table.
    """
    paths = get_paths(
        script_name=SCRIPT_NAME,
        figure_label=figure_label,
        figure_number=figure_number,
        source_data_ext=".xlsx",
    )

    # Process data (or load if already exists)
    df = process_data(
        source_data_path=paths.source_data,
        regenerate=regenerate,
    )

    # Create and save figure
    logger.info("Generating figure...")
    fig = plot_from_table(df)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("gnomAD NMD evasion mechanisms bar chart complete!")


if __name__ == "__main__":
    main()
