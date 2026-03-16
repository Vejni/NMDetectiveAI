#!/usr/bin/env python3
"""
Manuscript figure: gnomAD Rare PTC NMD Categories

This script analyzes rare (<0.1% MAF) PTCs from the gnomAD v4.1 database
and categorizes them by NMD status and evasion mechanisms:
- NMD triggering
- NMD evading:
  - Start-proximal (<150 nt from start)
  - Last exon
  - 50nt rule (>50-55 nt from last EJC)
  - Long exon (>400 nt)
  - Unexplained (not matching any known rule)
"""

import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from NMD.config import (
    PROCESSED_DATA_DIR,
    EVADING_2_TRIGGERING_COLOUR_GRAD,

)
from NMD.manuscript.output import get_paths, get_analysis_table_path


# ============================================================================
# CONFIGURATION
# ============================================================================

# Script identity (used for default standalone output filenames)
SCRIPT_NAME = "gnomad_nmd_ratios"

# Data paths
GNOMAD_ANNOTATED_FILE = PROCESSED_DATA_DIR / "gnomad_v4.1" / "annotated_rare" / "gnomad.v4.1.all_chromosomes.rare_stopgain_snv.mane.annotated.tsv"
GNOMAD_WITH_PREDICTIONS_FILE = PROCESSED_DATA_DIR / "gnomad_v4.1" / "annotated_rare" / "gnomad.v4.1.all_chromosomes.rare_stopgain_snv.mane.annotated_with_predictions.tsv"

# Analysis parameters
MAF_THRESHOLD = 0.001  # 0.1% = rare variants
LONG_EXON_THRESHOLD = 400  # nucleotides
START_PROX_THRESHOLD = 150  # nucleotides
LAST_EJC_THRESHOLD = 55  # nucleotides
PREDICTION_THRESHOLD_TRIGGERING = 0.43  # Predictions <= -0.1 are evading, > 0.1 are triggering
PREDICTION_THRESHOLD_EVADING = -0.17  # Predictions <= -0.1 are evading, > 0.1 are triggering

# Plot parameters
FIGURE_SIZE = (12, 10)  # Increased for 3 subplots
DPI = 300

# Colors for categories
COLORS = {
    'NMD_triggering': EVADING_2_TRIGGERING_COLOUR_GRAD[4],  # '#fb731d' - orange
    'Start_proximal': EVADING_2_TRIGGERING_COLOUR_GRAD[2],  # '#fcbb01' - yellow
    'Last_exon': EVADING_2_TRIGGERING_COLOUR_GRAD[0],  # '#ff9e9d' - pink
    'Last_EJC_50nt': EVADING_2_TRIGGERING_COLOUR_GRAD[1],  # '#ffdfcb' - light pink
    'Long_exon': EVADING_2_TRIGGERING_COLOUR_GRAD[3],  # '#2778ff' - blue
    'Unexplained': "gray",
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
    Process gnomAD data and categorize PTCs.

    Args:
        source_data_path: Path to the Excel source data file.
            Used for caching processed results.
        regenerate: If False and source_data_path exists, load
            from cache instead of reprocessing.

    Returns:
        DataFrame with category counts and percentages
    """
    # Check if output table already exists
    if not regenerate and source_data_path is not None and source_data_path.exists():
        logger.info(f"Loading existing results from {source_data_path}")
        # Load the evading categories sheet and reconstruct the full results
        bar_data = pd.read_excel(source_data_path, sheet_name='Panel_C_Evading_Categories', engine='openpyxl')
        pie_rule_data = pd.read_excel(source_data_path, sheet_name='Panel_A_Rule_Based_Pie', engine='openpyxl')
        
        # Reconstruct full results from the two sheets
        triggering_row = pd.DataFrame([{
            'category': 'NMD_triggering',
            'count': pie_rule_data[pie_rule_data['category'] == 'NMD_triggering']['count'].values[0],
            'percentage': 0  # Will be recalculated
        }])
        df_results = pd.concat([triggering_row, bar_data], ignore_index=True)
        
        # Recalculate percentages
        total = df_results['count'].sum()
        df_results['percentage'] = (df_results['count'] / total) * 100
        
        return df_results
    
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
    
    # Count categories
    category_counts = df_rare['category'].value_counts()
    total_ptcs = len(df_rare)
    
    # Calculate percentages
    results = []
    for category in ['NMD_triggering', 'Start_proximal', 'Last_exon', 'Last_EJC_50nt', 'Long_exon']:
        count = category_counts.get(category, 0)
        percentage = (count / total_ptcs * 100) if total_ptcs > 0 else 0
        results.append({
            'category': category,
            'count': count,
            'percentage': percentage
        })
    
    # Calculate triggering vs evading totals
    n_triggering = (df_rare['nmd_status'] == 'Triggering').sum()
    n_evading = (df_rare['nmd_status'] == 'Evading').sum()
    
    logger.info(f"\nResults:")
    logger.info(f"  Total rare PTCs: {total_ptcs}")
    logger.info(f"  NMD triggering: {n_triggering} ({n_triggering/total_ptcs*100:.1f}%)")
    logger.info(f"  NMD evading: {n_evading} ({n_evading/total_ptcs*100:.1f}%)")
    logger.info(f"\nEvasion mechanism breakdown:")
    for res in results:
        if res['category'] != 'NMD_triggering':
            logger.info(f"    {res['category']}: {res['count']} ({res['percentage']:.1f}%)")
    
    # Save detailed results as XLSX with sheets per panel
    if source_data_path is not None:
        source_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(source_data_path, engine='openpyxl') as writer:
        # Sheet 1: Rule-based pie chart data (triggering vs evading)
        triggering_count = sum(res['count'] for res in results if res['category'] == 'NMD_triggering')
        evading_count = sum(res['count'] for res in results if res['category'] != 'NMD_triggering')
        pie_rule_data = pd.DataFrame([
            {'category': 'NMD_triggering', 'count': triggering_count},
            {'category': 'NMD_evading', 'count': evading_count}
        ])
        pie_rule_data.to_excel(writer, sheet_name='Panel_A_Rule_Based_Pie', index=False)
        
        # Sheet 2: Placeholder for AI-based pie (will be filled when plot is generated)
        # This will be updated in the plot function if predictions are available
        pie_ai_placeholder = pd.DataFrame([
            {'category': 'Pending', 'count': 0, 'note': 'Generated during plotting if predictions available'}
        ])
        pie_ai_placeholder.to_excel(writer, sheet_name='Panel_B_AI_Based_Pie', index=False)
        
        # Sheet 3: Bar chart data (evading categories breakdown)
        evading_results = [res for res in results if res['category'] != 'NMD_triggering']
        bar_data = pd.DataFrame(evading_results)
        bar_data.to_excel(writer, sheet_name='Panel_C_Evading_Categories', index=False)
    
    logger.info(f"\nResults saved to {source_data_path}")
    
    # Save full annotated data with categories (analysis table, not source data)
    category_file = get_analysis_table_path("gnomad_rare_ptcs_categorized.csv")
    df_rare.to_csv(category_file, sep=',', index=False)
    logger.info(f"Full categorized data saved to {category_file}")
    
    df_results = pd.DataFrame(results)
    return df_results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_from_table(df):
    """
    Create visualization of PTC categories.
    
    Args:
        df: DataFrame with category counts and percentages
    """
    logger.info("Creating figure...")
    
    # Load the full data with predictions if available
    df_with_preds = None
    if GNOMAD_WITH_PREDICTIONS_FILE.exists():
        logger.info(f"Loading predictions from {GNOMAD_WITH_PREDICTIONS_FILE}")
        df_full = pd.read_csv(GNOMAD_WITH_PREDICTIONS_FILE, sep='\t')
        # Filter for rare variants
        df_with_preds = df_full[df_full['AF'] < MAF_THRESHOLD].copy()
        # Categorize
        df_with_preds['category'] = df_with_preds.apply(categorize_ptc, axis=1)
        df_with_preds['nmd_status'] = df_with_preds['category'].apply(determine_nmd_status)
        logger.info(f"Loaded {len(df_with_preds)} rare variants with predictions")
    
    # Prepare data
    categories = df['category'].tolist()
    counts = df['count'].tolist()
    percentages = df['percentage'].tolist()
    
    # Create figure with 3 subplots if predictions available, otherwise 2
    if df_with_preds is not None:
        fig = plt.figure(figsize=FIGURE_SIZE)
        gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.2)
        ax1 = fig.add_subplot(gs[0, 0])  # Top left: Rule-based pie
        ax2 = fig.add_subplot(gs[0, 1])  # Top right: Prediction-based pie
        ax3 = fig.add_subplot(gs[1, :])  # Bottom: Bar chart with shading
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    
    # --- Subplot 1: Pie chart showing triggering vs evading (rule-based) ---
    triggering_count = df[df['category'] == 'NMD_triggering']['count'].sum()
    evading_count = df[df['category'] != 'NMD_triggering']['count'].sum()
    total = triggering_count + evading_count
    
    pie_data = [triggering_count, evading_count]
    pie_labels = [
        f'Triggering\n{triggering_count}\n({triggering_count/total*100:.1f}%)',
        f'Evading\n{evading_count}\n({evading_count/total*100:.1f}%)'
    ]
    pie_colors = [COLORS['NMD_triggering'], COLORS['Last_exon']]  # Orange and another color
    
    ax1.pie(pie_data, labels=pie_labels, colors=pie_colors, autopct='',
           startangle=90, textprops={'fontsize': 16}, radius=0.8)
    ax1.set_title('NMD status', fontsize=18, fontweight='bold')
    
    if df_with_preds is not None:
        # --- Subplot 2: Pie chart showing prediction-based categorization ---
        # Filter for variants with successful predictions
        df_pred_success = df_with_preds[
            df_with_preds['NMDetectiveAI_status'] == 'processed'
        ].copy()
        
        if len(df_pred_success) > 0:
            # Categorize based on predictions
            df_pred_success['pred_category'] = df_pred_success['NMDetectiveAI_prediction'].apply(
                lambda x: 'Evading' if x <= PREDICTION_THRESHOLD_EVADING else ('Triggering' if x >= PREDICTION_THRESHOLD_TRIGGERING else 'Intermediate')
            )
            
            pred_evading = (df_pred_success['pred_category'] == 'Evading').sum()
            pred_triggering = (df_pred_success['pred_category'] == 'Triggering').sum()
            pred_unknown = (df_pred_success['pred_category'] == 'Intermediate').sum()
            pred_total = len(df_pred_success)
            
            # Create pie chart
            pred_pie_data = [pred_triggering, pred_evading, pred_unknown]
            pred_pie_labels = [
                f'Triggering\n{pred_triggering}\n({pred_triggering/pred_total*100:.1f}%)',
                f'Evading\n{pred_evading}\n({pred_evading/pred_total*100:.1f}%)',
                f'Intermediate\n{pred_unknown}\n({pred_unknown/pred_total*100:.1f}%)'
            ]
            pred_pie_colors = [COLORS['NMD_triggering'], COLORS['Last_exon'], '#7f7f7f']
            
            ax2.pie(pred_pie_data, labels=pred_pie_labels, colors=pred_pie_colors,
                   autopct='', startangle=90, textprops={'fontsize': 16}, radius=0.8)
            ax2.set_title(f'Predicted NMD status',
                         fontsize=18, fontweight='bold')
            
            logger.info(f"\nPrediction-based categorization:")
            logger.info(f"  Evading: {pred_evading} ({pred_evading/pred_total*100:.1f}%)")
            logger.info(f"  Triggering: {pred_triggering} ({pred_triggering/pred_total*100:.1f}%)")
            logger.info(f"  Intermediate: {pred_unknown} ({pred_unknown/pred_total*100:.1f}%)")
        
        # --- Subplot 3: Bar chart with shading based on predictions ---
        # Only show evading categories
        evading_df = df[df['category'] != 'NMD_triggering'].copy()
        evading_df = evading_df.sort_values('count', ascending=False)
        
        # Calculate how many in each category have predictions below threshold
        shaded_counts = []
        for cat in evading_df['category']:
            cat_variants = df_with_preds[df_with_preds['category'] == cat]
            cat_pred_success = cat_variants[cat_variants['NMDetectiveAI_status'] == 'processed']
            if len(cat_pred_success) > 0:
                below_threshold = (cat_pred_success['NMDetectiveAI_prediction'] <= PREDICTION_THRESHOLD_EVADING).sum()
                shaded_counts.append(below_threshold)
            else:
                shaded_counts.append(0)
        
        # Calculate AI-predicted evading but not in any rule-based category
        # These are variants categorized as 'NMD_triggering' but predicted as evading
        ai_only_evading = df_with_preds[
            (df_with_preds['category'] == 'NMD_triggering') &
            (df_with_preds['NMDetectiveAI_status'] == 'processed') &
            (df_with_preds['NMDetectiveAI_prediction'] <= PREDICTION_THRESHOLD_EVADING)
        ]
        ai_only_count = len(ai_only_evading)
        
        # Add AI-only category to the dataframe
        if ai_only_count > 0:
            ai_only_row = pd.DataFrame([{
                'category': 'AI_predicted_evading',
                'count': ai_only_count,
                'percentage': (ai_only_count / len(df_with_preds)) * 100
            }])
            evading_df = pd.concat([evading_df, ai_only_row], ignore_index=True)
            shaded_counts.append(ai_only_count)  # All of these are predicted evading
        
        # Create stacked bar chart
        y_pos = range(len(evading_df))
        total_counts = evading_df['count'].values
        
        # Base bars (full counts in light color)
        colors_list = [COLORS.get(cat, '#7f7f7f') for cat in evading_df['category']]
        ax3.barh(y_pos, total_counts, color=colors_list, alpha=0.3, label='Total')
        
        # Shaded bars (predicted evading)
        ax3.barh(y_pos, shaded_counts, color=colors_list, alpha=1.0,
                label=f'Predicted evading (≤ {PREDICTION_THRESHOLD_EVADING})')
        
    else:
        # Original 2-subplot layout without predictions
        # --- Subplot 2: Bar chart showing evasion mechanisms ---
        evading_df = df[df['category'] != 'NMD_triggering'].copy()
        evading_df = evading_df.sort_values('count', ascending=False)
        colors_list = [COLORS[cat] for cat in evading_df['category']]
        bars = ax2.barh(range(len(evading_df)), evading_df['count'], color=colors_list)
        ax3 = ax2  # Use ax2 as ax3 for common code below
    
    # Common bar chart formatting
    category_labels = {
        'Start_proximal': 'Start-proximal\n(<150 nt)',
        'Last_exon': 'Last exon',
        'Last_EJC_50nt': '55nt rule\n(≤55 nt from last EJC)',
        'Long_exon': 'Long exon\n(>400 nt)',
        'AI_predicted_evading': 'NMDetective-AI predicted\nevading (no rule match)',
        'Unexplained': 'Unexplained'
    }
    
    # Use the evading_df that was already created above (includes AI category if present)
    # Don't recreate it from df
    labels = [category_labels.get(cat, cat) for cat in evading_df['category']]
    ax3.set_yticks(range(len(evading_df)))
    ax3.set_yticklabels(labels, fontsize=15)
    ax3.set_xlabel('Number of PTCs', fontsize=16)
    ax3.set_title('NMD evasion mechanisms', fontsize=16, fontweight='bold')
    
    # Add count labels on bars
    for i, (count, pct) in enumerate(zip(evading_df['count'], evading_df['percentage'])):
        ax3.text(count + max(evading_df['count']) * 0.02, i,
                f'{count} ({pct:.1f}%)',
                va='center', fontsize=14)
    
    if df_with_preds is not None:
        ax3.legend(loc='upper right', fontsize=14)
    
    ax3.grid(axis='x', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Return figure and AI pie data if available
    ai_pie_data = None
    if df_with_preds is not None:
        df_pred_success = df_with_preds[df_with_preds['NMDetectiveAI_status'] == 'processed'].copy()
        if len(df_pred_success) > 0:
            df_pred_success['pred_category'] = df_pred_success['NMDetectiveAI_prediction'].apply(
                lambda x: 'Evading' if x <= PREDICTION_THRESHOLD_EVADING else ('Triggering' if x >= PREDICTION_THRESHOLD_TRIGGERING else 'Intermediate')
            )
            pred_counts = df_pred_success['pred_category'].value_counts()
            ai_pie_data = pd.DataFrame({
                'category': pred_counts.index,
                'count': pred_counts.values
            })
    
    return fig, ai_pie_data


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = True,
):
    """Generate gnomAD NMD categorisation figure (panels a+b).

    Args:
        figure_label: Panel label (e.g. "Fig7ab") when called from the
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
    fig, ai_pie_data = plot_from_table(df)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    # Update AI pie sheet if data is available
    if ai_pie_data is not None and paths.source_data.exists():
        logger.info("Updating AI-based pie chart data in source data...")
        with pd.ExcelWriter(paths.source_data, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            ai_pie_data.to_excel(writer, sheet_name='Panel_B_AI_Based_Pie', index=False)

    logger.success("gnomAD NMD categorization complete!")


if __name__ == "__main__":
    main()
