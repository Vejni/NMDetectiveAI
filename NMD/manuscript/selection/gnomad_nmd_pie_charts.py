#!/usr/bin/env python3
"""
Manuscript figure: gnomAD Rare PTC NMD Status Pie Charts (Fig7b)

This script creates two pie charts showing:
1. Rule-based NMD status (triggering vs evading)
2. AI-predicted NMD status (triggering vs evading vs intermediate)

Analyzes rare (<0.1% MAF) PTCs from the gnomAD v4.1 database.
"""

from pathlib import Path
import numpy as np
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
SCRIPT_NAME = "gnomad_nmd_pie_charts"

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
FIGURE_SIZE = (12, 12)  # Two pie charts side by side
DPI = 300

# Colors for categories
COLORS = {
    'NMD_triggering': EVADING_2_TRIGGERING_COLOUR_GRAD[4],  # '#fb731d' - orange
    'NMD_evading': EVADING_2_TRIGGERING_COLOUR_GRAD[0],  # '#ff9e9d' - pink
    'Intermediate': '#7f7f7f',  # gray
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
        Tuple of (rule_based_data, ai_based_data) DataFrames
    """
    # Check if output table already exists
    if not regenerate and source_data_path is not None and source_data_path.exists():
        logger.info(f"Loading existing results from {source_data_path}")
        rule_data = pd.read_excel(source_data_path, sheet_name='Rule_Based_Pie', engine='openpyxl')
        try:
            ai_data = pd.read_excel(source_data_path, sheet_name='AI_Based_Pie', engine='openpyxl')
        except:
            ai_data = None
        return rule_data, ai_data
    
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
    
    # Calculate triggering vs evading totals
    n_triggering = (df_rare['nmd_status'] == 'Triggering').sum()
    n_evading = (df_rare['nmd_status'] == 'Evading').sum()
    total_ptcs = len(df_rare)
    
    logger.info(f"\nRule-based categorization:")
    logger.info(f"  Total rare PTCs: {total_ptcs}")
    logger.info(f"  NMD triggering: {n_triggering} ({n_triggering/total_ptcs*100:.1f}%)")
    logger.info(f"  NMD evading: {n_evading} ({n_evading/total_ptcs*100:.1f}%)")
    
    # Create rule-based pie data
    rule_data = pd.DataFrame([
        {'category': 'NMD_triggering', 'count': n_triggering},
        {'category': 'NMD_evading', 'count': n_evading}
    ])
    
    # Load and process AI predictions if available
    ai_data = None
    if GNOMAD_WITH_PREDICTIONS_FILE.exists():
        logger.info(f"Loading predictions from {GNOMAD_WITH_PREDICTIONS_FILE}")
        df_full = pd.read_csv(GNOMAD_WITH_PREDICTIONS_FILE, sep='\t')
        df_with_preds = df_full[df_full['AF'] < MAF_THRESHOLD].copy()
        
        # Filter for variants with successful predictions
        df_pred_success = df_with_preds[
            df_with_preds['NMDetectiveAI_status'] == 'processed'
        ].copy()
        
        if len(df_pred_success) > 0:
            # Categorize based on predictions
            df_pred_success['pred_category'] = df_pred_success['NMDetectiveAI_prediction'].apply(
                lambda x: 'Evading' if x <= PREDICTION_THRESHOLD_EVADING else (
                    'Triggering' if x >= PREDICTION_THRESHOLD_TRIGGERING else 'Intermediate'
                )
            )
            
            pred_counts = df_pred_success['pred_category'].value_counts()
            pred_total = len(df_pred_success)
            
            logger.info(f"\nAI-based categorization:")
            logger.info(f"  Total variants with predictions: {pred_total}")
            for cat in ['Triggering', 'Evading', 'Intermediate']:
                count = pred_counts.get(cat, 0)
                logger.info(f"  {cat}: {count} ({count/pred_total*100:.1f}%)")
            
            ai_data = pd.DataFrame({
                'category': pred_counts.index,
                'count': pred_counts.values
            })
    
    # Save detailed results as XLSX with sheets per pie chart
    if source_data_path is not None:
        source_data_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(source_data_path, engine='openpyxl') as writer:
            rule_data.to_excel(writer, sheet_name='Rule_Based_Pie', index=False)
            if ai_data is not None:
                ai_data.to_excel(writer, sheet_name='AI_Based_Pie', index=False)
            else:
                # Placeholder for AI data
                pd.DataFrame([{
                    'category': 'Pending', 
                    'count': 0, 
                    'note': 'No predictions available'
                }]).to_excel(writer, sheet_name='AI_Based_Pie', index=False)
        logger.info(f"Results saved to {source_data_path}")
    
    return rule_data, ai_data


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_from_table(rule_data, ai_data=None):
    """
    Create pie chart visualization of NMD status.
    
    Args:
        rule_data: DataFrame with rule-based category counts
        ai_data: DataFrame with AI-based category counts (optional)
    """
    logger.info("Creating figure...")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    # --- Subplot 1: Pie chart showing triggering vs evading (rule-based) ---
    triggering_count = rule_data[rule_data['category'] == 'NMD_triggering']['count'].values[0]
    evading_count = rule_data[rule_data['category'] == 'NMD_evading']['count'].values[0]
    total = triggering_count + evading_count
    
    pie_data = [triggering_count, evading_count]
    pie_labels = [
        f'Triggering\n{triggering_count}\n({triggering_count/total*100:.1f}%)',
        f'Evading\n{evading_count}\n({evading_count/total*100:.1f}%)'
    ]
    pie_colors = [COLORS['NMD_triggering'], COLORS['NMD_evading']]
    
    ax1.pie(pie_data, labels=pie_labels, colors=pie_colors, autopct='',
           startangle=90, textprops={'fontsize': 20}, radius=0.8)
    ax1.set_title('Rule-based NMD status', fontsize=20, fontweight='bold')
    
    # --- Subplot 2: Pie chart showing prediction-based categorization ---
    if ai_data is not None and len(ai_data) > 0:
        # Get counts for each category
        pred_triggering = ai_data[ai_data['category'] == 'Triggering']['count'].values
        pred_evading = ai_data[ai_data['category'] == 'Evading']['count'].values
        pred_intermediate = ai_data[ai_data['category'] == 'Intermediate']['count'].values
        
        pred_triggering = pred_triggering[0] if len(pred_triggering) > 0 else 0
        pred_evading = pred_evading[0] if len(pred_evading) > 0 else 0
        pred_intermediate = pred_intermediate[0] if len(pred_intermediate) > 0 else 0
        pred_total = pred_triggering + pred_evading + pred_intermediate
        
        if pred_total > 0:
            # Create pie chart
            pred_pie_data = [pred_triggering, pred_evading, pred_intermediate]
            pred_pie_labels = [
                f'Triggering\n{pred_triggering}\n({pred_triggering/pred_total*100:.1f}%)',
                f'Evading\n{pred_evading}\n({pred_evading/pred_total*100:.1f}%)',
                f'Intermediate\n{pred_intermediate}\n({pred_intermediate/pred_total*100:.1f}%)'
            ]
            pred_pie_colors = [COLORS['NMD_triggering'], COLORS['NMD_evading'], COLORS['Intermediate']]
            
            ax2.pie(pred_pie_data, labels=pred_pie_labels, colors=pred_pie_colors,
                   autopct='', startangle=90, textprops={'fontsize': 18}, radius=0.8)
            ax2.set_title('NMDetective-AI predicted\nNMD status', fontsize=20, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No AI predictions available',
                    ha='center', va='center', fontsize=18, transform=ax2.transAxes)
            ax2.axis('off')
    else:
        ax2.text(0.5, 0.5, 'No AI predictions available',
                ha='center', va='center', fontsize=18, transform=ax2.transAxes)
        ax2.axis('off')
    
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
    """Generate gnomAD NMD status pie charts (Fig7b).

    Args:
        figure_label: Panel label (e.g. "Fig7b") when called from the
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
    rule_data, ai_data = process_data(
        source_data_path=paths.source_data,
        regenerate=regenerate,
    )

    # Create and save figure
    logger.info("Generating figure...")
    fig = plot_from_table(rule_data, ai_data)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("gnomAD NMD pie charts complete!")


if __name__ == "__main__":
    main()
