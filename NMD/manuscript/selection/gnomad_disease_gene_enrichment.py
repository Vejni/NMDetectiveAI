#!/usr/bin/env python3
"""
Manuscript figure: Disease gene NMD enrichment significance summary.

For genes classified as NMD-evading enriched (log_odds_t1 < 0) and
NMD-triggering enriched (log_odds_t1 > 0), the figure shows three bars each:

  Bar 1 – Test 1 nominal (p_value_t1 < 0.05):
      stacked: nominally-significant-only (lighter) + FDR-passing (darker) on top.
      The % of nominally-significant genes that pass FDR is annotated above the bar.
  Bar 2 – Test 1 sig & Test 2 n.s. (p_t1 < 0.05 & p_t2 >= 0.05):
      stacked: nominally-significant-only (lighter) + combined FDR-passing (darker) on top.
  Bar 3 – Pass both FDR thresholds (T1 & T2):
      single solid bar showing number of genes passing FDR for both tests.

Plus a single bar for non-significant genes (p >= 0.05 in both tests).

Direction is determined by the sign of log_odds_t1 throughout.
Depends on the supplementary table produced by gnomad_disease_genes.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger

from NMD.manuscript.output import get_paths
from NMD.manuscript.selection import gnomad_disease_genes

SCRIPT_NAME = "gnomad_disease_gene_enrichment"

# Significance thresholds (mirror those in gnomad_disease_genes.py)
FDR_T1_THRESHOLD = gnomad_disease_genes.FDR_T1_THRESHOLD
FDR_T2_THRESHOLD = gnomad_disease_genes.FDR_T2_THRESHOLD

PVAL_NOMINAL = 0.05

FULL_TABLE = gnomad_disease_genes.SUPP_TABLE

# ─── Colours ──────────────────────────────────────────────────────────────────
EVADING_NOMINAL_COLOR    = '#ffc8c8'   # nominal only – pale pink
EVADING_FDR_COLOR        = '#ff9e9d'   # FDR-passing – stronger pink/red
BOTH_FDR_EVADING_COLOR   = "#fc4340"   # evading – combined FDR-passing

TRIGGERING_NOMINAL_COLOR  = '#d6f3ff'  # nominal only – pale blue
TRIGGERING_FDR_COLOR      = '#022778'  # FDR-passing – deep navy
BOTH_FDR_TRIGGERING_COLOR = "#091A3F"  # triggering – combined FDR-passing

NON_SIG_COLOR = '#d4d4d4'


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_stats() -> pd.DataFrame:
    """Load the precomputed gene-level stats from gnomad_disease_genes.py."""
    if not FULL_TABLE.exists():
        raise FileNotFoundError(
            f"Gene stats table not found at {FULL_TABLE}. "
            "Please run gnomad_disease_genes.py first."
        )
    logger.info(f"Loading gene stats from {FULL_TABLE}")
    df = pd.read_csv(FULL_TABLE)
    logger.info(f"Loaded stats for {len(df)} genes")
    return df


def build_counts(stats: pd.DataFrame) -> dict:
    """Compute per-bar counts for the grouped bar chart.

    Direction is classified by the sign of log_odds_t1 (canonical).

    Returns:
        Dict with 'evading', 'triggering', 'non_sig', and 'total' keys.
        Each direction dict contains:
            t1_fdr, t1_nominal_only, t1_sig_t2_ns_fdr, t1_sig_t2_ns_nom_only, combined_fdr, combined_nom_only.
    """
    evading    = stats['log_odds_t1'] < 0
    triggering = stats['log_odds_t1'] > 0

    def _counts_for(mask):
        sub = stats[mask]
        nom_t1 = sub['p_value_t1'] < PVAL_NOMINAL
        fdr_t1 = sub['p_value_fdr_t1'] < FDR_T1_THRESHOLD
        nom_t2 = sub['p_value_t2'] < PVAL_NOMINAL
        fdr_t2 = sub['p_value_fdr_t2'] < FDR_T2_THRESHOLD

        # Genes significant in Test1 (nominal) but not nominal in Test2
        t1_sig_t2_ns = nom_t1 & ~nom_t2

        # Genes passing both FDR thresholds (T1 and T2)
        both_fdr = (fdr_t1 & fdr_t2)

        return {
            't1_fdr':          int(fdr_t1.sum()),
            't1_nominal_only': int((nom_t1 & ~fdr_t1).sum()),
            't2_fdr':          int(fdr_t2.sum()),
            't2_nominal_only': int((nom_t2 & ~fdr_t2).sum()),
            't1_sig_t2_ns_fdr':    int((t1_sig_t2_ns & both_fdr).sum()),
            't1_sig_t2_ns_nom_only': int((t1_sig_t2_ns & ~both_fdr).sum()),
            'both_fdr':        int(both_fdr.sum()),
        }

    non_sig = (~(stats['p_value_t1'] < PVAL_NOMINAL)) & (~(stats['p_value_t2'] < PVAL_NOMINAL))
    return {
        'evading':    _counts_for(evading),
        'triggering': _counts_for(triggering),
        'non_sig':    int(non_sig.sum()),
        'total':      len(stats),
    }


# ─── Plotting ──────────────────────────────────────────────────────────────────

def plot_enrichment_summary(counts: dict):
    """Draw a grouped bar chart with 3 bars per direction + 1 non-sig bar.

    Bars 1 and 2 within each direction are stacked (nominal-only + FDR).
    Bar 3 is stacked (nominal-only + combined FDR).

    Args:
        counts: Output of build_counts().

    Returns:
        matplotlib Figure.
    """
    total = counts['total']
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.55
    bar_gap    = 0.12   # gap between bars within a group (horizontal spacing)
    group_gap  = 1.0    # gap between direction groups

    step = bar_width + bar_gap
    xs_evading    = [0, step, 2 * step]
    x_nonsig      = xs_evading[-1] + bar_width + group_gap
    xs_triggering = [x_nonsig + bar_width + group_gap + i * step for i in range(3)]

    def _stacked_bar(ax, x, nominal_only, nominal_color, fdr_val, fdr_color):
        """Draw a two-segment vertical stacked bar; annotate counts."""
        if nominal_only > 0:
            ax.bar(x, nominal_only, width=bar_width, bottom=0,
                    color=nominal_color, edgecolor='white', linewidth=0.8)
        if fdr_val > 0:
            ax.bar(x, fdr_val, width=bar_width, bottom=nominal_only,
                    color=fdr_color, edgecolor='white', linewidth=0.8)
        total_h = nominal_only + fdr_val
        if total_h > 0:
            ax.text(x, total_h / 2, str(total_h),
                    ha='center', va='center', fontsize=10, fontweight='bold', color='black')
            ax.text(x, total_h + max(total * 0.005, 0.3), f'{fdr_val}/{total_h}',
                    ha='center', va='bottom', fontsize=8, color=fdr_color, fontweight='bold')

    def _solid_bar(ax, x, height, color):
        """Draw a single solid vertical bar (for combined FDR)."""
        if height > 0:
            ax.bar(x, height, width=bar_width, color=color,
                    edgecolor='white', linewidth=0.8)
            ax.text(x, height + max(total * 0.005, 0.3), str(height),
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    # ── NMD-evading bars (vertical) ──────────────────────────────────────
    ev = counts['evading']
    _stacked_bar(ax, xs_evading[2], ev['t1_nominal_only'], EVADING_NOMINAL_COLOR,
                  ev['t1_fdr'], EVADING_FDR_COLOR)
    _stacked_bar(ax, xs_evading[1], ev.get('t2_nominal_only', 0), EVADING_NOMINAL_COLOR,
                  ev.get('t2_fdr', 0), EVADING_FDR_COLOR)
    _solid_bar(ax, xs_evading[0], ev.get('both_fdr', 0), BOTH_FDR_EVADING_COLOR)

    # ── Non-significant bar (vertical) ───────────────────────────────────
    ns = counts['non_sig']
    if ns > 0:
        ax.bar(x_nonsig, ns, width=bar_width, color=NON_SIG_COLOR,
            edgecolor='white', linewidth=0.8)
        ax.text(x_nonsig, ns + max(total * 0.005, 0.3), str(ns),
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
        
    # ── NMD-triggering bars (vertical) ──────────────────────────────────
    tr = counts['triggering']
    _stacked_bar(ax, xs_triggering[0], tr['t1_nominal_only'], TRIGGERING_NOMINAL_COLOR,
                  tr['t1_fdr'], TRIGGERING_FDR_COLOR)
    _stacked_bar(ax, xs_triggering[1], tr.get('t2_nominal_only', 0), TRIGGERING_NOMINAL_COLOR,
                  tr.get('t2_fdr', 0), TRIGGERING_FDR_COLOR)
    _solid_bar(ax, xs_triggering[2], tr.get('both_fdr', 0), BOTH_FDR_TRIGGERING_COLOR)

    # ── Group bracket annotations (drawn above bars) ─────────────
    y_max    = ax.get_ylim()[1]
    y_br     = y_max + max(total * 0.01, 0.1)
    tick_h   = max(total * 0.018, 1.0)

    def _group_bracket(xs, label, color):
        x0 = xs[0] - bar_width / 2
        x1 = xs[-1] + bar_width / 2
        xm = (x0 + x1) / 2
        ax.plot([x0, x1], [y_br, y_br],             color=color, linewidth=1.5, clip_on=False)
        ax.plot([x0, x0], [y_br, y_br - tick_h],    color=color, linewidth=1.5, clip_on=False)
        ax.plot([x1, x1], [y_br, y_br - tick_h],    color=color, linewidth=1.5, clip_on=False)
        ax.text(xm, y_br - tick_h * 1.1, label,
                ha='center', va='top', fontsize=11, fontweight='bold', color=color, clip_on=False)

    # ── x-tick labels (categories horizontally) ───────────────────────────────
    tick_xs     = xs_evading + [x_nonsig] + xs_triggering
    tick_labels = [
        'Pass both FDR\n(T1 & T2)', 'Test 2\np<0.05', 'Test 1\np<0.05',
        'Non-sig',
        'Test 1\np<0.05', 'Test 2\np<0.05', 'Pass both FDR\n(T1 & T2)',
    ]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels(tick_labels, fontsize=14, rotation=45, ha='right')

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=EVADING_NOMINAL_COLOR,     label='Evading – nominal only (p<0.05)'),
        mpatches.Patch(color=EVADING_FDR_COLOR,         label='Evading – pass T1/T2 FDR (0.01/0.25)'),
        mpatches.Patch(color=BOTH_FDR_EVADING_COLOR,    label='Evading – pass both FDR (T1 & T2)'),
        mpatches.Patch(color=TRIGGERING_NOMINAL_COLOR,  label='Triggering – nominal only (p<0.05)'),
        mpatches.Patch(color=TRIGGERING_FDR_COLOR,      label='Triggering – pass T1/T2 FDR (0.01/0.25)'),
        mpatches.Patch(color=BOTH_FDR_TRIGGERING_COLOR, label='Triggering – pass both FDR (T1 & T2)'),
        mpatches.Patch(color=NON_SIG_COLOR,             label='Non-significant'),
    ]
    ax.legend(handles=legend_handles, fontsize=12, loc='upper center',
              frameon=True, edgecolor='#cccccc', ncol=3, bbox_to_anchor=(0.5, -0.35))

    # ── Axes style ───────────────────────────────────────────────────────────
    ax.set_ylabel('Number of genes (total tested n={})'.format(total), fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(xs_evading[0] - bar_width, xs_triggering[-1] + bar_width)
    plt.subplots_adjust(bottom=0.25, top=0.85, left=0.1, right=0.95)
    plt.title("Significance levels of NMD category enrichment in disease genes", fontsize=18, fontweight='bold')

    # Now that bars are drawn, compute bracket y position and draw group brackets
    y_max = ax.get_ylim()[1]
    y_br  = y_max + max(total * 0.01, 0.1)
    _group_bracket(xs_evading,    'NMD evading\nenriched',    EVADING_FDR_COLOR)
    _group_bracket(xs_triggering, 'NMD triggering\nenriched', TRIGGERING_FDR_COLOR)

    # Expand ylim to make room for bracket/labels
    ax.set_ylim(top=y_br + max(total * 0.01, 0.5))

    # LOEUF colorbar (not applicable here, but keeping placeholder)
    # Since this plot doesn't use LOEUF, remove colorbar

    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(figure_label=None, figure_number=None, regenerate=True):
    """Main execution function."""
    paths = get_paths(SCRIPT_NAME, figure_label=figure_label, figure_number=figure_number)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    if paths.source_data:
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting disease gene NMD enrichment summary...")

    stats = load_stats()
    counts = build_counts(stats)
    total = counts['total']

    logger.info(f"Total disease genes tested: {total}")
    for direction in ('evading', 'triggering'):
        logger.info(f"  {direction}:")
        for k, v in counts[direction].items():
            logger.info(f"    {k}: {v} ({v / total * 100:.1f}%)")
    logger.info(f"  non_sig: {counts['non_sig']} ({counts['non_sig'] / total * 100:.1f}%)")

    # Save source data
    if paths.source_data:
        rows = []
        for direction in ('evading', 'triggering'):
            for k, v in counts[direction].items():
                rows.append({'group': direction, 'category': k, 'count': v,
                             'pct': round(v / total * 100, 2)})
        rows.append({'group': 'non_sig', 'category': 'non_sig',
                     'count': counts['non_sig'],
                     'pct': round(counts['non_sig'] / total * 100, 2)})
        pd.DataFrame(rows).to_csv(paths.source_data, index=False)
        logger.info(f"Source data saved to {paths.source_data}")

    fig = plot_enrichment_summary(counts)
    fig.savefig(paths.figure_png, dpi=300, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Figure saved to {paths.figure_png}")
    logger.info(f"PDF saved to {paths.figure_pdf}")
    plt.close(fig)

    logger.info("Done.")


if __name__ == "__main__":
    main()
