"""
Supplementary: Correlation of log-odds scores for top gnomAD genes.

Scatter plot of log_odds_t1 vs log_odds_t2, colored by n_variants (capped).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from loguru import logger

from NMD.config import MANUSCRIPT_TABLES_DIR, MANUSCRIPT_SUPPLEMENTARY_FIGURES_DIR

# =============================
# CONFIGURATION
# =============================
INPUT_TABLE = MANUSCRIPT_TABLES_DIR / "gnomad_top_genes.csv"
OUTPUT_FIGURE = MANUSCRIPT_SUPPLEMENTARY_FIGURES_DIR / "gnomad_logodds_correlation.png"
OUTPUT_FIGURE_PDF = MANUSCRIPT_SUPPLEMENTARY_FIGURES_DIR / "gnomad_logodds_correlation.pdf"
PLOT_TITLE = "Correlation of log-odds scores"
MAX_PTC_COLOUR = 50
FIGURE_SIZE = (7, 6)
DPI = 300

# =============================
# DATA PROCESSING
# =============================
def process_data():
    logger.info(f"Loading gene table from {INPUT_TABLE}")
    df = pd.read_csv(INPUT_TABLE)
    logger.info(f"Loaded {len(df)} genes")
    return df


# =============================
# PLOTTING
# =============================
def plot_from_table(df):
    logger.info("Generating 2-panel plot: scatter + Pearson r barplot...")
    MIN_PTC_THRESHOLDS = [10, 15, 20, 25, 30]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1.2, 1]})
    ax_scatter, ax_bar = axes

    # --- Left panel: Scatter plot ---
    all_vals = pd.concat([df["log_odds_t1"], df["log_odds_t2"]])
    min_val = np.nanmin(all_vals)
    max_val = np.nanmax(all_vals)
    pad = 0.05 * (max_val - min_val)
    lims = (min_val - pad, max_val + pad)

    sc = ax_scatter.scatter(
        df["log_odds_t1"],
        df["log_odds_t2"],
        c=df["n_variants"].clip(upper=MAX_PTC_COLOUR),
        cmap="viridis",
        s=60,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5
    )
    ax_scatter.plot(lims, lims, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
    cbar = plt.colorbar(sc, ax=ax_scatter, pad=0.02)
    cbar.set_label(f"# PTCs", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    corr = np.corrcoef(df["log_odds_t1"], df["log_odds_t2"])[0, 1]
    n_genes = len(df)
    ax_scatter.text(0.98, 0.02, f"Pearson r = {corr:.2f}\nN = {n_genes}",
            ha="right", va="bottom", fontsize=13, fontweight="bold",
            transform=ax_scatter.transAxes, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # Manuscript style: no plot title, no axis titles
    ax_scatter.set_xlabel("log OR (Test 1)", fontsize=15, fontweight="bold")
    ax_scatter.set_ylabel("log OR (Test 2)", fontsize=15, fontweight="bold")
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.tick_params(labelsize=12)
    ax_scatter.grid(True, alpha=0.25, linestyle=":")
    ax_scatter.set_title(PLOT_TITLE, fontsize=14, fontweight="bold")

    # --- Right panel: Barplot of Pearson r for MIN_PTC thresholds ---
    r_vals = []
    n_vals = []
    for min_ptc in MIN_PTC_THRESHOLDS:
        df_filt = df[df["n_variants"] >= min_ptc]
        if len(df_filt) > 1:
            r = np.corrcoef(df_filt["log_odds_t1"], df_filt["log_odds_t2"])[0, 1]
        else:
            r = np.nan
        r_vals.append(r)
        n_vals.append(len(df_filt))

    bars = ax_bar.bar(range(len(MIN_PTC_THRESHOLDS)), r_vals, color="#4F8FC5", edgecolor="black", alpha=0.85)
    # Annotate bars with N
    for i, (bar, n) in enumerate(zip(bars, n_vals)):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"N={n}",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax_bar.set_xticks(range(len(MIN_PTC_THRESHOLDS)))
    ax_bar.set_xticklabels([f">= {t}" for t in MIN_PTC_THRESHOLDS], fontsize=13)
    ax_bar.set_ylabel("Pearson r", fontsize=15, fontweight="bold")
    ax_bar.set_title("Correlation by min # PTCs", fontsize=14, fontweight="bold")
    
    # x axis label
    ax_bar.set_xlabel("Minimum # PTCs", fontsize=15, fontweight="bold")

    ax_bar.set_ylim(0, 1.05)
    ax_bar.tick_params(labelsize=12)
    ax_bar.grid(axis="y", alpha=0.25, linestyle=":")

    # Remove top/right spines for both panels
    for ax in [ax_scatter, ax_bar]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=2.5)

    # Save
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")
    logger.info(f"Saved figure to {OUTPUT_FIGURE}")
    logger.info(f"Saved PDF to {OUTPUT_FIGURE_PDF}")
    plt.close(fig)

# =============================
# MAIN
# =============================
def main():
    logger.info("=" * 70)
    logger.info("gnomAD log-odds correlation supplementary plot")
    logger.info("=" * 70)
    df = process_data()
    plot_from_table(df)
    logger.success("Supplementary log-odds correlation plot complete!")

if __name__ == "__main__":
    main()
