"""
Reinitiation intercistronic distance plot (Fig 6f).

Scatter/boxplot of AUG-driven RNA stabilisation as a function of the
intercistronic distance (PTC-to-downstream-AUG distance in codons), coloured
by upstream ORF length (uORF length / PTC position).  Shows that longer
intercistronic distances and longer uORFs lead to greater transcript
stabilisation through translation reinitiation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from scipy import stats as scipy_stats
from loguru import logger

from NMD.config import RAW_DATA_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "reinit_intercistronic"

# ── Paths ─────────────────────────────────────────────────────────────────────
BRCA1_SPR_FILE = RAW_DATA_DIR / "DMS" / "BRCA1_SPR.csv"

# ── Filter ────────────────────────────────────────────────────────────────────
INTERCISTRONIC_DISTANCES = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80]

# ── Visual ────────────────────────────────────────────────────────────────────
FIGURE_SIZE = (10, 4)
DPI = 300
CMAP = plt.cm.viridis


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load BRCA1 SPR data, filter for UGA with specific intercistronic distances."""
    logger.info(f"Loading BRCA1 SPR data from {BRCA1_SPR_FILE}")
    df = pd.read_csv(BRCA1_SPR_FILE, index_col=0)
    df = df[
        (df["stop_type"] == "UGA")
        & (df["PTC_AUG_dist"].isin(INTERCISTRONIC_DISTANCES))
    ].copy()
    logger.info(f"Filtered to {len(df)} UGA variants with intercistronic distances "
                f"{INTERCISTRONIC_DISTANCES}")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_intercistronic(data: pd.DataFrame):
    """Draw scatter/boxplot of AUG-driven stabilisation by intercistronic distance.

    Args:
        data: Filtered DataFrame with 'PTC_AUG_dist', 'AUG_driven_fitness_enrichment',
              and 'uORF_length'.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Colormap for uORF length
    uorf_min = data["uORF_length"].min()
    uorf_max = data["uORF_length"].max()
    norm = mcolors.Normalize(vmin=uorf_min, vmax=uorf_max)

    # Boxplots at each intercistronic distance
    positions = sorted(data["PTC_AUG_dist"].unique())
    box_data = [
        data.loc[data["PTC_AUG_dist"] == d, "AUG_driven_fitness_enrichment"].dropna().values
        for d in positions
    ]
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=3,
        patch_artist=True,
        showfliers=False,
        zorder=1,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("gray")
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # Jittered scatter coloured by uORF length
    jitter = np.random.default_rng(42).uniform(-1.5, 1.5, size=len(data))
    sc = ax.scatter(
        data["PTC_AUG_dist"] + jitter,
        data["AUG_driven_fitness_enrichment"],
        c=data["uORF_length"],
        cmap=CMAP,
        norm=norm,
        s=25,
        alpha=0.7,
        edgecolors="none",
        zorder=2,
    )

    # Linear regression + Pearson r
    valid = data.dropna(subset=["AUG_driven_fitness_enrichment"])
    r, p = scipy_stats.pearsonr(valid["PTC_AUG_dist"], valid["AUG_driven_fitness_enrichment"])
    slope, intercept = np.polyfit(valid["PTC_AUG_dist"], valid["AUG_driven_fitness_enrichment"], 1)
    x_line = np.array([min(positions), max(positions)])
    ax.plot(x_line, slope * x_line + intercept, color="#ff9e9d", linewidth=2.5, zorder=3)

    ax.text(
        0.03, 0.95, f"R = {r:.2f}, p = {p:.1e}",
        transform=ax.transAxes, fontsize=14, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_title("Intercistronic distance and AUG-driven transcript stabilisation", fontsize=16, fontweight="bold")
    ax.set_xlabel("Intercistronic distance (codons)", fontsize=16, fontweight="bold")
    ax.set_ylabel("AUG-driven RNA stabilisation", fontsize=16, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(d) for d in positions], fontsize=12)
    ax.tick_params(axis="y", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.6, 1.7)

    # Colorbar for uORF length
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.30])
    cb = mcolorbar.ColorbarBase(cbar_ax, cmap=CMAP, norm=norm, orientation="vertical")
    cb.set_label("uORF length (codons)", fontsize=13, fontweight="bold")
    cb.ax.tick_params(labelsize=11)

    plt.subplots_adjust(right=0.85)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = False,
):
    """Generate the intercistronic distance stabilisation figure."""
    paths = get_paths(
        script_name=SCRIPT_NAME,
        figure_label=figure_label,
        figure_number=figure_number,
    )

    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        data = pd.read_csv(paths.source_data)
    else:
        data = load_data()
        source = data[["uORF_length", "PTC_AUG_dist", "AUG_driven_fitness_enrichment"]].copy()
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(paths.source_data, index=False)
        logger.info(f"Saved source data to {paths.source_data}")

    fig = plot_intercistronic(data)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("Reinitiation intercistronic distance plot complete!")


if __name__ == "__main__":
    main()
