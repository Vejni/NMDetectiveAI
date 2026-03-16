"""
Reinitiation boxplot: AUG vs non-AUG RNA levels (Fig 6e).

Boxplot comparing DMS-measured RNA levels (fitness) between UGA-PTC variants
with and without a downstream AUG methionine, demonstrating that translation
reinitiation through downstream AUGs stabilises transcripts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from loguru import logger

from NMD.config import RAW_DATA_DIR, CONTRASTING_2_COLOURS
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "reinit_boxplot"

# ── Paths ─────────────────────────────────────────────────────────────────────
BRCA1_SPR_FILE = RAW_DATA_DIR / "DMS" / "BRCA1_SPR.csv"

# ── Visual ────────────────────────────────────────────────────────────────────
FIGURE_SIZE = (4, 7)
DPI = 300

BOX_COLORS = {
    "no": CONTRASTING_2_COLOURS[1],   # pink
    "yes": CONTRASTING_2_COLOURS[0],  # dark blue
}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load BRCA1 SPR data and filter for UGA variants."""
    logger.info(f"Loading BRCA1 SPR data from {BRCA1_SPR_FILE}")
    df = pd.read_csv(BRCA1_SPR_FILE, index_col=0)
    df = df[df["stop_type"] == "UGA"].copy()
    logger.info(f"Filtered to {len(df)} UGA variants")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_boxplot(data: pd.DataFrame):
    """Draw boxplot of RNA levels for AUG vs non-AUG UGA variants.

    Args:
        data: Filtered UGA-only DataFrame with 'AUG_presence' and 'fitness'.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 3))

    groups = ["no", "yes"]
    group_data = [data.loc[data["AUG_presence"] == g, "fitness"].values for g in groups]

    bp = ax.boxplot(
        group_data,
        positions=[0, 1],
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        vert=False,
        flierprops=dict(marker="o", markersize=3, alpha=0.3),
    )

    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor(BOX_COLORS[group])
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # t-test annotation
    t_stat, p_val = scipy_stats.ttest_ind(group_data[1], group_data[0])
    x_max = max(d.max() for d in group_data if len(d) > 0)
    x_ann = x_max + 0.15
    ax.plot([x_ann - 0.05, x_ann, x_ann, x_ann - 0.05], [0, 0, 1, 1],
            color="black", linewidth=1.2)
    p_str = f"p = {p_val:.1e}" if p_val < 0.01 else f"p = {p_val:.3f}"
    ax.text(x_ann + 0.02, 0.5, p_str, ha="left", va="center",
            fontsize=13, fontweight="bold")

    ax.set_title("AUG downstream stabilisation of UGA-PTC transcripts", fontsize=16, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"], fontsize=15)
    ax.set_ylabel("AUG downstream", fontsize=16, fontweight="bold")
    ax.set_xlabel("RNA levels (fitness)", fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = False,
):
    """Generate the AUG vs non-AUG boxplot figure."""
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
        source = data[["AUG_presence", "fitness"]].copy()
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(paths.source_data, index=False)
        logger.info(f"Saved source data to {paths.source_data}")

    fig = plot_boxplot(data)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("Reinitiation boxplot complete!")


if __name__ == "__main__":
    main()
