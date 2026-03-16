"""
Reinitiation Kozak similarity correlation (Fig 6g).

Scatter plot of Kozak motif similarity score vs mean AUG-driven RNA
stabilisation for each AUG position along the BRCA1 5' region, showing
that AUGs embedded in stronger Kozak contexts drive greater transcript
stabilisation via translation reinitiation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from loguru import logger

from NMD.config import RAW_DATA_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "reinit_kozak"

# ── Paths ─────────────────────────────────────────────────────────────────────
BRCA1_SPR_FILE = RAW_DATA_DIR / "DMS" / "BRCA1_SPR.csv"

# ── Visual ────────────────────────────────────────────────────────────────────
FIGURE_SIZE = (5, 5)
DPI = 300
POINT_COLOR = "#022778"
LINE_COLOR = "#ff9e9d"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load BRCA1 SPR data and extract unique AUG-position summaries.

    Keeps one row per AUG position with the mean AUG-driven fitness enrichment
    (fitness_per_AUG) and Kozak similarity score.

    Returns:
        DataFrame with columns: AUG, fitness_per_AUG, kozak_similarity.
    """
    logger.info(f"Loading BRCA1 SPR data from {BRCA1_SPR_FILE}")
    df = pd.read_csv(BRCA1_SPR_FILE, index_col=0)

    # Keep only UGA variants with a computed fitness_per_AUG
    df = df[df["fitness_per_AUG"].notna()].copy()

    # Deduplicate to one row per AUG position
    df = df.drop_duplicates(subset="AUG").copy()
    df = df[["AUG", "fitness_per_AUG", "kozak_similarity"]].copy()
    df = df.dropna(subset=["kozak_similarity"])

    logger.info(f"Unique AUG positions with data: {len(df)}")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_kozak(data: pd.DataFrame):
    """Draw scatter plot of Kozak similarity vs mean AUG-driven stabilisation.

    Args:
        data: DataFrame with 'kozak_similarity' and 'fitness_per_AUG'.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.scatter(
        data["kozak_similarity"],
        data["fitness_per_AUG"],
        color=POINT_COLOR,
        s=40,
        alpha=0.7,
        edgecolors="none",
    )

    # Linear regression
    r, p = scipy_stats.pearsonr(data["kozak_similarity"], data["fitness_per_AUG"])
    slope, intercept = np.polyfit(data["kozak_similarity"], data["fitness_per_AUG"], 1)
    x_range = np.linspace(data["kozak_similarity"].min(), data["kozak_similarity"].max(), 100)
    ax.plot(x_range, slope * x_range + intercept, color=LINE_COLOR, linewidth=2.5)

    ax.text(
        0.03, 0.95, f"R = {r:.2f}, p = {p:.1e}",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_title("Kozak context and AUG-driven\ntranscript stabilisation", fontsize=14, fontweight="bold")
    ax.set_xlabel("Kozak similarity", fontsize=13, fontweight="bold")
    ax.set_ylabel("AUG-driven RNA stabilisation", fontsize=13, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
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
    """Generate the Kozak similarity correlation figure."""
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
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(paths.source_data, index=False)
        logger.info(f"Saved source data to {paths.source_data}")

    fig = plot_kozak(data)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("Reinitiation Kozak correlation complete!")


if __name__ == "__main__":
    main()
