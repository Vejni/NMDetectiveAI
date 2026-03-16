"""
PTC position effect on NMD prediction error (Fig 8l).

Boxplot of the loess-model residual (observed - predicted RNA levels),
grouped by PTC position (codon number). Red dashed line at zero indicates
no prediction error. Only NMD-sensitive variants (fitness_gene_specific < 0)
are shown.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from NMD.config import RAW_DATA_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "context_ptc_position"

# ── Paths ─────────────────────────────────────────────────────────────────────
GENES_139_FILE = RAW_DATA_DIR / "DMS" / "genes_139.csv"

# ── Visual ────────────────────────────────────────────────────────────────────
FIGURE_SIZE = (7, 5)
DPI = 300
BOX_COLOR = "#b3b3b3"
TREND_COLOR = "#022778"
Y_LIM = (-1.5, 2.0)
ROLLING_WINDOW = 7   # positions for the smoothed median trend


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load genes_139 data and filter for NMD-sensitive variants."""
    logger.info(f"Loading genes_139 data from {GENES_139_FILE}")
    df = pd.read_csv(GENES_139_FILE, index_col=0, low_memory=False)

    # Keep only NMD-sensitive variants (RNA levels below WT)
    df = df[df["fitness_gene_specific"] < 0].copy()
    # Compute residual
    df["residual"] = df["fitness_gene_specific"] - df["predicted"]

    logger.info(f"Filtered to {len(df)} NMD-sensitive variants")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ptc_position(data: pd.DataFrame):
    """Boxplot of loess residuals grouped by PTC position.

    Args:
        data: DataFrame with 'PTCposition' and 'residual' columns.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Get sorted unique positions
    positions = sorted(data["PTCposition"].dropna().unique())
    positions = [int(p) for p in positions if p > 0]  # exclude WT (0)

    # Prepare boxplot data
    box_data = [
        data.loc[data["PTCposition"] == pos, "residual"].dropna().values
        for pos in positions
    ]

    # Scale box widths by sqrt(n) to reflect reliability, normalised to max
    ns = np.array([len(d) for d in box_data], dtype=float)
    widths = 0.3 + 0.5 * (np.sqrt(ns) / np.sqrt(ns.max()))

    bp = ax.boxplot(
        box_data,
        positions=range(len(positions)),
        widths=widths,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        boxprops=dict(linewidth=0.8),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX_COLOR)
        patch.set_alpha(0.85)

    # Rolling-median trend line across box medians
    medians = np.array([np.median(d) for d in box_data])
    trend = (
        pd.Series(medians)
        .rolling(window=ROLLING_WINDOW, center=True, min_periods=3)
        .median()
    )
    ax.plot(
        range(len(positions)), trend,
        color=TREND_COLOR, linewidth=2, zorder=3,
        label=f"Rolling median (w={ROLLING_WINDOW})",
    )

    # Red dashed reference line at 0
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)

    # X-axis labels – show every 5th position to avoid crowding
    ax.set_xticks(range(len(positions)))
    tick_labels = [
        str(p) if (idx % 5 == 0) else ""
        for idx, p in enumerate(positions)
    ]
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8, va="top")

    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

    ax.set_title("PTC position effect on NMD prediction error", fontsize=11, fontweight="bold")
    ax.set_ylim(Y_LIM)
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_xlabel("PTC position (codon)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observed − Predicted\n(RNA levels)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
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
    """Generate the PTC position effect figure."""
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
        source = data[["gene", "PTCposition", "residual"]].copy()
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(paths.source_data, index=False)
        logger.info(f"Saved source data to {paths.source_data}")

    fig = plot_ptc_position(data)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("Context PTC position plot complete!")


if __name__ == "__main__":
    main()
