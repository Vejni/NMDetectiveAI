"""
DMS vs qPCR validation scatter plot (Fig 6e).

Pearson correlation of DMS-measured RNA levels against individual qPCR
measurements for 22 variants spanning the full NMD range.  The WT datapoint
is highlighted in blue.  Variant "21'" is excluded due to a failed qPCR
replicate (as in the original analysis).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from loguru import logger

from NMD.config import RAW_DATA_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "reinit_qpcr_validation"

# ── Paths ─────────────────────────────────────────────────────────────────────
QPCR_FILE = RAW_DATA_DIR / "DMS" / "qPCR_vs_DMS_val.csv"

# ── Visual ────────────────────────────────────────────────────────────────────
FIGURE_SIZE = (5, 5)
DPI = 300
POINT_COLOR = "#022778"
WT_COLOR = "#1a80c4"
LINE_COLOR = "#ff9e9d"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load qPCR vs DMS validation data, excluding the failed variant '21''.

    Returns:
        DataFrame with columns: new_name, DMS_fitness, mean_delta_CT, WT_factor.
    """
    logger.info(f"Loading qPCR validation data from {QPCR_FILE}")
    df = pd.read_csv(QPCR_FILE)
    df = df[df["new_name"] != "21'"].copy()
    df = df[["new_name", "DMS_fitness", "mean_delta_CT", "WT_factor"]].dropna()
    logger.info(f"Loaded {len(df)} variants for validation scatter")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_qpcr_validation(data: pd.DataFrame):
    """Draw scatter plot of DMS fitness vs qPCR measurement with WT highlighted.

    Args:
        data: DataFrame with 'mean_delta_CT', 'DMS_fitness', and 'WT_factor'.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    non_wt = data[data["WT_factor"] != "WT"]
    # wt = data[data["WT_factor"] == "WT"]

    # Only plot non-WT points
    ax.scatter(
        non_wt["mean_delta_CT"],
        non_wt["DMS_fitness"],
        color=POINT_COLOR,
        s=40,
        alpha=0.8,
        edgecolors="none",
        zorder=2,
    )
    # WT point and label are omitted

    # Linear regression line
    r, p = scipy_stats.pearsonr(data["mean_delta_CT"], data["DMS_fitness"])
    slope, intercept = np.polyfit(data["mean_delta_CT"], data["DMS_fitness"], 1)
    x_range = np.linspace(data["mean_delta_CT"].min(), data["mean_delta_CT"].max(), 100)
    ax.plot(x_range, slope * x_range + intercept, color=LINE_COLOR, linewidth=2.5, zorder=1)

    ax.text(
        0.03, 0.95, f"R = {r:.2f}, p = {p:.1e}",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_title("DMS vs individual qPCR measurements", fontsize=14, fontweight="bold")
    ax.set_xlabel("qPCR (−ΔCt)", fontsize=13, fontweight="bold")
    ax.set_ylabel("DMS (RNA levels)", fontsize=13, fontweight="bold")
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
    """Generate the DMS vs qPCR validation scatter figure."""
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

    fig = plot_qpcr_validation(data)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("DMS vs qPCR validation scatter complete!")


if __name__ == "__main__":
    main()
