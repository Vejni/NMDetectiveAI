"""
Downstream trinucleotide effect on NMD prediction error (Supplementary, Fig 8j).

Boxplot of the loess-model residual (observed - predicted RNA levels),
grouped by the three nucleotides immediately downstream of the PTC. Red
dashed line at zero indicates no prediction error. Only NMD-sensitive
variants (fitness_gene_specific < 0) with non-empty downstream context
are shown.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from NMD.config import RAW_DATA_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "context_downstream_trinuc"

# ── Paths ─────────────────────────────────────────────────────────────────────
GENES_139_FILE = RAW_DATA_DIR / "DMS" / "genes_139.csv"

# ── Visual ────────────────────────────────────────────────────────────────────
FIGURE_SIZE = (14, 6)
DPI = 300
BOX_COLOR = "#b3b3b3"
Y_LIM = (-1.5, 2.0)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load genes_139 data and filter for NMD-sensitive variants with downstream trinuc info."""
    logger.info(f"Loading genes_139 data from {GENES_139_FILE}")
    df = pd.read_csv(GENES_139_FILE, index_col=0, low_memory=False)

    # Keep only NMD-sensitive variants (RNA levels below WT)
    df = df[df["fitness_gene_specific"] < 0].copy()
    # Drop rows without downstream trinucleotide info
    df = df.dropna(subset=["nt1to3_downPTC"])
    df = df[df["nt1to3_downPTC"].str.strip() != ""].copy()
    # Compute residual
    df["residual"] = df["fitness_gene_specific"] - df["predicted"]

    logger.info(
        f"Filtered to {len(df)} NMD-sensitive variants with downstream trinuc data"
    )
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_downstream_trinuc(data: pd.DataFrame):
    """Boxplot of loess residuals grouped by downstream trinucleotide.

    Args:
        data: DataFrame with 'nt1to3_downPTC' and 'residual' columns.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Sort trinucleotides alphabetically
    ordered_trinucs = sorted(data["nt1to3_downPTC"].unique())

    # Prepare boxplot data
    box_data = [
        data.loc[data["nt1to3_downPTC"] == trinuc, "residual"].dropna().values
        for trinuc in ordered_trinucs
    ]

    bp = ax.boxplot(
        box_data,
        positions=range(len(ordered_trinucs)),
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX_COLOR)

    # Red dashed reference line at 0
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)

    # X-axis labels
    ax.set_xticks(range(len(ordered_trinucs)))
    ax.set_xticklabels(ordered_trinucs, rotation=90, fontsize=9, va="top")

    ax.set_ylim(Y_LIM)
    ax.set_xlim(-0.5, len(ordered_trinucs) - 0.5)
    ax.set_xlabel(
        "Three nucleotides downstream of the PTC",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_ylabel("Observed − Predicted (RNA levels)", fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13)
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
    """Generate the downstream trinucleotide effect figure."""
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
        source = data[
            ["gene", "PTCposition", "nt1to3_downPTC", "residual"]
        ].copy()
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(paths.source_data, index=False)
        logger.info(f"Saved source data to {paths.source_data}")

    fig = plot_downstream_trinuc(data)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("Context downstream trinucleotide plot complete!")


if __name__ == "__main__":
    main()
