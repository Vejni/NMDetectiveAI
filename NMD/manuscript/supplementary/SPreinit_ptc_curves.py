"""
Start-proximal PTC position curves by stop type (Supplementary, Fig 6c).

Line + point plot of DMS-measured RNA levels (fitness) as a function of
PTC position (in codons from 5'-end) for non-AUG BRCA1 variants, with
separate lines/colours for UGA, UAG, and UAA stop types.  Error bars
show DiMSum-estimated standard errors across replicates.
"""

import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from NMD.config import RAW_DATA_DIR, CONTRASTING_3_COLOURS
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "SPreinit_ptc_curves"

# ── Paths ─────────────────────────────────────────────────────────────────────
BRCA1_SPR_FILE = RAW_DATA_DIR / "DMS" / "BRCA1_SPR.csv"

# ── Visual ────────────────────────────────────────────────────────────────────
FIGURE_SIZE = (10, 6)
DPI = 300

STOP_COLORS = {
    "UGA": CONTRASTING_3_COLOURS[2],  # dark blue
    "UAG": CONTRASTING_3_COLOURS[1],  # green
    "UAA": CONTRASTING_3_COLOURS[0],  # pink
}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load BRCA1 SPR data and filter for non-AUG, non-WT variants."""
    logger.info(f"Loading BRCA1 SPR data from {BRCA1_SPR_FILE}")
    df = pd.read_csv(BRCA1_SPR_FILE, index_col=0)
    # Keep non-AUG variants (AUG==1 means no extra downstream AUG) and exclude WT
    df = df[(df["AUG"] == 1) & (df["WT"] != True)].copy()
    logger.info(f"Filtered to {len(df)} non-AUG, non-WT variants")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ptc_curves(data: pd.DataFrame):
    """Draw line+point plot of RNA levels vs PTC position coloured by stop type.

    Args:
        data: DataFrame with 'uORF_length', 'fitness', 'sigma', 'stop_type'.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for stop_type, color in STOP_COLORS.items():
        sub = data[data["stop_type"] == stop_type].sort_values("uORF_length")
        if len(sub) == 0:
            continue
        ax.errorbar(
            sub["uORF_length"],
            sub["fitness"],
            yerr=sub["sigma"],
            fmt="o-",
            color=color,
            markersize=4,
            linewidth=1.5,
            capsize=2,
            label=stop_type,
            alpha=0.85,
        )

    # WT reference line at 0
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("PTC position (codons from 5' end)", fontsize=16, fontweight="bold")
    ax.set_ylabel("RNA levels (fitness)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=14, title="Stop type", title_fontsize=14, framealpha=0.9)
    ax.tick_params(axis="both", labelsize=13)
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
    """Generate the PTC position by stop type supplementary figure."""
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
        source = data[["uORF_length", "stop_type", "fitness", "sigma"]].copy()
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(paths.source_data, index=False)
        logger.info(f"Saved source data to {paths.source_data}")

    fig = plot_ptc_curves(data)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("SPreinit PTC curves (stop type) complete!")


if __name__ == "__main__":
    main()
