"""
Start-proximal PTC position curves with AUGs (Supplementary, Fig 6d).

Line + point plot of DMS-measured RNA levels (fitness) as a function of
PTC position for UGA-type BRCA1 variants that possess a downstream AUG,
with separate lines coloured by intercistronic distance (PTC-to-AUG).
Error bars show DiMSum-estimated standard errors.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from loguru import logger

from NMD.config import RAW_DATA_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "SPreinit_ptc_curves_aug"

# ── Paths ─────────────────────────────────────────────────────────────────────
BRCA1_SPR_FILE = RAW_DATA_DIR / "DMS" / "BRCA1_SPR.csv"

# ── Visual ────────────────────────────────────────────────────────────────────
FIGURE_SIZE = (10, 6)
DPI = 300
PTC_AUG_DISTANCES = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load BRCA1 SPR data filtered for UGA variants with specific PTC-AUG distances."""
    logger.info(f"Loading BRCA1 SPR data from {BRCA1_SPR_FILE}")
    df = pd.read_csv(BRCA1_SPR_FILE, index_col=0)
    df = df[
        (df["stop_type"] == "UGA")
        & (df["PTC_AUG_dist"].isin(PTC_AUG_DISTANCES))
    ].copy()
    logger.info(f"Filtered to {len(df)} UGA variants with AUGs")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ptc_curves_aug(data: pd.DataFrame):
    """Draw line+point plot of RNA levels vs PTC position coloured by PTC-AUG distance.

    Args:
        data: DataFrame with 'uORF_length', 'fitness', 'sigma', 'PTC_AUG_dist'.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Continuous colour map for PTC-AUG distance
    unique_dists = sorted(data["PTC_AUG_dist"].unique())
    norm = mcolors.Normalize(vmin=min(unique_dists), vmax=max(unique_dists))
    cmap = cm.viridis

    for dist in unique_dists:
        sub = data[data["PTC_AUG_dist"] == dist].sort_values("uORF_length")
        if len(sub) == 0:
            continue
        color = cmap(norm(dist))
        ax.errorbar(
            sub["uORF_length"],
            sub["fitness"],
            yerr=sub["sigma"],
            fmt="o-",
            color=color,
            markersize=4,
            linewidth=1.5,
            capsize=2,
            alpha=0.85,
        )

    # WT reference line at 0
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Intercistronic distance (nt)", fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)

    ax.set_xlabel("PTC position (codons from 5' end)", fontsize=16, fontweight="bold")
    ax.set_ylabel("RNA levels (fitness)", fontsize=16, fontweight="bold")
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
    """Generate the PTC position with AUGs supplementary figure."""
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
            ["uORF_length", "PTC_AUG_dist", "fitness", "sigma"]
        ].copy()
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(paths.source_data, index=False)
        logger.info(f"Saved source data to {paths.source_data}")

    fig = plot_ptc_curves_aug(data)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("SPreinit PTC curves (with AUGs) complete!")


if __name__ == "__main__":
    main()
