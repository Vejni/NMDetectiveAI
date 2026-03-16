"""
Hexamer context effect on RNA levels (Fig 7h).

Faceted heatmap showing regression coefficients for each mutation at each
downstream hexamer position (+1, +2, +3) for three hexamers. Red indicates
lower RNA levels, blue indicates higher RNA levels relative to the reference
nucleotide, and white indicates no effect.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from loguru import logger

from NMD.config import RAW_DATA_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "context_hexamers"

# ── Paths ─────────────────────────────────────────────────────────────────────
HEXAMERS_CSV = RAW_DATA_DIR / "DMS" / "hexamers.csv"

# ── Visual ────────────────────────────────────────────────────────────────────
FIGURE_SIZE = (9, 3.5)
DPI = 300
FONT_SIZE = 16
TICK_SIZE = 13
LEGEND_TEXT_SIZE = 12
LEGEND_TITLE_SIZE = 14
CMAP = "RdBu"  # red (low/negative) → white (mid/0) → blue (high/positive)

MUTATIONS_ORDER = ["A", "C", "G", "T"]
POSITIONS_ORDER = ["+1_A", "+2_A", "+3_G"]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load hexamers data and filter out intercept rows (+NA_NA position).

    Returns:
        DataFrame with columns: hexamer, mutation, position_after_TGA, coeff.
    """
    logger.info(f"Loading hexamers data from {HEXAMERS_CSV}")
    df = pd.read_csv(HEXAMERS_CSV)
    df = df[df["position_after_TGA"] != "+NA_NA"].copy()
    logger.info(f"Loaded {len(df)} hexamer rows after filtering")
    return df


def _make_pivot(df: pd.DataFrame, hexamer: int) -> pd.DataFrame:
    """Pivot one hexamer's data into a mutation × position matrix.

    Args:
        df: Full filtered hexamers DataFrame.
        hexamer: Integer hexamer id (1, 2, or 3).

    Returns:
        DataFrame with MUTATIONS_ORDER rows and POSITIONS_ORDER columns,
        containing coeff values (NaN where the mutation matches the reference).
    """
    sub = df[df["hexamer"] == hexamer]
    pivot = sub.pivot_table(
        index="mutation",
        columns="position_after_TGA",
        values="coeff",
        aggfunc="first",
    )
    return pivot.reindex(index=MUTATIONS_ORDER, columns=POSITIONS_ORDER)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_hexamers(data: pd.DataFrame):
    """Faceted heatmap of hexamer regression coefficients.

    Each facet corresponds to one hexamer (1–3). Tiles are coloured by the
    regression coefficient (coeff): red = lower RNA levels; blue = higher.
    Reference-nucleotide cells (no mutation possible) are left grey.

    Args:
        data: DataFrame with columns hexamer, mutation, position_after_TGA, coeff.

    Returns:
        matplotlib Figure.
    """
    hexamers = sorted(data["hexamer"].unique())
    n = len(hexamers)

    fig, axes = plt.subplots(1, n, figsize=FIGURE_SIZE)
    if n == 1:
        axes = [axes]

    # Symmetric color range centred on 0
    abs_max = data["coeff"].abs().max()
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    # Custom colormap with specified colors
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_diverging", ['#022778', '#ffffff', '#ff9e9d'])

    for ax, hexamer in zip(axes, hexamers):
        pivot = _make_pivot(data, hexamer)
        mat = pivot.values  # shape: (4, 3)
        n_rows, n_cols = mat.shape

        for ri in range(n_rows):
            for ci in range(n_cols):
                val = mat[ri, ci]
                if np.isnan(val):
                    color = "#d0d0d0"  # grey for reference (missing) cells
                else:
                    color = cmap(norm(val))
                patch = plt.Rectangle(
                    (ci, n_rows - ri - 1), 1, 1,
                    facecolor=color, edgecolor="white", linewidth=0.8,
                )
                ax.add_patch(patch)

        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)

        # X ticks: position labels at cell centres
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels(POSITIONS_ORDER, fontsize=TICK_SIZE - 1, rotation=30, ha="right")

        # Y ticks: mutation labels (reversed so A is on top)
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels(list(reversed(MUTATIONS_ORDER)), fontsize=TICK_SIZE)

        ax.set_title(f"Hexamer {hexamer}", fontsize=FONT_SIZE, fontweight="bold", pad=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Shared y-axis label on leftmost subplot
    axes[0].set_ylabel("Mutation", fontsize=FONT_SIZE)

    # Shared colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="top", shrink=0.45, pad=0.15, aspect=25)
    cbar.set_label("RNA levels", fontsize=LEGEND_TITLE_SIZE)
    cbar.ax.tick_params(labelsize=LEGEND_TEXT_SIZE)

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = False,
):
    """Generate the hexamer context heatmap figure."""
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

    fig = plot_hexamers(data)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("Context hexamers plot complete!")


if __name__ == "__main__":
    main()
