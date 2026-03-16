"""
Upstream codon and amino acid effect on NMD prediction error (Fig 8k).

Boxplot of the loess-model residual (observed - predicted RNA levels),
grouped by the codon immediately upstream of the PTC, with amino acid
labels annotated below. Red dashed line at zero indicates no prediction
error. Only NMD-sensitive variants (fitness_gene_specific < 0) are shown.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Bio.Data.CodonTable import standard_dna_table
from loguru import logger

from NMD.config import RAW_DATA_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "context_upstream_codon"

# ── Paths ─────────────────────────────────────────────────────────────────────
GENES_139_FILE = RAW_DATA_DIR / "DMS" / "genes_139.csv"

# ── Visual ────────────────────────────────────────────────────────────────────
FIGURE_SIZE = (7, 5)
DPI = 300
Y_LIM = (-3, 2.0)

# ── Amino acid biochemical groups and colours ─────────────────────────────────
_AA_GROUPS: dict[str, str] = {
    # Nonpolar / hydrophobic
    "G": "Nonpolar", "A": "Nonpolar", "V": "Nonpolar", "L": "Nonpolar",
    "I": "Nonpolar", "P": "Nonpolar", "F": "Nonpolar", "W": "Nonpolar",
    "M": "Nonpolar",
    # Polar uncharged
    "S": "Polar", "T": "Polar", "N": "Polar", "Q": "Polar",
    "Y": "Polar", "C": "Polar",
    # Positively charged
    "K": "Positive", "R": "Positive", "H": "Positive",
    # Negatively charged
    "D": "Negative", "E": "Negative",
}
_GROUP_COLORS: dict[str, str] = {
    "Nonpolar": "#d4a96a",   # warm tan
    "Polar":    "#7ec8c8",   # teal
    "Positive": "#6a8fd4",   # blue
    "Negative": "#d46a6a",   # red
    "unknown":  "#b3b3b3",   # fallback grey
}

# ── Genetic code ──────────────────────────────────────────────────────────────
# forward_table maps all 61 sense codons → single-letter AA
_CODON_TO_AA: dict[str, str] = {
    codon.upper(): aa
    for codon, aa in standard_dna_table.forward_table.items()
}


def _codon_to_aa(codon: str) -> str:
    """Translate a DNA codon to its single-letter amino acid."""
    codon = codon.upper().replace("U", "T")
    return _CODON_TO_AA.get(codon, "?")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load genes_139 data and filter for NMD-sensitive variants with upstream codon info."""
    logger.info(f"Loading genes_139 data from {GENES_139_FILE}")
    df = pd.read_csv(GENES_139_FILE, index_col=0, low_memory=False)

    # Keep only NMD-sensitive variants (RNA levels below WT)
    df = df[df["fitness_gene_specific"] < 0].copy()
    # Drop rows without upstream codon info
    df = df.dropna(subset=["threents_upPTC"])
    # Compute residual column for safety (should match obs-pred)
    df["residual"] = df["fitness_gene_specific"] - df["predicted"]

    logger.info(f"Filtered to {len(df)} NMD-sensitive variants with upstream codon data")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_upstream_codon(data: pd.DataFrame):
    """Boxplot of loess residuals grouped by upstream codon, annotated with amino acids.

    Args:
        data: DataFrame with 'threents_upPTC' and 'residual' columns.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Sort codons alphabetically (matching R script)
    ordered_codons = sorted(data["threents_upPTC"].unique())
    # Translate each codon to amino acid
    ordered_aas = [_codon_to_aa(c) for c in ordered_codons]

    # Prepare boxplot data in order
    box_data = [
        data.loc[data["threents_upPTC"] == codon, "residual"].dropna().values
        for codon in ordered_codons
    ]

    bp = ax.boxplot(
        box_data,
        positions=range(len(ordered_codons)),
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        boxprops=dict(linewidth=0.8),
    )
    # Color each box by biochemical AA group
    for patch, aa in zip(bp["boxes"], ordered_aas):
        group = _AA_GROUPS.get(aa, "unknown")
        patch.set_facecolor(_GROUP_COLORS[group])
        patch.set_alpha(0.85)

    # Red dashed reference line at 0
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)

    # X-axis: codon labels (rotated)
    ax.set_xticks(range(len(ordered_codons)))
    ax.set_xticklabels(ordered_codons, rotation=90, fontsize=7, va="top")

    # Amino acid single-letter labels just below the rotated codon labels
    xform = ax.get_xaxis_transform()  # x=data coords, y=axes fraction
    for i, aa in enumerate(ordered_aas):
        ax.text(
            i, -0.22, aa,
            transform=xform,
            ha="center", va="top",
            fontsize=7, fontweight="bold",
            color=_GROUP_COLORS.get(_AA_GROUPS.get(aa, "unknown"), "#333333"),
            clip_on=False,
        )

    # Compact legend for AA groups
    legend_handles = [
        mpatches.Patch(facecolor=col, label=grp, alpha=0.85)
        for grp, col in _GROUP_COLORS.items() if grp != "unknown"
    ]
    ax.legend(
        handles=legend_handles,
        title="AA property",
        fontsize=7,
        title_fontsize=7,
        loc="upper right",
        framealpha=0.8,
        handlelength=1.2,
        handleheight=0.9,
    )

    ax.set_title("Upstream codon effect on NMD prediction error", fontsize=11, fontweight="bold")
    ax.set_ylim(Y_LIM)
    ax.set_xlim(-0.5, len(ordered_codons) - 0.5)
    ax.set_xlabel("Codon upstream of the PTC", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observed − Predicted\n(RNA levels)", fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = False,
):
    """Generate the upstream codon effect figure."""
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
        source = data[["gene", "PTCposition", "threents_upPTC", "up_aa", "residual"]].copy()
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(paths.source_data, index=False)
        logger.info(f"Saved source data to {paths.source_data}")

    fig = plot_upstream_codon(data)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("Context upstream codon plot complete!")


if __name__ == "__main__":
    main()
