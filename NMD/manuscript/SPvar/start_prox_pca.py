"""
PCA scatter plot of DMS SP fitness data (Fig 6c).

Reads the pre-computed PCA matrix (TABLES_DIR/SP/pca_matrix.csv) produced by
``dms_pca_analysis.main()`` and plots PC1 vs PC2 with highlighted genes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from NMD.config import TABLES_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "start_prox_pca"

# Genes to highlight
HIGHLIGHT_GENES = ["CDKL5", "MTOR"]
GENE_COLORS = ["#e41a1c", "#377eb8"]


def _load_pca_data() -> pd.DataFrame:
    """Load PCA matrix and return a 2-column (PC1, PC2) DataFrame."""
    pca_file = TABLES_DIR / "SP" / "pca_matrix.csv"
    if not pca_file.exists():
        raise FileNotFoundError(
            f"PCA matrix not found at {pca_file}. Run dms_pca_analysis.main() first."
        )
    pca_full = pd.read_csv(pca_file, index_col=0)
    pca_df = pca_full[["PC1", "PC2"]].copy()
    return pca_df


def plot_pca(pca_df: pd.DataFrame):
    """Plot PC1 vs PC2 with highlighted and extreme genes."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 4))

    highlight_mask = pca_df.index.isin(HIGHLIGHT_GENES)

    # Other genes
    ax.scatter(
        pca_df.loc[~highlight_mask, "PC1"],
        pca_df.loc[~highlight_mask, "PC2"],
        c="lightgray", s=50, alpha=0.6, label="Other genes",
    )

    # Highlighted genes
    for i, gene in enumerate(HIGHLIGHT_GENES):
        if gene in pca_df.index:
            ax.scatter(
                pca_df.loc[gene, "PC1"],
                pca_df.loc[gene, "PC2"],
                c=GENE_COLORS[i], s=150, marker="o",
                edgecolors="black", linewidths=1.5, zorder=10,
            )
            ax.annotate(
                gene,
                (pca_df.loc[gene, "PC1"], pca_df.loc[gene, "PC2"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=12, fontweight="bold",
            )
        else:
            logger.warning(f"Gene {gene} not found in PCA results")

    # Extreme genes on each axis
    extremes = {
        "PC1_max": pca_df["PC1"].idxmax(),
        "PC1_min": pca_df["PC1"].idxmin(),
        "PC2_max": pca_df["PC2"].idxmax(),
        "PC2_min": pca_df["PC2"].idxmin(),
    }
    extra_colors = ["#4daf4a", "#ff7f00", "#a65628", "#f781bf"]
    for i, (_, gene) in enumerate(extremes.items()):
        if gene not in HIGHLIGHT_GENES and gene in pca_df.index:
            ax.scatter(
                pca_df.loc[gene, "PC1"],
                pca_df.loc[gene, "PC2"],
                c=extra_colors[i], s=150, marker="D",
                edgecolors="black", linewidths=1.5, zorder=10,
            )
            ax.annotate(
                gene,
                (pca_df.loc[gene, "PC1"], pca_df.loc[gene, "PC2"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=12, fontweight="bold",
            )

    ax.set_xlabel("PC1", fontsize=14, fontweight="bold")
    ax.set_ylabel("PC2", fontsize=14, fontweight="bold")
    ax.legend(["Other genes"], fontsize=11, frameon=True, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = False,
):
    """Generate start-proximal PCA figure.

    Args:
        figure_label: Panel label when called from the manuscript app.
        figure_number: Figure number when called from the manuscript app.
        regenerate: If False and source data exists, load it instead of recomputing.
    """
    paths = get_paths(
        script_name=SCRIPT_NAME,
        figure_label=figure_label,
        figure_number=figure_number,
    )

    # Load or recompute
    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        pca_df = pd.read_csv(paths.source_data, index_col="gene")
    else:
        pca_df = _load_pca_data()
        # Save source data (only what is plotted)
        pca_df.index.name = "gene"
        pca_df.to_csv(paths.source_data)
        logger.info(f"Saved source data to {paths.source_data}")

    # Plot
    fig = plot_pca(pca_df)
    fig.savefig(paths.figure_png, dpi=300, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Figure saved to {paths.figure_png}")
    plt.close(fig)

    logger.success("PCA analysis complete!")


if __name__ == "__main__":
    main()
