"""
Correlation heatmap between PCA components and sigmoid parameters (Fig 6d).

Reads the pre-computed PCA matrix and sigmoid parameter table and shows the
Pearson correlation between each PC and each sigmoid parameter.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from loguru import logger

from NMD.config import TABLES_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "start_prox_correlations"

# Number of principal components to include
N_PCS = 4

FIGURE_SIZE = (12, 4)
DPI = 300


def _load_combined_data() -> pd.DataFrame:
    """Load and merge PCA matrix with sigmoid parameters."""
    pca_file = TABLES_DIR / "SP" / "pca_matrix.csv"
    sigmoid_file = TABLES_DIR / "SP" / "sigmoid_params_observations.csv"

    if not pca_file.exists():
        raise FileNotFoundError(f"PCA matrix not found at {pca_file}. Run dms_pca_analysis.main() first.")
    if not sigmoid_file.exists():
        raise FileNotFoundError(f"Sigmoid parameters not found at {sigmoid_file}. Run dms_sigmoid_fitting.fit_sigmoids_to_observations() first.")

    pca_df = pd.read_csv(pca_file, index_col=0)
    sigmoid_df = pd.read_csv(sigmoid_file).set_index("gene")[["A", "K", "B", "M", "r2"]]

    combined = pca_df.join(sigmoid_df, how="inner").dropna()
    logger.info(f"Combined dataset: {len(combined)} genes")
    return combined


def plot_correlation_heatmap(df: pd.DataFrame):
    """Create correlation matrix heatmap of PCs vs sigmoid params."""
    row_features = [f"PC{i+1}" for i in range(N_PCS)]
    col_features = ["A", "K", "B", "M", "r2"]
    col_labels = ["Min NMDeff", "Max NMDeff", "Steepness", "Midpoint", "Sigmoid R\u00b2"]

    corr = np.zeros((len(row_features), len(col_features)))
    for i, pc in enumerate(row_features):
        for j, feat in enumerate(col_features):
            corr[i, j] = df[pc].corr(df[feat])

    corr_df = pd.DataFrame(corr, index=row_features, columns=col_labels)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    cmap = LinearSegmentedColormap.from_list("custom", ["#022778", "white", "#ff9e9d"], N=100)

    sns.heatmap(
        corr_df, annot=True, fmt=".2f", cmap=cmap, center=0,
        vmin=-1, vmax=1, square=False, linewidths=0.5, cbar=False,
        ax=ax, annot_kws={"size": 16},
    )
    ax.set_ylabel("Principal Components", fontsize=14, weight="bold")
    ax.tick_params(labelsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    return fig


def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = False,
):
    """Generate PC–sigmoid correlation heatmap.

    Args:
        figure_label: Panel label when called from the manuscript app.
        figure_number: Figure number when called from the manuscript app.
        regenerate: If False and source data exists, load it instead of recomputing.
    """
    paths = get_paths(SCRIPT_NAME, figure_label=figure_label, figure_number=figure_number)

    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        df = pd.read_csv(paths.source_data, index_col="gene")
    else:
        df = _load_combined_data()
        df.index.name = "gene"
        df.to_csv(paths.source_data)
        logger.info(f"Saved source data to {paths.source_data}")

    fig = plot_correlation_heatmap(df)
    fig.savefig(paths.figure_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Saved figure to {paths.figure_png}")
    plt.close(fig)

    logger.success("Correlation analysis complete!")


if __name__ == "__main__":
    main()
