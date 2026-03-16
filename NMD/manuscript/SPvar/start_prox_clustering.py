"""
Clustered heatmap of sigmoid parameters (Fig 6g).

Reads pre-computed sigmoid parameters and cluster assignments (produced by the
analysis scripts ``dms_sigmoid_fitting`` and ``start_prox_clustering``) and
creates a clustered heatmap visualisation.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap

from NMD.config import TABLES_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "start_prox_clustering"

# Input paths (produced by analysis scripts)
SIGMOID_PARAMS_FILE = TABLES_DIR / "SP" / "sigmoid_params_observations.csv"
CLUSTER_TABLE = TABLES_DIR / "SP" / "cluster_assignments.csv"

# Visualisation parameters
PARAM_COLS = ["A", "K", "B", "M", "r2"]
PARAM_LABELS = ["Min NMDeff", "Max NMDeff", "Steepness", "Midpoint", "Sigmoid R\u00b2"]
HIGHLIGHT_GENES = ["CDKL5", "MTOR", "EPHA5", "NF2", "KDM6A", "TP63"]
GENE_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#a65628", "#f781bf"]
CLUSTER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]


def _load_data() -> pd.DataFrame:
    """Load sigmoid params and cluster assignments, returning a merged DataFrame."""
    if not SIGMOID_PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"Sigmoid parameters not found at {SIGMOID_PARAMS_FILE}. "
            "Run dms_sigmoid_fitting.fit_sigmoids_to_observations() first."
        )
    if not CLUSTER_TABLE.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found at {CLUSTER_TABLE}. "
            "Run NMD.analysis.start_prox_clustering.main() first."
        )

    params_df = pd.read_csv(SIGMOID_PARAMS_FILE)
    clusters_df = pd.read_csv(CLUSTER_TABLE)
    merged = params_df.merge(clusters_df, on="gene", how="inner")
    logger.info(f"Loaded {len(merged)} genes with sigmoid parameters and cluster assignments")
    return merged


def plot_clustered_heatmap(params_df: pd.DataFrame):
    """Create clustered heatmap of sigmoid parameters with cluster color bar."""
    X = params_df[PARAM_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    heatmap_data = pd.DataFrame(
        X_scaled.T,
        index=PARAM_LABELS,
        columns=params_df["gene"].values,
    )

    cluster_palette = {i + 1: CLUSTER_COLORS[i] for i in range(len(CLUSTER_COLORS))}
    gene_cluster_colors = pd.Series(
        params_df["cluster"].values, index=params_df["gene"].values,
    ).map(cluster_palette)

    cmap = LinearSegmentedColormap.from_list("custom", ["#022778", "white", "#ff9e9d"], N=100)

    g = sns.clustermap(
        heatmap_data,
        method="ward", metric="euclidean",
        cmap=cmap, center=0,
        figsize=(20, 4),
        linewidths=0.5, linecolor="white",
        yticklabels=True, xticklabels=True,
        row_cluster=False, col_cluster=True,
        col_colors=gene_cluster_colors,
        dendrogram_ratio=0.15,
        cbar_pos=(0.1, 0.2, 0.02, 0.6),
        cbar_kws={"label": "Standardized Value"},
    )

    g.ax_heatmap.tick_params(axis="y", labelsize=16)
    g.ax_heatmap.tick_params(axis="x", labelsize=12, rotation=90)
    current_labels = g.ax_heatmap.get_yticklabels()
    g.ax_heatmap.set_yticklabels(
        [lbl.get_text() for lbl in current_labels], va="center", rotation=20, fontsize=16,
    )
    title = f"Start-proximal sigmoid parameters (n={len(params_df)})"
    g.ax_heatmap.set_title(title, fontsize=18, pad=20, weight="bold", y=1.15)

    for lbl in g.ax_heatmap.get_xticklabels():
        gene_name = lbl.get_text()
        if gene_name in HIGHLIGHT_GENES:
            idx = HIGHLIGHT_GENES.index(gene_name)
            lbl.set_color(GENE_COLORS[idx])
            lbl.set_weight("bold")
            lbl.set_fontsize(14)

    return g


def main(
    figure_label: str | None = None,
    figure_number: str | None = None,
    regenerate: bool = False,
):
    """Generate clustered heatmap of start-proximal sigmoid parameters.

    Args:
        figure_label: Panel label when called from the manuscript app.
        figure_number: Figure number when called from the manuscript app.
        regenerate: If False and source data exists, load it instead of recomputing.
    """
    paths = get_paths(SCRIPT_NAME, figure_label=figure_label, figure_number=figure_number)

    if not regenerate and paths.source_data.exists():
        logger.info(f"Loading existing source data from {paths.source_data}")
        params_df = pd.read_csv(paths.source_data)
    else:
        params_df = _load_data()
        # Source data: only what is shown in the heatmap
        params_df[["gene"] + PARAM_COLS + ["cluster"]].to_csv(paths.source_data, index=False)
        logger.info(f"Saved source data to {paths.source_data}")

    for i in range(1, params_df["cluster"].max() + 1):
        genes = params_df.loc[params_df["cluster"] == i, "gene"].tolist()
        logger.info(f"Cluster {i}: {len(genes)} genes")

    g = plot_clustered_heatmap(params_df)
    g.savefig(paths.figure_png, dpi=300, bbox_inches="tight")
    g.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Saved figure to {paths.figure_png}")

    logger.success("Clustering heatmap complete!")


if __name__ == "__main__":
    main()
