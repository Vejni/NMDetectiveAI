"""
Hierarchical clustering of DMS SP genes based on sigmoid parameters.

Loads pre-computed sigmoid parameters (from dms_sigmoid_fitting), standardises
them, and performs Ward-linkage hierarchical clustering.  Saves a cluster
assignment table to TABLES_DIR / SP / cluster_assignments.csv.

This module is an analysis step that should run *after*
``dms_sigmoid_fitting.fit_sigmoids_to_observations()`` and *before* the
manuscript plotting scripts that require cluster labels.
"""

import pandas as pd
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

from NMD.config import TABLES_DIR

# Input / output
SIGMOID_PARAMS_FILE = TABLES_DIR / "SP" / "sigmoid_params_predictions.csv"
CLUSTER_TABLE = TABLES_DIR / "SP" / "cluster_assignments_predictions.csv"
DROP_GENES = ["FAT1"]  # List of genes to drop from clustering (e.g. outliers)

# Clustering configuration
N_CLUSTERS = 3
PARAM_COLS = ["A", "K", "B", "M", "r2"]


def perform_hierarchical_clustering(
    params_df: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
) -> pd.DataFrame:
    """Cluster genes by their sigmoid parameters.

    Args:
        params_df: DataFrame with at least columns ``gene`` and ``PARAM_COLS``.
        n_clusters: Number of clusters for Ward linkage.

    Returns:
        DataFrame with columns ``gene`` and ``cluster``.
    """

    # Drop specified genes
    if DROP_GENES:
        params_df = params_df[~params_df["gene"].isin(DROP_GENES)].copy()
        logger.info(f"Dropped genes from clustering: {', '.join(DROP_GENES)}")

    X = params_df[PARAM_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Z = linkage(X_scaled, method="ward", metric="euclidean")

    # Log silhouette scores for a range of k values
    for k in range(2, 8):
        labels = fcluster(Z, k, criterion="maxclust")
        sil = silhouette_score(X_scaled, labels)
        logger.info(f"k={k}: silhouette={sil:.3f}")

    cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")

    result = params_df[["gene"]].copy()
    result["cluster"] = cluster_labels

    for i in range(1, n_clusters + 1):
        genes = result.loc[result["cluster"] == i, "gene"].tolist()
        logger.info(f"Cluster {i}: {len(genes)} genes — {', '.join(genes[:5])}{'…' if len(genes) > 5 else ''}")

    return result


def main(force_recompute: bool = False):
    """Load sigmoid params, cluster genes, and save cluster_assignments.csv.

    Args:
        force_recompute: Recompute even if the output file already exists.
    """
    if CLUSTER_TABLE.exists() and not force_recompute:
        logger.info(f"Cluster assignments already exist at {CLUSTER_TABLE}; skipping (use force_recompute=True to override)")
        return pd.read_csv(CLUSTER_TABLE)

    if not SIGMOID_PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"Sigmoid parameters not found at {SIGMOID_PARAMS_FILE}. "
            "Run dms_sigmoid_fitting.fit_sigmoids_to_observations() first."
        )

    params_df = pd.read_csv(SIGMOID_PARAMS_FILE)
    logger.info(f"Loaded sigmoid parameters for {len(params_df)} genes")

    cluster_df = perform_hierarchical_clustering(params_df)

    CLUSTER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(CLUSTER_TABLE, index=False)
    logger.success(f"Saved cluster assignments ({len(cluster_df)} genes) to {CLUSTER_TABLE}")

    return cluster_df


if __name__ == "__main__":
    main(force_recompute=True)
