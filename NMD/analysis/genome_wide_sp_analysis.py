"""
Genome-wide start-proximal NMD evasion analysis.

Loads genome-wide PTC predictions, fits sigmoid curves to start-proximal regions,
classifies genes into DMS-derived clusters, and performs enrichment analyses
against gene annotation databases (essential genes, TSG/oncogenes, disease genes).

Outputs:
    - Sigmoid parameters for all genome-wide genes (TABLES_DIR/GW/gw_sigmoid_params.csv)
    - Cluster assignments for all genes (TABLES_DIR/GW/gw_cluster_assignments.csv)
    - Enrichment analysis results (TABLES_DIR/GW/gw_cluster_enrichments.csv)
    - Summary statistics (TABLES_DIR/GW/gw_sp_summary.csv)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from scipy.stats import fisher_exact
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm
import matplotlib.pyplot as plt
from genome_kit import Genome

from NMD.config import TABLES_DIR, RAW_DATA_DIR, FIGURES_DIR, GENCODE_VERSION
from NMD.analysis.dms_sigmoid_fitting import fit_logistic

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input paths
GW_PREDICTIONS_DIR = TABLES_DIR / "GW"

# Annotation files
ANNOTATIONS_DIR = RAW_DATA_DIR / "annotations"
ESSENTIAL_GENES_FILE = ANNOTATIONS_DIR / "hart_TableS2_jose.txt"
CANCER_GENES_FILE = ANNOTATIONS_DIR / "cancer_genes.csv"
DISEASE_GENES_FILE = ANNOTATIONS_DIR / "gene_condition_source_id"
ENSEMBL_ANNOTATION_FILE = ANNOTATIONS_DIR / "ensembl_v88_gene_transcript_genesymbol.txt"

# Output paths
OUTPUT_DIR = TABLES_DIR / "GW"
GW_SIGMOID_PARAMS_FILE = OUTPUT_DIR / "gw_sigmoid_params.csv"
GW_CLUSTER_FILE = OUTPUT_DIR / "gw_cluster_assignments.csv"
GW_ENRICHMENT_FILE = OUTPUT_DIR / "gw_cluster_enrichments.csv"
GW_SUMMARY_FILE = OUTPUT_DIR / "gw_sp_summary.csv"

# Analysis parameters
N_SP_PTCS = 83                # Number of start-proximal PTCs to analyse per gene
MAX_SP_POSITION = N_SP_PTCS * 3  # = 252 nt (CDS positions 3, 6, ..., 252)
MIN_POINTS_FOR_FIT = 10       # Minimum PTC positions in SP region for sigmoid fit
PARAM_COLS = ["A", "K", "B", "M", "r2"]  # Columns with sigmoid parameters for clustering
N_CLUSTERS = 3                # Number of clusters for Ward hierarchical clustering

# Exclusion criteria (applied via genomekit)
MAX_FIRST_EXON_LENGTH = 400   # Exclude if first CDS exon > 400 nt (no sigmoid shape possible)
MIN_CDS_LENGTH = 500          # Exclude short genes
NMD_55NT_MARGIN = 55          # Exclude if SP window reaches within 55 nt of last EJC


# ============================================================================
# DATA LOADING
# ============================================================================

def build_transcript_exon_info(transcript_ids: list[str]) -> dict:
    """Use genomekit to fetch first-CDS-exon length and last-EJC position for each transcript.

    For each transcript the function looks up:
    - ``first_exon_len``: length of the first CDS exon in nucleotides.
    - ``last_ejc_cds``: CDS coordinate of the last exon junction (= CDS length minus last CDS
      exon length). A PTC at CDS position ``p`` is within ``NMD_55NT_MARGIN`` nt of the last EJC
      when ``last_ejc_cds - p < NMD_55NT_MARGIN``.

    Args:
        transcript_ids: List of versioned transcript IDs (e.g. ``ENST00000263100.7``).

    Returns:
        Dict mapping transcript_id → dict with ``first_exon_len`` and ``last_ejc_cds``.
        Transcripts that cannot be found are omitted.
    """
    genome = Genome(GENCODE_VERSION)
    info = {}

    for tid in tqdm(transcript_ids, desc="Fetching exon info", leave=False):
        base_id = tid.split(".")[0]
        matches = [t for t in genome.transcripts if t.id.split(".")[0] == base_id]
        if not matches:
            continue
        t = matches[0]
        if not hasattr(t, "cdss") or not t.cdss:
            continue
        cdss = list(t.cdss)
        if len(cdss) < 2:          # single-exon gene — no EJC-based NMD
            continue
        first_exon_len = len(cdss[0])
        last_exon_len = len(cdss[-1])
        cds_total = sum(len(c) for c in cdss)
        last_ejc_cds = cds_total - last_exon_len   # CDS position of last EJC
        info[tid] = {
            "first_exon_len": first_exon_len,
            "last_ejc_cds": last_ejc_cds,
        }

    logger.info(
        f"Fetched exon info for {len(info)}/{len(transcript_ids)} transcripts "
        f"via {GENCODE_VERSION}"
    )
    return info


def load_gw_predictions() -> pd.DataFrame:
    """Load all genome-wide PTC prediction files and extract start-proximal positions.

    For each file:
    1. Computes CDS-relative positions by subtracting the per-gene 5' UTR offset
       (``cds_position = ptc_position - (min_ptc_position - 3)``).
    2. Keeps only the first ``N_SP_PTCS`` positions (``cds_position <= MAX_SP_POSITION``).
    3. Drops genes with ``cds_length < MIN_CDS_LENGTH``.
    4. Uses genomekit to drop genes where the first CDS exon is longer than
       ``MAX_FIRST_EXON_LENGTH`` nt, or where the SP window reaches within
       ``NMD_55NT_MARGIN`` nt of the last exon junction.

    Returns:
        DataFrame with columns: gene_name, transcript_id, ptc_position, cds_position,
        prediction, cds_length, num_exons, strand.
    """
    prediction_files = sorted(GW_PREDICTIONS_DIR.glob("*_ptc_predictions.csv"))
    logger.info(f"Found {len(prediction_files)} prediction files in {GW_PREDICTIONS_DIR}")

    all_dfs = []
    for f in tqdm(prediction_files, desc="Loading GW predictions"):
        try:
            df = pd.read_csv(f)

            # ---- 1. Compute CDS-relative positions --------------------------------
            # ptc_position is a transcript-level coordinate (includes 5' UTR).
            # The first in-frame codon of the CDS is at CDS position 3.
            utr_offset = df["ptc_position"].min() - 3
            df["cds_position"] = df["ptc_position"] - utr_offset

            # ---- 2. Keep only start-proximal PTCs ---------------------------------
            df_sp = df[df["cds_position"] <= MAX_SP_POSITION].copy()
            if len(df_sp) < MIN_POINTS_FOR_FIT:
                continue

            # ---- 3. Short-gene filter (no genomekit needed) -----------------------
            if df_sp["cds_length"].iloc[0] < MIN_CDS_LENGTH:
                continue

            all_dfs.append(df_sp)
        except Exception as e:
            logger.debug(f"Failed to read {f.name}: {e}")

    if not all_dfs:
        raise RuntimeError("No prediction files loaded — check GW_PREDICTIONS_DIR.")

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        f"Before exon-structure filters: {combined['gene_name'].nunique()} genes, "
        f"{len(combined)} SP PTCs"
    )

    # ---- 4. Exon-structure filters via genomekit ----------------------------------
    unique_tids = combined["transcript_id"].unique().tolist()
    exon_info = build_transcript_exon_info(unique_tids)

    def _passes_exon_filters(row):
        info = exon_info.get(row["transcript_id"])
        if info is None:
            return False   # Cannot verify — exclude
        # First CDS exon must not be longer than MAX_FIRST_EXON_LENGTH
        if info["first_exon_len"] > MAX_FIRST_EXON_LENGTH:
            return False
        # SP window must not reach within NMD_55NT_MARGIN nt of last EJC
        if info["last_ejc_cds"] - MAX_SP_POSITION < NMD_55NT_MARGIN:
            return False
        return True

    # Compute per-gene pass/fail (all rows of a gene share the same transcript)
    gene_pass = (
        combined.groupby("transcript_id")
        .first()
        .reset_index()[["transcript_id"]]
        .assign(passes=lambda df: df.apply(_passes_exon_filters, axis=1))
        .set_index("transcript_id")["passes"]
    )
    combined["_passes"] = combined["transcript_id"].map(gene_pass)
    n_before = combined["gene_name"].nunique()
    combined = combined[combined["_passes"]].drop(columns=["_passes"])
    n_after = combined["gene_name"].nunique()
    logger.info(
        f"Exon-structure filters removed {n_before - n_after} genes "
        f"(first exon > {MAX_FIRST_EXON_LENGTH} nt OR last EJC margin < {NMD_55NT_MARGIN} nt); "
        f"{n_after} genes remaining"
    )

    logger.info(
        f"Final SP dataset: {len(combined)} PTCs from {combined['gene_name'].nunique()} genes"
    )
    return combined


def load_essential_genes() -> set:
    """
    Load essential genes from Hart et al. TableS2 (Ensembl IDs) and convert to gene names.

    Returns:
        Set of essential gene names
    """
    # Load Ensembl ID to gene name mapping
    annot = pd.read_csv(ENSEMBL_ANNOTATION_FILE, sep="\t")
    ensembl_to_name = dict(zip(annot["gene_id"], annot["gene_name"]))

    # Load essential gene Ensembl IDs
    essential_ids = set()
    with open(ESSENTIAL_GENES_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("ENSG"):
                essential_ids.add(line)

    # Convert to gene names
    essential_names = set()
    for eid in essential_ids:
        if eid in ensembl_to_name:
            essential_names.add(ensembl_to_name[eid])

    logger.info(
        f"Loaded {len(essential_ids)} essential gene IDs, "
        f"mapped {len(essential_names)} to gene names"
    )
    return essential_names


def load_cancer_genes() -> tuple[set, set]:
    """
    Load TSG and oncogene annotations from cancer_genes.csv.

    Returns:
        Tuple of (tsg_genes, oncogenes) as sets of gene names
    """
    df = pd.read_csv(CANCER_GENES_FILE)
    # Clean the role column (some have extra quotes)
    df["role"] = df["role"].str.strip().str.strip('"')

    tsgs = set(df[df["role"] == "TSG"]["Gene"])
    oncogenes = set(df[df["role"] == "OG"]["Gene"])

    logger.info(f"Loaded {len(tsgs)} TSGs and {len(oncogenes)} oncogenes")
    return tsgs, oncogenes


def load_disease_genes() -> set:
    """
    Load disease-associated genes from NCBI gene_condition_source_id.

    Returns:
        Set of disease-associated gene names
    """
    df = pd.read_csv(DISEASE_GENES_FILE, sep="\t", comment="#", header=None,
                     names=["GeneID", "AssociatedGenes", "RelatedGenes",
                            "ConceptID", "DiseaseName", "SourceName",
                            "SourceID", "DiseaseMIM", "LastUpdated"])

    # Collect all unique gene names from AssociatedGenes column
    disease_genes = set()
    for genes_str in df["AssociatedGenes"].dropna():
        for gene in str(genes_str).split(","):
            gene = gene.strip()
            if gene:
                disease_genes.add(gene)

    logger.info(f"Loaded {len(disease_genes)} disease-associated genes")
    return disease_genes


# ============================================================================
# SIGMOID FITTING
# ============================================================================

def fit_sigmoids_to_gw_genes(gw_df: pd.DataFrame) -> pd.DataFrame:
    """Fit 4-parameter sigmoid curves to start-proximal predictions for each gene.

    Uses the CDS-relative ``cds_position`` column (1-based, in steps of 3) as
    the x-axis so the sigmoid midpoint M reflects actual CDS distance from the
    start codon rather than a transcript-level coordinate.

    Args:
        gw_df: DataFrame with genome-wide start-proximal predictions including
            ``cds_position`` and ``prediction`` columns.

    Returns:
        DataFrame with sigmoid parameters (gene, A, K, B, M, r2, cds_length, n_points)
    """
    if GW_SIGMOID_PARAMS_FILE.exists():
        logger.info(f"Loading cached sigmoid params from {GW_SIGMOID_PARAMS_FILE}")
        return pd.read_csv(GW_SIGMOID_PARAMS_FILE)

    genes = gw_df["gene_name"].unique()
    logger.info(f"Fitting sigmoids to {len(genes)} genes")

    results = []
    failed = []

    for gene in tqdm(genes, desc="Fitting sigmoids"):
        gene_df = gw_df[gw_df["gene_name"] == gene].sort_values("cds_position")
        positions = gene_df["cds_position"].values    # CDS-relative, starts at 3
        predictions = gene_df["prediction"].values

        if len(positions) < MIN_POINTS_FOR_FIT:
            failed.append(gene)
            continue

        try:
            result = fit_logistic(positions, predictions)
            results.append({
                "gene": gene,
                "A": result["params"][0],
                "K": result["params"][1],
                "B": result["params"][2],
                "M": result["params"][3],
                "r2": result["r2"],
                "x_min": result["x_min"],
                "x_max": result["x_max"],
                "cds_length": int(gene_df["cds_length"].iloc[0]),
                "n_points": len(positions),
            })
        except Exception:
            failed.append(gene)

    params_df = pd.DataFrame(results)
    logger.info(f"Fitted {len(params_df)}/{len(genes)} genes, {len(failed)} failed")

    # Save
    GW_SIGMOID_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    params_df.to_csv(GW_SIGMOID_PARAMS_FILE, index=False)
    logger.success(f"Saved GW sigmoid params to {GW_SIGMOID_PARAMS_FILE}")
    return params_df


# ============================================================================
# CLUSTERING
# ============================================================================

def cluster_gw_genes(gw_params: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.DataFrame:
    """Cluster genome-wide genes by sigmoid parameters using Ward hierarchical clustering.

    Standardises the sigmoid parameters, performs Ward-linkage agglomerative
    clustering, and logs silhouette scores for k=2..7 to aid cluster-count
    selection.

    Args:
        gw_params: DataFrame with genome-wide sigmoid parameters (must contain
            PARAM_COLS).
        n_clusters: Number of clusters to cut the dendrogram at.

    Returns:
        gw_params with added 'cluster' column (1-indexed integer labels).
    """
    X = gw_params[PARAM_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Z = linkage(X_scaled, method="ward", metric="euclidean")

    # Log silhouette scores for a range of k values to aid selection
    logger.info("Silhouette scores across cluster counts:")
    for k in range(2, 8):
        labels = fcluster(Z, k, criterion="maxclust")
        sil = silhouette_score(X_scaled, labels)
        marker = " <--" if k == n_clusters else ""
        logger.info(f"  k={k}: silhouette={sil:.3f}{marker}")

    cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")

    gw_params = gw_params.copy()
    gw_params["cluster"] = cluster_labels

    logger.info(f"Cluster assignments (n_clusters={n_clusters}):")
    for cid in range(1, n_clusters + 1):
        n = (gw_params["cluster"] == cid).sum()
        pct = 100 * n / len(gw_params)
        logger.info(f"  Cluster {cid}: {n} genes ({pct:.1f}%)")

    return gw_params


# ============================================================================
# PLOTTING
# ============================================================================

def plot_example_genes(gw_df: pd.DataFrame, gw_params: pd.DataFrame, n_examples: int = 3):
    """Create a grid plot showing example genes from each cluster plus a cluster-average column.

    Columns 1–``n_examples``: individual genes closest to the cluster centroid.
    Column ``n_examples + 1``: all individual sigmoid curves overlaid in light
    gray with the cluster-mean curve on top (all in normalised [0,1] x-space
    because the sigmoid parameters A, K, B, M were all fitted in that space).

    Args:
        gw_df: DataFrame with genome-wide predictions (gene_name, cds_position, prediction).
        gw_params: DataFrame with cluster assignments and PARAM_COLS columns.
        n_examples: Number of example genes to show per cluster.
    """
    from NMD.analysis.dms_sigmoid_fitting import logistic4

    clusters = sorted(gw_params["cluster"].unique())
    n_clusters = len(clusters)
    n_cols = n_examples + 1   # extra column for cluster average

    # Only keep genes that are actually present in gw_df
    genes_with_data = set(gw_df["gene_name"].unique())
    gw_params = gw_params[gw_params["gene"].isin(genes_with_data)].copy()

    # Standardise PARAM_COLS for centroid distance calculation
    scaler = StandardScaler()
    params_scaled = scaler.fit_transform(gw_params[PARAM_COLS].values)
    gw_params = gw_params.copy()
    gw_params["_scaled_idx"] = range(len(gw_params))

    fig, axes = plt.subplots(
        n_clusters, n_cols,
        figsize=(4 * n_cols, 3.5 * n_clusters),
        sharey=True,
    )
    if n_clusters == 1:
        axes = axes.reshape(1, -1)

    x_norm = np.linspace(0, 1, 300)   # shared x-axis for normalised space

    for row, cluster_id in enumerate(clusters):
        cluster_mask = gw_params["cluster"] == cluster_id
        cluster_df = gw_params[cluster_mask].copy()
        cluster_scaled = params_scaled[cluster_df["_scaled_idx"].values]

        # Centroid in scaled space
        centroid = cluster_scaled.mean(axis=0)
        dists = np.linalg.norm(cluster_scaled - centroid, axis=1)
        cluster_df = cluster_df.copy()
        cluster_df["_dist"] = dists

        # Pick n_examples closest to centroid
        selected = cluster_df.nsmallest(n_examples, "_dist")

        # ── Example gene columns ──────────────────────────────────────────────
        for col in range(n_examples):
            ax = axes[row, col]

            if col >= len(selected):
                ax.set_visible(False)
                continue

            gene_row = selected.iloc[col]
            gene = gene_row["gene"]
            A, K, B, M = gene_row["A"], gene_row["K"], gene_row["B"], gene_row["M"]
            r2 = gene_row["r2"]

            gene_data = gw_df[gw_df["gene_name"] == gene].sort_values("cds_position")
            positions = gene_data["cds_position"].values
            predictions = gene_data["prediction"].values

            if len(positions) == 0:
                ax.set_visible(False)
                continue

            # Scatter of raw predictions
            ax.scatter(positions, predictions, alpha=0.6, s=15, color="steelblue", zorder=2)

            # Fitted sigmoid curve — use the x_min/x_max stored at fit time so
            # the scaling exactly matches what fit_logistic used.
            x_min = gene_row["x_min"]
            x_max = gene_row["x_max"]
            x_fit = np.linspace(x_min, x_max, 300)
            x_scaled_fit = (x_fit - x_min) / (x_max - x_min)
            y_fit = logistic4(x_scaled_fit, A, K, B, M)
            ax.plot(x_fit, y_fit, color="tomato", linewidth=1.5, zorder=3)

            # Sigmoid parameter annotation
            param_text = (
                f"A={A:.2f}  K={K:.2f}\n"
                f"B={B:.2f}  M={M:.2f}\n"
                f"R²={r2:.2f}"
            )
            ax.text(
                0.97, 0.05, param_text,
                transform=ax.transAxes,
                fontsize=7, va="bottom", ha="right",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
            )

            ax.set_title(f"Cluster {cluster_id} · {gene}", fontsize=9, fontweight="bold")
            ax.set_ylim(-1, 1)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == n_clusters - 1:
                ax.set_xlabel("CDS position (nt)", fontsize=9)
            if col == 0:
                ax.set_ylabel("NMD prediction", fontsize=9)

        # ── Average sigmoid column ────────────────────────────────────────────
        ax_avg = axes[row, n_examples]

        # Plot every individual gene's sigmoid in the cluster (light gray)
        for _, gene_row in cluster_df.iterrows():
            A_i, K_i, B_i, M_i = gene_row["A"], gene_row["K"], gene_row["B"], gene_row["M"]
            y_i = logistic4(x_norm, A_i, K_i, B_i, M_i)
            ax_avg.plot(x_norm, y_i, color="gray", linewidth=0.4, alpha=0.15, zorder=1)

        # Cluster-mean curve (average of per-gene parameters)
        mean_params = cluster_df[["A", "K", "B", "M"]].mean()
        A_m, K_m, B_m, M_m = mean_params["A"], mean_params["K"], mean_params["B"], mean_params["M"]
        y_mean = logistic4(x_norm, A_m, K_m, B_m, M_m)
        ax_avg.plot(x_norm, y_mean, color="tomato", linewidth=2.0, zorder=3)

        # Mean parameter annotation
        mean_r2 = cluster_df["r2"].mean()
        param_text = (
            f"A={A_m:.2f}  K={K_m:.2f}\n"
            f"B={B_m:.2f}  M={M_m:.2f}\n"
            f"R²={mean_r2:.2f} (mean)"
        )
        ax_avg.text(
            0.97, 0.05, param_text,
            transform=ax_avg.transAxes,
            fontsize=7, va="bottom", ha="right",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

        n_genes = len(cluster_df)
        ax_avg.set_title(
            f"Cluster {cluster_id} average (n={n_genes})", fontsize=9, fontweight="bold"
        )
        ax_avg.set_ylim(-1, 1)
        ax_avg.grid(True, alpha=0.25, linestyle=":")
        ax_avg.spines["top"].set_visible(False)
        ax_avg.spines["right"].set_visible(False)

        if row == n_clusters - 1:
            ax_avg.set_xlabel("Normalised CDS position", fontsize=9)

    plt.tight_layout()

    output_dir = FIGURES_DIR / "SP"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "gw_cluster_examples.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.success(f"Saved example gene plots to {output_file}")
    plt.close(fig)


# ============================================================================
# ENRICHMENT ANALYSIS
# ============================================================================

def fisher_test(
    gene_set: set,
    cluster_genes: set,
    all_genes: set
) -> dict:
    """
    Perform Fisher's exact test for enrichment of a gene set in a cluster.

    Args:
        gene_set: Set of genes in the category (e.g. essential genes)
        cluster_genes: Set of genes in the cluster
        all_genes: Set of all genes in the analysis

    Returns:
        Dict with n_overlap, n_cluster, n_category, odds_ratio, p_value
    """
    in_both = len(gene_set & cluster_genes)
    in_cluster_only = len(cluster_genes - gene_set)

    # Correct contingency table
    a = in_both
    b = len(cluster_genes) - in_both
    c = len(gene_set & all_genes) - in_both
    d = len(all_genes) - a - b - c

    table = np.array([[a, b], [c, d]])
    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

    return {
        "n_overlap": a,
        "n_cluster": len(cluster_genes),
        "n_category": len(gene_set & all_genes),
        "odds_ratio": odds_ratio,
        "p_value": p_value,
    }


def run_enrichment_analysis(gw_params: pd.DataFrame) -> pd.DataFrame:
    """
    Run enrichment analyses for gene categories across all clusters.

    Args:
        gw_params: DataFrame with gene names and cluster assignments

    Returns:
        DataFrame with enrichment results for each cluster × category
    """
    all_genes = set(gw_params["gene"])

    # Load annotations
    essential_genes = load_essential_genes()
    tsgs, oncogenes = load_cancer_genes()
    disease_genes = load_disease_genes()

    cancer_genes = tsgs | oncogenes

    categories = {
        "essential": essential_genes,
        "TSG": tsgs,
        "oncogene": oncogenes,
        "cancer_gene": cancer_genes,
        "disease_gene": disease_genes,
    }

    results = []
    clusters = sorted(gw_params["cluster"].unique())

    for cluster_id in clusters:
        cluster_genes = set(gw_params[gw_params["cluster"] == cluster_id]["gene"])

        for cat_name, cat_genes in categories.items():
            test = fisher_test(cat_genes, cluster_genes, all_genes)
            results.append({
                "cluster": cluster_id,
                "category": cat_name,
                **test,
            })

    enrichment_df = pd.DataFrame(results)

    # Log significant results
    sig = enrichment_df[enrichment_df["p_value"] < 0.05]
    logger.info(f"\nSignificant enrichments (P < 0.05):")
    for _, row in sig.iterrows():
        direction = "enriched" if row["odds_ratio"] > 1 else "depleted"
        logger.info(
            f"  Cluster {row['cluster']} - {row['category']}: "
            f"OR={row['odds_ratio']:.2f}, P={row['p_value']:.2e} ({direction})"
        )

    return enrichment_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the genome-wide start-proximal NMD evasion analysis."""
    logger.info("Starting genome-wide start-proximal analysis")

    # Step 1: Load genome-wide predictions
    gw_df = load_gw_predictions()

    # Step 2: Fit sigmoids
    gw_params = fit_sigmoids_to_gw_genes(gw_df)

    # Print median values for each parameter
    medians = gw_params[PARAM_COLS].median()

    # Step 3: Cluster GW genes directly by their sigmoid parameters
    gw_params = cluster_gw_genes(gw_params)

    # Print median parameters for each cluster
    logger.info("Median sigmoid parameters by cluster:")
    for cluster_id in sorted(gw_params["cluster"].unique()):
        cluster_data = gw_params[gw_params["cluster"] == cluster_id]
        medians = cluster_data[PARAM_COLS].median()
        n_genes = len(cluster_data)
        logger.info(f"  Cluster {cluster_id} ({n_genes} genes): {medians.to_dict()}")

    # Plot example genes from each cluster
    plot_example_genes(gw_df, gw_params, n_examples=3)

    # Save cluster assignments
    GW_CLUSTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    gw_params.to_csv(GW_CLUSTER_FILE, index=False)
    logger.success(f"Saved cluster assignments to {GW_CLUSTER_FILE}")

    # Step 4: Enrichment analysis
    enrichment_df = run_enrichment_analysis(gw_params)

    # Save enrichment results
    GW_ENRICHMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    enrichment_df.to_csv(GW_ENRICHMENT_FILE, index=False)
    logger.success(f"Saved enrichment results to {GW_ENRICHMENT_FILE}")
    logger.success("Genome-wide SP analysis complete!")


if __name__ == "__main__":
    main()
