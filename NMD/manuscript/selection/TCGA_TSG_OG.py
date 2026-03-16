"""
Gene-category NMD selection pressure analysis (TCGA somatic PTCs).

For each gene a 2×2 Wald log OR is computed:

    |            | NMD-triggering | NMD-evading |
    |------------|----------------|-------------|
    | PTCs       | a              | b           |
    | Synonymous | c              | d           |

log OR_g = log((a+0.5)(d+0.5) / (b+0.5)(c+0.5))  (Haldane-Anscombe pseudocount)

Per-gene log ORs are then pooled within each gene category using
inverse-variance–weighted (fixed-effects) meta-analysis.  A random-effects
(DerSimonian-Laird) estimate is also computed and reported for reference.
Heterogeneity is summarised by Cochran's Q and I².

Each category is compared to the pooled "Other Genes" effect, which serves
as the genome-wide background.  Positive pooled log OR means PTCs are
relatively more concentrated in NMD-triggering positions than synonymous
variants are (selection for NMD-induced LoF); negative means the opposite.

The analysis is run twice -- once with rule-based NMD classification only,
and once with the NMDetectiveAI concordance filter that removes variants
where rule-based and AI predictions disagree.

Requires preprocessed data:
    - data/processed/selection/stopgain_variants_annotated_annotated.tsv
    - data/interim/selection/synonymous_variants_annotated.tsv
    - data/raw/annotations/ensembl_v88_gene_transcript_genesymbol.txt
    - data/raw/annotations/cancer_genes.csv
    - data/raw/annotations/hart_TableS2_jose.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from sklearn.mixture import GaussianMixture
from loguru import logger

from NMD.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR, TABLES_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "TCGA_TSG_OG"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NONSENSE_FILE = PROCESSED_DATA_DIR / "selection" / "stopgain_variants_annotated_annotated.tsv"
SYNONYMOUS_FILE = INTERIM_DATA_DIR / "selection" / "synonymous_variants_annotated.tsv"
GENE_MAP_FILE = RAW_DATA_DIR / "annotations" / "ensembl_v88_gene_transcript_genesymbol.txt"
CANCER_GENES_FILE = RAW_DATA_DIR / "annotations" / "cancer_genes.csv"
ESSENTIAL_GENES_FILE = RAW_DATA_DIR / "annotations" / "hart_TableS2_jose.txt"
DISEASE_GENES_FILE = RAW_DATA_DIR / "annotations" / "gene_condition_source_id"

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
MIN_PTC_GENE = 10   # Minimum PTCs per gene to enter meta-analysis
MIN_SYN_GENE = 5    # Minimum synonymous variants per gene
CI_LEVEL = 0.95     # Wald confidence interval level
PREDICTION_THRESHOLD_TRIGGERING = 0.43
PREDICTION_THRESHOLD_EVADING = -0.17

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOR_TSG = '#ff9e9d'       # pink/red
COLOR_OG = '#022778'        # deep navy
COLOR_ESSENTIAL = '#2d8b4d' # green
COLOR_OTHER = '#888888'     # gray


# ===================================================================
# Data loading
# ===================================================================

def load_gene_annotations() -> tuple[list, list, list, set, pd.DataFrame]:
    """Load gene category annotations and build the Other Genes exclusion set.

    The exclusion set contains every gene symbol appearing in *any* row of
    ``cancer_genes.csv`` (not just common drivers) and every gene in the
    NCBI gene-condition association file, so that "Other Genes" is a clean
    background of genes with no known cancer or disease association.

    Returns:
        Tuple of (oncogenes, tumor_suppressors, essential_genes,
        exclusion_genes, gene_map).
    """
    logger.info("Loading gene annotations...")

    gene_map = pd.read_csv(GENE_MAP_FILE, sep="\t", header=0)
    gene_map = gene_map[["gene_id", "gene_name"]].drop_duplicates()

    # Cancer genes used for TSG/OG categories (common drivers only)
    cgc = pd.read_csv(CANCER_GENES_FILE)
    cgc_common = cgc[(cgc.cancer_gene == "common mutational cancer driver")] #  | (cgc.cancer_gene == "rare mutational cancer driver")
    oncogenes = cgc_common[cgc_common.role == "OG"].Gene.tolist()
    tumor_suppressors = cgc_common[cgc_common.role == "TSG"].Gene.tolist()

    # All cancer genes (any annotation) for exclusion from Other Genes
    all_cancer_genes = set(cgc["Gene"].dropna().unique())

    # Disease genes from NCBI gene-condition associations
    disease_df = pd.read_csv(DISEASE_GENES_FILE, sep="\t", dtype=str)
    # Column name has a leading '#' in the raw file; normalise
    disease_df.columns = [c.lstrip("#").strip() for c in disease_df.columns]
    disease_genes = set(disease_df["AssociatedGenes"].dropna().str.strip().unique())

    exclusion_genes = all_cancer_genes | disease_genes
    logger.info(
        f"Exclusion set: {len(all_cancer_genes)} cancer-gene entries + "
        f"{len(disease_genes)} disease-gene entries = "
        f"{len(exclusion_genes)} unique genes excluded from Other Genes"
    )

    essential = pd.read_csv(ESSENTIAL_GENES_FILE, header=0)
    essential.dropna(inplace=True)
    essential.columns = ["Gene"]
    essential = essential.merge(gene_map, left_on="Gene", right_on="gene_id", how="left")
    essential_genes = essential.gene_name.unique().tolist()

    logger.info(
        f"Loaded {len(oncogenes)} oncogenes, "
        f"{len(tumor_suppressors)} tumor suppressors, "
        f"{len(essential_genes)} essential genes"
    )
    return oncogenes, tumor_suppressors, essential_genes, exclusion_genes, gene_map


# ===================================================================
# NMD classification
# ===================================================================

def fit_per_gene_gmm_thresholds(
    df: pd.DataFrame,
    gene_col: str = "gene_name",
    min_ptcs: int = MIN_PTC_GENE,
) -> dict:
    """Fit a 2-component GMM per gene to find the AI decision boundary.

    Args:
        df: DataFrame with nonsense variants including
            ``NMDetectiveAI_prediction``.
        gene_col: Column containing gene symbols.
        min_ptcs: Minimum valid predictions per gene.

    Returns:
        ``{gene_name: threshold}`` dictionary.
    """
    thresholds = {}
    valid_df = df[df["NMDetectiveAI_prediction"].notna()].copy()
    n_skipped = 0

    for gene, group in valid_df.groupby(gene_col):
        if len(group) < min_ptcs:
            continue
        preds = group["NMDetectiveAI_prediction"].values.reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(preds)
            means = gmm.means_.flatten()
            low_idx, high_idx = int(np.argmin(means)), int(np.argmax(means))
            if abs(means[high_idx] - means[low_idx]) < 0.05:
                n_skipped += 1
                continue
            x = np.linspace(means[low_idx], means[high_idx], 1000).reshape(-1, 1)
            probs = gmm.predict_proba(x)[:, high_idx]
            threshold = float(x[int(np.argmin(np.abs(probs - 0.5))), 0])
            thresholds[gene] = threshold
        except Exception as e:
            logger.debug(f"GMM fitting failed for {gene}: {e}")

    logger.info(
        f"Fitted per-gene GMM thresholds for {len(thresholds)} genes "
        f"({n_skipped} skipped -- unimodal)"
    )
    return thresholds


def classify_nmd_regions(
    df: pd.DataFrame,
    use_concordance_filter: bool = False,
    gene_thresholds: dict | None = None,
    gene_col: str = "gene_name",
) -> pd.DataFrame:
    """Assign each variant to NMD-triggering, NMD-evading, or unknown.

    Rule-based classification is always applied first.  When
    ``use_concordance_filter=True``, variants where the AI prediction
    **disagrees** with the rule-based label are set to ``'unknown'`` and
    excluded from downstream counting.

    Args:
        df: Variant table with ``NMD_status`` column.
        use_concordance_filter: If True, apply the concordance filter.
        gene_thresholds: Per-gene thresholds from
            :func:`fit_per_gene_gmm_thresholds`.
        gene_col: Column name for gene identifiers.

    Returns:
        Copy of *df* with added ``nmd_region`` column.
    """
    df = df.copy()

    # Step 1: rule-based
    df["nmd_region"] = df["NMD_status"].map({
        "NMD_triggering": "NMD_triggering",
        "NMD_evading_last_exon": "NMD_evading",
        "NMD_evading_long_exon": "NMD_evading",
        "NMD_evading_150nt": "NMD_evading",
        "NMD_evading_55nt": "NMD_evading",
        "NMD_unknown": "unknown",
    })

    # Step 2 (optional): concordance filter -- mark disagreements as unknown
    if (
        use_concordance_filter
        and gene_thresholds
        and "NMDetectiveAI_prediction" in df.columns
    ):
        df["_thr"] = df[gene_col].map(gene_thresholds)
        eligible = df["_thr"].notna() & df["NMDetectiveAI_prediction"].notna()

        ai_trig = df["NMDetectiveAI_prediction"] >= df["_thr"]
        ai_evad = df["NMDetectiveAI_prediction"] < df["_thr"]

        # AI says triggering but rule says evading -> unknown
        df.loc[eligible & ai_trig & (df["nmd_region"] == "NMD_evading"), "nmd_region"] = "unknown"
        # AI says evading but rule says triggering -> unknown
        df.loc[eligible & ai_evad & (df["nmd_region"] == "NMD_triggering"), "nmd_region"] = "unknown"

        # first local thresholds
        #df.loc[eligible & ai_trig, "nmd_region"] = "NMD_triggering"
        #df.loc[eligible & ai_evad, "nmd_region"] = "NMD_evading"

        n_unknown = (df.loc[eligible, "nmd_region"] == "unknown").sum()
        logger.info(
            f"Concordance filter: {n_unknown} discordant variants -> unknown "
            f"(out of {eligible.sum()} eligible)"
        )
        df.drop(columns=["_thr"], inplace=True)

    return df


# ===================================================================
# Statistics
# ===================================================================

def _wald_log_or(a: int, b: int, c: int, d: int, ci_level: float = CI_LEVEL) -> dict:
    """Compute log OR, SE, CI, and z-test p-value from a 2x2 table.

    Cells represent:
        a = PTC triggering, b = PTC evading
        c = SYN triggering, d = SYN evading

    Haldane-Anscombe 0.5 pseudocount is added to every cell.

    Returns:
        dict with keys log_or, se, ci_lower, ci_upper, z, p_value.
    """
    eps = 0.5
    a_, b_, c_, d_ = a + eps, b + eps, c + eps, d + eps
    log_or = np.log(a_ * d_ / (b_ * c_))
    se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    z_crit = scipy_stats.norm.ppf(1 - (1 - ci_level) / 2)
    z = log_or / se
    p = 2 * scipy_stats.norm.sf(abs(z))
    return {
        "log_or": log_or,
        "se": se,
        "ci_lower": log_or - z_crit * se,
        "ci_upper": log_or + z_crit * se,
        "z": z,
        "p_value": p,
    }


def per_gene_log_or(
    nonsense_df: pd.DataFrame,
    synonymous_df: pd.DataFrame,
    gene_list: list,
    gene_col: str = "gene_name",
    min_ptc: int = MIN_PTC_GENE,
    min_syn: int = MIN_SYN_GENE,
) -> pd.DataFrame:
    """Compute a Wald log OR for each gene in *gene_list*.

    Genes with fewer than *min_ptc* combined PTCs or fewer than *min_syn*
    combined synonymous variants are excluded to keep SE estimates stable.

    Args:
        nonsense_df: Classified variant table (has ``nmd_region``).
        synonymous_df: Classified variant table (has ``nmd_region``).
        gene_list: Genes to analyse.
        gene_col: Column containing gene symbols.
        min_ptc: Minimum total PTC count per gene.
        min_syn: Minimum total synonymous variant count per gene.

    Returns:
        DataFrame with one row per gene and columns:
        ``gene``, ``n_ptc_trig``, ``n_ptc_evad``, ``n_syn_trig``,
        ``n_syn_evad``, ``log_or``, ``se``, ``ci_lower``, ``ci_upper``,
        ``z``, ``p_value``.
    """
    rows = []

    ns_counts = (
        nonsense_df[nonsense_df[gene_col].isin(gene_list)]
        .groupby([gene_col, "nmd_region"])
        .size()
        .unstack(fill_value=0)
    )
    sy_counts = (
        synonymous_df[synonymous_df[gene_col].isin(gene_list)]
        .groupby([gene_col, "nmd_region"])
        .size()
        .unstack(fill_value=0)
    )

    for gene in gene_list:
        a = int(ns_counts.get("NMD_triggering", pd.Series(dtype=int)).get(gene, 0))
        b = int(ns_counts.get("NMD_evading", pd.Series(dtype=int)).get(gene, 0))
        c = int(sy_counts.get("NMD_triggering", pd.Series(dtype=int)).get(gene, 0))
        d = int(sy_counts.get("NMD_evading", pd.Series(dtype=int)).get(gene, 0))

        if a + b < min_ptc or c + d < min_syn:
            continue

        wald = _wald_log_or(a, b, c, d)
        rows.append(
            {"gene": gene, "n_ptc_trig": a, "n_ptc_evad": b,
             "n_syn_trig": c, "n_syn_evad": d, **wald}
        )

    return pd.DataFrame(rows)


def pool_category_meta(
    gene_df: pd.DataFrame,
    category_name: str,
    other_pooled: dict | None = None,
    ci_level: float = CI_LEVEL,
) -> dict:
    """Inverse-variance fixed-effects meta-analysis of per-gene log ORs.

    Also computes a DerSimonian-Laird random-effects estimate, Cochran's Q,
    and I².  If *other_pooled* is provided (the pooled result for "Other
    Genes"), a z-test comparing this category to that background is added.

    Args:
        gene_df: Output of :func:`per_gene_log_or`.
        category_name: Label for this gene category.
        other_pooled: Pooled meta-analysis result for "Other Genes" (to test
            against the background).
        ci_level: Confidence interval level.

    Returns:
        Dictionary with pooled log OR, SE, CI, heterogeneity statistics,
        and comparison vs. Other Genes when available.
    """
    if gene_df.empty:
        return {"category": category_name, "n_genes": 0,
                "log_or": np.nan, "se": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan,
                "z": np.nan, "p_value": np.nan,
                "Q": np.nan, "i2": np.nan,
                "n_ptc_trig": 0, "n_ptc_evad": 0}

    log_ors = gene_df["log_or"].values
    ses = gene_df["se"].values
    w = 1.0 / ses ** 2  # inverse-variance weights

    # Fixed-effects pooled estimate
    fe_log_or = np.sum(w * log_ors) / np.sum(w)
    fe_se = 1.0 / np.sqrt(np.sum(w))

    # Cochran's Q and I²
    Q = float(np.sum(w * (log_ors - fe_log_or) ** 2))
    k = len(log_ors)
    df = k - 1
    i2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0

    # DerSimonian-Laird between-study variance
    tau2 = max(0.0, (Q - df) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))

    # Random-effects pooled estimate
    w_re = 1.0 / (ses ** 2 + tau2)
    re_log_or = np.sum(w_re * log_ors) / np.sum(w_re)
    re_se = 1.0 / np.sqrt(np.sum(w_re))

    z_crit = scipy_stats.norm.ppf(1 - (1 - ci_level) / 2)

    # Use RE if heterogeneous (I² > 50%), FE otherwise
    use_re = i2 > 0.50
    pooled_log_or = re_log_or if use_re else fe_log_or
    pooled_se = re_se if use_re else fe_se
    estimator = "RE" if use_re else "FE"

    z = pooled_log_or / pooled_se
    p_vs_zero = 2 * scipy_stats.norm.sf(abs(z))

    result: dict = {
        "category": category_name,
        "log_or": pooled_log_or,
        "se": pooled_se,
        "ci_lower": pooled_log_or - z_crit * pooled_se,
        "ci_upper": pooled_log_or + z_crit * pooled_se,
        "z": z,
        "p_value": p_vs_zero,
        "Q": Q,
        "i2": i2,
        "tau2": tau2,
        "estimator": estimator,
        "n_genes": k,
        "n_ptc_trig": int(gene_df["n_ptc_trig"].sum()),
        "n_ptc_evad": int(gene_df["n_ptc_evad"].sum()),
        "fe_log_or": fe_log_or,
        "re_log_or": re_log_or,
    }

    # Comparison vs. Other Genes background
    if other_pooled is not None and not np.isnan(other_pooled.get("log_or", np.nan)):
        delta = pooled_log_or - other_pooled["log_or"]
        se_delta = np.sqrt(pooled_se ** 2 + other_pooled["se"] ** 2)
        z_vs_other = delta / se_delta
        result["delta_vs_other"] = delta
        result["z_vs_other"] = z_vs_other
        result["p_vs_other"] = float(2 * scipy_stats.norm.sf(abs(z_vs_other)))

    return result


# ===================================================================
# Plotting
# ===================================================================

def _fmt_p(p: float) -> str:
    """Format a p-value for annotation."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def plot_forest(results_rule: list[dict], results_ai: list[dict]) -> plt.Figure:
    """Forest plot of pooled meta-analysis log OR per gene category.

    Each category shows two estimates (circle = rule-based FE/RE pooled OR,
    open square = AI-filtered FE/RE pooled OR) with 95% CI whiskers.
    The "Other Genes" CI is shown as a grey background band (genome-wide
    reference).  I² and p-vs-Other are annotated to the right of each row.

    Args:
        results_rule: Per-category meta-analysis results (rule-based).
        results_ai: Per-category meta-analysis results (AI concordance filter).

    Returns:
        matplotlib Figure.
    """
    from matplotlib.lines import Line2D

    color_map = {
        "Tumor Suppressors": COLOR_TSG,
        "Oncogenes": COLOR_OG,
        "Essential Genes": COLOR_ESSENTIAL,
        "Other Genes": COLOR_OTHER,
    }

    categories = [r["category"] for r in results_rule]
    n = len(categories)
    y_positions = np.arange(n)
    offset = 0.18  # vertical offset to separate rule vs AI within a category

    fig, ax = plt.subplots(figsize=(9, 5))

    # Background band: "Other Genes" rule-based CI as reference
    other_rule = next(r for r in results_rule if r["category"] == "Other Genes")
    ax.axvspan(
        other_rule["ci_lower"], other_rule["ci_upper"],
        color="#cccccc", alpha=0.35, zorder=0,
    )

    for i, (rule, ai) in enumerate(zip(results_rule, results_ai)):
        cat = rule["category"]
        color = color_map.get(cat, "#666666")

        # ------- Rule-based (upper offset, circle marker) -------
        y_rule = y_positions[i] + offset
        if not np.isnan(rule.get("log_or", np.nan)):
            ax.errorbar(
                rule["log_or"], y_rule,
                xerr=[[rule["log_or"] - rule["ci_lower"]],
                      [rule["ci_upper"] - rule["log_or"]]],
                fmt="o", color=color, markersize=8,
                elinewidth=2, capsize=4, zorder=3,
                label="Rule-based" if i == 0 else None,
            )

        # ------- AI concordance filter (lower offset, open square) -------
        y_ai = y_positions[i] - offset
        if not np.isnan(ai.get("log_or", np.nan)):
            ax.errorbar(
                ai["log_or"], y_ai,
                xerr=[[ai["log_or"] - ai["ci_lower"]],
                      [ai["ci_upper"] - ai["log_or"]]],
                fmt="s", color=color, markersize=7, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.5,
                elinewidth=2, capsize=4, alpha=0.85, zorder=3,
                label="Rule + AI filter" if i == 0 else None,
            )

    # Null line
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6, zorder=1)

    ax.set_yticks(y_positions)
    labels = [f"{cat} ({next(r for r in results_rule if r['category'] == cat)['n_genes']} genes)" for cat in categories]
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel(
        "Pooled log OR  [NMD-triggering enrichment relative to synonymous]",
        fontsize=12, fontweight="bold"
    )
    ax.tick_params(axis="x", labelsize=14)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title("NMD Selection Pressure by Gene Category", fontsize=16, fontweight="bold")

    legend_elements = [
        Line2D([0], [0], marker="o", color="#444", markersize=7, linestyle="none",
               label="Rule-based (FE/RE pooled)"),
        Line2D([0], [0], marker="s", color="#444", markersize=6.5, linestyle="none",
               markerfacecolor="white", markeredgewidth=1.5,
               label="Rule + AI filter (FE/RE pooled)"),
        plt.Rectangle((0, 0), 1, 1, fc="#cccccc", alpha=0.35, edgecolor="none",
                       label="Other genes 95% CI"),
    ]
    ax.legend(handles=legend_elements, fontsize=9.5, loc="lower right", frameon=True)

    fig.tight_layout()
    return fig


# ===================================================================
# Main
# ===================================================================

def main(figure_label=None, figure_number=None, regenerate=True):
    """Run the gene-category NMD selection analysis."""
    paths = get_paths(SCRIPT_NAME, figure_label=figure_label, figure_number=figure_number)
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    if paths.source_data:
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting gene-category NMD selection analysis")

    # ------------------------------------------------------------------
    # Check for cached results
    # ------------------------------------------------------------------
    if paths.source_data and paths.source_data.exists() and not regenerate:
        logger.info(f"Loading existing results from {paths.source_data}")
        results_df = pd.read_csv(paths.source_data)
        results_rule = results_df[~results_df["use_ai_filter"]].to_dict("records")
        results_ai = results_df[results_df["use_ai_filter"]].to_dict("records")
    else:
        # ------------------------------------------------------------------
        # Load annotations and variant data
        # ------------------------------------------------------------------
        oncogenes, tumor_suppressors, essential_genes, exclusion_genes, gene_map = load_gene_annotations()

        logger.info(f"Loading nonsense variants from {NONSENSE_FILE}")
        df = pd.read_csv(NONSENSE_FILE, sep="\t")
        logger.info(f"Loaded {len(df)} nonsense variants")

        logger.info(f"Loading synonymous variants from {SYNONYMOUS_FILE}")
        syn = pd.read_csv(SYNONYMOUS_FILE, sep="\t")
        logger.info(f"Loaded {len(syn)} synonymous variants")

        # Map gene IDs -> gene names
        df = df.merge(gene_map, left_on="Gene", right_on="gene_id", how="left")
        syn = syn.merge(gene_map, left_on="Gene", right_on="gene_id", how="left")

        # Filter to genes with enough PTCs
        ptc_per_gene = df.groupby("gene_name").size()
        genes_ok = set(ptc_per_gene[ptc_per_gene >= MIN_PTC_GENE].index)
        logger.info(f"Genes with >= {MIN_PTC_GENE} PTCs: {len(genes_ok)}")

        # Define gene sets (only from genes with enough PTCs)
        all_genes = set(df["gene_name"].dropna()) & set(syn["gene_name"].dropna()) & genes_ok
        og_set = {g for g in oncogenes if g in all_genes}
        tsg_set = {g for g in tumor_suppressors if g in all_genes}
        ess_set = {g for g in essential_genes if g in all_genes}
        # Other Genes: exclude TSG/OG/Essential AND any gene with a known
        # cancer or disease annotation (all cancer_genes.csv rows +
        # gene_condition_source_id)
        pre_exclusion = all_genes - og_set - tsg_set - ess_set
        other_set = pre_exclusion - exclusion_genes
        logger.info(
            f"Other Genes after exclusion filter: {len(other_set)} "
            f"(removed {len(pre_exclusion) - len(other_set)} "
            f"additional disease/cancer-annotated genes)"
        )

        logger.info(
            f"Category sizes: TSG={len(tsg_set)}, OG={len(og_set)}, "
            f"Essential={len(ess_set)}, Other={len(other_set)}"
        )

        # Fit GMM thresholds for AI concordance filter
        gene_thresholds = fit_per_gene_gmm_thresholds(
            df[df["gene_name"].isin(all_genes)], gene_col="gene_name"
        )

        # ------------------------------------------------------------------
        # Run analysis twice: rule-based only, then rule + AI filter
        # ------------------------------------------------------------------
        all_rows = []
        all_gene_rows = []
        results_rule = []
        results_ai = []

        category_defs = [
            ("Tumor Suppressors", list(tsg_set)),
            ("Oncogenes", list(og_set)),
            ("Essential Genes", list(ess_set)),
            ("Other Genes", list(other_set)),
        ]

        for use_ai in [False, True]:
            label = "Rule + AI filter" if use_ai else "Rule-based"
            logger.info(f"\n{'='*60}\n{label}\n{'='*60}")

            df_cls = classify_nmd_regions(
                df, use_concordance_filter=use_ai, gene_thresholds=gene_thresholds
            )
            syn_cls = classify_nmd_regions(syn, use_concordance_filter=False)

            logger.info(f"PTC region distribution:\n{df_cls['nmd_region'].value_counts()}")

            # Per-gene log ORs for every category
            gene_dfs: dict[str, pd.DataFrame] = {}
            for cat_name, cat_genes in category_defs:
                gdf = per_gene_log_or(df_cls, syn_cls, cat_genes)
                gdf["category"] = cat_name
                gdf["use_ai_filter"] = use_ai
                gene_dfs[cat_name] = gdf
                all_gene_rows.append(gdf)
                logger.info(
                    f"  {cat_name}: {len(gdf)} genes pass filters "
                    f"(of {len(cat_genes)} total)"
                )

            # Pool Other Genes first (used as reference for all categories)
            other_pooled = pool_category_meta(
                gene_dfs["Other Genes"], "Other Genes", other_pooled=None
            )

            run_results = []
            for cat_name, _ in category_defs:
                ref = None if cat_name == "Other Genes" else other_pooled
                res = pool_category_meta(gene_dfs[cat_name], cat_name, other_pooled=ref)
                res["use_ai_filter"] = use_ai
                run_results.append(res)
                all_rows.append(res)

                log_msg = (
                    f"  {cat_name} ({res['estimator']}): "
                    f"log OR = {res['log_or']:.3f} "
                    f"[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}], "
                    f"p_vs_zero = {res['p_value']:.2e}, "
                    f"I\u00b2 = {res['i2']:.0%}, "
                    f"{res['n_genes']} genes"
                )
                if "p_vs_other" in res:
                    log_msg += f", p_vs_other = {res['p_vs_other']:.2e}"
                logger.info(log_msg)

            if use_ai:
                results_ai = run_results
            else:
                results_rule = run_results

        # Save source data
        if paths.source_data:
            pd.DataFrame(all_rows).to_csv(paths.source_data, index=False)
            logger.info(f"Saved category results to {paths.source_data}")
            gene_path = paths.source_data.with_name(
                paths.source_data.stem + "_per_gene" + paths.source_data.suffix
            )
            pd.concat(all_gene_rows, ignore_index=True).to_csv(gene_path, index=False)
            logger.info(f"Saved per-gene log ORs to {gene_path}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig = plot_forest(results_rule, results_ai)
    fig.savefig(paths.figure_png, dpi=300, bbox_inches="tight")
    fig.savefig(paths.figure_pdf, bbox_inches="tight")
    logger.info(f"Saved figure to {paths.figure_png}")
    plt.close(fig)

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
