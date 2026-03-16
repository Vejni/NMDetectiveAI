"""
gnomAD NMD triggering/evading ratio by disease gene — combined Test 1 + Test 2.

# ---------------------------------------------------------------------------
# Variant Count vs AC Sum Tag
# ---------------------------------------------------------------------------
# Set USE_AC_SUM = True to use sum of AC for logodds calculations (default behavior).
# Set USE_AC_SUM = False to use count of variants instead.
USE_AC_SUM = True

Two complementary tests are computed for each qualifying gene:

Test 1 (synonymous-normalised):
    PTCs in NMD-triggering vs NMD-evading regions, normalised by synonymous
    variant allele counts in the same regions.

        |            | NMD-triggering | NMD-evading |
        |------------|----------------|-------------|
        | PTCs       | AC_trig        | AC_evad     |
        | Synonymous | AC_syn_trig    | AC_syn_evad |

Test 2 (common-vs-rare PTCs, per-gene median-AF split):
    For each gene, PTC variants are ranked by AF and split at the median into
    a "rare" half (lower AF) and a "common" half (higher AF).

        |        | NMD-triggering | NMD-evading  |
        |--------|----------------|--------------|
        | Rare   | AC_com_trig    | AC_com_evad |
        | Common | AC_rare_trig   | AC_rare_evad  |

Both tests use the Wald method on log(OR) with Haldane 0.5 pseudocounts.
A combined p-value is derived via Fisher's method (-2·sum(ln(p)) ~ chi²(4)).

GENE_SELECTION_PVALUE controls which FDR-corrected p-value is used to select
genes for the figure ('p_value_fdr_t1', 'p_value_fdr_t2', or
'p_value_fdr_combined').

The figure shows two horizontal bars per gene: Test 1 (solid) and Test 2
(hatched), both with 95 % Wald CIs, coloured by LOEUF decile.

Requires preprocessed data:
    - data/processed/gnomad_v4.1/annotated_rare/...rare_stopgain_snv.mane.annotated_with_predictions.tsv
    - data/processed/gnomad_v4.1/annotated_common/...common_stopgain_snv.mane.annotated_with_predictions.tsv
    - data/processed/gnomad_v4.1/synonymous/...synonymous.mane.nmd_region_summary.tsv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from matplotlib.patches import Patch
from loguru import logger
from scipy import stats as scipy_stats

from NMD.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, MANUSCRIPT_TABLES_DIR
from NMD.manuscript.output import get_paths

SCRIPT_NAME = "gnomad_top_genes"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

USE_AI_PREDICTIONS = True
AI_THRESHOLD_TRIGGERING = 0.43
AI_THRESHOLD_EVADING = -0.17

TOP_N = 10 
MAF_THRESHOLD = 1.0  # Genes with any variant AF >= this value are dropped entirely
CI_LEVEL = 0.95

# Which FDR column to use when ranking and selecting genes for the figure:
#   'p_value_fdr_t1', 'p_value_fdr_t2', or 'p_value_fdr_combined'
GENE_SELECTION_PVALUE = "p_value_fdr_combined"

MIN_PTC = 10   # Minimum unique PTC variants per gene
MIN_SYN = 1   # Minimum synonymous AC in each NMD region

# Filtering for Test 2
FDR_T1_THRESHOLD = 0.01  # Drop genes with p_value_fdr_t1 >= this before Test 2
FDR_T2_THRESHOLD = 0.25  # Drop genes with p_value_fdr_t2 >= this
USE_SIGN_AGREEMENT = True  # Drop genes where log_odds_t1 and log_odds_t2 have different signs

ADD_MISSENSE_AS_BACKGROUND = True  # When True, add missense variant counts to the synonymous denominator in Test 1
USE_AC_SUM_T1 = True  # When True, use sum of AC for log-odds calculations; if False, use count of variants instead (for both tests)
USE_AC_SUM_T2 = True  # When True, use sum of AC for log-odds calculations; if False, use count of variants instead (for both tests)
MAX_MAF_T1 = 0.01

SPLIT_MAF_THRESHOLD_T2 = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RARE_PTC_FILE = PROCESSED_DATA_DIR / "gnomad_v4.1" / "annotated_rare" / "gnomad.v4.1.all_chromosomes.rare_stopgain_snv.mane.annotated_with_predictions.tsv"
COMMON_PTC_FILE = PROCESSED_DATA_DIR / "gnomad_v4.1" / "annotated_common" / "gnomad.v4.1.all_chromosomes.common_stopgain_snv.mane.annotated_with_predictions.tsv"
SYNONYMOUS_FILE = PROCESSED_DATA_DIR / "gnomad_v4.1" / "synonymous" / "gnomad.v4.1.all_chromosomes.synonymous.mane.nmd_region_summary.tsv"
MISSENSE_FILE   = PROCESSED_DATA_DIR / "gnomad_v4.1" / "missense"   / "gnomad.v4.1.all_chromosomes.missense.mane.nmd_region_summary.tsv"
CLINVAR_GENE_FILE = RAW_DATA_DIR / "annotations" / "gene_condition_source_id"
CONSTRAINT_FILE = RAW_DATA_DIR / "annotations" / "supplementary_dataset_11_full_constraint_metrics.tsv"

# Supplementary table (all passing genes, both tests) saved to manuscript tables dir
SUPP_TABLE = MANUSCRIPT_TABLES_DIR / f"{SCRIPT_NAME}.csv"
# Internal cache to skip re-processing when regenerate=False
# _CACHE_TABLE = TABLES_DIR / "selection" / "gnomad_nmd_ratio.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_nmd(df: pd.DataFrame, use_ai: bool,
                  ai_thr_trig: float, ai_thr_evad: float) -> pd.DataFrame:
    """Add 'is_nmd_triggering' bool column; drop intermediate AI-score variants.

    Args:
        df: PTC DataFrame.
        use_ai: Use AI predictions if True, rule-based otherwise.
        ai_thr_trig: AI score >= this → NMD-triggering.
        ai_thr_evad: AI score <= this → NMD-evading.

    Returns:
        Annotated (and filtered) copy of df.
    """
    df = df.copy()
    if use_ai:
        df['is_nmd_triggering'] = df['NMDetectiveAI_prediction'].apply(
            lambda x: True if x >= ai_thr_trig else (False if x <= ai_thr_evad else None)
        )
        n_drop = df['is_nmd_triggering'].isna().sum()
        if n_drop:
            logger.info(f"  Dropping {n_drop} variants with intermediate AI scores")
            df = df.dropna(subset=['is_nmd_triggering']).copy()
        df['is_nmd_triggering'] = df['is_nmd_triggering'].astype(bool)
    else:
        df['is_nmd_triggering'] = df['predicted_nmd_status'] == 'NMD_triggering'
    return df


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values.

    Returns:
        Array of BH-corrected q-values, capped at 1.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.argsort(order) + 1
    fdr = p_values * n / ranks
    fdr_ordered = fdr[order]
    for i in range(n - 2, -1, -1):
        fdr_ordered[i] = min(fdr_ordered[i], fdr_ordered[i + 1])
    out = np.empty(n)
    out[order] = fdr_ordered
    return np.minimum(out, 1.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_disease_genes() -> set:
    """Load disease gene list from ClinVar."""
    logger.info(f"Loading disease genes from {CLINVAR_GENE_FILE}")
    df = pd.read_csv(CLINVAR_GENE_FILE, sep='\t', comment='#',
                     names=['GeneID', 'AssociatedGenes', 'RelatedGenes', 'ConceptID',
                            'DiseaseName', 'SourceName', 'SourceID', 'DiseaseMIM', 'LastUpdated'],
                     skiprows=1)
    disease_genes = set(df['AssociatedGenes'].dropna().unique())
    disease_genes.discard('')
    logger.info(f"Loaded {len(disease_genes)} unique disease genes from ClinVar")
    return disease_genes


def load_ptcs() -> pd.DataFrame:
    """Load and merge rare + common PTCs; drop genes with any variant AF >= MAF_THRESHOLD.

    Returns:
        Combined PTC DataFrame with high-AF genes removed and AF cast to float.
    """
    logger.info(f"Loading rare PTCs from {RARE_PTC_FILE}")
    rare_df = pd.read_csv(RARE_PTC_FILE, sep='\t')
    logger.info(f"  Loaded {len(rare_df)} rare PTCs")

    logger.info(f"Loading common PTCs from {COMMON_PTC_FILE}")
    common_df = pd.read_csv(COMMON_PTC_FILE, sep='\t')
    logger.info(f"  Loaded {len(common_df)} common PTCs")

    combined = pd.concat([rare_df, common_df], ignore_index=True)
    logger.info(f"Combined PTC dataset: {len(combined)} variants")

    combined['AF'] = combined['AF'].astype(float)
    above_mask = combined['AF'] >= MAF_THRESHOLD
    if above_mask.any():
        genes_to_drop = combined.loc[above_mask, 'gene_symbol'].unique()
        logger.info(f"Dropping {len(genes_to_drop)} genes with variants AF >= {MAF_THRESHOLD}")
        combined = combined[~combined['gene_symbol'].isin(genes_to_drop)]
        logger.info(f"After dropping: {len(combined)} variants remaining")
    else:
        logger.info(f"No variants with AF >= {MAF_THRESHOLD}; no genes dropped")
    return combined


def load_synonymous_counts() -> pd.DataFrame:
    """Load NMD-region-stratified synonymous variant counts by gene.

    When ADD_MISSENSE_AS_BACKGROUND is True, the missense NMD-region counts
    are added to the synonymous counts to form the combined denominator.
    """
    logger.info(f"Loading synonymous counts from {SYNONYMOUS_FILE}")
    df = pd.read_csv(SYNONYMOUS_FILE, sep='\t')
    logger.info(f"Loaded region-stratified synonymous counts for {len(df)} genes")

    if ADD_MISSENSE_AS_BACKGROUND:
        if not MISSENSE_FILE.exists():
            raise FileNotFoundError(
                f"Missense background file not found at {MISSENSE_FILE}. "
                "Run: python -m NMD.data.process_gnomad_synonymous --chr all --variant-type missense"
            )
        logger.info(f"ADD_MISSENSE_AS_BACKGROUND=True – loading missense counts from {MISSENSE_FILE}")
        mis = pd.read_csv(MISSENSE_FILE, sep='\t')
        logger.info(f"  Loaded missense counts for {len(mis)} genes")
        ac_cols = ['ac_synonymous_nmd_triggering', 'ac_synonymous_nmd_evading']
        n_cols  = ['n_synonymous_variants_nmd_triggering', 'n_synonymous_variants_nmd_evading']
        merged = df.merge(mis[['gene_symbol'] + ac_cols + n_cols], on='gene_symbol',
                          how='outer', suffixes=('_syn', '_mis'))
        for col in ac_cols + n_cols:
            merged[col] = merged[f'{col}_syn'].fillna(0) + merged[f'{col}_mis'].fillna(0)
        df = merged[['gene_symbol'] + ac_cols + n_cols].copy()
        logger.info(f"Combined synonymous + missense denominator: {len(df)} genes")

    logger.info(f"  NMD-triggering region variants: {df['n_synonymous_variants_nmd_triggering'].sum():.0f}")
    logger.info(f"  NMD-evading region variants: {df['n_synonymous_variants_nmd_evading'].sum():.0f}")
    return df


def load_constraint_metrics() -> pd.DataFrame:
    """Load gnomAD gene constraint metrics (LOEUF decile 0–9)."""
    logger.info(f"Loading constraint metrics from {CONSTRAINT_FILE}")
    df = pd.read_csv(CONSTRAINT_FILE, sep='\t')
    df = df[['gene', 'oe_lof_upper_bin', 'oe_lof_upper_bin_6', 'canonical']].copy()
    df_canon = df[df['canonical'] == True].drop_duplicates(subset=['gene'], keep='first')
    df_other = df[~df['gene'].isin(df_canon['gene'])].drop_duplicates(subset=['gene'], keep='first')
    df_unique = pd.concat([df_canon, df_other], ignore_index=True)
    df_unique = df_unique.rename(columns={
        'gene': 'gene_symbol',
        'oe_lof_upper_bin': 'loeuf_decile',  # 0=most constrained, 9=least
    })
    logger.info(f"Loaded constraint metrics for {len(df_unique)} genes")
    return df_unique[['gene_symbol', 'loeuf_decile']]


# ---------------------------------------------------------------------------
# Test 1: PTC allele counts vs synonymous normalisation
# ---------------------------------------------------------------------------

def calculate_test1(original_ptc_df: pd.DataFrame, synonymous_df: pd.DataFrame,
                    ci_level: float = 0.95) -> pd.DataFrame:
    """Compute log OR (NMD-triggering / NMD-evading) normalised by synonymous AC.

    2×2 Wald table per gene (Haldane 0.5 pseudocount):
        a = AC of NMD-triggering PTCs
        b = AC of NMD-evading PTCs
        c = AC of NMD-triggering synonymous variants
        d = AC of NMD-evading synonymous variants
    log OR = log(a/c) − log(b/d)

    Args:
        ptc_df: PTC DataFrame with 'is_nmd_triggering' column already set.
        synonymous_df: Region-stratified synonymous counts DataFrame.
        ci_level: Confidence interval level.

    Returns:
        Per-gene DataFrame with Test 1 statistics (columns suffixed _t1).
    """
    ptc_df = original_ptc_df.copy()

    if MAX_MAF_T1 is not None:
        # Drop PTCs not genes with AF >= MAX_MAF_T1 to prevent extreme outliers dominating the plot and combined p-values
        above_mask = ptc_df['AF'] >= MAX_MAF_T1
        ptc_df = ptc_df[~above_mask]
        logger.info(f"After dropping PTCs with AF >= {MAX_MAF_T1}: {len(ptc_df)} variants remaining")

    if USE_AC_SUM_T1:
        trig = ptc_df[ptc_df['is_nmd_triggering']].groupby('gene_symbol').agg(
            ac_nmd_triggering=('AC', 'sum'),
            n_triggering=('gene_symbol', 'size'),
        ).reset_index()
        evad = ptc_df[~ptc_df['is_nmd_triggering']].groupby('gene_symbol').agg(
            ac_nmd_evading=('AC', 'sum'),
            n_evading=('gene_symbol', 'size'),
        ).reset_index()
        total = ptc_df.groupby('gene_symbol').agg(
            ac_total=('AC', 'sum'),
            n_variants=('gene_symbol', 'size'),
        ).reset_index()
    else:
        trig = ptc_df[ptc_df['is_nmd_triggering']].groupby('gene_symbol').agg(
            ac_nmd_triggering=('AC', lambda x: len(x)),
            n_triggering=('gene_symbol', 'size'),
        ).reset_index()
        evad = ptc_df[~ptc_df['is_nmd_triggering']].groupby('gene_symbol').agg(
            ac_nmd_evading=('AC', lambda x: len(x)),
            n_evading=('gene_symbol', 'size'),
        ).reset_index()
        total = ptc_df.groupby('gene_symbol').agg(
            ac_total=('AC', lambda x: len(x)),
            n_variants=('gene_symbol', 'size'),
        ).reset_index()

    stats = (total
             .merge(trig, on='gene_symbol', how='left')
             .merge(evad, on='gene_symbol', how='left'))
    for col in ['ac_nmd_triggering', 'ac_nmd_evading', 'n_triggering', 'n_evading']:
        stats[col] = stats[col].fillna(0)

    syn_cols = ['gene_symbol',
                'ac_synonymous_nmd_triggering', 'ac_synonymous_nmd_evading',
                'n_synonymous_variants_nmd_triggering', 'n_synonymous_variants_nmd_evading']
    stats = stats.merge(synonymous_df[syn_cols], on='gene_symbol', how='left')
    for col in syn_cols[1:]:
        stats[col] = stats[col].fillna(0)

    pseudocount = 0.5
    a = stats['ac_nmd_triggering'] + pseudocount
    b = stats['ac_nmd_evading'] + pseudocount
    c = stats['ac_synonymous_nmd_triggering'] + pseudocount
    d = stats['ac_synonymous_nmd_evading'] + pseudocount

    stats['log_odds_t1'] = np.log(a / c) - np.log(b / d)
    stats['se_log_or_t1'] = np.sqrt(1/a + 1/b + 1/c + 1/d)

    z_crit = scipy_stats.norm.ppf(1 - (1 - ci_level) / 2)
    stats['ci_lower_t1'] = stats['log_odds_t1'] - z_crit * stats['se_log_or_t1']
    stats['ci_upper_t1'] = stats['log_odds_t1'] + z_crit * stats['se_log_or_t1']
    stats['z_score_t1'] = stats['log_odds_t1'] / stats['se_log_or_t1']
    stats['p_value_t1'] = 2 * scipy_stats.norm.sf(np.abs(stats['z_score_t1']))

    logger.info(f"Test 1 computed for {len(stats)} genes "
                f"(median log OR = {stats['log_odds_t1'].median():.3f})")
    return stats


# ---------------------------------------------------------------------------
# Test 2: rare-vs-common PTCs using a per-gene median-AF split
# ---------------------------------------------------------------------------

def calculate_test2(ptc_df: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    """Compute log OR using a per-gene median-AF split of PTC variants.

    For each gene, variants are sorted by AF and split at the median:
    the lower half is the "rare" test set, the upper half is the "common"
    background.  The 2×2 Wald table (Haldane 0.5 pseudocount):
        a = AC of rare  NMD-triggering PTCs
        b = AC of rare  NMD-evading PTCs
        c = AC of common NMD-triggering PTCs
        d = AC of common NMD-evading PTCs
    log OR = log(a/c) − log(b/d)

    Args:
        ptc_df: PTC DataFrame with 'is_nmd_triggering' and 'AF' columns.
        ci_level: Confidence interval level.

    Returns:
        Per-gene DataFrame with Test 2 statistics (columns suffixed _t2).
    """
    rows = []
    if SPLIT_MAF_THRESHOLD_T2 is not None:
        # Global split: rare = AF < threshold, common = AF >= threshold
        for gene, group in ptc_df.groupby('gene_symbol'):
            rare = group[group['AF'] < SPLIT_MAF_THRESHOLD_T2]
            common = group[group['AF'] >= SPLIT_MAF_THRESHOLD_T2]
            af_split = SPLIT_MAF_THRESHOLD_T2

            if USE_AC_SUM_T2:
                ac_rare_triggering = rare.loc[rare['is_nmd_triggering'], 'AC'].sum()
                ac_rare_evading = rare.loc[~rare['is_nmd_triggering'], 'AC'].sum()
                ac_common_triggering = common.loc[common['is_nmd_triggering'], 'AC'].sum() if len(common) else 0
                ac_common_evading = common.loc[~common['is_nmd_triggering'], 'AC'].sum() if len(common) else 0
            else:
                ac_rare_triggering = rare.loc[rare['is_nmd_triggering'], 'AC'].count()
                ac_rare_evading = rare.loc[~rare['is_nmd_triggering'], 'AC'].count()
                ac_common_triggering = common.loc[common['is_nmd_triggering'], 'AC'].count() if len(common) else 0
                ac_common_evading = common.loc[~common['is_nmd_triggering'], 'AC'].count() if len(common) else 0

            rows.append({
                'gene_symbol': gene,
                'n_rare_t2': len(rare),
                'n_common_t2': len(common),
                'af_split_t2': af_split,
                'ac_rare_triggering_t2': ac_rare_triggering,
                'ac_rare_evading_t2': ac_rare_evading,
                'ac_common_triggering_t2': ac_common_triggering,
                'ac_common_evading_t2': ac_common_evading,
                'n_rare_triggering_t2': int(rare['is_nmd_triggering'].sum()),
                'n_rare_evading_t2': int((~rare['is_nmd_triggering']).sum()),
                'n_common_triggering_t2': int(common['is_nmd_triggering'].sum()) if len(common) else 0,
                'n_common_evading_t2': int((~common['is_nmd_triggering']).sum()) if len(common) else 0,
            })
    else:
        # Per-gene median split (default)
        for gene, group in ptc_df.groupby('gene_symbol'):
            group_sorted = group.sort_values('AF').reset_index(drop=True)
            n = len(group_sorted)
            n_rare = (n + 1) // 2  # lower half (extra variant goes to rare if odd)
            rare = group_sorted.iloc[:n_rare]
            common = group_sorted.iloc[n_rare:]
            af_split = group_sorted['AF'].iloc[n_rare - 1]

            if USE_AC_SUM_T2:
                ac_rare_triggering = rare.loc[rare['is_nmd_triggering'], 'AC'].sum()
                ac_rare_evading = rare.loc[~rare['is_nmd_triggering'], 'AC'].sum()
                ac_common_triggering = common.loc[common['is_nmd_triggering'], 'AC'].sum() if len(common) else 0
                ac_common_evading = common.loc[~common['is_nmd_triggering'], 'AC'].sum() if len(common) else 0
            else:
                ac_rare_triggering = rare.loc[rare['is_nmd_triggering'], 'AC'].count()
                ac_rare_evading = rare.loc[~rare['is_nmd_triggering'], 'AC'].count()
                ac_common_triggering = common.loc[common['is_nmd_triggering'], 'AC'].count() if len(common) else 0
                ac_common_evading = common.loc[~common['is_nmd_triggering'], 'AC'].count() if len(common) else 0

            rows.append({
                'gene_symbol': gene,
                'n_rare_t2': len(rare),
                'n_common_t2': len(common),
                'af_split_t2': af_split,
                'ac_rare_triggering_t2': ac_rare_triggering,
                'ac_rare_evading_t2': ac_rare_evading,
                'ac_common_triggering_t2': ac_common_triggering,
                'ac_common_evading_t2': ac_common_evading,
                'n_rare_triggering_t2': int(rare['is_nmd_triggering'].sum()),
                'n_rare_evading_t2': int((~rare['is_nmd_triggering']).sum()),
                'n_common_triggering_t2': int(common['is_nmd_triggering'].sum()) if len(common) else 0,
                'n_common_evading_t2': int((~common['is_nmd_triggering']).sum()) if len(common) else 0,
            })

    stats = pd.DataFrame(rows)

    pseudocount = 0.5
    a = stats['ac_rare_triggering_t2'] + pseudocount
    b = stats['ac_rare_evading_t2'] + pseudocount
    c = stats['ac_common_triggering_t2'] + pseudocount
    d = stats['ac_common_evading_t2'] + pseudocount

    stats['log_odds_t2'] = np.log(c / a) - np.log(d / b) # flipped to match Test 1 direction
    stats['se_log_or_t2'] = np.sqrt(1/a + 1/b + 1/c + 1/d)

    z_crit = scipy_stats.norm.ppf(1 - (1 - ci_level) / 2)
    stats['ci_lower_t2'] = stats['log_odds_t2'] - z_crit * stats['se_log_or_t2']
    stats['ci_upper_t2'] = stats['log_odds_t2'] + z_crit * stats['se_log_or_t2']
    stats['z_score_t2'] = stats['log_odds_t2'] / stats['se_log_or_t2']
    stats['p_value_t2'] = 2 * scipy_stats.norm.sf(np.abs(stats['z_score_t2']))

    logger.info(f"Test 2 computed for {len(stats)} genes "
                f"(median log OR = {stats['log_odds_t2'].median():.3f})")
    return stats


# ---------------------------------------------------------------------------
# Combine tests
# ---------------------------------------------------------------------------

def combine_tests(t1: pd.DataFrame, t2: pd.DataFrame) -> pd.DataFrame:
    """Merge Test 1 and Test 2 statistics and compute combined p-value (Fisher).

    Fisher's method: chi² = −2·(ln p₁ + ln p₂) ~ chi²(4 df).
    BH FDR correction is applied separately to p_value_t1, p_value_t2,
    and p_value_combined.

    Args:
        t1: Output of calculate_test1.
        t2: Output of calculate_test2.

    Returns:
        Merged DataFrame with _fdr columns appended.
    """
    stats = t1.merge(t2, on='gene_symbol', how='inner')

    chi2 = -2 * (np.log(stats['p_value_t1'].clip(lower=1e-300)) +
                 np.log(stats['p_value_t2'].clip(lower=1e-300)))
    stats['p_value_combined'] = scipy_stats.chi2.sf(chi2, df=4)

    stats['p_value_fdr_t1'] = _bh_fdr(stats['p_value_t1'].values)

    # Compute FDR for test2 only on those passing FDR_T1_THRESHOLD, NA elsewhere
    mask_t1 = stats['p_value_fdr_t1'] < FDR_T1_THRESHOLD
    fdr_t2 = np.full(len(stats), np.nan)
    if mask_t1.sum() > 0:
        fdr_t2[mask_t1] = _bh_fdr(stats.loc[mask_t1, 'p_value_t2'].values)
    stats['p_value_fdr_t2'] = fdr_t2

    # Combined FDR: still on all genes (or could restrict similarly if needed)
    stats['p_value_fdr_combined'] = _bh_fdr(stats['p_value_combined'].values)

    logger.info(f"Combined results: {len(stats)} genes")
    logger.info(f"  FDR(T1) < {FDR_T1_THRESHOLD}: {(stats['p_value_fdr_t1'] < FDR_T1_THRESHOLD).sum()}")
    logger.info(f"  FDR(T2) < {FDR_T2_THRESHOLD}: {(stats['p_value_fdr_t2'] < FDR_T2_THRESHOLD).sum()}")
    return stats


# ---------------------------------------------------------------------------
# Gene selection
# ---------------------------------------------------------------------------

def select_top_enriched_genes(stats: pd.DataFrame, top_n: int,
                               ranking_col: str) -> list:
    """Select top N most significant genes per direction based on ranking column.

    Direction is determined by the sign of log_odds_t1.
    No FDR threshold applied here; assumes stats are pre-filtered.

    Args:
        stats: Combined gene-level statistics DataFrame (pre-filtered).
        top_n: Number of genes per direction (triggering & evading).
        ranking_col: Column to rank by (e.g. 'p_value_t1').

    Returns:
        List of gene symbols to plot.
    """
    logger.info(f"Selecting top {top_n} genes per direction from {len(stats)} genes "
                f"based on {ranking_col}")

    top_evading = (stats[stats['log_odds_t1'] < 0]
                   .sort_values(ranking_col).head(top_n)['gene_symbol'].tolist())
    top_triggering = (stats[stats['log_odds_t1'] > 0]
                      .sort_values(ranking_col).head(top_n)['gene_symbol'].tolist())
    selected = list(set(top_evading + top_triggering))

    logger.info(f"  Top {top_n} evading-enriched : {top_evading}")
    logger.info(f"  Top {top_n} triggering-enriched: {top_triggering}")
    return selected


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_barplot(stats: pd.DataFrame, genes_to_plot: list,
                 constraint_df: pd.DataFrame = None,
                 n_tested: int = None, n_pass_t1: int = None, n_pass_t2: int = None,
                 r_all: float = None, p_all: float = None):
    """Horizontal bar chart with two bars per gene (Test 1 solid, Test 2 hatched).

    Genes are ordered by Test 1 log OR (most evading at top).
    LOEUF decile determines bar colour; Test 2 bars carry a '//' hatch pattern.

    Args:
        stats: Combined gene-level statistics DataFrame.
        genes_to_plot: Gene symbols to include.
        constraint_df: Optional DataFrame with 'gene_symbol' and 'loeuf_decile'.
        n_tested: Number of genes tested.
        n_pass_t1: Number of genes passing Test 1 FDR threshold.
        n_pass_t2: Number of genes passing Test 2 FDR threshold.
        r_all: Pearson r on all genes tested.
        p_all: p-value for r_all.

    Returns:
        Tuple (fig, plot_data) or None if no genes matched.
    """
    # Merge LOEUF data to stats
    if constraint_df is not None:
        stats_with_loeuf = stats.merge(constraint_df, on='gene_symbol', how='left')
    else:
        stats_with_loeuf = stats.copy()
        stats_with_loeuf['loeuf_decile'] = None

    # Compute Spearman correlation between LOEUF and combined p-value on all FDR-filtered genes
    rho, p_spearman = None, None
    if stats_with_loeuf['loeuf_decile'].notna().any():
        rho, p_spearman = scipy_stats.spearmanr(stats_with_loeuf['loeuf_decile'].dropna(), 
                                                stats_with_loeuf.loc[stats_with_loeuf['loeuf_decile'].notna(), 'log_odds_t1'].abs() * (-1))

    plot_data = stats[stats['gene_symbol'].isin(genes_to_plot)].copy()
    if len(plot_data) == 0:
        logger.error(f"None of the selected genes found: {genes_to_plot}")
        return None

    if constraint_df is not None:
        plot_data = plot_data.merge(constraint_df, on='gene_symbol', how='left')
        logger.info(f"Merged LOEUF data: "
                    f"{plot_data['loeuf_decile'].notna().sum()}/{len(plot_data)} genes")
    else:
        plot_data['loeuf_decile'] = None

    plot_data = plot_data.sort_values('log_odds_t1', ascending=True).reset_index(drop=True)
    logger.info(f"Plotting {len(plot_data)} genes: {plot_data['gene_symbol'].tolist()}")

    # Compute correlation between Test 1 and Test 2 log odds on all FDR-filtered genes
    r, p = scipy_stats.pearsonr(stats['log_odds_t1'], stats['log_odds_t2'])

    n_genes = len(plot_data)
    bar_height = 0.35
    fig, ax = plt.subplots(figsize=(12, max(6, n_genes * 0.6 + 1.5)))

    cmap = mcolors.LinearSegmentedColormap.from_list('loeuf', ['#ff9e9d', '#022778'])

    for i, (_, row) in enumerate(plot_data.iterrows()):
        loeuf = row.get('loeuf_decile')
        color = cmap(loeuf / 9.0) if pd.notna(loeuf) else '#808080'
        y_t1 = i + bar_height / 2
        y_t2 = i - bar_height / 2

        # Test 1 – solid bar
        ax.barh(y_t1, row['log_odds_t1'], height=bar_height,
                color=color, edgecolor='white', linewidth=0.8, alpha=0.85)
        ax.errorbar(row['log_odds_t1'], y_t1,
                    xerr=[[max(0, row['log_odds_t1'] - row['ci_lower_t1'])],
                          [max(0, row['ci_upper_t1'] - row['log_odds_t1'])]],
                    fmt='none', ecolor='black', elinewidth=1.4, capsize=3, capthick=1.4)

        # Test 2 – hatched bar
        ax.barh(y_t2, row['log_odds_t2'], height=bar_height,
                color=color, edgecolor='white', linewidth=0.8, alpha=0.50, hatch='//')
        ax.errorbar(row['log_odds_t2'], y_t2,
                    xerr=[[max(0, row['log_odds_t2'] - row['ci_lower_t2'])],
                          [max(0, row['ci_upper_t2'] - row['log_odds_t2'])]],
                    fmt='none', ecolor='black', elinewidth=1.4, capsize=3, capthick=1.4)

    # Add horizontal dashed line between evading and triggering groups
    ax.axhline(y=10 - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_yticks(np.arange(n_genes))
    labels = [f"{r['gene_symbol']} (#PTCs={int(r['n_variants'])})"
              for _, r in plot_data.iterrows()]
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel('log OR$_2$ (NMD triggering vs NMD evading)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Gene', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    legend_handles = [
        Patch(facecolor='gray', edgecolor='white', alpha=0.85,
              label=f'Test 1 (all PTC vs synonymous{" + missense" if ADD_MISSENSE_AS_BACKGROUND else ""})'),
        Patch(facecolor='gray', edgecolor='white', alpha=0.50, hatch='//',
              label='Test 2 (common vs rare PTCs)'),
    ]
    ax.legend(handles=legend_handles, fontsize=12, loc='lower right')

    # Add correlation annotation
    text_lines = []
    if n_tested is not None:
        text_lines.append(f'Genes tested: {n_tested}')
    if n_pass_t1 is not None:
        text_lines.append(f'Pass T1 FDR ({FDR_T1_THRESHOLD * 100:.0f}%): {n_pass_t1}')
    if n_pass_t2 is not None:
        text_lines.append(f'Pass T1+T2 FDR ({FDR_T2_THRESHOLD * 100:.0f}%): {n_pass_t2}')
    if r_all is not None and p_all is not None:
        text_lines.append(f'Pearson r (all): {r_all:.3f} (p={p_all:.2e})')
    text_lines.append(f'Pearson r (FDR filtered): {r:.3f} (p={p:.2e})')
    #if rho is not None and p_spearman is not None:
    #    text_lines.append(f'Spearman ρ\n(LOEUF vs OR-T1): {rho:.3f} (p={p_spearman:.2e})')
    text = '\n'.join(text_lines)
    ax.text(0.03, 0.95, text, 
            transform=ax.transAxes, fontsize=12, verticalalignment='top', 
            horizontalalignment='left', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # LOEUF colorbar
    cbar_ax = fig.add_axes([0.83, 0.15, 0.02, 0.30])
    norm = mcolors.Normalize(vmin=0, vmax=9)
    cb = mcolorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('LOEUF decile', fontsize=13, fontweight='bold')
    cb.set_ticks([0, 4.5, 9])
    cb.set_ticklabels(['Constrained\n(0)', 'Intermediate\n(4.5)', 'Unconstrained\n(9)'])
    cb.ax.tick_params(labelsize=11)
    plt.suptitle('Enrichment of NMD-evading and NMD-triggering PTCs in disease genes', fontsize=18, fontweight='bold')

    plt.tight_layout()
    return fig, plot_data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(figure_label=None, figure_number=None, regenerate=True):
    """Run both tests, save supplementary table and figure."""
    paths = get_paths(SCRIPT_NAME, figure_label=figure_label, figure_number=figure_number)
    _CACHE_TABLE = paths.source_data
    paths.figure_png.parent.mkdir(parents=True, exist_ok=True)
    if paths.source_data:
        paths.source_data.parent.mkdir(parents=True, exist_ok=True)
    SUPP_TABLE.parent.mkdir(parents=True, exist_ok=True)
    # _CACHE_TABLE.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting gnomAD NMD ratio analysis (combined Test 1 + Test 2)")
    logger.info(f"  USE_AI_PREDICTIONS    : {USE_AI_PREDICTIONS}")
    logger.info(f"  MAF_THRESHOLD         : {MAF_THRESHOLD}")
    logger.info(f"  MIN_PTC / MIN_SYN     : {MIN_PTC} / {MIN_SYN}")
    logger.info(f"  TOP_N                 : {TOP_N}")
    logger.info(f"  FDR_T2_THRESHOLD      : {FDR_T2_THRESHOLD}")
    logger.info(f"  USE_SIGN_AGREEMENT    : {USE_SIGN_AGREEMENT}")

    if SUPP_TABLE.exists() and not regenerate:
        logger.info(f"Loading cached results from {SUPP_TABLE}")
        stats = pd.read_csv(SUPP_TABLE)
    else:
        disease_genes = load_disease_genes()
        ptc_df = load_ptcs()
        synonymous_df = load_synonymous_counts()

        # Classify NMD status
        ptc_df = _classify_nmd(ptc_df, USE_AI_PREDICTIONS, AI_THRESHOLD_TRIGGERING, AI_THRESHOLD_EVADING)

        # Keep disease genes only
        before = ptc_df['gene_symbol'].nunique()
        ptc_df = ptc_df[ptc_df['gene_symbol'].isin(disease_genes)].copy()
        logger.info(f"Filtered for disease genes: {before - ptc_df['gene_symbol'].nunique()} removed, "
                    f"{ptc_df['gene_symbol'].nunique()} remaining")

        # Require MIN_PTC unique variants
        gene_counts = ptc_df.groupby('gene_symbol').size()
        valid_genes = gene_counts[gene_counts >= MIN_PTC].index
        before = ptc_df['gene_symbol'].nunique()
        ptc_df = ptc_df[ptc_df['gene_symbol'].isin(valid_genes)].copy()
        logger.info(f"Filtered genes with < {MIN_PTC} unique PTCs: "
                    f"{before - ptc_df['gene_symbol'].nunique()} removed, "
                    f"{ptc_df['gene_symbol'].nunique()} remaining")

        # Require MIN_SYN variants in each NMD region
        before = len(synonymous_df)
        synonymous_df = synonymous_df[(synonymous_df['n_synonymous_variants_nmd_triggering'] >= MIN_SYN) &
                           (synonymous_df['n_synonymous_variants_nmd_evading'] >= MIN_SYN)].copy()
        logger.info(f"Filtered synonymous data for >= {MIN_SYN} variants per NMD region: "
                f"{before - len(synonymous_df)} genes removed, {len(synonymous_df)} remaining")
        ptc_df = ptc_df[ptc_df['gene_symbol'].isin(synonymous_df['gene_symbol'])].copy()

        # Test 1
        logger.info("Computing Test 1 (PTC vs synonymous normalisation)")
        t1 = calculate_test1(ptc_df, synonymous_df, ci_level=CI_LEVEL)

        # Test 2 (only for genes kept after T1 filter)
        logger.info("Computing Test 2 (per-gene median AF split)")
        t2 = calculate_test2(ptc_df, ci_level=CI_LEVEL)

        # Combine
        stats = combine_tests(t1, t2)

        stats.to_csv(SUPP_TABLE, index=False)
        logger.info(f"Saved supplementary table (all genes tested) to {SUPP_TABLE}")

        # Compute stats for plot annotation
        stats_full = stats.copy()
        r_all, p_all = scipy_stats.pearsonr(stats_full['log_odds_t1'], stats_full['log_odds_t2'])
        n_tested = len(stats_full)

    # Apply FDR 1
    before = len(stats)
    stats = stats[stats['p_value_fdr_t1'] < FDR_T1_THRESHOLD].copy()
    logger.info(f"Filtered genes with p_value_fdr_t1 >= {FDR_T1_THRESHOLD}: "
                f"{before - len(stats)} removed, {len(stats)} remaining")
    n_pass_t1 = len(stats)

    # Apply FDR 2
    before = len(stats)
    stats = stats[stats['p_value_fdr_t2'] < FDR_T2_THRESHOLD].copy()
    logger.info(f"Filtered genes with p_value_fdr_t2 >= {FDR_T2_THRESHOLD}: "
                f"{before - len(stats)} removed, {len(stats)} remaining")
    n_pass_t2 = len(stats)

    # Optional: filter for sign agreement
    if USE_SIGN_AGREEMENT:
        before = len(stats)
        sign_agree = np.sign(stats['log_odds_t1']) == np.sign(stats['log_odds_t2'])
        stats = stats[sign_agree].copy()
        logger.info(f"Filtered genes with discordant log_odds signs: "
                    f"{before - len(stats)} removed, {len(stats)} remaining")

    # Load constraint metrics for colouring
    constraint_df = load_constraint_metrics()

    # Select genes for the figure based on Test 1 p-value
    genes_to_plot = select_top_enriched_genes(stats, TOP_N, ranking_col=GENE_SELECTION_PVALUE)

    # Plot
    result = plot_barplot(stats, genes_to_plot, constraint_df, n_tested, n_pass_t1, n_pass_t2, r_all, p_all)
    if result is None:
        logger.warning("No genes to plot; figure not saved.")
        return

    fig, plot_data = result

    # Source data – only plotted genes and the columns shown in the figure
    if _CACHE_TABLE:
        source_cols = [
            'gene_symbol', 'n_variants',
            'log_odds_t1', 'ci_lower_t1', 'ci_upper_t1', 'p_value_t1', 'p_value_fdr_t1',
            'log_odds_t2', 'ci_lower_t2', 'ci_upper_t2', 'p_value_t2', 'p_value_fdr_t2',
            'p_value_combined', 'p_value_fdr_combined',
        ]
        available = [c for c in source_cols if c in plot_data.columns]
        plot_data[available].to_csv(_CACHE_TABLE, index=False)
        logger.info(f"Saved source data to {_CACHE_TABLE}")

    fig.savefig(paths.figure_png, dpi=300, bbox_inches='tight')
    fig.savefig(paths.figure_pdf, bbox_inches='tight')
    logger.info(f"Saved figure to {paths.figure_png}")
    logger.info(f"Saved PDF to {paths.figure_pdf}")
    plt.close(fig)

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
