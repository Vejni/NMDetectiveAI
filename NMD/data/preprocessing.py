from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

from NMD.config import (
    INTERIM_DATA_DIR,
    FIGURES_DIR,
    VAL_CHRS,
)
from NMD.plots import plot_germline_somatic_overlap_comparison
from NMD.data.DatasetConfig import DatasetConfig


def create_rule_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary NMD rule labels.
    
    Args:
        df: DataFrame with PTC data
        
    Returns:
        DataFrame with added rule columns
    """
    df = df.copy()
    
    # Binary rule labels (fixed thresholds: 100nt for start, 400nt for long exon)
    df["Last_Exon"] = df["last_exon"].apply(lambda x: 1 if x == "yes" else 0)
    df["Penultimate_Exon"] = df["X55_nt"].apply(lambda x: 1 if x == "yes" else 0)
    df["Start_Prox"] = df["PTC_CDS_pos"].apply(lambda x: 1 if pd.notna(x) and x <= 150 else 0)
    df["Long_Exon"] = df["PTC_CDS_exon_length"].apply(lambda x: 1 if pd.notna(x) and x >= 400 else 0)
    df["NMD_Triggering"] = 1 - df[["Last_Exon", "Penultimate_Exon", "Start_Prox", "Long_Exon"]].max(axis=1)
    
    # Standardized column names for NMDetectiveA
    df["InLastExon"] = df["Last_Exon"]
    df["50ntToLastEJ"] = df["Penultimate_Exon"]
    df["DistanceToStart"] = df["PTC_CDS_pos"].clip(upper=1000)
    df["ExonLength"] = df["PTC_CDS_exon_length"]
    df["DistanceToWTStop"] = df["normal_stop_codon_CDS_pos"] - df["PTC_CDS_pos"]
    
    return df


def impute_rna_halflife(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing RNA half-life values with median.
    
    Args:
        df: DataFrame with PTC data
        
    Returns:
        DataFrame with imputed RNAHalfLife column
    """
    df = df.copy()
    median_halflife = df["half_life"].median()
    df["RNAHalfLife"] = df["half_life"].fillna(median_halflife)
    logger.info(f"Imputed missing RNA half-life values with median: {median_halflife:.2f}")
    return df


def remove_somatic_overlap(
    df: pd.DataFrame, 
    input_file: Path,
) -> pd.DataFrame:
    """
    Remove variants that overlap with somatic TCGA dataset.
    
    Args:
        df: DataFrame with PTC data
        input_file: Path to input file (for naming plots)
        
    Returns:
        DataFrame with overlapping variants removed
    """
    df = df.copy()
    somatic_interim_path = INTERIM_DATA_DIR / "PTC" / "somatic_TCGA.csv"
    
    if not somatic_interim_path.exists():
        logger.warning(f"Somatic TCGA interim file not found at {somatic_interim_path}, skipping overlap removal")
        return df
    
    logger.info(f"Removing variants that overlap with somatic TCGA...")
    somatic_df = pd.read_csv(somatic_interim_path)
    
    # Create variant IDs (chr:pos:ref:alt)
    df['variant_id'] = df['chr'].astype(str) + ':' + df['start_pos'].astype(str) + ':' + df['Ref'] + ':' + df['Alt']
    somatic_df['variant_id'] = somatic_df['chr'].astype(str) + ':' + somatic_df['start_pos'].astype(str) + ':' + somatic_df['Ref'] + ':' + somatic_df['Alt']
    
    # Determine overlap mask based on detection method
    somatic_variants = set(somatic_df['variant_id'])
    somatic_sequences = set(somatic_df['fasta_sequence_mut'])
    
    overlap_mask_coords = df['variant_id'].isin(somatic_variants)
    overlap_mask_seqs = df['fasta_sequence_mut'].isin(somatic_sequences)
    overlap_mask = overlap_mask_coords | overlap_mask_seqs
    
    # Log detailed statistics
    n_overlap_coords = overlap_mask_coords.sum()
    n_overlap_seqs = overlap_mask_seqs.sum()
    logger.info(f"Overlapping variants by genomic coordinates: {n_overlap_coords}")
    logger.info(f"Overlapping variants by fasta_sequence_mut: {n_overlap_seqs}")
    
    if n_overlap_coords > 0:
        overlapping_df = df[overlap_mask_coords].copy()
        somatic_dict = dict(zip(somatic_df['variant_id'], somatic_df['fasta_sequence_mut']))
        overlapping_df['somatic_seq'] = overlapping_df['variant_id'].map(somatic_dict)
        seq_match = (overlapping_df['fasta_sequence_mut'] == overlapping_df['somatic_seq']).sum()
        logger.info(f"  -> Of coordinate overlaps, {seq_match}/{n_overlap_coords} also have matching sequences")
    
    n_overlap_total = overlap_mask.sum()
    logger.info(f"Total overlapping variants: {n_overlap_total}")
    
    # Plot comparison before removing overlaps
    if n_overlap_total > 0:
        plot_output_dir = FIGURES_DIR / "data" / "PTC"
        os.makedirs(plot_output_dir, exist_ok=True)
        dataset_name = "GTEx" if "GTEx" in str(input_file) else "germline_TCGA"
        overlap_plot_path = plot_output_dir / f"{input_file.stem}_somatic_overlap_comparison.png"
        plot_germline_somatic_overlap_comparison(df, somatic_df, overlap_mask, overlap_plot_path, dataset_name)
    
    # Remove overlapping variants
    df = df[~overlap_mask].copy()
    df = df.drop(columns=['variant_id'])
    
    logger.info(f"Removed {n_overlap_total} overlapping variants with somatic TCGA")
    logger.info(f"Number of rows after removing overlaps: {len(df)}")
    
    return df


def apply_lenient_expression_filter(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Apply lenient expression and CV filtering to all chromosomes (somatic only).

    Uses the train-tier thresholds (``min_tpm_train``, ``max_cv_train``) and is
    applied globally, so the resulting data does not depend on the train/val split.
    This step should run before saving any intermediate data.

    Args:
        df: DataFrame with PTC data.
        config: Dataset configuration.

    Returns:
        Filtered DataFrame.
    """
    df = df.copy()
    mask = (
        (df["median_TPM_exp_transcript"] >= config.min_tpm_train) &
        (df["coeff_var"] <= config.max_cv_train)
    )
    df = df[mask]
    logger.info(f"After lenient expression/CV filter (all chrs): {len(df)} rows")
    return df


def apply_strict_val_expression_filter(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Apply strict expression and CV filtering to validation chromosomes only.

    Uses the val-tier thresholds (``min_tpm_val``, ``max_cv_val``) restricted to
    ``VAL_CHRS``. Non-validation chromosomes are not touched. This step should run
    *after* saving intermediate data so that the intermediate file reflects only
    the split-agnostic lenient filter.

    Args:
        df: DataFrame with PTC data.
        config: Dataset configuration.

    Returns:
        Filtered DataFrame.
    """
    df = df.copy()
    val_chr_mask = df["chr"].isin(VAL_CHRS)

    cv_mask = (df["coeff_var"] <= config.max_cv_val) | (~val_chr_mask)
    df = df[cv_mask]

    tpm_mask = (df["median_TPM_exp_transcript"] >= config.min_tpm_val) | (~val_chr_mask)
    df = df[tpm_mask]

    logger.info(f"After strict expression/CV filter (val chrs only): {len(df)} rows")
    return df


def apply_splice_site_filter(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """
    Filter PTCs adjacent to splice sites.
    
    Args:
        df: DataFrame with PTC data
        config: Dataset configuration
        
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    
    # Calculate exon positions
    df['exons_length_prePTC_sum'] = df['exons_length_prePTC'].apply(_sum_exon_lengths)
    df['PTC_pos_in_exon'] = df['PTC_CDS_pos'] - df['exons_length_prePTC_sum']
    df['dist_to_exon_end'] = df['PTC_CDS_exon_length'] - df['PTC_pos_in_exon'] - 3
    
    # Filter PTCs within threshold of either splice site
    threshold = config.splice_proximity_threshold
    near_start = (df['PTC_pos_in_exon'] < threshold) & (df['PTC_pos_in_exon'] >= 0)
    near_end = (df['dist_to_exon_end'] < threshold) & (df['dist_to_exon_end'] >= 0)
    near_splice = near_start | near_end
    
    logger.info(f"Filtering PTCs adjacent to splice sites (within {threshold}nt):")
    logger.info(f"  PTCs near exon start: {near_start.sum()} ({near_start.sum()/len(df)*100:.2f}%)")
    logger.info(f"  PTCs near exon end: {near_end.sum()} ({near_end.sum()/len(df)*100:.2f}%)")
    logger.info(f"  Total PTCs filtered: {near_splice.sum()} ({near_splice.sum()/len(df)*100:.2f}%)")
    
    df = df[~near_splice]
    logger.info(f"Number of rows after splice site filtering: {len(df)}")
    
    return df


def apply_vaf_filter(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """
    Apply variant allele frequency filtering (germline only).
    
    Args:
        df: DataFrame with PTC data
        config: Dataset configuration
        
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    df = df[df["VAF"] <= config.max_vaf]
    df = df[df["VAF"] > config.min_vaf]
    logger.info(f"After VAF filter ({config.min_vaf} < VAF <= {config.max_vaf}): {len(df)} rows")
    return df


def apply_frameshift_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct frameshift bias by adjusting mean to match SNV PTCs.
    
    Args:
        df: DataFrame with PTC data
        
    Returns:
        DataFrame with corrected NMD efficiency values
    """
    df = df.copy()
    
    if not (df["stopgain"] != "nonsense").any():
        logger.info("No frameshift variants found, skipping FS correction")
        return df
    
    fs_mask = (df["stopgain"] != "nonsense") & (df["NMD_Triggering"] == 1)
    snv_mask = (df["stopgain"] == "nonsense") & (df["NMD_Triggering"] == 1)
    
    mean_diff = (
        df.loc[snv_mask, "ASE_NMD_efficiency_TPM"].mean()
        - df.loc[fs_mask, "ASE_NMD_efficiency_TPM"].mean()
    )
    logger.info(f"Mean difference between SNV and FS triggering PTCs: {mean_diff:.4f}")
    
    df.loc[fs_mask, "ASE_NMD_efficiency_TPM"] = (
        df.loc[fs_mask, "ASE_NMD_efficiency_TPM"] + mean_diff
    )
    
    return df


def apply_regression_correction(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """
    Apply regression correction to remove technical confounders.
    
    Args:
        df: DataFrame with PTC data
        config: Dataset configuration
        
    Returns:
        DataFrame with regression-corrected NMD efficiency values
    """
    df = df.copy()
    
    # Build list of predictor columns
    predictor_cols = []
    for predictor in config.regression_predictors:
        if predictor == "tissue_PCs":
            # Add tissue PC1-4
            tissue_pcs = [col for col in df.columns if col.startswith("tissue_PC") and col[-1] in "1234"]
            predictor_cols.extend(tissue_pcs)
        else:
            if predictor in df.columns:
                predictor_cols.append(predictor)
            else:
                logger.warning(f"Predictor '{predictor}' not found in dataframe, skipping")
    
    if not predictor_cols:
        logger.warning("No valid predictors found, skipping regression correction")
        return df
    
    # Drop rows with NA in predictor columns
    df_clean = df.dropna(subset=predictor_cols)
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped} rows with missing predictor values")
    
    # Fit regression model(s)
    if config.regression_separate_by_variant_type:
        # Separate models for SNVs and indels
        snv_mask = df_clean["stopgain"] == "nonsense"
        indel_mask = df_clean["stopgain"] != "nonsense"
        
        # SNVs
        if snv_mask.sum() > 0:
            X_snv = df_clean.loc[snv_mask, predictor_cols]
            y_snv = df_clean.loc[snv_mask, "ASE_NMD_efficiency_TPM"]
            reg_snv = LinearRegression()
            reg_snv.fit(X_snv, y_snv)
            predictions_snv = reg_snv.predict(X_snv)
            df.loc[df_clean.loc[snv_mask].index, "ASE_NMD_efficiency_TPM"] = y_snv - predictions_snv
            logger.info(f"SNV model - R²: {reg_snv.score(X_snv, y_snv):.3f}, N={len(X_snv)}")
        
        # Indels
        if indel_mask.sum() > 0:
            X_indel = df_clean.loc[indel_mask, predictor_cols]
            y_indel = df_clean.loc[indel_mask, "ASE_NMD_efficiency_TPM"]
            reg_indel = LinearRegression()
            reg_indel.fit(X_indel, y_indel)
            predictions_indel = reg_indel.predict(X_indel)
            df.loc[df_clean.loc[indel_mask].index, "ASE_NMD_efficiency_TPM"] = y_indel - predictions_indel
            logger.info(f"Indel model - R²: {reg_indel.score(X_indel, y_indel):.3f}, N={len(X_indel)}")
    else:
        # Combined model
        X = df_clean[predictor_cols]
        y = df_clean["ASE_NMD_efficiency_TPM"]
        reg = LinearRegression()
        reg.fit(X, y)
        predictions = reg.predict(X)
        df.loc[df_clean.index, "ASE_NMD_efficiency_TPM"] = y - predictions
        logger.info(f"Combined model - R²: {reg.score(X, y):.3f}, N={len(X)}")
    
    logger.info(f"Regression correction applied using predictors: {predictor_cols}")
    
    return df


def center_nmd_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Center NMD efficiency values by subtracting the mean.
    
    Args:
        df: DataFrame with PTC data
        
    Returns:
        DataFrame with centered NMD efficiency
    """
    df = df.copy()
    mean_val = df["ASE_NMD_efficiency_TPM"].mean()
    df["ASE_NMD_efficiency_TPM"] = df["ASE_NMD_efficiency_TPM"] - mean_val
    logger.info(f"Centered NMD efficiency values (subtracted mean: {mean_val:.4f})")
    return df


def aggregate_variants(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """
    Aggregate variants by grouping identical variants across samples.
    
    Args:
        df: DataFrame with PTC data
        config: Dataset configuration
        
    Returns:
        Aggregated DataFrame
    """
    # Columns to group by
    groupby_cols = [
        "gene_id",
        "transcript_id",
        "chr",
        "start_pos",
        "Ref",
        "Alt",
        "stopgain",
        "fasta_sequence_mut",
        "fasta_sequence_wt",
        "seq_5UTR",
        "seq_3UTR",
        "UTR5s_length",
        "UTR3s_length",
        "original_stop_codon",
        "exons_length_postPTC",
        "exons_length_prePTC",
        "PTC_CDS_exon_length",
        "PTC_stop_codon_type",
        "Last_Exon",
        "Penultimate_Exon",
        "Start_Prox",
        "Long_Exon",
        "NMD_Triggering",
        "CDS_num_exons_downstream",
        "PTC_CDS_exon_num",
        "PTC_EJC_dist",
        "PTC_CDS_pos",
        "CDS_mut_length",
        "normal_stop_codon_CDS_pos",
        "InLastExon",
        "50ntToLastEJ",
        "DistanceToStart",
        "ExonLength",
        "DistanceToWTStop",
        "transcript_length",
    ]
    
    # Aggregation functions
    agg_method = config.aggregation_method
    agg_dict = {
        "ASE_NMD_efficiency_TPM": [agg_method, "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        "RNAHalfLife": "median",
        "transcript_id": "count",
    }
    
    df_agg = df.groupby(groupby_cols).agg(agg_dict).reset_index()
    
    # Flatten column names
    df_agg.columns = [
        "_".join(col).strip("_") if col[1] else col[0] for col in df_agg.columns.values
    ]
    
    # Rename columns
    df_agg = df_agg.rename(columns={
        f"ASE_NMD_efficiency_TPM_{agg_method}": "ASE_NMD_efficiency_TPM",
        "ASE_NMD_efficiency_TPM_std": "ASE_NMD_efficiency_std",
        "ASE_NMD_efficiency_TPM_<lambda_0>": "ASE_NMD_efficiency_Q25",
        "ASE_NMD_efficiency_TPM_<lambda_1>": "ASE_NMD_efficiency_Q75",
        "transcript_id_count": "count",
        "RNAHalfLife_median": "RNAHalfLife"
    })
    
    # Calculate confidence intervals
    df_agg["ASE_NMD_efficiency_CI_lower"] = df_agg["ASE_NMD_efficiency_Q25"]
    df_agg["ASE_NMD_efficiency_CI_upper"] = df_agg["ASE_NMD_efficiency_Q75"]
    
    logger.info(f"Aggregated variants using {agg_method}: {len(df)} -> {len(df_agg)} rows")
    logger.info(f"Number of unique fasta_sequence_mut: {df_agg['fasta_sequence_mut'].nunique()}")
    
    return df_agg


def apply_threshold_filter(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """
    Filter extreme NMD efficiency values.
    
    Args:
        df: DataFrame with PTC data
        config: Dataset configuration
        
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    threshold = config.nmd_efficiency_threshold
    df = df[
        (df["ASE_NMD_efficiency_TPM"] < threshold) &
        (df["ASE_NMD_efficiency_TPM"] > -threshold)
    ]
    logger.info(f"After threshold filter (±{threshold}): {len(df)} rows")
    return df


def normalize_nmd_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize NMD efficiency using triggering vs last exon PTCs.
    
    Args:
        df: DataFrame with PTC data
        
    Returns:
        DataFrame with normalized NMDeff_Norm column
    """
    df = df.copy()
    
    triggering_mask = (df["NMD_Triggering"] == 1)
    last_exon_mask = (df["Last_Exon"] == 1)
    
    triggering_mean = df.loc[triggering_mask, "ASE_NMD_efficiency_TPM"].mean()
    last_exon_mean = df.loc[last_exon_mask, "ASE_NMD_efficiency_TPM"].mean()
    
    # Calculate center and scale
    center = (triggering_mean + last_exon_mean) / 2
    scale = np.abs(triggering_mean - last_exon_mean)
    
    # Apply normalization
    df["NMDeff_Norm"] = (df["ASE_NMD_efficiency_TPM"] - center) / scale
    
    logger.info(f"Normalized NMD efficiency - center: {center:.4f}, scale: {scale:.4f}")
    logger.info(f"  Triggering mean: {triggering_mean:.4f}, Last exon mean: {last_exon_mean:.4f}")
    
    return df


def apply_transcript_length_filter(
    df: pd.DataFrame, 
    var_type: str,
    config: DatasetConfig
) -> pd.DataFrame:
    """
    Filter transcripts that exceed maximum length.
    
    Args:
        df: DataFrame with PTC data
        var_type: Type of variants ("somatic", "germline", or "gtex")
        config: Dataset configuration
        
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    max_length = config.max_transcript_length
    
    if var_type == "germline" and not config.apply_length_filter_to_germline:
        logger.info("Transcript length filter not applied to germline (disabled in config)")
        return df
    
    if config.apply_length_filter_to_val_chrs:
        # Apply filter to all chromosomes
        df = df[df["transcript_length"] <= max_length]
        logger.info(f"After transcript length filter (≤{max_length}nt, all chrs): {len(df)} rows")
    else:
        # Exempt validation chromosomes
        df = df[(df["transcript_length"] <= max_length) | (df["chr"].isin(VAL_CHRS))]
        logger.info(f"After transcript length filter (≤{max_length}nt, train chrs only): {len(df)} rows")
    
    return df


def _sum_exon_lengths(exon_str):
    """Sum comma-separated exon lengths."""
    if pd.isna(exon_str):
        return 0
    try:
        return sum([int(x) for x in str(exon_str).split(',')])
    except:
        return 0
