from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import typer
from NMD.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
)
from NMD.data.transcripts import (
    get_UTR_sequences,
    get_stopcodons,
    create_6track_onehot_sequence,
)
from NMD.data.preprocessing import (
    create_rule_labels,
    impute_rna_halflife,
    remove_somatic_overlap,
    apply_lenient_expression_filter,
    apply_strict_val_expression_filter,
    apply_splice_site_filter,
    apply_vaf_filter,
    apply_frameshift_correction,
    apply_regression_correction,
    center_nmd_efficiency,
    aggregate_variants,
    apply_threshold_filter,
    normalize_nmd_efficiency,
    apply_transcript_length_filter,
)
from NMD.data.DatasetConfig import DatasetConfig
from NMD.plots import (
    plot_nmd_efficiency_distributions,
    plot_preprocessing_steps,
)
import os


app = typer.Typer()


def _calculate_preprocessing_stats(df, step_name):
    """
    Calculate statistics for a preprocessing step.
    
    Args:
        df: DataFrame with processed PTC data
        step_name: Name of the preprocessing step
        
    Returns:
        Dictionary with statistics
    """
    # Use NMDeff column if it exists, otherwise use ASE_NMD_efficiency_TPM
    nmd_col = "NMDeff" if "NMDeff" in df.columns else "ASE_NMD_efficiency_TPM"
    
    # Calculate counts
    n_snvs = len(df[df["stopgain"] == "nonsense"])
    n_indels = len(df[df["stopgain"] != "nonsense"])
    
    # Calculate statistics for different groups
    stats_dict = {
        "step": step_name,
        "n_snvs": n_snvs,
        "n_indels": n_indels,
    }
    
    # Masks for variant types
    snv_mask = df["stopgain"] == "nonsense"
    indel_mask = df["stopgain"] != "nonsense"
    
    # Last exon statistics - SNVs
    last_exon_snv_mask = (df["Last_Exon"] == 1) & snv_mask
    if last_exon_snv_mask.sum() > 0:
        last_exon_snv_values = df.loc[last_exon_snv_mask, nmd_col].dropna()
        stats_dict["last_exon_snv_mean"] = last_exon_snv_values.mean()
        stats_dict["last_exon_snv_ci_lower"] = last_exon_snv_values.mean() - 1.96 * last_exon_snv_values.sem()
        stats_dict["last_exon_snv_ci_upper"] = last_exon_snv_values.mean() + 1.96 * last_exon_snv_values.sem()
    else:
        stats_dict["last_exon_snv_mean"] = np.nan
        stats_dict["last_exon_snv_ci_lower"] = np.nan
        stats_dict["last_exon_snv_ci_upper"] = np.nan
    
    # Last exon statistics - Indels
    last_exon_indel_mask = (df["Last_Exon"] == 1) & indel_mask
    if last_exon_indel_mask.sum() > 0:
        last_exon_indel_values = df.loc[last_exon_indel_mask, nmd_col].dropna()
        stats_dict["last_exon_indel_mean"] = last_exon_indel_values.mean()
        stats_dict["last_exon_indel_ci_lower"] = last_exon_indel_values.mean() - 1.96 * last_exon_indel_values.sem()
        stats_dict["last_exon_indel_ci_upper"] = last_exon_indel_values.mean() + 1.96 * last_exon_indel_values.sem()
    else:
        stats_dict["last_exon_indel_mean"] = np.nan
        stats_dict["last_exon_indel_ci_lower"] = np.nan
        stats_dict["last_exon_indel_ci_upper"] = np.nan
    
    # Evading groups (any rule triggered) - SNVs
    evading_snv_mask = (df[["Last_Exon", "Penultimate_Exon", "Start_Prox", "Long_Exon"]].max(axis=1) == 1) & snv_mask
    if evading_snv_mask.sum() > 0:
        evading_snv_values = df.loc[evading_snv_mask, nmd_col].dropna()
        stats_dict["evading_snv_mean"] = evading_snv_values.mean()
        stats_dict["evading_snv_ci_lower"] = evading_snv_values.mean() - 1.96 * evading_snv_values.sem()
        stats_dict["evading_snv_ci_upper"] = evading_snv_values.mean() + 1.96 * evading_snv_values.sem()
    else:
        stats_dict["evading_snv_mean"] = np.nan
        stats_dict["evading_snv_ci_lower"] = np.nan
        stats_dict["evading_snv_ci_upper"] = np.nan
    
    # Evading groups (any rule triggered) - Indels
    evading_indel_mask = (df[["Last_Exon", "Penultimate_Exon", "Start_Prox", "Long_Exon"]].max(axis=1) == 1) & indel_mask
    if evading_indel_mask.sum() > 0:
        evading_indel_values = df.loc[evading_indel_mask, nmd_col].dropna()
        stats_dict["evading_indel_mean"] = evading_indel_values.mean()
        stats_dict["evading_indel_ci_lower"] = evading_indel_values.mean() - 1.96 * evading_indel_values.sem()
        stats_dict["evading_indel_ci_upper"] = evading_indel_values.mean() + 1.96 * evading_indel_values.sem()
    else:
        stats_dict["evading_indel_mean"] = np.nan
        stats_dict["evading_indel_ci_lower"] = np.nan
        stats_dict["evading_indel_ci_upper"] = np.nan
    
    # Triggering (no rules triggered) - SNVs
    triggering_snv_mask = (df["NMD_Triggering"] == 1) & snv_mask
    if triggering_snv_mask.sum() > 0:
        triggering_snv_values = df.loc[triggering_snv_mask, nmd_col].dropna()
        stats_dict["triggering_snv_mean"] = triggering_snv_values.mean()
        stats_dict["triggering_snv_ci_lower"] = triggering_snv_values.mean() - 1.96 * triggering_snv_values.sem()
        stats_dict["triggering_snv_ci_upper"] = triggering_snv_values.mean() + 1.96 * triggering_snv_values.sem()
    else:
        stats_dict["triggering_snv_mean"] = np.nan
        stats_dict["triggering_snv_ci_lower"] = np.nan
        stats_dict["triggering_snv_ci_upper"] = np.nan
    
    # Triggering (no rules triggered) - Indels
    triggering_indel_mask = (df["NMD_Triggering"] == 1) & indel_mask
    if triggering_indel_mask.sum() > 0:
        triggering_indel_values = df.loc[triggering_indel_mask, nmd_col].dropna()
        stats_dict["triggering_indel_mean"] = triggering_indel_values.mean()
        stats_dict["triggering_indel_ci_lower"] = triggering_indel_values.mean() - 1.96 * triggering_indel_values.sem()
        stats_dict["triggering_indel_ci_upper"] = triggering_indel_values.mean() + 1.96 * triggering_indel_values.sem()
    else:
        stats_dict["triggering_indel_mean"] = np.nan
        stats_dict["triggering_indel_ci_lower"] = np.nan
        stats_dict["triggering_indel_ci_upper"] = np.nan
    
    return stats_dict


def save_and_stats_after_main_filters(df, output_path):
    os.makedirs(output_path.parent, exist_ok=True)

    # log number of variants, SNVs and indels
    n_variants = len(df)
    n_snvs = len(df[df["stopgain"] == "nonsense"])
    n_indels = len(df[df["stopgain"] != "nonsense"])
    logger.info(f"Number of variants after initial filters: {n_variants} (SNVs: {n_snvs}, Indels: {n_indels})")

    # log mean NMD efficiency for triggering vs evading groups
    if "NMDeff" in df.columns:
        triggering_mean = df[df["NMD_Triggering"] == 1]["NMDeff"].mean()
        evading_mean = df[df[["Last_Exon", "Penultimate_Exon", "Start_Prox", "Long_Exon"]].max(axis=1) == 1]["NMDeff"].mean()
        logger.info(f"Mean NMD efficiency - Triggering group: {triggering_mean:.4f}, Evading group: {evading_mean:.4f}")
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved intermediate data after initial filters to {output_path}")


def process_PTC_dataset_with_config(
    input_file: Path,
    output_name: str = None,
    config: DatasetConfig = None
):
    """
    Process PTC dataset with modular preprocessing pipeline (programmatic API).
    
    Args:
        input_file: Input filename in RAW_DATA_DIR/PTC/
        output_name: Optional output name (defaults to input_file stem)
        config: DatasetConfig instance (defaults to DatasetConfig())
    """
    if config is None:
        config = DatasetConfig()
    
    input_file = Path(input_file)
    input_path = RAW_DATA_DIR / "PTC" / input_file
    output_path: Path = PROCESSED_DATA_DIR / "PTC" / (output_name if output_name else input_path.name.replace(".txt", ""))
    
    # Determine variant type
    var_type = "somatic" if "somatic" in str(input_file) else "germline"
    if "gtex" in str(input_file).lower():
        var_type = "gtex"
    
    # Read the input file
    df = pd.read_csv(input_path, sep="\t")
    logger.info(f"Starting preprocessing for {input_file.name}")
    logger.info(f"Number of rows before filtering: {len(df)}")

    # ===== STEP 0: Fill NA values =====
    df["exons_length_postPTC"] = df["exons_length_postPTC"].fillna(0)
    df["exons_length_prePTC"] = df["exons_length_prePTC"].fillna(0)
    df["UTR3s_length"] = df["UTR3s_length"].fillna(0)
    df["UTR5s_length"] = df["UTR5s_length"].fillna(0)
    df = impute_rna_halflife(df)
    
    # Initialize preprocessing tracking list
    preprocessing_stats = []
    os.makedirs(output_path.parent, exist_ok=True)

    # ===== STEP 1: Create rule labels =====
    df = create_rule_labels(df)
    
    # Track initial state
    df["NMDeff"] = df["ASE_NMD_efficiency_TPM"]
    preprocessing_stats.append(_calculate_preprocessing_stats(df, "Initial"))

    # ===== STEP 2: Lenient expression/CV filtering (somatic only, all chrs) =====
    if config.apply_expression_filter and var_type == "somatic":
        df = apply_lenient_expression_filter(df, config)
        preprocessing_stats.append(_calculate_preprocessing_stats(df, "Expression/CV Filtered (lenient)"))

    # ===== STEP 3: Splice site filtering =====
    if config.apply_splice_filter:
        df = apply_splice_site_filter(df, config)
        preprocessing_stats.append(_calculate_preprocessing_stats(df, "Splice Site Filtered"))

    # ===== STEP 4: VAF filtering (germline only) =====
    if config.apply_vaf_filter and var_type == "germline":
        df = apply_vaf_filter(df, config)
        preprocessing_stats.append(_calculate_preprocessing_stats(df, "VAF Filtered"))

    # Save intermediate data after split-agnostic filtering steps
    save_and_stats_after_main_filters(df, TABLES_DIR / "data" / "PTC" / f"{input_file.stem}_after_initial_filters.csv")

    # ===== STEP 4b: Strict expression/CV filtering on val chrs (somatic only) =====
    if config.apply_expression_filter and var_type == "somatic":
        df = apply_strict_val_expression_filter(df, config)
        preprocessing_stats.append(_calculate_preprocessing_stats(df, "Expression/CV Filtered (strict val chrs)"))

    # ===== STEP 5: Remove somatic overlap =====
    if config.somatic_overlap_removal == "remove_from_germline" and var_type != "somatic":
        df = remove_somatic_overlap(df, input_file)
        preprocessing_stats.append(_calculate_preprocessing_stats(df, "Removed Somatic Overlaps"))
    elif config.somatic_overlap_removal == "remove_from_somatic" and var_type == "somatic":
        df = remove_somatic_overlap(df, input_file)
        preprocessing_stats.append(_calculate_preprocessing_stats(df, "Removed Germline Overlaps"))

    # ===== STEP 6: Frameshift correction =====
    if config.apply_frameshift_correction:
        df = apply_frameshift_correction(df)
        df["NMDeff"] = df["ASE_NMD_efficiency_TPM"]
        preprocessing_stats.append(_calculate_preprocessing_stats(df, "FS Corrected"))

    # ===== STEP 7: Get UTR sequences and stop codons =====
    df = get_UTR_sequences(df)
    df = get_stopcodons(df)

    # ===== STEP 8: Center NMD values =====
    if config.center_nmd_efficiency:
        df = center_nmd_efficiency(df)
        df["NMDeff"] = df["ASE_NMD_efficiency_TPM"]
        preprocessing_stats.append(_calculate_preprocessing_stats(df, "Centered"))

    # ===== STEP 9: Regression correction =====
    if config.apply_regression_correction:
        df = apply_regression_correction(df, config)
        df["NMDeff"] = df["ASE_NMD_efficiency_TPM"]
        preprocessing_stats.append(_calculate_preprocessing_stats(df, "Regression Corrected"))

    # ===== STEP 10: Aggregate variants =====
    df = aggregate_variants(df, config)
    df["NMDeff"] = df["ASE_NMD_efficiency_TPM"]
    preprocessing_stats.append(_calculate_preprocessing_stats(df, "Grouped"))

    # ===== STEP 11: Threshold filtering =====
    if config.apply_threshold_filter:
        df = apply_threshold_filter(df, config)
        preprocessing_stats.append(_calculate_preprocessing_stats(df, f"Threshold Filtered {config.nmd_efficiency_threshold}"))

    # ===== STEP 12: Transcript length filtering =====
    if config.apply_transcript_length_filter:
        df = apply_transcript_length_filter(df, var_type, config)
        preprocessing_stats.append(_calculate_preprocessing_stats(df, "Transcript Length Filtered"))

    # ===== STEP 13: Normalization =====
    # Save plots before normalization
    plot_output_dir = FIGURES_DIR / "data" / "PTC"
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_output_path = plot_output_dir / f"{input_file.stem}_nmd_distributions.png"
    df["NMDeff"] = df["ASE_NMD_efficiency_TPM"]
    plot_nmd_efficiency_distributions(df, plot_output_path)

    if config.normalize_nmd_efficiency:
        df = normalize_nmd_efficiency(df)
        
        # Save plots after normalization
        plot_output_path = plot_output_dir / f"{input_file.stem}_nmd_distributions_norm.png"
        df["NMDeff"] = df["NMDeff_Norm"]
        plot_nmd_efficiency_distributions(df, plot_output_path)
        
        preprocessing_stats.append(_calculate_preprocessing_stats(df, "Normalized"))

    # Reset NMDeff for saving
    df["NMDeff"] = df["ASE_NMD_efficiency_TPM"]

    # ===== Save preprocessing statistics and plots =====
    preprocessing_df = pd.DataFrame(preprocessing_stats)
    preprocessing_table_dir = TABLES_DIR / "data" / "PTC"
    os.makedirs(preprocessing_table_dir, exist_ok=True)
    preprocessing_table_path = preprocessing_table_dir / f"{input_file.stem}_preprocessing_stats.csv"
    preprocessing_df.to_csv(preprocessing_table_path, index=False)
    logger.info(f"Saved preprocessing statistics to {preprocessing_table_path}")
    
    # Plot preprocessing steps
    preprocessing_plot_path = plot_output_dir / f"{input_file.stem}_preprocessing_steps.png"
    plot_preprocessing_steps(preprocessing_df, preprocessing_plot_path)

    # ===== Summary statistics =====
    logger.info(f"Number of stopgain variants: {len(df[df['stopgain'] == 'nonsense'])}")
    logger.info(f"Number of frameshift variants: {len(df[df['stopgain'] != 'nonsense'])}")
    logger.info(f"Longest transcript length: {df['transcript_length'].max()}")

    # ===== Save interim data =====
    interim_output_path = INTERIM_DATA_DIR / "PTC" / f"{input_file.stem}.csv"
    os.makedirs(interim_output_path.parent, exist_ok=True)
    df.to_csv(interim_output_path, index=False)

    # ===== Save processed data =====
    df[
        [
            "gene_id",
            "transcript_id",
            "chr",
            "start_pos",
            "Ref",
            "Alt",
            "count",
            "PTC_CDS_pos",
            "Last_Exon",
            "Penultimate_Exon",
            "Start_Prox",
            "Long_Exon",
            "InLastExon",
            "50ntToLastEJ",
            "DistanceToStart",
            "ExonLength",
            "PTC_EJC_dist",
            "DistanceToWTStop",
            "RNAHalfLife",
            "normal_stop_codon_CDS_pos",
            "NMDeff",
            "NMDeff_Norm" if config.normalize_nmd_efficiency else "NMDeff"
        ]
    ].to_csv(f"{output_path}.csv", index=False)

    # ===== Create and save sequences =====
    seqs_original = []
    for i in tqdm(range(len(df)), desc="Processing variants"):
        row = df.iloc[i]
        seq_original = create_6track_onehot_sequence(row)
        seqs_original.append(seq_original.T)

    with open(f"{output_path}.pkl", "wb") as f:
        pickle.dump(seqs_original, f)
    
    # ===== Save preprocessing configuration =====
    config_output_path = Path(f"{output_path}_preprocessing_config.txt")
    config.save_to_file(config_output_path)
    logger.info(f"Saved preprocessing configuration to {config_output_path}")
    
    logger.info(f"Preprocessing complete. Saved to {output_path}")


@app.command()
def process_PTC_dataset(
    input_file: str = "somatic_TCGA.txt",
    output_name: str = None
):
    """
    Process PTC dataset with default preprocessing configuration (CLI command).
    
    Args:
        input_file: Input filename in RAW_DATA_DIR/PTC/
        output_name: Optional output name (defaults to input_file stem)
    """
    process_PTC_dataset_with_config(Path(input_file), output_name)


@app.command()
def all():
    """Process all three PTC datasets with default configuration."""
    process_PTC_dataset_with_config(Path("somatic_TCGA.txt"))
    process_PTC_dataset_with_config(Path("germline_TCGA.txt"))
    process_PTC_dataset_with_config(Path("GTEx.txt"))


if __name__ == "__main__":
    app()
