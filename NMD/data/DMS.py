from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import typer
from NMD.config import (
    GENCODE_VERSION,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    FIGURES_DIR,
)
from NMD.data.transcripts import (
    seq_to_oh,
    create_cds_track,
    create_splice_track,
)
from NMD.utils import loess_smooth, prepare_for_interp
from NMD.plots import (
    plot_dms_sp_normalization_comparison,
    plot_dms_le_normalization_comparison,
    plot_dms_pe_normalization_comparison,
)
import os

from genome_kit import Genome
from scipy.interpolate import interp1d


app = typer.Typer()

def apply_nmd_scaling(df: pd.DataFrame, current_5prime_median: float, target_5prime: float = -0.5, anchor_3prime: float = 0.5):
    """
    Linearly scales NMDeff values using a 3' anchor point.
    
    Args:
        df: The DataFrame containing 'NMDeff_shifted'.
        current_5prime_median: The median of the first 5 medians (M5' in your code).
        target_5prime: The desired global median for the 5' end.
        anchor_3prime: The fixed point at the 3' end (0.5).
    """
    # Calculate the scaling factor k
    # Formula: k = (Target_5 - Anchor) / (Current_5 - Anchor)
    k = (target_5prime - anchor_3prime) / (current_5prime_median - anchor_3prime)
    
    print(f"Calculated scaling factor (k): {k:.4f}")
    
    # Apply transformation: k * (x - anchor) + anchor
    df['NMDeff_Norm'] = k * (df['NMDeff_shifted'] - anchor_3prime) + anchor_3prime
    
    return df

def scale_DMS_SP(dms):
    result = {}
    for gene in dms['gene'].unique():
        gene_df = dms[dms['gene'] == gene].sort_values('PTCposition', ascending=False)
        top15 = gene_df.head(15)["NMDeff"].values
        first5 = top15[:5]
        middle5 = top15[5:10]
        last5 = top15[10:]
        medians = [np.median(first5), np.median(middle5), np.median(last5)]
        result[gene] = max(medians)
    utr3_nmd = pd.DataFrame.from_dict(result, orient='index', columns=['max_median_nmdeff'])
    utr3_nmd = utr3_nmd.reset_index().rename(columns={'index': 'gene'})
    utr3_nmd["shift_amount"] = 0.5 - utr3_nmd["max_median_nmdeff"]

    # Shift NMDeff for each gene by its shift_amount
    shift_map = dict(zip(utr3_nmd['gene'], utr3_nmd['shift_amount']))
    dms['NMDeff_shifted'] = dms.apply(
        lambda row: row['NMDeff'] + shift_map.get(row['gene'], 0), axis=1
    )

    # Compute the median of the first 5 NMDeff_shifted values for each gene, ordered by PTCposition
    first5_medians = (
        dms.sort_values(['gene', 'PTCposition'])
        .groupby('gene')
        .head(5)
        .groupby('gene')['NMDeff_shifted']
        .median()
    )

    first5_medians.median()

    #  Get the current global 5' median from your existing code
    m5_prime = first5_medians.median()

    #  Apply the final scaling
    target_5prime = -0.5
    anchor_3prime = 0.5
    k = (target_5prime - anchor_3prime) / (m5_prime - anchor_3prime)
    
    print(f"Calculated scaling factor (k): {k:.4f}")
    
    # Apply transformation: k * (x - anchor) + anchor
    dms['NMDeff_Norm'] = k * (dms['NMDeff_shifted'] - anchor_3prime) + anchor_3prime
    return dms

@app.command()
def process_DMS_SP_dataset(csv_path: Path = RAW_DATA_DIR / "DMS/SP.csv", df_col:str = "NMDeff"):
    # 0. Read Data
    dms = pd.read_csv(csv_path)
    df = pd.read_csv(
        "/g/strcombio/fsupek_franklin/mveiner/Projects/NMD/data/processed/PTC/somatic_TCGA.csv"
    )
    df = df[df.Last_Exon == False]
    df = df[df.Penultimate_Exon == False]
    df = df[df.Long_Exon == False]
    df = df[df.PTC_CDS_pos <= 250]
    df = df[df.Ref != "-"]
    df = df[df.Alt != "-"]

    # map genes to chromosomes
    logger.info("Mapping genes to chromosomes using genome_kit...")
    genome = Genome(GENCODE_VERSION)

    # 1. Filtering
    dms = pd.read_csv("/g/strcombio/fsupek_franklin/mveiner/Projects/NMD/data/raw/DMS/SP.csv")

    dms["PTCposition_nt"] = dms["PTCposition"] * 3

    wts = dms[dms.wild_type == "yes"]
    dms = dms[dms.wild_type == "no"]

    # 1. Filtering
    dms = dms[dms.sigma <= 1]
    dms = dms[dms["fitness"] >= -3]
    dms = dms[dms["fitness"] <= 3]

    # drop genes where more than 50% of variants have fitness > than the wild-type fitness
    wt_fitness = wts.groupby("gene")["fitness"].first()
    gene_variant_counts = dms.groupby("gene").size()
    dms["wt_fitness"] = dms["gene"].map(wt_fitness)
    gene_high_fitness_mask = dms["fitness"] > dms["wt_fitness"]
    gene_high_fitness_counts = dms[gene_high_fitness_mask].groupby("gene").size()
    # Align indices before comparison
    aligned_variant_counts = gene_variant_counts.reindex(gene_high_fitness_counts.index)
    genes_to_drop = gene_high_fitness_counts[gene_high_fitness_counts > (aligned_variant_counts / 2)].index
    dms = dms[~dms.gene.isin(genes_to_drop)]
    print(f"Dropped {len(genes_to_drop)} genes where >50% of variants have fitness > wild-type fitness.")
    dms = dms.drop(columns="wt_fitness")

    # drop genes with fewer than 50 variants
    gene_counts = dms["gene"].value_counts()
    genes_to_keep = gene_counts[gene_counts > 49].index
    dms = dms[dms.gene.isin(genes_to_keep)]
    print(f"Kept {len(genes_to_keep)} genes with at least 50 variants.")

    # 2. Invert fitness to NMDeff
    dms["NMDeff"] = dms["fitness"] * (-1)

    # Normalize
    dms = scale_DMS_SP(dms)

    # print how many genes are left
    logger.info(f"Number of genes left: {len(dms.gene.unique())}")
    logger.info(f"Number of variants left: {len(dms)}")

    # 3. ensembl id map and MANE
    gene_map = pd.read_csv("/g/strcombio/fsupek_franklin/mveiner/Projects/NMD/data/raw/annotations/ensembl_v88_gene_transcript_genesymbol.txt", sep="\t", header=0)[["gene_id","gene_name"]].drop_duplicates()
    dms = dms.merge(gene_map, left_on='gene', right_on="gene_name", how='left')

    # 4. Get gene CDS + UTR sequences from gencode
    genome = Genome(GENCODE_VERSION)
    indices_to_drop, data, chrs = [], [], []
    for idx, row in tqdm(dms.iterrows(), total=len(dms)):
        nt_seq = row.nt_seq.upper()
        trs = genome.genes[row.gene_id].transcripts
        # pick the longest transcript with a CDS
        tr_obj = None
        max_cds_length = 0
        for tr in trs:
            if tr.cdss is not None:
                cds_length = sum([len(x) for x in tr.cdss])
                if cds_length > max_cds_length:
                    max_cds_length = cds_length
                    tr_obj = tr
        
        # skip gene if no transcript with cdss found or it has < 2 coding exons, or is < 300nt long
        if (tr_obj is None):
            logger.warning(f"No valid transcript found for gene {row.gene} at index {idx}, skipping...")
            indices_to_drop.append(idx)
            continue

        if (len(tr_obj.cdss) < 2):
            logger.warning(f"Transcript {tr_obj.id} for gene {row.gene} has less than 2 coding exons, skipping...")
            indices_to_drop.append(idx)
            continue

        if (sum([len(x) for x in tr_obj.cdss]) < 300):
            logger.warning(f"Transcript {tr_obj.id} for gene {row.gene} has CDS length < 300nt, skipping...")
            indices_to_drop.append(idx)
            continue

        chrs.append(tr_obj.chrom)

        # Check if PTC position falls in the last exon
        # Calculate PTC position in transcript coordinates
        mut_start = sum([len(x) for x in tr_obj.utr5s])

        # Use PTCposition from the row data (position relative to CDS start)
        ptc_cds_position = row.PTCposition  # This should be in CDS coordinates
        ptc_transcript_pos = mut_start + ptc_cds_position

        # Get exon boundaries in transcript coordinates
        exon_boundaries = []
        current_pos = 0
        for exon in tr_obj.exons:
            exon_start = current_pos
            exon_end = current_pos + len(exon)
            exon_boundaries.append((exon_start, exon_end))
            current_pos = exon_end

        # Check if PTC position falls in the last exon
        last_exon_start, last_exon_end = exon_boundaries[-1]

        if last_exon_start <= ptc_transcript_pos < last_exon_end:
            indices_to_drop.append(idx)
            continue  # Skip this row

        cds_track = create_cds_track(tr_obj)
        splice_track = create_splice_track(tr_obj)

        # Get the full wild-type exon sequence (no mutations for WT)
        seq = "".join([genome.dna(exon) for exon in tr_obj.exons])
        ohe = seq_to_oh(seq)
        mut_end = mut_start + len(nt_seq)

        # Get the full wild-type exon sequence
        ohe = seq_to_oh("".join([genome.dna(exon) for exon in tr_obj.exons]))
        mut_start = sum([len(x) for x in tr_obj.utr5s]) if tr_obj.utr5s is not None else 0
        mut_end = mut_start + len(nt_seq)
        mut_ohe = seq_to_oh(nt_seq)

        if (ohe.shape[0] != cds_track.shape[0]) or (ohe.shape[0] < mut_end):
            logger.warning(
                f"Sequence length mismatch for gene {row.gene} at index {idx}, skipping..."
            )
            indices_to_drop.append(idx)
            continue

        ohe[mut_start:mut_end] = mut_ohe
        six_track = np.concatenate([ohe, cds_track[:, None], splice_track[:, None]], axis=1)
        data.append(six_track)

    # add tag for rows questioned in the above loop and drop them later
    dms["no_seq"] = False
    dms.loc[indices_to_drop, "no_seq"] = True
    dms["chr"] = chrs

    # 6. Plot comparison of distributions
    plot_output_dir = FIGURES_DIR / "data" / "DMS"
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_output_path = plot_output_dir / "DMS_SP_normalization_comparison.png"
    plot_dms_sp_normalization_comparison(df, dms, plot_output_path, df_col)

    # 7. Save the processed sequences as npy
    output_path = PROCESSED_DATA_DIR / "DMS_SP" / "processed_sequences"
    os.makedirs(output_path.parent, exist_ok=True)
    with open(f"{output_path}.pkl", "wb") as f:
        pickle.dump(data, f)
    dms.to_csv(PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv", index=False)

    logger.success(
        f"Finished processing DMS dataset. Processed sequences saved to {output_path}.pkl"
    )


@app.command()
def process_DMS_LE_dataset(csv_path: Path = RAW_DATA_DIR / "DMS/LE.csv", df_col:str = "NMDeff"):
    # 0. Get Gene names
    dms = pd.read_csv(csv_path)
    dms["chr"] = "chr17"

    df = pd.read_csv(
        "/g/strcombio/fsupek_franklin/mveiner/Projects/NMD/data/processed/PTC/somatic_TCGA.csv"
    )
    df = df[df.Last_Exon == False]
    df = df[df.Penultimate_Exon == False]
    df = df[df.Start_Prox == False]
    df = df[df.Ref != "-"]
    df = df[df.Alt != "-"]

    # 1. Filters
    dms = dms.loc[dms.fifty_nt_boundary == "no"]
    dms = dms.dropna()
    dms["fitness"] = dms["fitness"] - dms["fitness"].mean()
    dms = dms[dms.fitness >= -1]
    dms = dms[dms.fitness <= 1]

    # 2. Invert fitness to NMDeff
    dms["NMDeff"] = dms["fitness"] * (-1)

    # 3. Get transcript
    genome = Genome(GENCODE_VERSION)
    tr_obj = genome.transcripts["ENST00000357654.7"]
    logger.info(f"Processing transcript {tr_obj.id} for DMS LE dataset")
    logger.info(f"Transcript exons: {tr_obj.exons}")
    logger.info(f"Transcript exon Lengths: {[len(exon) for exon in tr_obj.exons]}")

    # 4. Create 6-track one-hot sequences
    data = []
    cds_track_wt = create_cds_track(tr_obj)
    splice_track_wt = create_splice_track(tr_obj)
    for _, row in tqdm(dms.iterrows(), total=len(dms)):
        ohe_wt = seq_to_oh("".join([genome.dna(exon) for exon in tr_obj.exons]))
        target_exon = tr_obj.exons[9]
        target_exon_len = len(target_exon)
        mut_start = sum([len(x) for x in tr_obj.exons[:9]])
        mut_end = mut_start + target_exon_len

        add_seq = "CT"
        mut_seq = add_seq + row.nt_seq.upper()
        mut_ohe = seq_to_oh(mut_seq)

        ohe = np.zeros((ohe_wt.shape[0] + mut_ohe.shape[0] - target_exon_len, 4))
        ohe[:mut_start, :] = ohe_wt[:mut_start]
        ohe[mut_start : (mut_start + mut_ohe.shape[0]), :] = mut_ohe
        ohe[(mut_start + mut_ohe.shape[0]) :, :] = ohe_wt[mut_end:, :]

        cds_track = np.zeros(ohe.shape[0])
        splice_track = np.zeros(ohe.shape[0])

        cds_track[:mut_start] = cds_track_wt[:mut_start]
        cds_track[mut_start : (mut_start + mut_ohe.shape[0])] = [1, 0, 0] * (
            mut_ohe.shape[0] // 3
        ) + ([1] if mut_ohe.shape[0] % 3 == 1 else [1, 0] if mut_ohe.shape[0] % 3 == 2 else [])
        cds_track[(mut_start + mut_ohe.shape[0]) :] = cds_track_wt[mut_end:]

        splice_track[:mut_start] = splice_track_wt[:mut_start]
        splice_track[mut_start : (mut_start + mut_ohe.shape[0])] = [0] * len(mut_seq)
        splice_track[(mut_start + mut_ohe.shape[0]) :] = splice_track_wt[mut_end:]

        six_track = np.concatenate([ohe, cds_track[:, None], splice_track[:, None]], axis=1)

        data.append(six_track)

    # 5. Normalization to PTC dataset
    #ase_mean, ase_std = df[df_col].mean(), df[df_col].std()
    #dms["NMDeff_Norm"] = (dms["NMDeff"] - dms["NMDeff"].mean()) / dms[
    #    "NMDeff"
    #].std() * ase_std + ase_mean
    dms["NMDeff_Norm"] = dms.NMDeff
    dms["gene"] = "BRCA1"

    # 6. Plot comparison of distributions
    plot_output_dir = FIGURES_DIR / "data" / "DMS"
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_output_path = plot_output_dir / "DMS_LE_normalization_comparison.png"
    plot_dms_le_normalization_comparison(df, dms, plot_output_path, df_col)

    # 7. save the processed sequences as npy
    output_path = PROCESSED_DATA_DIR / "DMS_LE" / "processed_sequences"
    os.makedirs(output_path.parent, exist_ok=True)
    with open(f"{output_path}.pkl", "wb") as f:
        pickle.dump(data, f)
    dms.to_csv(PROCESSED_DATA_DIR / "DMS_LE" / "fitness.csv", index=False)
    logger.success(
        f"Finished processing DMS dataset. Processed sequences saved to {output_path}.pkl"
    )


@app.command()
def process_DMS_PE_dataset(csv_path: Path = RAW_DATA_DIR / "DMS/PE.csv", df_col:str = "NMDeff"):
    """
    Process the DMS Penultimate Exon dataset.

    FIXED: Ensures PTC positions are properly aligned to codon boundaries.
    The original implementation had a frameshift issue where PTCs were not
    aligned to codon starts, making them not true premature termination codons.
    """
    # 0. Read Data
    df = pd.read_csv(
        "/g/strcombio/fsupek_franklin/mveiner/Projects/NMD/data/interim/PTC/somatic_TCGA.csv"
    )
    df = df[df.Last_Exon == False]
    df = df[df.CDS_num_exons_downstream == 1]
    df = df[df.Long_Exon == False]
    df = df[df.Ref != "-"]
    df = df[df.Alt != "-"]
    df["PTC_EJC_dist"] *= -1
    dms = pd.read_csv(csv_path)

    # Add chr column BRCA1=17, ATP7A=X
    gene_to_chr = {"BRCA1": "chr17", "ATP7A": "chrX"}
    dms["chr"] = dms["gene"].map(gene_to_chr)

    # 1. Filtering
    dms = dms[dms.fitness_TGG >= -2]
    dms = dms[dms.sigma <= 0.15]

    # 2. Invert fitness to NMDeff
    dms["fitness_TGG"] = dms["fitness_TGG"] - dms["fitness_TGG"].mean()
    dms["NMDeff"] = dms["fitness_TGG"] * (-1)

    # 3. Get Transcript ids
    genome = Genome(GENCODE_VERSION)
    brca1_tr_obj = genome.transcripts["ENST00000357654.7"]
    logger.info(f"Using BRCA1 transcript: {brca1_tr_obj.id}")
    logger.info(f"BRCA1 has {len(brca1_tr_obj.exons)} exons")
    logger.info(f"BRCA1 exon lengths: {[len(exon) for exon in brca1_tr_obj.exons]}")

    # Also get ATP7A transcript
    atp7a_tr_obj = genome.transcripts["ENST00000341514.10"]
    logger.info(f"Using ATP7A transcript: {atp7a_tr_obj.id}")
    logger.info(f"ATP7A has {len(atp7a_tr_obj.exons)} exons")
    logger.info(f"ATP7A exon lengths: {[len(exon) for exon in atp7a_tr_obj.exons]}")

    # 4. Get gene CDS + UTR sequences from gencode
    data = []
    for _, row in tqdm(dms.iterrows(), total=len(dms)):
        # Select transcript based on gene
        if row.gene == "ATP7A":
            tr_obj = atp7a_tr_obj
        elif row.gene == "BRCA1":
            tr_obj = brca1_tr_obj
        else:
            logger.warning(f"Unknown gene: {row.gene}, skipping")
            continue

        # Get the full wild-type transcript sequence
        ohe = seq_to_oh("".join([genome.dna(exon) for exon in tr_obj.exons]))
        cds_track = create_cds_track(tr_obj)
        splice_track = create_splice_track(tr_obj)

        # Calculate PTC position relative to the end of the penultimate exon
        ptc_pos_rev_codons = int(row.PTC_pos_rev)

        # Find the penultimate exon (second to last exon)
        penultimate_exon_idx = len(tr_obj.exons) - 2

        # Calculate the end of the penultimate exon in transcript coordinates
        pos_before_penultimate = sum([len(exon) for exon in tr_obj.exons[:penultimate_exon_idx]])
        penultimate_exon_len = len(tr_obj.exons[penultimate_exon_idx])
        penultimate_exon_end = pos_before_penultimate + penultimate_exon_len
        ptc_nt_pos_in_transcript = (penultimate_exon_end + ptc_pos_rev_codons * 3) - 1
        if row.gene == "ATP7A":
            ptc_nt_pos_in_transcript -= 1  # Adjust for BRCA1 as reading frame

        # Insert stop codon at the codon-aligned PTC position
        stop_type = row.stop_type.replace("U", "T")  # Replace U with T if present
        stop_codon_ohe = seq_to_oh(stop_type)
        ohe[ptc_nt_pos_in_transcript : ptc_nt_pos_in_transcript + 3] = stop_codon_ohe

        six_track = np.concatenate([ohe, cds_track[:, None], splice_track[:, None]], axis=1)
        data.append(six_track)

    # 5. Range-based normalization to PTC dataset
    logger.info("Applying range-based normalization using LOESS fits...")

    # Fit LOESS to PTC data
    ptc_ejc_x_smooth, ptc_ejc_y_smooth = loess_smooth(
        df["PTC_EJC_dist"].values, df[df_col].values, frac=0.4
    )

    # Fit LOESS to all DMS data combined
    dms["nt_position"] = dms["PTC_pos_rev"].astype(int) * 3
    dms_nt_x_smooth, dms_nt_y_smooth = loess_smooth(
        dms["nt_position"].values, dms["NMDeff"].values, frac=0.4
    )

    # Prepare data for interpolation (remove duplicates)
    ptc_ejc_x_unique, ptc_ejc_y_unique = prepare_for_interp(ptc_ejc_x_smooth, ptc_ejc_y_smooth)
    dms_nt_x_unique, dms_nt_y_unique = prepare_for_interp(dms_nt_x_smooth, dms_nt_y_smooth)

    # Create interpolation functions from the LOESS fits
    ptc_ejc_interp = interp1d(
        ptc_ejc_x_unique, ptc_ejc_y_unique, kind="linear", fill_value="extrapolate"
    )
    dms_nt_interp = interp1d(
        dms_nt_x_unique, dms_nt_y_unique, kind="linear", fill_value="extrapolate"
    )

    # Calculate range of LOESS curves at common positions
    common_positions = np.linspace(-300, -1, 100)
    ptc_ejc_loess_values = ptc_ejc_interp(common_positions)
    dms_nt_loess_values = dms_nt_interp(common_positions)

    # Calculate ranges and means
    ptc_ejc_range = ptc_ejc_loess_values.max() - ptc_ejc_loess_values.min()
    dms_nt_range = dms_nt_loess_values.max() - dms_nt_loess_values.min()
    ptc_ejc_mean = ptc_ejc_loess_values.mean()

    # Normalize DMS PE to match PTC range and mean
    # First center, then scale, then shift to PTC mean
    dms["NMDeff_Norm"] = (dms["NMDeff"] - dms["NMDeff"].mean()) * (
        ptc_ejc_range / dms_nt_range
    ) + ptc_ejc_mean

    # 6. Plot DMS PE normalization comparison
    plot_output_dir = FIGURES_DIR / "data" / "DMS"
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_output_path = plot_output_dir / "DMS_PE_normalization_comparison.png"
    plot_dms_pe_normalization_comparison(df, dms, plot_output_path, df_col)

    # 7. Save the processed sequences as npy
    output_path = PROCESSED_DATA_DIR / "DMS_PE" / "processed_sequences"
    os.makedirs(output_path.parent, exist_ok=True)
    with open(f"{output_path}.pkl", "wb") as f:
        pickle.dump(data, f)
    dms.to_csv(PROCESSED_DATA_DIR / "DMS_PE" / "fitness.csv", index=False)
    logger.success(
        f"Finished processing DMS dataset. Processed sequences saved to {output_path}.pkl"
    )


@app.command()
def generate_DMS_LE_all_positions(output_path: Path = PROCESSED_DATA_DIR / "DMS_LE_all_positions"):
    """
    Generate sequences for ALL possible stop codon insertions in DMS LE dataset.
    
    For each exon length category (125bps, 250bps, 750bps, 2500bps, 3426bps):
    - For each possible codon position (every 3 nucleotides)
    - For each stop codon type (UAA, UAG, UGA)
    - Create the 6-track sequence with that stop codon inserted
    
    Also creates a fitness file that contains actual observations where available,
    and NaN where no observation exists.
    
    Args:
        output_path: Directory to save the generated sequences and fitness file
    """
    logger.info("Generating all possible DMS LE stop codon insertion sequences...")
    
    # Load existing DMS LE data to get actual observations
    existing_dms = pd.read_csv(PROCESSED_DATA_DIR / "DMS_LE" / "fitness.csv")
    logger.info(f"Loaded {len(existing_dms)} existing DMS LE observations")
    
    # Get transcript
    genome = Genome(GENCODE_VERSION)
    tr_obj = genome.transcripts["ENST00000357654.7"]
    logger.info(f"Using BRCA1 transcript: {tr_obj.id}")
    
    # Define exon length categories and their corresponding downstream sequences
    exon_lengths = existing_dms['exon_length'].unique()
    exon_lengths = {length: int(length.replace("bps", "")) for length in exon_lengths}
    
    # Create wildtype tracks once
    cds_track_wt = create_cds_track(tr_obj)
    splice_track_wt = create_splice_track(tr_obj)
    ohe_wt = seq_to_oh("".join([genome.dna(exon) for exon in tr_obj.exons]))
    
    # Get the target exon info (exon 10, index 9)
    target_exon = tr_obj.exons[9]
    target_exon_len = len(target_exon)
    mut_start = sum([len(x) for x in tr_obj.exons[:9]])
    mut_end = mut_start + target_exon_len
    
    # Get downstream sequence for constructing different lengths
    downstream_seq = "".join([genome.dna(exon) for exon in tr_obj.exons[10:]])
    logger.info(f"Target exon length: {target_exon_len}nt")
    logger.info(f"Downstream sequence length: {len(downstream_seq)}nt")
    
    # Stop codons to insert (in DNA, not RNA)
    stop_codons = {'UAA': 'TAA', 'UAG': 'TAG', 'UGA': 'TGA'}
    
    all_sequences = []
    all_metadata = []
    
    for exon_len_str, exon_len_nt in tqdm(exon_lengths.items(), desc="Exon lengths"):
        # For each codon position in the exon (every 3 nt)
        # Start at position 1 (after CT prefix) to align with codon boundaries
        for codon_pos in range(1, (exon_len_nt // 3)):
            nt_pos = codon_pos * 3
            
            for stop_type, stop_seq in stop_codons.items():
                # Construct the mutated exon sequence
                # Format: CT + sequence_before_stop + STOP + 123nt_downstream
                
                # Calculate how many nucleotides we need before the stop codon
                nt_before_stop = nt_pos
                
                # Get 123nt downstream sequence
                down_123nt = downstream_seq[:123]
                
                # Create sequence before stop (from target exon)
                seq_before = genome.dna(target_exon)[:nt_before_stop]
                
                # Construct full mutated sequence
                add_seq = "CT"
                mut_seq = add_seq + seq_before + stop_seq + down_123nt
                
                # Create one-hot encoding for mutated sequence
                mut_ohe = seq_to_oh(mut_seq)
                
                # Build the full 6-track sequence
                ohe = np.zeros((ohe_wt.shape[0] + mut_ohe.shape[0] - target_exon_len, 4))
                ohe[:mut_start, :] = ohe_wt[:mut_start]
                ohe[mut_start : (mut_start + mut_ohe.shape[0]), :] = mut_ohe
                ohe[(mut_start + mut_ohe.shape[0]) :, :] = ohe_wt[mut_end:, :]
                
                # Create CDS track
                cds_track = np.zeros(ohe.shape[0])
                cds_track[:mut_start] = cds_track_wt[:mut_start]
                cds_track[mut_start : (mut_start + mut_ohe.shape[0])] = [1, 0, 0] * (
                    mut_ohe.shape[0] // 3
                ) + ([1] if mut_ohe.shape[0] % 3 == 1 else [1, 0] if mut_ohe.shape[0] % 3 == 2 else [])
                cds_track[(mut_start + mut_ohe.shape[0]) :] = cds_track_wt[mut_end:]
                
                # Create splice track
                splice_track = np.zeros(ohe.shape[0])
                splice_track[:mut_start] = splice_track_wt[:mut_start]
                splice_track[mut_start : (mut_start + mut_ohe.shape[0])] = [0] * len(mut_seq)
                splice_track[(mut_start + mut_ohe.shape[0]) :] = splice_track_wt[mut_end:]
                
                # Combine into 6-track
                six_track = np.concatenate([ohe, cds_track[:, None], splice_track[:, None]], axis=1)
                
                all_sequences.append(six_track)
                
                # Calculate PTC-EJC distance (distance from stop to end of exon)
                # The stop codon position in the mutated sequence is at: 2 (CT prefix) + nt_pos
                # So the distance from stop to end of exon is: exon_len_nt - (2 + nt_pos)
                ptc_ejc_dist = exon_len_nt - (2 + nt_pos)
                
                # Check if we have an actual observation for this case
                # First calculate PTC_EJC_dist for existing observations the same way as in plotting code
                if 'PTC_EJC_dist' not in existing_dms.columns:
                    existing_dms['stop_down_123nt'] = (existing_dms.stop_type + existing_dms.down_123nt).str.replace("U", "T")
                    existing_dms['stop_pos'] = existing_dms.apply(lambda row: row.nt_seq.find(row.stop_down_123nt), axis=1)
                    existing_dms["PTC_EJC_dist"] = existing_dms["exon_length"].str.replace("bps", "").astype(int) - existing_dms["stop_pos"]
                
                matching_obs = existing_dms[
                    (existing_dms['exon_length'] == exon_len_str) &
                    (existing_dms['stop_type'] == stop_type) &
                    (np.abs(existing_dms['PTC_EJC_dist'] - ptc_ejc_dist) < 3)  # Allow small tolerance
                ]
                
                if len(matching_obs) > 0:
                    obs_row = matching_obs.iloc[0]
                    nmdeff = obs_row['NMDeff']
                    nmdeff_norm = obs_row.get('NMDeff_Norm', nmdeff)
                    has_observation = True
                else:
                    nmdeff = np.nan
                    nmdeff_norm = np.nan
                    has_observation = False
                
                all_metadata.append({
                    'exon_length': exon_len_str,
                    'exon_length_nt': exon_len_nt,
                    'codon_position': codon_pos,
                    'nt_position': nt_pos,
                    'stop_type': stop_type,
                    'PTC_EJC_dist': ptc_ejc_dist,
                    'NMDeff': nmdeff,
                    'NMDeff_Norm': nmdeff_norm,
                    'has_observation': has_observation,
                    'chr': 'chr17',
                    'gene': 'BRCA1'
                })
    
    logger.info(f"Generated {len(all_sequences)} total sequences")
    logger.info(f"  Sequences with observations: {sum(m['has_observation'] for m in all_metadata)}")
    logger.info(f"  Sequences without observations: {sum(not m['has_observation'] for m in all_metadata)}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save sequences
    sequences_file = output_path / "processed_sequences.pkl"
    with open(sequences_file, "wb") as f:
        pickle.dump(all_sequences, f)
    logger.info(f"Saved sequences to {sequences_file}")
    
    # Save metadata/fitness file
    metadata_df = pd.DataFrame(all_metadata)
    fitness_file = output_path / "fitness.csv"
    metadata_df.to_csv(fitness_file, index=False)
    logger.info(f"Saved fitness data to {fitness_file}")
    
    # Print summary by exon length
    logger.info("\nSummary by exon length:")
    for exon_len in exon_lengths.keys():
        subset = metadata_df[metadata_df['exon_length'] == exon_len]
        n_obs = subset['has_observation'].sum()
        n_total = len(subset)
        logger.info(f"  {exon_len}: {n_obs}/{n_total} have observations")
    
    logger.success("Finished generating all DMS LE positions!")


@app.command()
def combine_DMS_datasets():
    """
    Combine and shuffle the outputs from process_DMS_dataset,
    process_DMS_LE_dataset, and process_DMS_PE_dataset.

    Reads the processed sequences and fitness data from each dataset,
    combines them, shuffles the combined data, and writes out the result.
    """
    logger.info("Loading DMS datasets...")

    # Load sequence data
    datasets = []
    fitness_data = []

    # Load DMS dataset (SPrule)
    try:
        with open(PROCESSED_DATA_DIR / "DMS_SP" / "processed_sequences.pkl", "rb") as f:
            dms_seqs = pickle.load(f)
        dms_fitness = pd.read_csv(PROCESSED_DATA_DIR / "DMS_SP" / "fitness.csv")[
            ["NMDeff", "NMDeff_Norm", "gene", "chr"]
        ]

        datasets.extend(dms_seqs)
        fitness_data.append(dms_fitness)
        logger.info(
            f"Loaded DMS dataset: {len(dms_seqs)} sequences {len(dms_fitness)} fitness entries"
        )
    except FileNotFoundError:
        logger.warning("DMS dataset not found, skipping...")

    # Load DMS_LE dataset (LErule)
    try:
        with open(PROCESSED_DATA_DIR / "DMS_LE" / "processed_sequences.pkl", "rb") as f:
            dms_le_seqs = pickle.load(f)
        dms_le_fitness = pd.read_csv(PROCESSED_DATA_DIR / "DMS_LE" / "fitness.csv")[
            ["NMDeff", "NMDeff_Norm", "gene", "chr"]
        ]

        datasets.extend(dms_le_seqs)
        fitness_data.append(dms_le_fitness)
        logger.info(
            f"Loaded DMS_LE dataset: {len(dms_le_seqs)} sequences {len(dms_le_fitness)} fitness entries"
        )
    except FileNotFoundError:
        logger.warning("DMS_LE dataset not found, skipping...")

    # Load DMS_PE dataset (50ntsrule)
    try:
        with open(PROCESSED_DATA_DIR / "DMS_PE" / "processed_sequences.pkl", "rb") as f:
            dms_pe_seqs = pickle.load(f)
        dms_pe_fitness = pd.read_csv(PROCESSED_DATA_DIR / "DMS_PE" / "fitness.csv")[
            ["NMDeff", "NMDeff_Norm", "gene", "chr"]
        ]

        datasets.extend(dms_pe_seqs)
        fitness_data.append(dms_pe_fitness)
        logger.info(
            f"Loaded DMS_PE dataset: {len(dms_pe_seqs)} sequences {len(dms_pe_fitness)} fitness entries"
        )
    except FileNotFoundError:
        logger.warning("DMS_PE dataset not found, skipping...")

    if not datasets:
        logger.error("No datasets found to combine!")
        return

    # Combine fitness data
    combined_fitness = pd.concat(fitness_data, ignore_index=True)
    logger.info(f"Combined fitness data shape: {combined_fitness.shape}")
    logger.info(f"Combined fitness data columns: {list(combined_fitness.columns)}")

    # Create indices for shuffling
    total_samples = len(datasets)
    indices = np.random.permutation(total_samples)

    # Shuffle sequences
    shuffled_sequences = [datasets[i] for i in indices]

    # Shuffle fitness data
    shuffled_fitness = combined_fitness.iloc[indices].reset_index(drop=True)

    logger.info(f"Shuffled {total_samples} total samples")

    # Create output directory
    output_dir = PROCESSED_DATA_DIR / "DMS_combined"
    os.makedirs(output_dir, exist_ok=True)

    # Save combined and shuffled sequences
    output_seqs_path = output_dir / "processed_sequences.pkl"
    with open(output_seqs_path, "wb") as f:
        pickle.dump(shuffled_sequences, f)

    # Save combined and shuffled fitness data
    output_fitness_path = output_dir / "fitness.csv"
    shuffled_fitness.to_csv(output_fitness_path, index=False)

    logger.success(f"Combined and shuffled datasets saved:")
    logger.success(f"  Sequences: {output_seqs_path}")
    logger.success(f"  Fitness data: {output_fitness_path}")
    logger.success(f"  Total samples: {total_samples}")


@app.command()
def all():
    process_DMS_SP_dataset(csv_path=RAW_DATA_DIR / "DMS/SP.csv", df_col="NMDeff_Norm")
    process_DMS_LE_dataset(csv_path=RAW_DATA_DIR / "DMS/LE.csv", df_col="NMDeff_Norm")
    process_DMS_PE_dataset(csv_path=RAW_DATA_DIR / "DMS/PE.csv", df_col="NMDeff_Norm")
    combine_DMS_datasets()


if __name__ == "__main__":
    app()
