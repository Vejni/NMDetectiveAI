import genome_kit as gk
from genome_kit import Genome, Variant, VariantGenome
from loguru import logger
import numpy as np
from tqdm import tqdm

from NMD.config import GENCODE_VERSION

# Source: https://github.com/bowang-lab/Orthrus/blob/main/nbs/start_here.ipynb
revcomp = {"A": "T", "C": "G", "G": "C", "T": "A"}


def find_transcript(genome, transcript_id):
    """Find a transcript in a genome by transcript ID.

    Args:
        genome (object): The genome object containing a list of transcripts.
        transcript_id (str): The ID of the transcript to find.

    Returns:
        object: The transcript object, if found.

    Raises:
        ValueError: If no transcript with the given ID is found.

    Example:
        >>> # Create sample transcripts and a genome
        >>> transcript1 = 'ENST00000263946'
        >>> genome = Genome(GENCODE_VERSION)
        >>> result = find_transcript(genome, 'ENST00000335137')
        >>> print(result.id)
        <Transcript ENST00000263946.7 of PKP1>
        >>> # If transcript ID is not found
        >>> find_transcript(genome, 'ENST00000000000')
        ValueError: Transcript with ID ENST00000000000 not found.
    """
    transcripts = [x for x in genome.transcripts if x.id.split(".")[0] == transcript_id]
    if not transcripts:
        raise ValueError(f"Transcript with ID {transcript_id} not found.")

    return transcripts[0]


def find_transcript_by_gene_name(genome, gene_name):
    """Find all transcripts in a genome by gene name.

    Args:
        genome (object): The genome object containing a list of transcripts.
        gene_name (str): The name of the gene whose transcripts are to be found.

    Returns:
        list: A list of transcript objects corresponding to the given gene name.

    Raises:
        ValueError: If no transcripts for the given gene name are found.

    Example:
        >>> # Find transcripts by gene name
        >>> transcripts = find_transcript_by_gene_name(genome, 'PKP1')
        >>> print(transcripts)
        [<Transcript ENST00000367324.7 of PKP1>,
        <Transcript ENST00000263946.7 of PKP1>,
        <Transcript ENST00000352845.3 of PKP1>,
        <Transcript ENST00000475988.1 of PKP1>,
        <Transcript ENST00000477817.1 of PKP1>]
        >>> # If gene name is not found
        >>> find_transcript_by_gene_name(genome, 'XYZ')
        ValueError: No transcripts found for gene name XYZ.
    """
    genes = [x for x in genome.genes if x.name == gene_name]
    if not genes:
        raise ValueError(f"No genes found for gene name {gene_name}.")
    if len(genes) > 1:
        print(f"Warning: More than one gene found for gene name {gene_name}.")
        print("Concatenating transcripts from all genes.")

    transcripts = []
    for gene in genes:
        transcripts += gene.transcripts
    return transcripts


def create_cds_track(t):
    """Create a track of the coding sequence of a transcript.
    Use the exons of the transcript to create a track where the first position of the codon is one.

    Args:
        t (gk.Transcript): The transcript object.
    """
    cds_intervals = t.cdss if t.cdss is not None else []
    utr3_intervals = t.utr3s if t.utr3s is not None else []
    utr5_intervals = t.utr5s if t.utr5s is not None else []

    len_utr3 = sum([len(x) for x in utr3_intervals])
    len_utr5 = sum([len(x) for x in utr5_intervals])
    len_cds = sum([len(x) for x in cds_intervals])

    # create a track where first position of the codon is one
    cds_track = np.zeros(len_cds, dtype=int)
    # set every third position to 1
    cds_track[0::3] = 1
    # concat with zeros of utr3 and utr5
    cds_track = np.concatenate(
        [np.zeros(len_utr5, dtype=int), cds_track, np.zeros(len_utr3, dtype=int)]
    )
    return cds_track


def create_splice_track(t):
    """Create a track of the splice sites of a transcript.
    The track is a 1D array where the positions of the splice sites are 1.

    Args:
        t (gk.Transcript): The transcript object.
    """
    utr3_intervals = t.utr3s if t.utr3s is not None else []
    utr5_intervals = t.utr5s if t.utr5s is not None else []
    cds_intervals = t.cdss if t.cdss is not None else []
    
    len_utr3 = sum([len(x) for x in utr3_intervals]) if utr3_intervals else 0
    len_utr5 = sum([len(x) for x in utr5_intervals]) if utr5_intervals else 0
    len_cds = sum([len(x) for x in cds_intervals]) if cds_intervals else 0

    len_mrna = len_utr3 + len_utr5 + len_cds
    splicing_track = np.zeros(len_mrna, dtype=int)
    cumulative_len = 0
    for exon in t.exons:
        cumulative_len += len(exon)
        splicing_track[cumulative_len - 1 : cumulative_len] = 1

    return splicing_track


def oh_to_seq(onehot_seq):
    """Convert one-hot encoded sequence to DNA string.

    Args:
        onehot_seq: numpy array of shape (length, 6) where columns represent:
                   [A, T, C, G, N, Stop/Pad]

    Returns:
        str: DNA sequence string
    """
    # Define mapping from one-hot positions to nucleotides
    nucleotide_map = {0: "A", 1: "C", 2: "G", 3: "T"}

    # Get the index of the maximum value in each row (argmax)
    indices = np.argmax(onehot_seq[:, :4], axis=1)

    # Convert indices to nucleotides
    dna_sequence = "".join([nucleotide_map[idx] for idx in indices])

    # Remove trailing stop/pad characters if desired
    dna_sequence = dna_sequence.rstrip("*")

    return dna_sequence


def seq_to_oh(seq):
    oh = np.zeros((len(seq), 4), dtype=int)
    for i, base in enumerate(seq):
        base = base.upper()
        if base == "A":
            oh[i, 0] = 1
        elif base == "C":
            oh[i, 1] = 1
        elif base == "G":
            oh[i, 2] = 1
        elif base == "T" or base == "U":
            oh[i, 3] = 1
    return oh


def create_one_hot_encoding(t, genome):
    """Create a track of the sequence of a transcript.
    The track is a 2D array where the rows are the positions
    and the columns are the one-hot encoding of the bases.

    Args:
        t (gk.Transcript): The transcript object.
    """
    seq = "".join([genome.dna(exon) for exon in t.exons])
    oh = seq_to_oh(seq)
    return oh

def find_transcript(genome, t):
    """Find a transcript in a genome by transcript ID.

    Args:
        genome (object): The genome object containing a list of transcripts.
        t (str): The ID of the transcript to find.
    Returns:
        object: The transcript object, if found.
    """
    transcripts = [x for x in genome.transcripts if x.id.split(".")[0] == t]
    if not transcripts:
        raise ValueError(f"Transcript with ID {t} not found.")
    return transcripts[0]


def create_six_track_encoding_with_variant(t, var_str, gencode_version=GENCODE_VERSION):
    """Create a track of the sequence of a transcript.
    The track is a 2D array where the rows are the positions
    and the columns are the one-hot encoding of the bases.
    Concatenate the one-hot encoding with the cds track and the splice track.

    Args:
        t (str): The transcript ID.
        var_str (str): Variant string in format chr:pos:ref:alt
    """
    chrom, pos, ref, alt = var_str.split(":")
    if "chr" not in chrom:
        chrom = "chr" + chrom
    ref, alt = ref.replace("-", ""), alt.replace("-", "")
    genome = Genome(gencode_version)
    if "." in t:
        tr = genome.transcripts[t]
    else: 
        tr = find_transcript(genome, t)

    # For negative strand, reverse complement both ref and alt
    # if tr.strand == "-":
    #    ref = "".join([revcomp[base] for base in ref[::-1]])
    #    alt = "".join([revcomp[base] for base in alt[::-1]])

    variant = Variant(chrom, int(pos) - 1, ref, alt, genome)
    variant_genome = VariantGenome(genome, variant)

    tr = variant_genome.transcripts[t]

    oh = create_one_hot_encoding(tr, variant_genome)
    cds_track = create_cds_track(tr)
    splice_track = create_splice_track(tr)
    six_track = np.concatenate([oh, cds_track[:, None], splice_track[:, None]], axis=1)

    return six_track


def create_six_track_encoding(t, gencode_version=GENCODE_VERSION):
    """Create a track of the sequence of a transcript.
    The track is a 2D array where the rows are the positions
    and the columns are the one-hot encoding of the bases.
    Concatenate the one-hot encoding with the cds track and the splice track.

    Args:
        t (gk.Transcript): The transcript object.
    """
    genome = Genome(gencode_version)
    tr = genome.transcripts[t]

    oh = create_one_hot_encoding(tr, genome)
    cds_track = create_cds_track(tr)
    splice_track = create_splice_track(tr)
    six_track = np.concatenate([oh, cds_track[:, None], splice_track[:, None]], axis=1)

    return six_track


def generate_ptc_sequences(tr, stop_codon="TAG", gencode_version=GENCODE_VERSION):
    """
    Generate all stop-gained mutated sequences by inserting premature termination codons (PTCs).

    Args:
        tr (gk.Transcript): The transcript object
        stop_codon (str): Stop codon to insert (default: "TAG")

    Returns:
        list: List of mutated sequences (numpy arrays)
    """
    mutated_sequences = []
    if stop_codon == "TAG":
        stop_codon_ohe = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])  # T  # A
    elif stop_codon == "TAA":
        stop_codon_ohe = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]])  # T  # A
    elif stop_codon == "TGA":
        stop_codon_ohe = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])  # T  # G
    else:
        raise ValueError(f"Invalid stop codon: {stop_codon}. Must be TAG, TAA, or TGA.")

    # Check if transcript has UTR5s
    if not hasattr(tr, "utr5s") or tr.utr5s is None:
        raise ValueError(f"Transcript {tr.id} has no 5' UTR information")

    start_index = sum(len(exon) for exon in tr.utr5s)  # Start after 5' UTR
    wt_sequence = create_six_track_encoding(tr.id.split(".")[0], gencode_version=gencode_version)  # Use transcript ID

    # Generate PTCs at every third nucleotide (codon positions) for 250 positions
    for i in range(0, 249, 3):  # 0, 3, 6, ..., 246
        ptc_position = start_index + i

        # Check if the position is within the sequence bounds
        if ptc_position + 3 > len(wt_sequence):
            break  # Stop if we exceed sequence length

        # Create mutated sequence by replacing the codon at this position
        mutated_seq = wt_sequence.copy()
        # Replace the codon at the specified position with the stop codon
        mutated_seq[ptc_position : ptc_position + 3, :4] = stop_codon_ohe
        mutated_sequences.append(mutated_seq)

    return mutated_sequences


def generate_penultimate_exon_ptc_sequences(tr, stop_codon="TAG", gencode_version=GENCODE_VERSION):
    """
    Generate PTC sequences in the penultimate exon from -100th codon to the last exon junction.

    Args:
        tr (gk.Transcript): The transcript object
        stop_codon (str): Stop codon to insert (default: "TAG")
        gencode_version (str): GENCODE version to use for genome initialization

    Returns:
        list: List of mutated sequences (numpy arrays)
    """
    mutated_sequences = []

    # Define stop codon one-hot encoding
    if stop_codon == "TAG":
        stop_codon_ohe = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])  # T A G
    elif stop_codon == "TAA":
        stop_codon_ohe = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]])  # T A A
    elif stop_codon == "TGA":
        stop_codon_ohe = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])  # T G A
    else:
        raise ValueError(f"Invalid stop codon: {stop_codon}. Must be TAG, TAA, or TGA.")

    # Check if transcript has required exon structure
    if not hasattr(tr, "exons") or len(tr.exons) < 2:
        raise ValueError(f"Transcript {tr.id} must have at least 2 exons")

    if not hasattr(tr, "cdss") or tr.cdss is None or len(tr.cdss) == 0:
        raise ValueError(f"Transcript {tr.id} has no CDS information")

    # Get the wild-type sequence
    wt_sequence = create_six_track_encoding(tr.id.split(".")[0], gencode_version=gencode_version)

    # Calculate positions
    utr5_length = sum(len(exon) for exon in tr.utr5s) if tr.utr5s is not None else 0
    cds_length = sum(len(exon) for exon in tr.cdss)

    # Find the last exon junction position (end of penultimate exon)
    # This is the start of the last exon in CDS coordinates
    exon_lengths = [len(exon) for exon in tr.exons]

    # Calculate the position of the last exon junction in the mRNA
    last_exon_junction_pos = sum(exon_lengths[:-1])  # End of penultimate exon

    # Convert to CDS coordinates if the junction is within CDS
    if last_exon_junction_pos > utr5_length:
        # Junction is in CDS region
        cds_junction_pos = last_exon_junction_pos - utr5_length

        # Calculate the -100th codon position (300 nucleotides upstream)
        start_codon_pos = max(0, (cds_junction_pos // 3 - 100) * 3)
        start_pos = utr5_length + start_codon_pos

        # End at the last exon junction
        end_pos = last_exon_junction_pos

        # Generate PTCs at every codon position from start to end
        for pos in range(start_pos, min(end_pos, len(wt_sequence) - 2), 3):
            # Check if we have enough space for a full codon
            if pos + 3 > len(wt_sequence):
                break

            # Create mutated sequence
            mutated_seq = wt_sequence.copy()
            mutated_seq[pos : pos + 3, :4] = stop_codon_ohe
            mutated_sequences.append(mutated_seq)

    return mutated_sequences


def generate_all_ptc_sequences(tr, stop_codon="TAG", max_positions=None, gencode_version=GENCODE_VERSION):
    """
    Generate PTC sequences for all possible positions in the coding sequence.

    Args:
        tr (gk.Transcript): The transcript object
        stop_codon (str): Stop codon to insert (default: "TAG")
        max_positions (int): Maximum number of positions to generate (None for all)
        gencode_version (str): GENCODE version to use for genome initialization

    Returns:
        tuple: (list of mutated sequences, list of CDS positions)
               CDS positions are 1-indexed, starting from 1 at the first CDS nucleotide
    """
    mutated_sequences = []
    ptc_positions = []

    # Define stop codon one-hot encoding
    if stop_codon == "TAG":
        stop_codon_ohe = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])  # T A G
    elif stop_codon == "TAA":
        stop_codon_ohe = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]])  # T A A
    elif stop_codon == "TGA":
        stop_codon_ohe = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])  # T G A
    else:
        raise ValueError(f"Invalid stop codon: {stop_codon}. Must be TAG, TAA, or TGA.")

    # Check if transcript has required structure
    if not hasattr(tr, "cdss") or tr.cdss is None or len(tr.cdss) == 0:
        raise ValueError(f"Transcript {tr.id} has no CDS information")

    # Get the wild-type sequence
    wt_sequence = create_six_track_encoding(tr.id.split(".")[0], gencode_version=gencode_version)

    # Calculate positions
    utr5_length = sum(len(exon) for exon in tr.utr5s) if tr.utr5s is not None else 0
    cds_length = sum(len(exon) for exon in tr.cdss) if tr.cdss is not None else 0

    # Generate PTCs at every codon position in the CDS
    cds_start = utr5_length

    # Determine how many positions to generate
    max_cds_codons = cds_length // 3
    if max_positions is None:
        positions_to_generate = max_cds_codons
    else:
        positions_to_generate = min(max_positions, max_cds_codons)

    for codon_idx in range(positions_to_generate):
        ptc_position = cds_start + (codon_idx * 3)

        # Check if we have enough space for a full codon
        if ptc_position + 3 > len(wt_sequence):
            break

        # Create mutated sequence
        mutated_seq = wt_sequence.copy()
        mutated_seq[ptc_position : ptc_position + 3, :4] = stop_codon_ohe
        mutated_sequences.append(mutated_seq)

        # Convert to transcript coordinates (0-based)
        # Position = UTR5_length + PTC_CDS_pos (in nucleotides)
        # This is for the plotting code only, leave it
        transcript_nucleotide_pos = utr5_length + (codon_idx * 3) + 1
        ptc_positions.append(transcript_nucleotide_pos)

    return mutated_sequences, ptc_positions


def create_6track_onehot_sequence(row):
    # Define mapping for nucleotides to one-hot encoding
    nuc_map = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }

    # Combine sequence parts (5'UTR + CDS + 3'UTR)
    full_seq = (
        row["seq_5UTR"]
        + row["fasta_sequence_mut"]
        + row["original_stop_codon"]
        + row["seq_3UTR"]  #  we don't know the og stop codon
    ).upper()
    seq_len = len(full_seq)

    # Create one-hot encoding matrix for sequence
    onehot_seq = np.zeros((4, seq_len))
    for i, nuc in enumerate(full_seq):
        if nuc in nuc_map:
            onehot_seq[:, i] = nuc_map[nuc]

    # Create CDS track (1 at start of each codon in CDS region)
    cds_track = np.zeros(seq_len)
    utr5_len = len(row["seq_5UTR"])
    cds_len = len(row["fasta_sequence_mut"])

    # Mark codon starts in CDS region
    for i in range(utr5_len, utr5_len + cds_len + 3, 3):
        cds_track[i] = 1

    # Create splice site track
    splice_track = np.zeros(seq_len)

    # Parse exon lengths
    utr5s = [int(x) for x in str(row["UTR5s_length"]).split(",") if x.strip()]
    pre_ptc_exons = [int(x) for x in str(row["exons_length_prePTC"]).split(",") if x.strip()]
    ptc_exon_length = row["PTC_CDS_exon_length"]
    post_ptc_exons = [int(x) for x in str(row["exons_length_postPTC"]).split(",") if x.strip()]
    utr3s = [int(x) for x in str(row["UTR3s_length"]).split(",") if x.strip()]

    current_pos = 0
    for utr5 in utr5s[:-1]:
        current_pos += utr5
        if current_pos < seq_len:
            splice_track[current_pos - 1] = 1

    # Mark splice sites
    current_pos = utr5_len  # Start after 5'UTR

    # Pre-PTC exons
    for exon_len in pre_ptc_exons:
        current_pos += exon_len
        if current_pos < seq_len:
            splice_track[current_pos - 1] = 1  # Mark splice junction

    if post_ptc_exons != [0]:
        # PTC-containing exon
        current_pos += ptc_exon_length
        if current_pos < seq_len:
            splice_track[current_pos - 1] = 1

        # Post-PTC exons
        for exon_len in post_ptc_exons[:-1]:  # Exclude last exon
            current_pos += exon_len
            if current_pos < seq_len:
                splice_track[current_pos - 1] = 1

    for utr3 in utr3s[:-1]:
        current_pos += utr3
        if current_pos < seq_len:
            splice_track[current_pos - 1] = 1

    splice_track[-1] = 1  # Mark end of last exon

    # Combine all tracks
    final_encoding = np.vstack([onehot_seq, cds_track, splice_track])

    return final_encoding


def get_UTR_sequences(df, gencode_version=GENCODE_VERSION):
    # Initialize new columns
    df["seq_5UTR"] = ""
    df["seq_3UTR"] = ""

    genome = Genome(gencode_version)

    # Process each transcript
    for idx, row in tqdm(df.iterrows()):
        transcript_id = row["transcript_id"]
        try:
            # Get the transcript object
            tr = genome.transcripts[transcript_id]
        except KeyError:
            df.at[idx, "seq_5UTR"] = None
            df.at[idx, "seq_3UTR"] = None
            continue

        # Get the 5' UTR sequence
        utr5_seq = "".join([genome.dna(utr) for utr in tr.utr5s])
        # Get the 3' UTR sequence
        utr3_seq = "".join([genome.dna(utr) for utr in tr.utr3s])

        # Store sequences and length in the DataFrame
        df.at[idx, "seq_5UTR"] = utr5_seq
        df.at[idx, "seq_3UTR"] = utr3_seq
        df.at[idx, "UTR5s_length"] = ",".join([str(len(utr)) for utr in tr.utr5s])
        df.at[idx, "UTR3s_length"] = ",".join([str(len(utr)) for utr in tr.utr3s])

    # Drop rows with NA in 5' or 3' UTR
    print(f"Number of rows with NA in 5' UTR: {df['seq_5UTR'].isna().sum()}")
    print(f"Number of rows with NA in 3' UTR: {df['seq_3UTR'].isna().sum()}")
    df = df.dropna(subset=["seq_5UTR", "seq_3UTR"]).reset_index(drop=True)
    return df


def get_stopcodons(df, gencode_version=GENCODE_VERSION):
    """
    Extract the original stop codon sequence from the transcript's CDS using GenomeKit.

    Args:
        df (pd.DataFrame): DataFrame containing transcript information with 'transcript_id' column
        gencode_version (str): GENCODE version to use for genome initialization

    Returns:
        pd.DataFrame: DataFrame with added 'original_stop_codon' column
    """
    # Initialize new column
    df["original_stop_codon"] = ""

    genome = Genome(gencode_version)
    matching = 0

    # Process each transcript to get the original stop codon
    for idx, row in tqdm(df.iterrows(), desc="Extracting stop codons"):
        transcript_id = row["transcript_id"]
        tr = genome.transcripts[transcript_id]
        cds_sequence = "".join([genome.dna(cds) for cds in tr.cdss])
        stop_codon = cds_sequence[-3:].upper()
        df.at[idx, "original_stop_codon"] = stop_codon

        seq1 = row.fasta_sequence_wt.upper()
        seq2 = cds_sequence[:-3]
        if seq1 == seq2:
            matching += 1
        else:
            logger.warning(f"Transcript {transcript_id} NOT matching Gencode")

    logger.info(f"Total transcripts matching Gencode: {matching} out of {len(df)}")

    # count how many stop codons are ""
    empty_stop_codons = (df["original_stop_codon"] == "").sum()
    logger.info(f"Found {empty_stop_codons} transcripts with empty stop codons")

    # set them to TAG
    df.loc[df["original_stop_codon"] == "", "original_stop_codon"] = "TAG"

    return df


def get_exon_boundaries_in_cds(transcript, exon_idx):
    """
    Get the start and end positions of an exon within the CDS.
    
    Args:
        transcript: genome_kit Transcript object
        exon_idx: 1-indexed CDS exon number (as stored in PTC_CDS_exon_num)
        
    Returns:
        tuple: (exon_start_nt, exon_end_nt) in CDS coordinates, or (None, None) if not found
    """
    if transcript is None or not hasattr(transcript, 'cdss') or transcript.cdss is None:
        return None, None
    
    # PTC_CDS_exon_num refers to CDS exons, not all transcript exons
    # So we should use transcript.cdss instead of transcript.exons
    if exon_idx < 1 or exon_idx > len(transcript.cdss):
        return None, None
    
    # Calculate cumulative CDS length up to this exon
    cumulative_length = 0
    for i, cds_exon in enumerate(transcript.cdss):
        exon_start = cumulative_length
        cumulative_length += len(cds_exon)
        exon_end = cumulative_length
        
        # Check if this is our target exon (1-indexed)
        if i + 1 == exon_idx:
            return exon_start, exon_end
    
    return None, None
