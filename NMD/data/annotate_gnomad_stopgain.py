"""
Annotate gnomAD v4.1 stopgain variants with NMD-relevant information.

This script processes gnomAD stopgain TSV files (rare or common variants)
and annotates PTC positions with NMD-relevant metrics including:
- PTC position in coding sequence
- Exon length containing the PTC
- Distance to downstream exon-exon junction
- NMD prediction based on established rules

Processes both SNVs and indels that create premature termination codons.
For SNVs, the PTC position is extracted from VEP annotations.
For indels, the PTC position is determined by translating the mutated sequence.
SNVs and indels are saved to separate output files.

Focuses on MANE Select transcripts only.

Usage:
    # Process rare variants (AF < 0.001)
    python -m NMD.data.annotate_gnomad_stopgain --variant-type rare --chr chr21
    python -m NMD.data.annotate_gnomad_stopgain --variant-type rare --chr all
    
    # Process common variants (AF >= 0.001)
    python -m NMD.data.annotate_gnomad_stopgain --variant-type common --chr chr21
    python -m NMD.data.annotate_gnomad_stopgain --variant-type common --chr all
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gzip
import re
import argparse
from loguru import logger
import pandas as pd
import genome_kit as gk
from tqdm import tqdm

from NMD.config import LARGE_DATA_DIR, PROCESSED_DATA_DIR


# Base paths
GNOMAD_BASE_DIR = Path(LARGE_DATA_DIR) / "gnomad_v4.1"
OUTPUT_BASE_DIR = Path(PROCESSED_DATA_DIR) / "gnomad_v4.1"
MANE_GTF_PATH = Path(LARGE_DATA_DIR) / "MANE" / "MANE.GRCh38.v1.0.ensembl_genomic.gtf.gz"

# VEP annotation field indices (from VCF header inspection)
VEP_FIELDS = [
    "Allele", "Consequence", "IMPACT", "SYMBOL", "Gene", "Feature_type", 
    "Feature", "BIOTYPE", "EXON", "INTRON", "HGVSc", "HGVSp", 
    "cDNA_position", "CDS_position", "Protein_position", "Amino_acids", 
    "Codons", "ALLELE_NUM", "DISTANCE", "STRAND", "FLAGS", "VARIANT_CLASS",
    "SYMBOL_SOURCE", "HGNC_ID", "CANONICAL", "MANE_SELECT", "MANE_PLUS_CLINICAL",
    "TSL", "APPRIS", "CCDS", "ENSP", "UNIPROT_ISOFORM", "SOURCE"
]


def load_mane_transcripts() -> set:
    """
    Load MANE Select transcript IDs from GTF file.
    
    Returns:
        Set of MANE Select transcript IDs (e.g., 'ENST00000673477.1')
    """
    logger.info(f"Loading MANE Select transcripts from {MANE_GTF_PATH}")
    mane_transcripts = set()
    
    with gzip.open(MANE_GTF_PATH, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            # Check if this is a MANE Select transcript
            attributes = fields[8]
            if 'MANE_Select' in attributes:
                # Extract transcript_id
                match = re.search(r'transcript_id "([^"]+)"', attributes)
                if match:
                    transcript_id = match.group(1)
                    # Remove version number for matching
                    transcript_id_base = transcript_id.split('.')[0]
                    mane_transcripts.add(transcript_id_base)
    
    logger.info(f"Loaded {len(mane_transcripts)} MANE Select transcripts")
    return mane_transcripts


def parse_vep_annotation(vep_string: str, alt_allele: str) -> List[Dict]:
    """
    Parse VEP annotation string into list of transcript annotations.
    
    Args:
        vep_string: VEP annotation string from INFO field
        alt_allele: The alternate allele to match
        
    Returns:
        List of dictionaries, one per transcript annotation
    """
    annotations = []
    
    # Split by comma to get individual transcript annotations
    transcript_annots = vep_string.split(',')
    
    for annot in transcript_annots:
        fields = annot.split('|')
        
        # Skip if not matching the alt allele
        if len(fields) > 0 and fields[0] != alt_allele:
            continue
        
        # Create dictionary from fields
        annot_dict = {}
        for i, field_name in enumerate(VEP_FIELDS):
            if i < len(fields):
                annot_dict[field_name] = fields[i] if fields[i] else None
            else:
                annot_dict[field_name] = None
        
        annotations.append(annot_dict)
    
    return annotations


def find_ptc_position_from_indel(transcript_id: str, variant_pos: int, ref: str, alt: str, genome: gk.Genome) -> Optional[int]:
    """
    Find the PTC position in CDS for an indel variant by translating the mutated sequence.
    
    Args:
        transcript_id: Ensembl transcript ID (without version)
        variant_pos: Genomic position of the variant (1-based)
        ref: Reference allele
        alt: Alternate allele
        genome: GenomeKit genome object
        
    Returns:
        CDS position of the PTC (1-based), or None if no PTC found
    """
    try:
        tr = genome.transcripts[transcript_id]
    except KeyError:
        logger.warning(f"Transcript {transcript_id} not found in genome")
        return None
    
    # Get the CDS sequence
    try:
        cds_seq = tr.cds.sequence.upper()
    except Exception as e:
        logger.warning(f"Could not get CDS sequence for {transcript_id}: {e}")
        return None
    
    # Convert genomic position to CDS position
    try:
        # Create a genomic interval for the variant position
        variant_interval = gk.Interval(tr.chromosome, variant_pos - 1, variant_pos, tr.strand)
        
        # Map to CDS coordinates
        cds_intervals = tr.cds.liftover(variant_interval)
        
        if not cds_intervals or len(cds_intervals) == 0:
            logger.warning(f"Variant at {variant_pos} not in CDS of {transcript_id}")
            return None
        
        # Get the CDS position (0-based)
        cds_pos_0based = cds_intervals[0].start
        
    except Exception as e:
        logger.warning(f"Could not map variant position to CDS for {transcript_id}: {e}")
        return None
    
    # Apply the indel to the CDS sequence
    # For insertions: ref is shorter than alt (e.g., C -> CAGTA means insert AGTA after position)
    # For deletions: ref is longer than alt (e.g., CAG -> C means delete AG)
    
    ref_len = len(ref)
    alt_len = len(alt)
    
    if tr.strand == '+':
        # Positive strand: straightforward
        if ref_len < alt_len:  # Insertion
            # Insert bases after position
            inserted_bases = alt[ref_len:]
            mutated_cds = cds_seq[:cds_pos_0based + 1] + inserted_bases + cds_seq[cds_pos_0based + 1:]
        elif ref_len > alt_len:  # Deletion
            # Delete bases
            deletion_length = ref_len - alt_len
            mutated_cds = cds_seq[:cds_pos_0based + 1] + cds_seq[cds_pos_0based + 1 + deletion_length:]
        else:
            # Should not happen for indels
            logger.warning(f"Variant is not an indel: {ref} -> {alt}")
            return None
    else:
        # Negative strand: need to reverse complement
        def reverse_complement(seq: str) -> str:
            complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
            return ''.join(complement.get(b, 'N') for b in reversed(seq))
        
        ref_rc = reverse_complement(ref)
        alt_rc = reverse_complement(alt)
        
        if len(ref_rc) < len(alt_rc):  # Insertion
            inserted_bases = alt_rc[len(ref_rc):]
            mutated_cds = cds_seq[:cds_pos_0based + 1] + inserted_bases + cds_seq[cds_pos_0based + 1:]
        elif len(ref_rc) > len(alt_rc):  # Deletion
            deletion_length = len(ref_rc) - len(alt_rc)
            mutated_cds = cds_seq[:cds_pos_0based + 1] + cds_seq[cds_pos_0based + 1 + deletion_length:]
        else:
            logger.warning(f"Variant is not an indel: {ref} -> {alt}")
            return None
    
    # Translate and find the first stop codon
    # Ensure we start from a valid reading frame
    # Find the position within the codon
    start_translation_pos = (cds_pos_0based // 3) * 3  # Start of the affected codon
    
    # Translate from the affected position onwards
    translation_start = start_translation_pos
    
    try:
        # Translate the mutated sequence
        from Bio.Seq import Seq
        
        # Start translating from the beginning to maintain frame
        mutated_protein = str(Seq(mutated_cds).translate(to_stop=False))
        
        # Find the first stop codon in the translated sequence
        ptc_aa_position = mutated_protein.find('*')
        
        if ptc_aa_position == -1:
            logger.warning(f"No PTC found in mutated sequence for {transcript_id}")
            return None
        
        # Convert amino acid position to CDS position (1-based)
        ptc_cds_position = (ptc_aa_position * 3) + 1
        
        return ptc_cds_position
        
    except Exception as e:
        logger.warning(f"Could not translate sequence for {transcript_id}: {e}")
        return None


def calculate_nmd_metrics(transcript_id: str, cds_position: int, genome: gk.Genome) -> Dict:
    """
    Calculate NMD-relevant metrics for a PTC position.
    
    Args:
        transcript_id: Ensembl transcript ID (without version)
        cds_position: Position in CDS (1-based)
        genome: GenomeKit genome object
        
    Returns:
        Dictionary with NMD metrics
    """
    try:
        tr = genome.transcripts[transcript_id]
    except KeyError:
        logger.warning(f"Transcript {transcript_id} not found in genome")
        return None
    
    # Calculate total CDS length
    cds_length = sum(len(exon) for exon in tr.cdss)
    
    # Find which exon contains the PTC
    cumulative_pos = 0
    ptc_exon_idx = None
    ptc_exon_length = None
    position_in_exon = None
    
    for idx, exon in enumerate(tr.cdss):
        exon_len = len(exon)
        if cumulative_pos < cds_position <= cumulative_pos + exon_len:
            ptc_exon_idx = idx
            ptc_exon_length = exon_len
            position_in_exon = cds_position - cumulative_pos
            break
        cumulative_pos += exon_len
    
    if ptc_exon_idx is None:
        logger.warning(f"Could not find exon containing position {cds_position} in {transcript_id}")
        return None
    
    # Calculate distance to downstream junction
    distance_to_downstream_junction = ptc_exon_length - position_in_exon
    
    # Calculate distance from last EEJ (for 50nt rule)
    # last_ejc_position is the CDS position right at the last exon-exon junction
    last_ejc_position = sum(len(exon) for exon in tr.cdss[:-1])
    
    # Distance from last EJC for all PTCs
    # - If in last exon: positive value = how far into last exon
    # - If before last exon: negative value = how far before the junction
    # We'll store absolute distance for the rule check
    distance_from_last_ejc = abs(cds_position - last_ejc_position)
    
    # Check if in last exon
    is_in_last_exon = (ptc_exon_idx == len(tr.cdss) - 1)
    
    # Determine NMD status based on rules (priority order)
    # Rule 1: In last exon (highest priority)
    if is_in_last_exon:
        nmd_status = "NMD_evading_last_exon"
    # Rule 2: Within 50-55 nt of last EEJ (in penultimate or earlier exon, close to last junction)
    elif not is_in_last_exon and distance_from_last_ejc <= 55:
        nmd_status = "NMD_evading_55nt"
    # Rule 3: < 150 nt from start codon (start-proximal)
    elif cds_position <= 150:
        nmd_status = "NMD_evading_150nt"
    # Rule 4: Long exon (>400 nt)
    elif ptc_exon_length > 400:
        nmd_status = "NMD_evading_long_exon"
    else:
        nmd_status = "NMD_triggering"
    
    return {
        'ptc_cds_position': cds_position,
        'cds_length': cds_length,
        'ptc_exon_idx': ptc_exon_idx,
        'ptc_exon_length': ptc_exon_length,
        'position_in_exon': position_in_exon,
        'distance_to_downstream_junction': distance_to_downstream_junction,
        'distance_from_last_ejc': distance_from_last_ejc,
        'is_in_last_exon': is_in_last_exon,
        'predicted_nmd_status': nmd_status,
        'num_cds_exons': len(tr.cdss)
    }


def parse_info_field(info_str: str) -> Dict:
    """
    Parse INFO field to extract AC and AF values.
    
    Args:
        info_str: INFO field string from VCF
        
    Returns:
        Dictionary with AC, AF, and other relevant fields
    """
    info_dict = {}
    
    # Extract AC (allele count)
    ac_match = re.search(r'AC=(\d+)', info_str)
    if ac_match:
        info_dict['AC'] = int(ac_match.group(1))
    
    # Extract AF (allele frequency)
    af_match = re.search(r'AF=([0-9.e-]+)', info_str)
    if af_match:
        info_dict['AF'] = float(af_match.group(1))
    
    # Extract AN (allele number)
    an_match = re.search(r'AN=(\d+)', info_str)
    if an_match:
        info_dict['AN'] = int(an_match.group(1))
    
    return info_dict


def process_chromosome(chrom: str, variant_type: str, mane_transcripts: set, genome: gk.Genome) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all stopgain variants for a single chromosome.
    
    Args:
        chrom: Chromosome name (e.g., 'chr21')
        variant_type: 'rare' or 'common'
        mane_transcripts: Set of MANE Select transcript IDs
        genome: GenomeKit genome object
        
    Returns:
        Tuple of (SNV DataFrame, indel DataFrame) with annotated variants
    """
    # Determine input directory and file pattern based on variant type
    if variant_type == 'rare':
        input_dir = GNOMAD_BASE_DIR / "rare_stopgain"
        file_pattern = f"gnomad.genomes.v4.1.sites.{chrom}.rare_stopgain.tsv"
    else:  # common
        input_dir = GNOMAD_BASE_DIR / "common_stopgain"
        file_pattern = f"gnomad.genomes.v4.1.sites.{chrom}.common_stopgain.tsv"
    
    input_file = input_dir / file_pattern
    
    if not input_file.exists():
        logger.warning(f"Input file not found: {input_file}")
        return pd.DataFrame()
    
    logger.info(f"Processing {chrom} ({variant_type}) from {input_file}")
    
    snv_results = []
    indel_results = []
    total_lines = 0
    snv_count = 0
    indel_count = 0
    mane_match_count_snv = 0
    mane_match_count_indel = 0
    annotated_count_snv = 0
    annotated_count_indel = 0
    
    with open(input_file, 'r') as f:
        for line in tqdm(f, desc=f"Processing {chrom}"):
            total_lines += 1
            
            # Parse TSV line (no header)
            fields = line.strip().split('\t')
            if len(fields) < 6:
                continue
            
            chr_col = fields[0]
            pos = int(fields[1])
            rsid = fields[2]
            ref = fields[3]
            alt = fields[4]
            info = fields[5]
            
            # Extract VEP annotation from INFO field
            vep = ""
            vep_match = re.search(r'vep=([^;\t]+)', info)
            if vep_match:
                vep = vep_match.group(1)
            
            # Determine if SNV or indel
            is_snv = (len(ref) == 1 and len(alt) == 1 and ref in "ACGT" and alt in "ACGT")
            is_indel = not is_snv and (len(ref) != len(alt))
            
            if not is_snv and not is_indel:
                # Skip if neither SNV nor indel
                continue
            
            if is_snv:
                snv_count += 1
            else:
                indel_count += 1
            
            # Parse INFO field
            info_dict = parse_info_field(info)
            
            # Parse VEP annotations
            vep_annotations = parse_vep_annotation(vep, alt)
            
            # Process each transcript annotation
            for annot in vep_annotations:
                # Check if this is a stop_gained consequence
                consequence = annot.get('Consequence', '')
                if 'stop_gained' not in consequence:
                    continue
                
                # Get transcript ID (remove version if present)
                transcript_id = annot.get('Feature', '')
                if not transcript_id:
                    continue
                
                transcript_id_base = transcript_id.split('.')[0]
                
                # Check if MANE Select
                # VEP has MANE_SELECT field or we can check against our set
                mane_select_flag = annot.get('MANE_SELECT', '')
                is_mane = (mane_select_flag and mane_select_flag != '') or (transcript_id_base in mane_transcripts)
                
                if not is_mane:
                    continue
                
                if is_snv:
                    mane_match_count_snv += 1
                else:
                    mane_match_count_indel += 1
                
                # Get CDS position
                if is_snv:
                    # For SNVs, we can use the CDS_position from VEP
                    cds_pos_str = annot.get('CDS_position', '')
                    if not cds_pos_str or cds_pos_str == '-':
                        continue
                    
                    # Parse CDS position (format can be "123/456")
                    try:
                        cds_position = int(cds_pos_str.split('/')[0])
                    except (ValueError, IndexError):
                        continue
                else:
                    # For indels, we need to find the PTC position by translating the sequence
                    cds_position = find_ptc_position_from_indel(transcript_id_base, pos, ref, alt, genome)
                    if cds_position is None:
                        continue
                
                # Calculate NMD metrics
                metrics = calculate_nmd_metrics(transcript_id_base, cds_position, genome)
                
                if metrics is None:
                    continue
                
                if is_snv:
                    annotated_count_snv += 1
                else:
                    annotated_count_indel += 1
                
                # Compile result
                result = {
                    'chr': chr_col,
                    'pos': pos,
                    'rsid': rsid,
                    'ref': ref,
                    'alt': alt,
                    'variant_type': 'SNV' if is_snv else 'indel',
                    'gene_id': annot.get('Gene', '').split('.')[0],  # Remove version
                    'gene_symbol': annot.get('SYMBOL', ''),
                    'transcript_id': transcript_id,
                    'AC': info_dict.get('AC', 0),
                    'AF': info_dict.get('AF', 0.0),
                    'AN': info_dict.get('AN', 0),
                    **metrics
                }
                
                if is_snv:
                    snv_results.append(result)
                else:
                    indel_results.append(result)
    
    logger.info(f"{chrom} statistics:")
    logger.info(f"  Total lines: {total_lines}")
    logger.info(f"  SNVs: {snv_count}")
    logger.info(f"    MANE matches: {mane_match_count_snv}")
    logger.info(f"    Successfully annotated: {annotated_count_snv}")
    logger.info(f"  Indels: {indel_count}")
    logger.info(f"    MANE matches: {mane_match_count_indel}")
    logger.info(f"    Successfully annotated: {annotated_count_indel}")
    
    return pd.DataFrame(snv_results), pd.DataFrame(indel_results)


def main():
    """Main function to process gnomAD stopgain variants."""
    parser = argparse.ArgumentParser(description='Annotate gnomAD stopgain variants with NMD metrics')
    parser.add_argument('--variant-type', type=str, required=True, choices=['rare', 'common'],
                       help='Type of variants to process: "rare" (AF < 0.001) or "common" (AF >= 0.001)')
    parser.add_argument('--chr', type=str, required=True, 
                       help='Chromosome to process (e.g., chr21) or "all" for all chromosomes')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: gnomad_v4.1/annotated[_common])')
    
    args = parser.parse_args()
    
    # Setup output directory based on variant type
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.variant_type == 'rare':
            output_dir = OUTPUT_BASE_DIR / "annotated_rare"
        else:
            output_dir = OUTPUT_BASE_DIR / "annotated_common"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MANE transcripts
    mane_transcripts = load_mane_transcripts()
    
    # Load genome
    genome = gk.Genome("gencode.v41")
    logger.info("Genome loaded successfully")
    
    # Determine which chromosomes to process
    if args.chr.lower() == 'all':
        chromosomes = [f"chr{i}" for i in range(1, 23)] + ['chrX', 'chrY']
    else:
        chromosomes = [args.chr]
    
    # Process each chromosome
    for chrom in chromosomes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {chrom} ({args.variant_type} variants)")
        logger.info(f"{'='*60}")
        
        df_snv, df_indel = process_chromosome(chrom, args.variant_type, mane_transcripts, genome)
        
        # Save SNV results
        if len(df_snv) > 0:
            if args.variant_type == 'rare':
                output_file_snv = output_dir / f"gnomad.v4.1.{chrom}.rare_stopgain_snv.mane.annotated.tsv"
            else:
                output_file_snv = output_dir / f"gnomad.v4.1.{chrom}.common_stopgain_snv.mane.annotated.tsv"
            
            df_snv.to_csv(output_file_snv, sep='\t', index=False)
            logger.info(f"Saved {len(df_snv)} annotated SNVs to {output_file_snv}")
        else:
            logger.warning(f"No SNVs to save for {chrom}")
        
        # Save indel results to separate files
        if len(df_indel) > 0:
            if args.variant_type == 'rare':
                output_file_indel = output_dir / f"gnomad.v4.1.{chrom}.rare_stopgain_indel.mane.annotated.tsv"
            else:
                output_file_indel = output_dir / f"gnomad.v4.1.{chrom}.common_stopgain_indel.mane.annotated.tsv"
            
            df_indel.to_csv(output_file_indel, sep='\t', index=False)
            logger.info(f"Saved {len(df_indel)} annotated indels to {output_file_indel}")
        else:
            logger.warning(f"No indels to save for {chrom}")
    
    logger.info("\n" + "="*60)
    logger.info("Processing complete!")
    logger.info("="*60)

    if args.chr.lower() == 'all':
        # Combine all chromosome files into one for SNVs
        if args.variant_type == 'rare':
            combined_file_snv = output_dir / "gnomad.v4.1.all_chromosomes.rare_stopgain_snv.mane.annotated.tsv"
            file_pattern_snv = "gnomad.v4.1.{chrom}.rare_stopgain_snv.mane.annotated.tsv"
            combined_file_indel = output_dir / "gnomad.v4.1.all_chromosomes.rare_stopgain_indel.mane.annotated.tsv"
            file_pattern_indel = "gnomad.v4.1.{chrom}.rare_stopgain_indel.mane.annotated.tsv"
        else:
            combined_file_snv = output_dir / "gnomad.v4.1.all_chromosomes.common_stopgain_snv.mane.annotated.tsv"
            file_pattern_snv = "gnomad.v4.1.{chrom}.common_stopgain_snv.mane.annotated.tsv"
            combined_file_indel = output_dir / "gnomad.v4.1.all_chromosomes.common_stopgain_indel.mane.annotated.tsv"
            file_pattern_indel = "gnomad.v4.1.{chrom}.common_stopgain_indel.mane.annotated.tsv"
        
        # Combine SNV files
        all_dfs_snv = []
        for chrom in chromosomes:
            file_path = output_dir / file_pattern_snv.format(chrom=chrom)
            if file_path.exists():
                df_chrom = pd.read_csv(file_path, sep='\t')
                all_dfs_snv.append(df_chrom)
        
        if all_dfs_snv:
            combined_df_snv = pd.concat(all_dfs_snv, ignore_index=True)
            combined_df_snv.to_csv(combined_file_snv, sep='\t', index=False)
            logger.info(f"Saved {len(combined_df_snv)} combined annotated SNVs to {combined_file_snv}")
        
        # Combine indel files
        all_dfs_indel = []
        for chrom in chromosomes:
            file_path = output_dir / file_pattern_indel.format(chrom=chrom)
            if file_path.exists():
                df_chrom = pd.read_csv(file_path, sep='\t')
                all_dfs_indel.append(df_chrom)
        
        if all_dfs_indel:
            combined_df_indel = pd.concat(all_dfs_indel, ignore_index=True)
            combined_df_indel.to_csv(combined_file_indel, sep='\t', index=False)
            logger.info(f"Saved {len(combined_df_indel)} combined annotated indels to {combined_file_indel}")


if __name__ == "__main__":
    main()
