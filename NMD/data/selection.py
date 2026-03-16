from pathlib import Path
from typing import List, Tuple
import gzip
from loguru import logger
import genome_kit as gk
from tqdm import tqdm

from NMD.config import INTERIM_DATA_DIR, GENCODE_VERSION
from NMD.modeling.predict import annotate_vcf_with_predictions


def find_vep_vcf_files(base_path: str) -> List[Path]:
    """
    Find all VCF files in the VEP directory structure.
    
    Args:
        base_path: Base path to search for VCF files (e.g., "/path/to/VEP/*/*.vcf")
        
    Returns:
        List of Path objects for found VCF files
    """
    base_dir = Path(base_path).parent.parent  # Remove "/*/*.vcf" to get base directory
    vcf_files = []
    
    # Search for VCF files in the pattern: base_dir/tissue_type/*.vcf
    for tissue_dir in base_dir.iterdir(): 
        if tissue_dir.is_dir():
            # Look for .vcf and .vcf.gz files directly in tissue directories
            vcf_pattern = list(tissue_dir.glob("*.vcf")) + list(tissue_dir.glob("*.vcf.gz"))
            vcf_files.extend(vcf_pattern)
    
    logger.info(f"Found {len(vcf_files)} VCF files")
    return vcf_files


def determine_nmd_status(chromosome: str, tr: gk.Transcript, cds_position: int) -> str:
    """
    Determine if a variant is in an NMD evading or triggering region.
    
    NMD Evading rules:
    1. Within the last exon
    2. Within 55 nucleotides from last exon-exon junction 
    3. Within first 150 coding nucleotides of transcript
    4. Within a long exon (e.g., > 400 nt)
    
    Returns:
        "NMD_evading" or "NMD_triggering"
    """
    # Ensure chromosome format is correct for GenomeKit
    if not chromosome.startswith('chr'):
        chromosome = f"chr{chromosome}"
    cds_position = int(cds_position)

    if len(tr.cdss) == 0:
        logger.warning(f"Transcript {tr.id} has no CDS. Defaulting to Unknown.")
        return "NMD_unknown"

    # Build CDS-relative exon start/end positions (1-based)
    exon_lengths = [len(exon) for exon in tr.cdss]
    cds_starts = []
    cds_ends = []
    cur = 1
    for L in exon_lengths:
        cds_starts.append(cur)
        cds_ends.append(cur + L - 1)
        cur += L

    # Rule 1: Within the last CDS exon
    last_start = cds_starts[-1]
    last_end = cds_ends[-1]
    if cds_position >= last_start and cds_position <= last_end:
        return "NMD_evading_last_exon"

    # Rule 2: Within last 55 nt of penultimate exon (i.e., within 55 nt upstream of last EJC)
    if len(cds_starts) > 1:
        last_ejc_pos = last_start - 1  # position of last exon-exon junction in CDS coords
        pen_start = cds_starts[-2]
        pen_end = cds_ends[-2]
        if cds_position >= pen_start and cds_position <= pen_end:
            # last 55 nt of penultimate exon: positions > last_ejc_pos - 55 and <= last_ejc_pos
            if cds_position > (last_ejc_pos - 55) and cds_position <= last_ejc_pos:
                return "NMD_evading_55nt"

    # Rule 3: Within first 150 coding nucleotides
    if cds_position <= 150:
        return "NMD_evading_150nt"

    # Rule 4: Within a long exon (> 400 nt) (use CDS-relative exon lengths)
    for start, end, L in zip(cds_starts, cds_ends, exon_lengths):
        if L > 400 and cds_position >= start and cds_position <= end:
            return "NMD_evading_long_exon"

    return "NMD_triggering"


def annotate_variant_with_nmd_status(variant_line: str, genome: gk.Genome) -> str:
    """
    Annotate a variant line with NMD status.
    
    Args:
        variant_line: Tab-separated variant line from VCF
        genome: GenomeKit genome object
        
    Returns:
        Annotated variant line with NMD status appended
    """
    columns = variant_line.split('\t')
    
    # Parse VEP format columns
    location = columns[1]  # e.g., "1:12345"
    transcript_id = columns[4]  # Feature (Ensembl transcript ID)
    cds_position = columns[8]  # CDS_position
    
    # Parse chromosome and position from location
    chrom, pos_str = location.split(':')
    cds_position = int(cds_position)

    try:
        tr = genome.transcripts[transcript_id]
    except KeyError:
        logger.warning(f"Transcript {transcript_id} not found in genome. Skipping NMD annotation.")
        return variant_line + '\t' + "NMD_unknown" + '\t0\t0\t0'

    # Determine NMD status
    nmd_status = determine_nmd_status(chrom, tr, cds_position)
    cds_length = sum(len(exon) for exon in tr.cdss) if hasattr(tr, 'cdss') and tr.cdss else 0
    n_exons = len(tr.cdss) if hasattr(tr, 'cdss') and tr.cdss else 0

    # Determine exon number containing the PTC (CDS-relative)
    exon_number = 0
    if n_exons > 0:
        exon_lengths = [len(exon) for exon in tr.cdss]
        cur = 1
        for i, L in enumerate(exon_lengths):
            start = cur
            end = cur + L - 1
            if cds_position >= start and cds_position <= end:
                exon_number = i + 1
                break
            cur += L

    return variant_line + '\t' + nmd_status + '\t' + str(cds_length) + '\t' + str(n_exons) + '\t' + str(exon_number)


def process_and_write_variants(vcf_file: Path, stopgain_writer, synonymous_writer, genome: gk.Genome,
                               require_mane: bool = True) -> Tuple[int, int]:
    """
    Process a single VCF file and write variants directly to output files.
    
    Selects stop_gained variants in protein_coding transcripts (optionally MANE Select only)
    and synonymous variants.
    
    Args:
        vcf_file: Path to the VCF file
        stopgain_writer: Open file handle for stop-gained variants
        synonymous_writer: Open file handle for synonymous variants  
        genome: GenomeKit genome object
        require_mane: If True, only keep MANE Select transcripts
        
    Returns:
        Tuple of (stopgain_count, synonymous_count)
    """
    stopgain_count = 0
    synonymous_count = 0
    
    # Handle both .vcf and .vcf.gz files
    open_func = gzip.open if vcf_file.suffix == '.gz' else open
    mode = 'rt' if vcf_file.suffix == '.gz' else 'r'
    
    with open_func(vcf_file, mode) as f:
        for line in f:
            # Skip header lines
            if line.startswith('#'):
                continue
            # Split line by tabs to check CDS_position column (9th column, index 8)
            columns = line.rstrip('\n').split('\t')
            # Check if we have at least 9 columns and the 9th column is not empty or "-"
            if len(columns) < 9 or columns[8] == "-" or not columns[8].strip():
                continue
            consequence = columns[6]  # Consequence column
            extra = columns[13] if len(columns) > 13 else ""  # Extra column
            # Only keep protein_coding biotype
            if "BIOTYPE=protein_coding" not in extra:
                continue
            # Optionally require MANE Select transcript
            if require_mane and "MANE_SELECT" not in extra:
                continue
            variant_line = line.rstrip('\n')
            if "stop_gained" in consequence:
                annotated_line = annotate_variant_with_nmd_status(variant_line, genome)
                stopgain_writer.write(annotated_line + '\n')
                stopgain_count += 1
            elif "synonymous_variant" in consequence:
                annotated_line = annotate_variant_with_nmd_status(variant_line, genome)
                synonymous_writer.write(annotated_line + '\n')
                synonymous_count += 1
    
    return stopgain_count, synonymous_count


def combine_variants(vep_base_path: str, output_dir: Path = None, require_mane: bool = True) -> Tuple[Path, Path]:
    """
    Process all VEP annotated VCF files to extract and combine stop_gained and 
    synonymous variants from protein_coding transcripts (MANE Select by default).
    Output as TSV files with NMD status annotation.
    Uses streaming to avoid keeping all variants in memory.
    
    Args:
        vep_base_path: Base path pattern for VEP files (e.g., "/path/to/VEP/*/*.vcf")
        output_dir: Optional output directory path. If None, uses default location.
        require_mane: If True, only keep MANE Select transcripts.
        
    Returns:
        Tuple of (stopgain_output_file, synonymous_output_file) paths
    """
    # Set default output directory
    if output_dir is None:
        output_dir = INTERIM_DATA_DIR / "selection"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    stopgain_output_file = output_dir / "stopgain_variants_annotated.tsv"
    synonymous_output_file = output_dir / "synonymous_variants_annotated.tsv"
    
    # Initialize GenomeKit genome
    logger.info("Loading GenomeKit genome...")
    genome = gk.Genome("gencode.v47") # VEP used in TCGA was actually 45
    logger.info("GenomeKit genome loaded successfully")
    
    # Find all VCF files
    vcf_files = find_vep_vcf_files(vep_base_path)
    
    # Get column header from first file
    column_header = None
    for vcf_file in tqdm(vcf_files):
        open_func = gzip.open if vcf_file.suffix == '.gz' else open
        mode = 'rt' if vcf_file.suffix == '.gz' else 'r'
        with open_func(vcf_file, mode) as f:
            for line in f:
                if line.startswith('#Uploaded_variation'):
                    column_header = line.rstrip('\n')[1:] + '\tNMD_status\tCDS_length\tn_exons\texon_number'
                    break
        if column_header:
            break
    
    # Open output files and write headers
    total_stopgain_variants = 0
    total_synonymous_variants = 0
    
    with open(stopgain_output_file, 'w') as stopgain_writer, open(synonymous_output_file, 'w') as synonymous_writer:
        # Write headers
        stopgain_writer.write(column_header + '\n')
        synonymous_writer.write(column_header + '\n')
        
        # Process each VCF file and stream variants to output
        for vcf_file in tqdm(vcf_files):
            logger.info(f"Processing {vcf_file}")
            
            stopgain_count, synonymous_count = process_and_write_variants(
                vcf_file, stopgain_writer, synonymous_writer, genome,
                require_mane=require_mane
            )
            
            total_stopgain_variants += stopgain_count
            total_synonymous_variants += synonymous_count
            
            logger.debug(f"Processed {stopgain_count} stop-gained and {synonymous_count} synonymous variants from {vcf_file}")
    
    logger.info(f"Successfully created annotated TSV files:")
    logger.info(f"  {stopgain_output_file} with {total_stopgain_variants} stop-gained variants (protein_coding{', MANE Select' if require_mane else ''})")
    logger.info(f"  {synonymous_output_file} with {total_synonymous_variants} synonymous variants")
    
    return stopgain_output_file, synonymous_output_file


def main():
    """
    Main function to process VEP annotated VCF files.
    """

    # Default behavior: process VCF files
    vep_base_path = "/g/strcombio/fsupek_franklin/igalvan/hotspots_TCGA_v43/17k_WGS/VEP/*/*.vcf"
    combine_variants(vep_base_path)

    # annotate the combined file
    tsv_file = INTERIM_DATA_DIR / "selection" / "stopgain_variants_annotated.tsv"
    annotate_vcf_with_predictions(tsv_file)


if __name__ == "__main__":
    main()