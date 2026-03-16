"""Process gnomAD v4.1 background (synonymous or missense) variants to count allele counts by gene.

This script processes gnomAD variant TSV files and aggregates allele counts
by gene symbol, annotated with NMD region (triggering vs evading) using the
same positional rules as the stopgain annotation pipeline. The region-stratified
counts serve as position-matched denominators when normalizing NMD ratios.

Focuses on MANE Select transcripts only to match the stopgain analysis.

Usage:
    python -m NMD.data.process_gnomad_synonymous --chr chr21
    python -m NMD.data.process_gnomad_synonymous --chr all
    python -m NMD.data.process_gnomad_synonymous --chr all --variant-type missense
"""

from pathlib import Path
from typing import Dict, List, Optional
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

# VEP consequence strings per variant type
CONSEQUENCE_FILTERS = {
    "synonymous": "synonymous_variant",
    "missense":   "missense_variant",
}

# Use gencode.v41 - closest available to gnomAD v4.1's v39 in genome_kit
GENCODE_VERSION = "gencode.v41"

# VEP annotation field indices (from VCF header inspection)
VEP_FIELDS = [
    "Allele", "Consequence", "IMPACT", "SYMBOL", "Gene", "Feature_type", 
    "Feature", "BIOTYPE", "EXON", "INTRON", "HGVSc", "HGVSp", 
    "cDNA_position", "CDS_position", "Protein_position", "Amino_acids", 
    "Codons", "ALLELE_NUM", "DISTANCE", "STRAND", "FLAGS", "VARIANT_CLASS",
    "SYMBOL_SOURCE", "HGNC_ID", "CANONICAL", "MANE_SELECT", "MANE_PLUS_CLINICAL",
    "TSL", "APPRIS", "CCDS", "ENSP", "UNIPROT_ISOFORM", "SOURCE"
]

MANE_GTF_PATH = Path(LARGE_DATA_DIR) / "MANE" / "MANE.GRCh38.v1.0.ensembl_genomic.gtf.gz"

# Chromosomes to process
CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def load_mane_transcripts() -> set:
    """
    Load MANE Select transcript IDs from GTF file.
    
    Returns:
        Set of MANE Select transcript IDs (base, without version numbers)
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


def classify_nmd_region(transcript_id: str, cds_position: int, genome: gk.Genome) -> Optional[str]:
    """
    Classify which NMD region a CDS position falls in, using the same positional
    rules as the stopgain annotation pipeline.

    Args:
        transcript_id: Ensembl transcript ID (without version)
        cds_position: 1-based nucleotide position in the CDS
        genome: GenomeKit genome object

    Returns:
        'NMD_triggering', 'NMD_evading', or None if classification fails
    """
    try:
        tr = genome.transcripts[transcript_id]
    except KeyError:
        return None

    cumulative_pos = 0
    ptc_exon_idx = None
    ptc_exon_length = None

    for idx, exon in enumerate(tr.cdss):
        exon_len = len(exon)
        if cumulative_pos < cds_position <= cumulative_pos + exon_len:
            ptc_exon_idx = idx
            ptc_exon_length = exon_len
            break
        cumulative_pos += exon_len

    if ptc_exon_idx is None:
        return None

    is_in_last_exon = (ptc_exon_idx == len(tr.cdss) - 1)
    last_ejc_position = sum(len(exon) for exon in tr.cdss[:-1])
    distance_from_last_ejc = abs(cds_position - last_ejc_position)

    # Same NMD rules as in annotate_gnomad_stopgain.py
    if is_in_last_exon:
        return 'NMD_evading'
    elif distance_from_last_ejc <= 55:
        return 'NMD_evading'
    elif cds_position <= 150:
        return 'NMD_evading'
    elif ptc_exon_length > 400:
        return 'NMD_evading'
    else:
        return 'NMD_triggering'


def process_chromosome(chr_name: str, mane_transcripts: set, genome: gk.Genome, variant_type: str) -> pd.DataFrame:
    """
    Process a single chromosome file to extract synonymous variant counts.

    Args:
        chr_name: Chromosome name (e.g., 'chr21')
        mane_transcripts: Set of MANE Select transcript IDs
        genome: GenomeKit genome object for NMD region classification

    Returns:
        DataFrame with columns: gene_symbol, transcript_id, chr, pos, ref, alt, AC, nmd_region
    """
    input_file = GNOMAD_BASE_DIR / variant_type / f"gnomad.genomes.v4.1.sites.{chr_name}.{variant_type}.tsv"
    consequence_key = CONSEQUENCE_FILTERS.get(variant_type)

    logger.info(f"Processing {input_file}")
    
    if not input_file.exists():
        logger.warning(f"File not found: {input_file}")
        return pd.DataFrame()
    
    records = []
    
    with open(input_file, 'r') as f:
        for line in tqdm(f, desc=f"Processing {chr_name}"):
            fields = line.strip().split('\t')
            
            if len(fields) < 6:
                continue
            
            chrom = fields[0]
            pos = fields[1]
            ref = fields[3]
            alt = fields[4]
            info = fields[5]
            
            # Extract AC from INFO field
            ac_match = re.search(r'AC=(\d+)', info)
            if not ac_match:
                continue
            ac = int(ac_match.group(1))
            
            # Extract VEP annotations
            vep_match = re.search(r'vep=(.+?)(?:\t|$)', info)
            if not vep_match:
                continue
            vep_string = vep_match.group(1)
            
            # Parse VEP annotations
            vep_annotations = parse_vep_annotation(vep_string, alt)
            
            # Filter for MANE Select transcripts with synonymous_variant consequence
            for annot in vep_annotations:
                # Check if desired variant type
                consequence = annot.get('Consequence', '')
                if consequence_key not in consequence:
                    continue
                
                # Get transcript ID
                transcript_id = annot.get('Feature')
                if not transcript_id:
                    continue
                
                # Remove version number
                transcript_id_base = transcript_id.split('.')[0]
                
                # Check if MANE Select
                if transcript_id_base not in mane_transcripts:
                    continue
                
                # Get gene symbol
                gene_symbol = annot.get('SYMBOL')
                if not gene_symbol:
                    continue

                # Get CDS position and classify NMD region
                cds_pos_raw = annot.get('CDS_position')
                nmd_region = None
                if cds_pos_raw:
                    # VEP CDS_position may be a range or fraction (e.g. "123-125" or "123/1500")
                    try:
                        cds_pos = int(str(cds_pos_raw).split('-')[0].split('/')[0])
                        nmd_region = classify_nmd_region(transcript_id_base, cds_pos, genome)
                    except (ValueError, TypeError):
                        pass

                # Store record
                records.append({
                    'gene_symbol': gene_symbol,
                    'transcript_id': transcript_id_base,
                    'chr': chrom,
                    'pos': int(pos),
                    'ref': ref,
                    'alt': alt,
                    'AC': ac,
                    'nmd_region': nmd_region
                })
    
    logger.info(f"Extracted {len(records)} MANE Select {variant_type} variants from {chr_name}")
    
    return pd.DataFrame(records)


def process_all_chromosomes(mane_transcripts: set, genome: gk.Genome, chromosomes: List[str] = None,
                            variant_type: str = "synonymous"):
    """Process all chromosomes and save aggregated results.

    Args:
        mane_transcripts: Set of MANE Select transcript IDs
        genome: GenomeKit genome object for NMD region classification
        chromosomes: List of chromosome names to process (default: all)
        variant_type: 'synonymous' or 'missense'
    """
    if chromosomes is None:
        chromosomes = CHROMOSOMES

    output_dir = Path(PROCESSED_DATA_DIR) / "gnomad_v4.1" / variant_type
    all_records = []

    for chr_name in chromosomes:
        chr_df = process_chromosome(chr_name, mane_transcripts, genome, variant_type=variant_type)
        if not chr_df.empty:
            all_records.append(chr_df)
    
    if not all_records:
        logger.error("No records extracted from any chromosome")
        return
    
    # Combine all chromosomes
    logger.info("Combining all chromosomes")
    combined_df = pd.concat(all_records, ignore_index=True)

    logger.info(f"Total {variant_type} variants across all chromosomes: {len(combined_df)}")
    logger.info(f"Total allele count: {combined_df['AC'].sum()}")
    logger.info(f"Unique genes: {combined_df['gene_symbol'].nunique()}")
    
    # Save detailed variant-level data
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"gnomad.v4.1.all_chromosomes.{variant_type}.mane.tsv"
    combined_df.to_csv(output_file, sep='\t', index=False)
    logger.info(f"Saved variant-level data to {output_file}")

    # Aggregate by gene (overall summary)
    logger.info("Aggregating by gene")
    gene_summary = combined_df.groupby('gene_symbol').agg(
        ac_synonymous=('AC', 'sum'),
        n_synonymous_variants=('gene_symbol', 'size')
    ).reset_index()

    gene_summary = gene_summary.sort_values('ac_synonymous', ascending=False)

    logger.info(f"Gene-level summary for {variant_type}:")
    logger.info(f"  Total genes: {len(gene_summary)}")

    gene_output_file = output_dir / f"gnomad.v4.1.all_chromosomes.{variant_type}.mane.gene_summary.tsv"
    logger.info(f"Saved gene-level summary to {gene_output_file}")

    # Aggregate by gene AND NMD region (for position-matched normalization)
    logger.info("Aggregating by gene and NMD region")
    classified_df = combined_df[combined_df['nmd_region'].notna()].copy()
    n_unclassified = len(combined_df) - len(classified_df)
    if n_unclassified > 0:
        logger.warning(f"{n_unclassified} variants could not be classified into an NMD region and are excluded from region summary")

    region_long = classified_df.groupby(['gene_symbol', 'nmd_region']).agg(
        ac_synonymous=('AC', 'sum'),
        n_synonymous_variants=('gene_symbol', 'size')
    ).reset_index()

    # Pivot to wide format: one row per gene
    region_wide = region_long.pivot(index='gene_symbol', columns='nmd_region', values=['ac_synonymous', 'n_synonymous_variants'])
    region_wide.columns = [f"{metric}_{region.lower()}" for metric, region in region_wide.columns]
    region_wide = region_wide.reset_index()

    # Fill missing regions with 0
    for col in ['ac_synonymous_nmd_triggering', 'n_synonymous_variants_nmd_triggering',
                'ac_synonymous_nmd_evading', 'n_synonymous_variants_nmd_evading']:
        if col not in region_wide.columns:
            region_wide[col] = 0
        else:
            region_wide[col] = region_wide[col].fillna(0).astype(int)

    region_wide = region_wide.sort_values('gene_symbol')

    logger.info(f"NMD region summary: {len(region_wide)} genes with classified synonymous variants")
    logger.info(f"  NMD-triggering region variants: {region_wide['n_synonymous_variants_nmd_triggering'].sum():.0f}")
    logger.info(f"  NMD-evading region variants: {region_wide['n_synonymous_variants_nmd_evading'].sum():.0f}")

    region_output_file = output_dir / f"gnomad.v4.1.all_chromosomes.{variant_type}.mane.nmd_region_summary.tsv"
    region_wide.to_csv(region_output_file, sep='\t', index=False)
    logger.info(f"Saved NMD region summary to {region_output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Process gnomAD background (synonymous or missense) variants')
    parser.add_argument('--chr', type=str, default='all',
                      help='Chromosome to process (e.g., chr21) or "all" for all chromosomes')
    parser.add_argument('--variant-type', type=str, default='synonymous',
                      choices=['synonymous', 'missense'],
                      help='Variant type to process (default: synonymous)')
    args = parser.parse_args()
    variant_type = args.variant_type
    
    # Load MANE transcripts
    mane_transcripts = load_mane_transcripts()

    # Load genome for NMD region classification
    logger.info(f"Loading GenomeKit genome: {GENCODE_VERSION}")
    genome = gk.Genome(GENCODE_VERSION)
    logger.info("Genome loaded successfully")

    # Determine which chromosomes to process
    if args.chr == 'all':
        chromosomes = CHROMOSOMES
        logger.info(f"Processing all {len(chromosomes)} chromosomes ({variant_type})")
    else:
        chromosomes = [args.chr]
        logger.info(f"Processing {args.chr} ({variant_type})")

    # Process chromosomes
    process_all_chromosomes(mane_transcripts, genome, chromosomes, variant_type=variant_type)
    
    logger.success("Processing complete!")


if __name__ == '__main__':
    main()
