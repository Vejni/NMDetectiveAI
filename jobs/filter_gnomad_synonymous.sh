#!/bin/bash
#SBATCH --job-name=gnomad_synonymous
#SBATCH --output=/g/strcombio/fsupek_franklin/mveiner/Projects/NMD/jobs/logs/gnomad_synonymous_%A_%a.out
#SBATCH --error=/g/strcombio/fsupek_franklin/mveiner/Projects/NMD/jobs/logs/gnomad_synonymous_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-24

# Configuration
INPUT_DIR="/g/strcombio/fsupek_franklin/mveiner/Data/gnomad_v4.1/raw"
OUTPUT_DIR="/g/strcombio/fsupek_franklin/mveiner/Data/gnomad_v4.1/synonymous"

# Create directories if they don't exist
mkdir -p /g/strcombio/fsupek_franklin/mveiner/Projects/NMD/jobs/logs
mkdir -p $OUTPUT_DIR

# Map array task ID to chromosome
# 1-22 = chr1-chr22, 23 = chrX, 24 = chrY
if [ $SLURM_ARRAY_TASK_ID -le 22 ]; then
    CHR=$SLURM_ARRAY_TASK_ID
elif [ $SLURM_ARRAY_TASK_ID -eq 23 ]; then
    CHR="X"
elif [ $SLURM_ARRAY_TASK_ID -eq 24 ]; then
    CHR="Y"
fi

echo "Processing chromosome: $CHR"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Filtering for all synonymous variants (irrespective of AF)"
echo "Start time: $(date)"

# Define input and output files
INPUT_FILE="${INPUT_DIR}/gnomad.genomes.v4.1.sites.chr${CHR}.vcf.bgz"
OUTPUT_FILE="${OUTPUT_DIR}/gnomad.genomes.v4.1.sites.chr${CHR}.synonymous.tsv"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# Check if output already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Output file already exists: $OUTPUT_FILE"
    echo "Skipping processing for chromosome $CHR"
    exit 0
fi

echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"

# Filter for synonymous variants
# Extract variants with:
# - FILTER = PASS
# - Contains synonymous_variant in VEP consequence
# - No AF filtering (all variants irrespective of frequency)
echo "Filtering synonymous variants..."

zcat $INPUT_FILE | awk -F'\t' 'BEGIN {OFS="\t"} {
    # Skip header lines
    if ($1 ~ /^#/) {next}
    
    # Only process PASS variants
    if ($7 == "PASS") {
        info = $8
        
        # Check if vep annotation contains synonymous_variant
        if (info ~ /vep=/) {
            # Extract vep field
            match(info, /vep=[^;]*(;|$)/)
            vep_field = substr(info, RSTART, RLENGTH)
            
            # Clean up vep field
            gsub(/vep=/, "", vep_field)
            gsub(/;$/, "", vep_field)
            
            # Split transcript annotations by comma
            n = split(vep_field, transcripts, ",")
            for (j = 1; j <= n; j++) {
                # Split each transcript by pipe
                split(transcripts[j], fields, "|")
                # Check if 2nd field (Consequence) contains synonymous_variant
                if (fields[2] ~ /synonymous_variant/) {
                    print $1, $2, $3, $4, $5, info
                    break
                }
            }
        }
    }
}' > $OUTPUT_FILE

# Check if output file was created successfully
if [ -f "$OUTPUT_FILE" ]; then
    LINE_COUNT=$(wc -l < $OUTPUT_FILE)
    echo "Filtering complete. Found $LINE_COUNT synonymous variants"
else
    echo "ERROR: Output file was not created"
    exit 1
fi

echo "End time: $(date)"
echo "Chromosome $CHR processing complete"
