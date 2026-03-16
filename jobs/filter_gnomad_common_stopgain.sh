#!/bin/bash
#SBATCH --job-name=gnomad_common_stopgain
#SBATCH --output=/g/strcombio/fsupek_franklin/mveiner/Projects/NMD/jobs/logs/gnomad_common_stopgain_%A_%a.out
#SBATCH --error=/g/strcombio/fsupek_franklin/mveiner/Projects/NMD/jobs/logs/gnomad_common_stopgain_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-24

# Configuration
INPUT_DIR="/g/strcombio/fsupek_franklin/mveiner/Data/gnomad_v4.1/raw"
OUTPUT_DIR="/g/strcombio/fsupek_franklin/mveiner/Data/gnomad_v4.1/common_stopgain"
THRESHOLD=0.001  # AF >= 0.001 for common variants (complement of rare)

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
echo "Threshold: AF >= $THRESHOLD (common variants)"
echo "Start time: $(date)"

# Define input and output files
INPUT_FILE="${INPUT_DIR}/gnomad.genomes.v4.1.sites.chr${CHR}.vcf.bgz"
OUTPUT_FILE="${OUTPUT_DIR}/gnomad.genomes.v4.1.sites.chr${CHR}.common_stopgain.tsv"

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

# Filter for common stopgain variants
# Extract variants with:
# - FILTER = PASS
# - AF >= threshold (common)
# - Contains stop_gained in VEP consequence
echo "Filtering common stopgain variants..."

zcat $INPUT_FILE | awk -F'\t' -v threshold=$THRESHOLD 'BEGIN {OFS="\t"} {
    # Skip header lines
    if ($1 ~ /^#/) {next}
    
    # Only process PASS variants
    if ($7 == "PASS") {
        info = $8
        
        # Parse INFO field for AF and subpopulation AFs
        split(info, info_array, ";");
        af = 0; af_afr = 0; af_amr = 0; af_fin = 0; af_nfe = 0; af_eas = 0; has_stopgain = 0;

        for (i in info_array) {
            if (info_array[i] ~ /^AF=/) { split(info_array[i], v, "="); af = v[2] }
            else if (info_array[i] ~ /^AF_afr=/) { split(info_array[i], v, "="); af_afr = v[2] }
            else if (info_array[i] ~ /^AF_amr=/) { split(info_array[i], v, "="); af_amr = v[2] }
            else if (info_array[i] ~ /^AF_fin=/) { split(info_array[i], v, "="); af_fin = v[2] }
            else if (info_array[i] ~ /^AF_nfe=/) { split(info_array[i], v, "="); af_nfe = v[2] }
            else if (info_array[i] ~ /^AF_eas=/) { split(info_array[i], v, "="); af_eas = v[2] }
        }

        # Check if AF is >= threshold in any population (common if any subpop is common)
        if ((af+0) >= threshold || (af_afr+0) >= threshold || (af_amr+0) >= threshold || (af_fin+0) >= threshold || (af_nfe+0) >= threshold || (af_eas+0) >= threshold) {
            # Check if vep annotation contains stop_gained
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
                    # Check if 2nd field (Consequence) contains stop_gained
                    if (fields[2] ~ /stop_gained/) {
                        has_stopgain = 1
                        break
                    }
                }
                
                # Output if stop_gained found
                if (has_stopgain == 1) {
                    print $1, $2, $3, $4, $5, info
                }
            }
        }
    }
}' > $OUTPUT_FILE

# Check if output file was created successfully
if [ -f "$OUTPUT_FILE" ]; then
    LINE_COUNT=$(wc -l < $OUTPUT_FILE)
    echo "Filtering complete. Found $LINE_COUNT common stopgain variants"
else
    echo "ERROR: Output file was not created"
    exit 1
fi

echo "End time: $(date)"
echo "Chromosome $CHR processing complete"
