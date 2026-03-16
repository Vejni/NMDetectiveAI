#!/bin/bash
#SBATCH --job-name=gnomad_rare_stopgain
#SBATCH --output=/g/strcombio/fsupek_franklin/mveiner/Projects/NMD/jobs/logs/gnomad_rare_stopgain_%A_%a.out
#SBATCH --error=/g/strcombio/fsupek_franklin/mveiner/Projects/NMD/jobs/logs/gnomad_rare_stopgain_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-24

# Configuration
INPUT_DIR="/g/strcombio/fsupek_franklin/mveiner/Data/gnomad_v4.1/raw"
OUTPUT_DIR="/g/strcombio/fsupek_franklin/mveiner/Data/gnomad_v4.1/rare_stopgain"
THRESHOLD=0.001

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
echo "Threshold: AF < $THRESHOLD"
echo "Start time: $(date)"

# Define input and output files
INPUT_FILE="${INPUT_DIR}/gnomad.genomes.v4.1.sites.chr${CHR}.vcf.bgz"
OUTPUT_FILE="${OUTPUT_DIR}/gnomad.genomes.v4.1.sites.chr${CHR}.rare_stopgain.tsv"

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

# Filter for rare stop_gained variants in one pass
echo "Filtering rare stop_gained variants..."

zcat $INPUT_FILE | awk -F'\t' -v threshold=$THRESHOLD 'BEGIN {OFS="\t"} {
    # Skip header lines
    if ($1 ~ /^#/) {next}
    
    # Only process PASS variants
    if ($7 == "PASS") {
        info = $8

        # Parse INFO for AF and subpopulation AFs (default: 1 to be conservative)
        split(info, info_array, ";");
        af = 1; af_afr = 1; af_amr = 1; af_fin = 1; af_nfe = 1; af_eas = 1;
        for (i in info_array) {
            if (info_array[i] ~ /^AF=/) { split(info_array[i], v, "="); af = v[2] }
            else if (info_array[i] ~ /^AF_afr=/) { split(info_array[i], v, "="); af_afr = v[2] }
            else if (info_array[i] ~ /^AF_amr=/) { split(info_array[i], v, "="); af_amr = v[2] }
            else if (info_array[i] ~ /^AF_fin=/) { split(info_array[i], v, "="); af_fin = v[2] }
            else if (info_array[i] ~ /^AF_nfe=/) { split(info_array[i], v, "="); af_nfe = v[2] }
            else if (info_array[i] ~ /^AF_eas=/) { split(info_array[i], v, "="); af_eas = v[2] }
        }

        # Variant is rare only if AF < threshold in ALL populations, and at least one AF > 0
        if ((af+0) < threshold && (af_afr+0) < threshold && (af_amr+0) < threshold && (af_fin+0) < threshold && (af_nfe+0) < threshold && (af_eas+0) < threshold && ((af+0) > 0 || (af_afr+0) > 0 || (af_amr+0) > 0 || (af_fin+0) > 0 || (af_nfe+0) > 0 || (af_eas+0) > 0)) {
            # Check VEP annotations for stop_gained
            if (info ~ /vep=/) {
                match(info, /vep=[^;]*(;|$)/)
                vep_field = substr(info, RSTART, RLENGTH)
                gsub(/vep=/, "", vep_field)
                gsub(/;$/, "", vep_field)
                n = split(vep_field, transcripts, ",")
                for (j = 1; j <= n; j++) {
                    split(transcripts[j], fields, "|")
                    if (fields[2] ~ /stop_gained/) {
                        print $1, $2, $3, $4, $5, info
                        break
                    }
                }
            }
        }
    }
}' > $OUTPUT_FILE

# Check if output file was created successfully
if [ -f "$OUTPUT_FILE" ]; then
    LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
    echo "Filtering complete. Found $LINE_COUNT rare stop_gained variants"
else
    echo "ERROR: Output file was not created"
    exit 1
fi

echo "End time: $(date)"
echo "Chromosome $CHR processing complete"
