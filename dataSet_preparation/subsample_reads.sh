#!/bin/bash

# Defaults
COVS=(40 20 10)
DOWNSAMPLER=./downsample_reads.py

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --INDIR) INDIR="$2"; shift ;;
        --OUTDIR) OUTDIR="$2"; shift ;;
        --GENOME_SIZE) GENOME_SIZE="$2"; shift ;;
        *) echo "❌ Opción desconocida: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$INDIR" || -z "$OUTDIR" || -z "$GENOME_SIZE" ]]; then
    echo "Uso: bash subsamples_reads.sh --INDIR <input_dir> --OUTDIR <output_dir> --GENOME_SIZE <e.g. 4m>"
    exit 1
fi

mkdir -p "$OUTDIR"

for fq in "$INDIR"/*.fastq.gz; do
    sample=$(basename "$fq" .fastq.gz)

    echo "Procesando $sample ..."
    for cov in "${COVS[@]}"; do
        out="$OUTDIR/${sample}_${cov}x.fastq.gz"
        echo "  ➡️ Downsampling a ${cov}x -> $out"
        python "$DOWNSAMPLER" \
            --fastq "$fq" \
            --genome-size "$GENOME_SIZE" \
            --target-cov "$cov" \
            --out "$out"
    done
done

echo " Done"
