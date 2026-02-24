# ESDP Dataset Preparation

## Build Dataset

This section describes how get the **training dataset** used in ESDP, starting from raw Oxford Nanopore (ONT) reads.  
The goal is to go from **FASTQ** files to a clean tabular dataset with per-round polishing metrics and final QUAST/BUSCO statistics, ready to be used by the [**training pipeline**](../docs/USAGE.md).

The process consists of three main stages:

- [**1. Read quality filtering and coverage subsampling**](#1-read-filtering-and-coverage-subsampling)
- [**2. Assembly + polishing (per-round metrics)**](#2-assembly-and-polishing-pipeline)

- [**3. Consolidation of all runs into a single CSV dataset**](#3-consolidation-into-the-training-dataset)


---

## 1. Read filtering and coverage subsampling

### 1.1. Long-read filtering (QC)

Starting from raw ONT reads (`in.fastq.gz`), we first apply a length and quality-based filtering step using **Filtlong**.  
This removes very short or low-quality reads while keeping most of the informative data.

```bash
# 0. Filter raw ONT reads
filtlong --min_length 1000 --keep_percent 95 in.fastq.gz | gzip > sample_id_filtered.fastq.gz
```
params:

`--min_length` 1000 keeps only reads ≥ 1 kb

`--keep_percent` 95 retains the best 95% of bases by quality

**Output:** 
- sample_id_filtered.fastq.gz (used in the next step)

## 1.2. Subsampling to target coverages (10×, 20×, 40×)

To study the effect of coverage and build a diverse dataset, we generate subsamples of the filtered reads at different target coverages (e.g. 10×, 20×, 40×) using a helper script:

```bash
bash subsample_reads.sh \
  --INDIR  ../path/to/folder/Corynebacterium_glutamicum/ \
  --OUTDIR ../path/to/folder/out_pipeline_Salmonella_enterica/ \
  --GENOME_SIZE 4.8m
```

**params**:

`--INDIR`: directory containing all the raw/SRA runs for a given species or sample set with the correct format (.fastq.gz)

`--OUTDIR`: output directory where the subsampled FASTQ files will be written

`--GENOME_SIZE`: average estimated genome size (e.g. 4.8m for ~4.8 Mb)

This script produces, for each sample, multiple FASTQ files with different effective coverages, e.g.:

SRRXXXXX_10x.fastq.gz

SRRXXXXX_20x.fastq.gz

SRRXXXXX_40x.fastq.gz

Each of these files will be processed independently in the polishing pipeline.

## 2. Assembly and polishing pipeline

For each subsampled readset (e.g. SRR23239696_20x.fastq.gz), we run a full assembly + polishing pipeline implemented in
polish_advisor/run_pipeline.py. This script:

1 - Assembles the genome with Flye.

2 - Performs up to 5 rounds of Racon polishing (with minimap2 and samtools)

3 - Runs a final Medaka polishing step

4 - the `--strict-metrics`, runs QUAST and BUSCO at each polishing round.

5 - Extracts per-round metrics and computes a suggested optimal round (r*).

6 - Runs final **QUAST + BUSCO** on the polished consensus.

Example call:

```bash
python dataSet_preparation/src/polish_advisor/run_pipeline.py \
  --reads out_pipeline_Acinetobacter_baumanii/SRR23239696_20x.fastq.gz \
  --genome-size 4.2m \
  --threads 16 \
  --outdir out_pipeline_Acinetobacter_baumanii/SRR23239696/SRR23239696_20x_strict \
  --strict-metrics
```
This command will create an output folder, e.g.:
```
out_pipeline_Acinetobacter_baumanii/
└── out_pipeline_SRR23239696/
    └── SRR23239696_20x_stric/
        ├── busco/
        ├── flye/
        ├── polish/
        └── quast/
```
**output:**
- `flye/assembly.fasta` - Draft assembly produced by Flye.
Tab-delimited table with **one row per Racon polishing round**, including metrics computed from the
  minimap2 + Racon + samtools stats pipeline, such as:
  - `round`, `n50`, `qv`, `indels_per_100kb`, `error_rate`
  - `--strict-metrics` give to you `busco_complete`, `genome_fraction`, `num_contigs`,
    `total_length`, `gc`, `misassemblies`

- `polish/medaka/consensus.fasta`
Final polished consensus **obtained after the 5 Racon rounds**, further refined by Medaka

- `quast/report.tsv`
Final QUAST report for the Medaka consensus (plus HTML/PDF/other reports in the same folder).

- `busco/short_summary.specific.bacteria_*.txt`
Final BUSCO summary for the Medaka consensus (together with the corresponding .json and run directory).

In addition, the polish/ directory contains intermediate files for each round:

- r1.sam … r5.sam, r1.bam … r5.bam, r*.stats.txt

- racon_r1.fasta … racon_r5.fasta

- Per-round QUAST and BUSCO outputs under `polish/quast_r*/` and `polish/busco_r*/`

These intermediates are used internally to compute the per-round metrics but are not required by the downstream ESDP training pipeline.

## 3. Consolidation into the training dataset

Once all sample × coverage combinations have been processed, each run produces the following key output files:

- `polish/per_round_metrics.tsv` – per-round polishing metrics

- `quast/report.tsv` – final QUAST assembly metrics

- `busco/short_summary*.txt` – final BUSCO completeness metrics

These files are then merged into a final CSV using in `1_csv_merge.py`:

```bash
python 1_csv_merge.py # -> data/all_samples_polishing_metrics.csv
```

The resulting file data/all_samples_polishing_metrics.csv is the starting point for the ESDP analysis and [**modelling pipeline**](../docs/USAGE.md)

In this way, the final dataset used to train ESDP is built reproducibly from raw ONT reads, going through QC, de novo assembly, iterative polishing and comprehensive evaluation with QUAST/BUSCO.