#!/usr/bin/env python3
from pathlib import Path
import subprocess, csv, math, re, argparse
from Bio import SeqIO
import pandas as pd
from polish_advisor.features import build_feature_table
from polish_advisor.rstar import compute_rstar

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline polishing con modo estricto opcional")
    p.add_argument("--reads", required=True, help="FASTQ de lecturas ONT")
    p.add_argument("--genome-size", required=True, help="Tamaño estimado del genoma (ej. 4m)")
    p.add_argument("--threads", type=int, default=16, help="Número de hilos")
    p.add_argument("--outdir", default="out_pipeline", help="Directorio de salida")
    p.add_argument("--strict-metrics", action="store_true",
                   help="Ejecuta QUAST/BUSCO en cada ronda y añade esas métricas")
    return p.parse_args()

def extraer_stats_samtools(stats_path):
    err_rate = None; bases_cigar = None; mismatches = None
    sci_num = r"([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)"
    with open(stats_path) as f:
        for line in f:
            if line.startswith("SN"):
                if "error rate:" in line:
                    m = re.search(sci_num, line)
                    if m: err_rate = float(m.group(1))
                elif "bases mapped (cigar)" in line:
                    m = re.search(r"(\d+)", line)
                    if m: bases_cigar = int(m.group(1))
                elif "mismatches" in line:
                    m = re.search(r"(\d+)", line)
                    if m: mismatches = int(m.group(1))
    return {
        "error_rate": err_rate if err_rate is not None else 1.0,
        "bases_mapped_cigar": bases_cigar if bases_cigar is not None else 1,
        "mismatches": mismatches if mismatches is not None else 0
    }

def parse_busco_complete(busco_dir: Path) -> float:
    summaries = list(busco_dir.glob("short_summary.specific.bacteria_odb12*.txt"))
    if not summaries:
        return float("nan")
    summary_path = summaries[0]
    with open(summary_path) as f:
        for line in f:
            if "C:" in line:
                m = re.search(r"C:([\d\.]+)%", line)
                if m:
                    return float(m.group(1))
    return float("nan")

def extraer_metricas_quast(report_path: Path) -> dict:
    datos = {}
    with open(report_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: 
                continue
            name, value = parts[0], parts[1]
            if name.startswith("Genome fraction"):
                datos["genome_fraction"] = float(value.replace('%', ''))
            elif name.startswith("# contigs"):
                datos["num_contigs"] = int(value)
            elif name.startswith("Total length"):
                datos["total_length"] = int(value)
            elif name.startswith("GC (%)"):
                datos["gc"] = float(value)
            elif name.startswith("Misassemblies"):
                datos["misassemblies"] = int(value)
    return datos

def registrar_metricas_por_ronda(csv_path, ronda, n50, sam_stats, complete=None, quast_metrics=None):
    bases = max(1, sam_stats["bases_mapped_cigar"])
    indels_per_100kb = (sam_stats["mismatches"] / bases) * 1e5
    qv = -10 * math.log10(sam_stats["error_rate"])
    header = ["round","n50","qv","indels_per_100kb","error_rate"]
    row = [ronda,n50,qv,indels_per_100kb,sam_stats["error_rate"]]
    if complete is not None:
        header.append("busco_complete")
        row.append(complete)
    if quast_metrics is not None:
        header += ["genome_fraction","num_contigs","total_length","gc","misassemblies"]
        row += [
            quast_metrics.get("genome_fraction", float("nan")),
            quast_metrics.get("num_contigs", float("nan")),
            quast_metrics.get("total_length", float("nan")),
            quast_metrics.get("gc", float("nan")),
            quast_metrics.get("misassemblies", float("nan"))
        ]
    write_header = not csv_path.exists()
    with open(csv_path,"a",newline="") as f:
        w = csv.writer(f,delimiter="\t")
        if write_header: w.writerow(header)
        w.writerow(row)

def calcular_n50(fasta_path: Path) -> int:
    contigs = sorted([len(r.seq) for r in SeqIO.parse(fasta_path, "fasta")], reverse=True)
    half = sum(contigs)/2; acc = 0
    for l in contigs:
        acc += l
        if acc >= half: return l
    return 0

def run_flye(reads, genome_size, outdir, threads=16):
    cmd = ["flye","--nano-raw",str(reads),"--genome-size",str(genome_size),
           "--out-dir",str(outdir),"--threads",str(threads)]
    subprocess.run(cmd,check=True)
    return outdir / "assembly.fasta"

def run_polishing(reads,draft,outdir,threads=16,max_rounds=5,strict_metrics=False):
    outdir.mkdir(parents=True, exist_ok=True)
    current = draft
    per_round_csv = outdir / "per_round_metrics.tsv"

    for r in range(1,max_rounds+1):
        sam = outdir / f"r{r}.sam"
        polished = outdir / f"racon_r{r}.fasta"

        subprocess.run(f"minimap2 -ax map-ont -t {threads} {current} {reads} > {sam}",
                       shell=True, check=True)
        subprocess.run(f"racon -t {threads} {reads} {sam} {current} > {polished}",
                       shell=True, check=True)

        bam = sam.with_suffix(".bam")
        stats_file = sam.with_suffix(".stats.txt")
        subprocess.run(f"samtools view -Sb {sam} | samtools sort -o {bam}", shell=True, check=True)
        subprocess.run(f"samtools stats {bam} > {stats_file}", shell=True, check=True)

        n50 = calcular_n50(polished)
        sam_stats = extraer_stats_samtools(stats_file)

        complete = None; quast_metrics = None
        if strict_metrics:
            r_quast = outdir / f"quast_r{r}"
            r_busco = outdir / f"busco_r{r}"
            r_quast.mkdir(exist_ok=True)
            r_busco.mkdir(exist_ok=True)
            subprocess.run(f"quast {polished} -o {r_quast}", shell=True, check=True)
            subprocess.run(
                f"busco -i {polished} -o {r_busco} -l bacteria_odb12 -m genome -f --cpu {threads}",
                shell=True, check=True
            )
            complete = parse_busco_complete(r_busco)
            quast_metrics = extraer_metricas_quast(r_quast / "report.tsv")

        registrar_metricas_por_ronda(per_round_csv, r, n50, sam_stats, complete, quast_metrics)
        current = polished

    medaka_out = outdir / "medaka"
    subprocess.run(
        f"medaka_consensus -i {reads} -d {current} -o {medaka_out} -t {threads}",
        shell=True, check=True
    )
    return medaka_out / "consensus.fasta"

def main():
    args = parse_args()
    reads = Path(args.reads); genome_size = args.genome_size
    threads = args.threads; work = Path(args.outdir)
    work.mkdir(exist_ok=True)

    print("==> 1. Ensamblaje con Flye")
    draft = run_flye(reads, genome_size, work / "flye", threads)

    print("==> 2. Polishing con Racon + Medaka")
    final_fasta = run_polishing(
        reads, draft, work / "polish", threads,
        strict_metrics=args.strict_metrics
    )

    print("==> 3. Métricas finales con QUAST + BUSCO")
    quast_dir = work / "quast"; quast_dir.mkdir(exist_ok=True)
    subprocess.run(f"quast {final_fasta} -o {quast_dir}", shell=True, check=True)

    busco_dir = work / "busco"; busco_dir.mkdir(exist_ok=True)
    subprocess.run(
        f"busco -i {final_fasta} -o {busco_dir} -l bacteria_odb12 -m genome -f --cpu {threads}",
        shell=True, check=True
    )

    print("==> 4. Extraer features y calcular r*")
    feat = build_feature_table(
        "sample1",
        quast_dir / "report.tsv",
        list((busco_dir).glob("short_summary.specific.bacteria_odb12*.txt"))[0]
    )
    print(feat)

    per_round = work / "polish" / "per_round_metrics.tsv"
    if per_round.exists():
        metrics_df = pd.read_csv(per_round, sep="\t")
        metrics_df = metrics_df.drop_duplicates(subset=["round"], keep="last").sort_values("round")
        r_star = compute_rstar(metrics_df, use_busco=args.strict_metrics)
        print(f"⭐ Número óptimo de rondas sugerido: {r_star}")
    else:
        print("⚠️  No se encontró per_round_metrics.tsv; r* no calculado")

    print(f"\n✅ Pipeline finalizado. Ensamblaje en: {final_fasta}")

if __name__ == "__main__":
    main()
