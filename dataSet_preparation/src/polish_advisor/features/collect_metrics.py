import argparse, csv
from pathlib import Path
import pandas as pd

def parse_samtools_stats(p):
    stats = {"avg_read_length": None, "avg_read_quality": None}
    if not Path(p).exists():
        return stats
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line=line.strip()
            if line.startswith('average length:'):
                try: stats["avg_read_length"] = float(line.split(':')[-1].strip())
                except: pass
            if line.startswith('average quality:'):
                try: stats["avg_read_quality"] = float(line.split(':')[-1].strip())
                except: pass
    return stats

def parse_quast_report(p):
    metrics = {"n_contigs": None, "n50": None, "l50": None, "total_len": None, "gc": None}
    if not Path(p).exists():
        return metrics
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        rdr = csv.DictReader(f, delimiter='\t')
        row = next(rdr, None)
        if row:
            def get(keys):
                for k in keys:
                    if k in row and row[k]:
                        return row[k]
                return None
            metrics["n_contigs"] = get(["# contigs (>= 0 bp)", "# contigs"])
            metrics["n50"]       = get(["N50"])
            metrics["l50"]       = get(["L50"])
            metrics["total_len"] = get(["Total length", "Total length (>= 0 bp)"])
            metrics["gc"]        = get(["GC (%)"])
    for k in ["n_contigs", "n50", "l50", "total_len", "gc"]:
        try:
            metrics[k] = float(str(metrics[k]).replace(',', '')) if metrics[k] is not None else None
        except: pass
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample-id', required=True)
    ap.add_argument('--mapping', required=True)
    ap.add_argument('--quast', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sam = parse_samtools_stats(args.mapping)
    qua = parse_quast_report(args.quast)

    row = {
        "sample_id": args.sample_id,
        "avg_read_length": sam["avg_read_length"],
        "avg_read_quality": sam["avg_read_quality"],
        "n_contigs": qua["n_contigs"],
        "n50": qua["n50"],
        "l50": qua["l50"],
        "total_len": qua["total_len"],
        "gc": qua["gc"],
    }
    df = pd.DataFrame([row])
    df.to_parquet(args.out, index=False)

if __name__ == "__main__":
    main()
