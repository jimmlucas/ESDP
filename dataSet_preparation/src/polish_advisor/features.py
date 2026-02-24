from pathlib import Path
import pandas as pd
import re

def parse_quast(quast_report: Path) -> pd.DataFrame:
    """
    Lee el report.tsv de QUAST cuando tiene formato clave-valor (dos columnas).
    Devuelve n_contigs, n50, l50, total_len, gc.
    """
    if not quast_report.exists():
        return pd.DataFrame()

    rows = []
    with quast_report.open() as fh:
        for line in fh:
            if line.strip() and not line.startswith("Assembly"):
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    rows.append(parts)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric").T

    rename = {
        "# contigs (>= 0 bp)": "n_contigs",
        "N50": "n50",
        "L50": "l50",
        "Total length": "total_len",
        "GC (%)": "gc",
    }

    # Renombrar solo las que existan
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Convertir a numérico
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def parse_busco(busco_report: Path) -> pd.DataFrame:
    """
    Parsea el archivo short_summary.txt de BUSCO v5+.
    Extrae C, S, D, F, M, n desde la línea tipo:
    C:98.3%[S:97.5%,D:0.8%],F:1.2%,M:0.5%,n:100
    """
    if not busco_report.exists():
        return pd.DataFrame()

    text = busco_report.read_text()
    match = re.search(
        r"C:(?P<C>[\d\.]+)%\[S:(?P<S>[\d\.]+)%,D:(?P<D>[\d\.]+)%\],"
        r"F:(?P<F>[\d\.]+)%,M:(?P<M>[\d\.]+)%,n:(?P<n>\d+)",
        text
    )
    if not match:
        return pd.DataFrame()

    data = {
        "complete": float(match.group("C")),
        "single_copy": float(match.group("S")),
        "duplicated": float(match.group("D")),
        "fragmented": float(match.group("F")),
        "missing": float(match.group("M")),
        "total": int(match.group("n")),
    }
    return pd.DataFrame([data])

def build_feature_table(sample_id: str, quast_report: Path, busco_report: Path) -> pd.DataFrame:
    """
    Une métricas de QUAST y BUSCO en un solo DataFrame con sample_id.
    """
    quast_df = parse_quast(quast_report)
    busco_df = parse_busco(busco_report)

    if quast_df.empty and busco_df.empty:
        return pd.DataFrame()

    feature_data = {"sample_id": sample_id}

    if not quast_df.empty:
        for col in quast_df.columns:
            feature_data[col] = quast_df.iloc[0][col]

    if not busco_df.empty:
        for col in busco_df.columns:
            feature_data[col] = busco_df.iloc[0][col]

    return pd.DataFrame([feature_data])
