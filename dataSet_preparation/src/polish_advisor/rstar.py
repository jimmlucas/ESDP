from typing import Optional
import numpy as np
import pandas as pd

def compute_rstar(metrics: pd.DataFrame,
                  use_busco: bool = False,
                  max_err: float = 0.03,
                  alpha_rel: float = 0.01,
                  need_consecutive: int = 2,
                  busco_complete_min: float = 95.0,
                  non_decreasing_n50: bool = True) -> Optional[int]:
    """
    Determina r* (número óptimo de rondas de polishing) por rendimientos decrecientes.

    Parámetros
    ----------
    metrics : DataFrame con columnas obligatorias:
        'round','qv','indels_per_100kb','error_rate','n50'
        opcional 'complete' si use_busco=True
    use_busco : Si True y la columna 'complete' existe, exige BUSCO>=busco_complete_min
    max_err : error_rate máximo aceptable (ej. 0.03 = 3 %)
    alpha_rel : umbral de mejora relativa acumulada en QV (<alpha_rel en 2 rondas consecutivas para parar)
    need_consecutive : nº de rondas consecutivas que deben cumplir el criterio de mejora marginal
    busco_complete_min : % mínimo de BUSCO complete si use_busco
    non_decreasing_n50 : Si True, N50 no debe decrecer de forma significativa

    Retorna
    -------
    r* (int) o None si no hay plateau claro.
    """
    req = {"round","qv","indels_per_100kb","error_rate","n50"}
    if not req.issubset(metrics.columns):
        raise ValueError(f"Faltan columnas requeridas: {req - set(metrics.columns)}")

    df = metrics.sort_values("round").reset_index(drop=True)
    has_busco = "complete" in df.columns
    use_busco = use_busco and has_busco

    # normalizamos QV para mejoras relativas
    total_gain = max(df["qv"].max() - df["qv"].min(), 1e-6)
    small_improv = []

    for i in range(1, len(df)):
        rel_gain = (df.loc[i,"qv"] - df.loc[i-1,"qv"]) / total_gain
        small_improv.append(rel_gain < alpha_rel)

    # buscamos la primera posición donde haya `need_consecutive` mejoras pequeñas consecutivas
    def consecutive_true(lst, k):
        return any(all(lst[j:j+k]) for j in range(len(lst)-k+1))

    for i in range(len(df)):
        row = df.iloc[i]
        conds = [
            np.isfinite(row["qv"]),
            np.isfinite(row["error_rate"]) and row["error_rate"] <= max_err,
            np.isfinite(row["indels_per_100kb"]),  # se mantiene por compatibilidad
        ]
        if non_decreasing_n50 and i>0:
            conds.append(row["n50"] >= df.iloc[i-1]["n50"])
        if use_busco:
            conds.append(np.isfinite(row.get("complete", np.nan)) and
                         row["complete"] >= busco_complete_min)
        if not all(conds):
            continue

        # ¿hay plateau a partir de aquí?
        if i+need_consecutive-1 <= len(small_improv) and consecutive_true(small_improv[i:], need_consecutive):
            return int(row["round"])

    # fallback: última ronda si ninguna cumple
    return int(df["round"].max()) if len(df) else None
