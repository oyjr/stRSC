"""TLS functions from the paper scoring implementation, with unchanged function bodies."""
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

def canonical_gene(values: Any) -> pd.Index:
    return pd.Index(values).astype(str).str.upper().str.replace(" ", "", regex=False)

def load_coefficients(path: Path) -> pd.Series:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    if frame.shape[1] != 1:
        raise RuntimeError(f"TLS_COEFFICIENT_COLUMN_COUNT:{path}:{frame.shape[1]}")
    values = frame.iloc[:, 0].astype(float)
    values.index = canonical_gene(values.index)
    if values.index.duplicated().any():
        raise RuntimeError(f"TLS_COEFFICIENT_DUPLICATE_GENE:{path}")
    return values

def official_tls_transform(
    values: np.ndarray,
    gene_index: np.ndarray,
    weights: np.ndarray,
    intercept: float,
) -> np.ndarray:
    """Apply the operation order in the official HER2ST predict.py."""
    x = np.asarray(values, dtype=np.float64)
    row_sum = x.sum(axis=1, keepdims=True)
    x = np.divide(x, row_sum, out=np.zeros_like(x), where=np.abs(row_sum) > 1e-12)
    gene_sd = x.std(axis=0, keepdims=True)
    x = np.divide(x, gene_sd, out=np.zeros_like(x), where=gene_sd > 1e-12)
    return x[:, gene_index] @ weights + intercept

def coefficient_binding(
    genes: tuple[str, ...], coefficients: pd.Series
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    canonical = canonical_gene(genes)
    overlap = canonical.intersection(coefficients.index.drop("INTERCEPT", errors="ignore"))
    gene_index = canonical.get_indexer(overlap)
    weights = coefficients.loc[overlap].to_numpy(dtype=np.float64)
    intercept = float(coefficients.get("INTERCEPT", 0.0))
    return gene_index, weights, intercept, overlap.tolist()
