#!/usr/bin/env python3
"""Shared, model-blind machinery for G0-050.

The graph filter bank is scalable to native Xenium cell graphs.  Let S be the
lazy symmetric diffusion operator ``(I + D^-1/2 A D^-1/2) / 2``.  Its spectrum
is in [0, 1].  The five positive semidefinite filters are

    B1 = S^16
    B2 = S^8  - S^16
    B3 = S^4  - S^8
    B4 = S^2  - S^4
    B5 = I    - S^2

They sum exactly to I.  B1 therefore contains changes shared across more graph
steps and B5 contains changes between nearby cells.  All RNA halves and model
predictions use the same graph and the same filters.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


BANDS = ("B1", "B2", "B3", "B4", "B5")
MILESTONES = (0, 2, 4, 8, 16)
NUMERIC_TOL = 1e-12


class ContractError(RuntimeError):
    pass


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def load_protocol(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, sep="\t", index=False, lineterminator="\n")
    os.replace(tmp, path)


def decode_text(values: Iterable[Any]) -> np.ndarray:
    return np.asarray(
        [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in values],
        dtype=object,
    )


def save_canonical(
    output: Path,
    counts: sparse.csr_matrix,
    cells: pd.DataFrame,
    genes: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output / "counts_cells_by_genes.npz", counts, compressed=True)
    cells.to_parquet(output / "cell_axis.parquet", index=False, compression="zstd")
    genes.to_parquet(output / "gene_axis.parquet", index=False, compression="zstd")
    atomic_json(output / "metadata.json", metadata)


def load_canonical(path: Path) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = (
        path / "counts_cells_by_genes.npz",
        path / "cell_axis.parquet",
        path / "gene_axis.parquet",
        path / "metadata.json",
    )
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise ContractError(f"CANONICAL_FILES_MISSING:{missing}")
    counts = sparse.load_npz(required[0]).tocsr()
    cells = pd.read_parquet(required[1])
    genes = pd.read_parquet(required[2])
    metadata = json.loads(required[3].read_text(encoding="utf-8"))
    if counts.shape != (len(cells), len(genes)):
        raise ContractError(f"CANONICAL_SHAPE:{counts.shape}:{len(cells)}:{len(genes)}")
    return counts, cells, genes, metadata


def split_counts(counts: sparse.csr_matrix, specimen: str, seed: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    counts = counts.tocsr()
    rng = np.random.default_rng(int(seed))
    a = counts.copy().astype(np.int64)
    a.data = rng.binomial(a.data.astype(np.int64), 0.5).astype(np.int64)
    a.eliminate_zeros()
    b = counts.astype(np.int64) - a
    b.eliminate_zeros()
    if (a + b != counts).nnz:
        raise ContractError(f"MOLECULE_CONSERVATION:{specimen}:{seed}")
    return a.tocsr(), b.tocsr()


def aggregate_cells(
    matrices: list[sparse.csr_matrix], coords_um: np.ndarray, bin_um: int
) -> tuple[list[sparse.csr_matrix], np.ndarray, np.ndarray]:
    xy = np.asarray(coords_um, dtype=np.float64)
    origin = xy.min(axis=0)
    ij = np.floor((xy - origin[None, :]) / float(bin_um)).astype(np.int64)
    _, inverse = np.unique(ij, axis=0, return_inverse=True)
    n_bins = int(inverse.max()) + 1
    membership = sparse.coo_matrix(
        (np.ones(len(inverse), np.float64), (inverse, np.arange(len(inverse)))),
        shape=(n_bins, len(inverse)),
    ).tocsr()
    cell_count = np.asarray(membership.sum(axis=1)).reshape(-1)
    coords = np.column_stack(
        [
            np.asarray(membership @ xy[:, axis]).reshape(-1) / cell_count
            for axis in (0, 1)
        ]
    )
    return [(membership @ matrix).tocsr() for matrix in matrices], coords, cell_count


def build_lazy_graph(
    coords_um: np.ndarray,
    neighbors: int,
    max_edge_um: float,
    minimum_component_cells: int,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, dict[str, Any]]:
    xy = np.asarray(coords_um, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ContractError("COORDINATE_CONTRACT")
    if len(xy) < minimum_component_cells:
        raise ContractError(f"TOO_FEW_NODES:{len(xy)}")
    k = min(int(neighbors) + 1, len(xy))
    distance, index = cKDTree(xy).query(xy, k=k, workers=-1)
    if k == 1:
        raise ContractError("GRAPH_NO_NEIGHBORS")
    rows = np.repeat(np.arange(len(xy)), k - 1)
    cols = index[:, 1:].reshape(-1)
    dist = distance[:, 1:].reshape(-1)
    keep_edge = np.isfinite(dist) & (dist > 0) & (dist <= float(max_edge_um))
    rows, cols, dist = rows[keep_edge], cols[keep_edge], dist[keep_edge]
    if not len(rows):
        raise ContractError("GRAPH_NO_EDGES")
    sigma = float(np.median(dist))
    weights = np.exp(-0.5 * np.square(dist / max(sigma, NUMERIC_TOL)))
    adjacency = sparse.coo_matrix((weights, (rows, cols)), shape=(len(xy), len(xy))).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    n_raw, labels_raw = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels_raw, minlength=n_raw)
    retained = sizes[labels_raw] >= int(minimum_component_cells)
    if int(retained.sum()) < minimum_component_cells:
        raise ContractError("GRAPH_NO_RETAINED_COMPONENT")
    adjacency = adjacency[retained][:, retained].tocsr()
    n_components, labels = connected_components(adjacency, directed=False)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    if np.any(degree <= 0):
        raise ContractError("GRAPH_ZERO_DEGREE")
    inv = 1.0 / np.sqrt(degree)
    normalized = sparse.diags(inv) @ adjacency @ sparse.diags(inv)
    diffusion = (
        0.5 * sparse.eye(adjacency.shape[0], format="csr") + 0.5 * normalized
    ).astype(np.float32)
    nearest = distance[:, 1]
    receipt = {
        "nodes_input": int(len(xy)),
        "nodes_retained": int(retained.sum()),
        "retained_fraction": float(retained.mean()),
        "edges_retained_undirected": int(adjacency.nnz // 2),
        "components_retained": int(n_components),
        "component_sizes": np.bincount(labels, minlength=n_components).astype(int).tolist(),
        "nearest_distance_um_median": float(np.median(nearest[np.isfinite(nearest)])),
        "edge_distance_um_median": sigma,
        "max_edge_um": float(max_edge_um),
        "neighbors": int(neighbors),
        "filter_bank": "LAZY_DIFFUSION_16_8_4_2_IDENTITY",
    }
    return diffusion, retained, labels.astype(np.int32), receipt


def center_components(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    for component in np.unique(labels):
        local = labels == component
        values[local] -= values[local].mean(axis=0, keepdims=True)
    return values


def transform_rna_pair(
    a: sparse.csr_matrix,
    b: sparse.csr_matrix,
    exposure: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exposure = np.asarray(exposure, dtype=np.float64).reshape(-1, 1)
    aa = np.log1p(2.0 * a.toarray().astype(np.float64) / exposure)
    bb = np.log1p(2.0 * b.toarray().astype(np.float64) / exposure)
    aa = center_components(aa, labels)
    bb = center_components(bb, labels)
    scale = np.sqrt(0.5 * (np.square(aa).sum(axis=0) + np.square(bb).sum(axis=0)))
    eligible = scale > NUMERIC_TOL
    aa[:, eligible] /= scale[eligible]
    bb[:, eligible] /= scale[eligible]
    aa[:, ~eligible] = 0
    bb[:, ~eligible] = 0
    return aa.astype(np.float32), bb.astype(np.float32), eligible


def transform_prediction(prediction: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pp = center_components(np.asarray(prediction, dtype=np.float32), labels)
    scale = np.sqrt(np.square(pp).sum(axis=0))
    eligible = scale > NUMERIC_TOL
    pp[:, eligible] /= scale[eligible]
    pp[:, ~eligible] = 0
    return pp.astype(np.float32), eligible


def _milestone_grams_cpu(
    diffusion: sparse.csr_matrix, matrices: dict[str, np.ndarray]
) -> dict[int, dict[tuple[str, str], float]]:
    names = list(matrices)
    widths = {matrix.shape[1] for matrix in matrices.values()}
    if len(widths) != 1:
        raise ContractError(f"GENE_BLOCK_WIDTH:{widths}")
    original = {name: np.asarray(matrices[name], np.float32) for name in names}
    state = {name: value.copy() for name, value in original.items()}
    output: dict[int, dict[tuple[str, str], float]] = {}
    for step in range(17):
        if step in MILESTONES:
            gram: dict[tuple[str, str], float] = {}
            for i, left in enumerate(names):
                for right in names[i:]:
                    value = 0.5 * (
                        np.sum(original[left] * state[right], dtype=np.float64)
                        + np.sum(original[right] * state[left], dtype=np.float64)
                    )
                    gram[(left, right)] = float(value)
            output[step] = gram
        if step < 16:
            for name in names:
                state[name] = diffusion @ state[name]
    return output


def _milestone_grams_torch(
    diffusion: sparse.csr_matrix, matrices: dict[str, np.ndarray], device: str
) -> dict[int, dict[tuple[str, str], float]]:
    import torch

    names = list(matrices)
    widths = {matrix.shape[1] for matrix in matrices.values()}
    if len(widths) != 1:
        raise ContractError(f"GENE_BLOCK_WIDTH:{widths}")
    coo = diffusion.tocoo()
    index = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    value = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    operator = torch.sparse_coo_tensor(index, value, coo.shape, device=device).coalesce()
    original = {
        name: torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32, device=device)
        for name, values in matrices.items()
    }
    state = {name: values.clone() for name, values in original.items()}
    output: dict[int, dict[tuple[str, str], float]] = {}
    for step in range(17):
        if step in MILESTONES:
            gram: dict[tuple[str, str], float] = {}
            for i, left in enumerate(names):
                for right in names[i:]:
                    value_lr = torch.sum(original[left] * state[right], dtype=torch.float64)
                    value_rl = torch.sum(original[right] * state[left], dtype=torch.float64)
                    gram[(left, right)] = float((0.5 * (value_lr + value_rl)).cpu())
            output[step] = gram
        if step < 16:
            for name in names:
                state[name] = torch.sparse.mm(operator, state[name])
    del operator, original, state
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output


def lazy_band_grams(
    diffusion: sparse.csr_matrix,
    matrices: dict[str, np.ndarray],
    device: str,
) -> dict[str, dict[tuple[str, str], float]]:
    if device.startswith("cuda"):
        milestones = _milestone_grams_torch(diffusion, matrices, device)
    else:
        milestones = _milestone_grams_cpu(diffusion, matrices)
    output: dict[str, dict[tuple[str, str], float]] = {}
    for band, high, low in (
        ("B1", 16, None),
        ("B2", 8, 16),
        ("B3", 4, 8),
        ("B4", 2, 4),
        ("B5", 0, 2),
    ):
        output[band] = {}
        for pair, value in milestones[high].items():
            output[band][pair] = value if low is None else value - milestones[low][pair]
    return output


def empty_terms(pairs: Iterable[tuple[str, str]]) -> dict[str, dict[tuple[str, str], float]]:
    return {band: {pair: 0.0 for pair in pairs} for band in BANDS}


def add_terms(
    total: dict[str, dict[tuple[str, str], float]],
    addition: dict[str, dict[tuple[str, str], float]],
) -> None:
    for band in BANDS:
        for pair, value in addition[band].items():
            total[band][pair] += float(value)


def support_from_terms(terms: dict[str, dict[tuple[str, str], float]]) -> list[dict[str, float]]:
    rows = []
    for band in BANDS:
        aa = terms[band][("A", "A")]
        bb = terms[band][("B", "B")]
        ab = terms[band][("A", "B")]
        denominator = math.sqrt(max(aa, 0.0) * max(bb, 0.0))
        rows.append(
            {
                "band": band,
                "A_energy": aa,
                "B_energy": bb,
                "V_RNA": ab,
                "rna_split_agreement": ab / denominator if denominator > NUMERIC_TOL else math.nan,
            }
        )
    return rows


def model_q_from_terms(terms: dict[str, dict[tuple[str, str], float]]) -> list[dict[str, float]]:
    rows = []
    for band in BANDS:
        aa = terms[band][("A", "A")]
        bb = terms[band][("B", "B")]
        vrna = terms[band][("A", "B")]
        vmodel = terms[band][("P", "P")]
        covariance = 0.5 * (terms[band][("A", "P")] + terms[band][("B", "P")])
        denominator = math.sqrt(max(vrna, 0.0) * max(vmodel, 0.0))
        rows.append(
            {
                "band": band,
                "A_energy": aa,
                "B_energy": bb,
                "V_RNA": vrna,
                "rna_split_agreement": (
                    vrna / math.sqrt(max(aa, 0.0) * max(bb, 0.0))
                    if math.sqrt(max(aa, 0.0) * max(bb, 0.0)) > NUMERIC_TOL
                    else math.nan
                ),
                "C_model": covariance,
                "V_model": vmodel,
                "q": covariance / denominator if denominator > NUMERIC_TOL else math.nan,
            }
        )
    return rows


def combine_bands(rows: pd.DataFrame, bands: tuple[str, ...], kind: str) -> dict[str, float]:
    local = rows.set_index("band").loc[list(bands)]
    if kind == "support":
        aa, bb, ab = local[["A_energy", "B_energy", "V_RNA"]].sum(axis=0)
        denominator = math.sqrt(max(aa, 0.0) * max(bb, 0.0))
        return {
            "band": "+".join(bands),
            "A_energy": float(aa),
            "B_energy": float(bb),
            "V_RNA": float(ab),
            "rna_split_agreement": float(ab / denominator) if denominator > NUMERIC_TOL else math.nan,
        }
    vrna, covariance, vmodel = local[["V_RNA", "C_model", "V_model"]].sum(axis=0)
    denominator = math.sqrt(max(vrna, 0.0) * max(vmodel, 0.0))
    output = {
        "band": "+".join(bands),
        "V_RNA": float(vrna),
        "C_model": float(covariance),
        "V_model": float(vmodel),
        "q": float(covariance / denominator) if denominator > NUMERIC_TOL else math.nan,
    }
    if {"A_energy", "B_energy"}.issubset(local.columns):
        aa, bb = local[["A_energy", "B_energy"]].sum(axis=0)
        rna_denominator = math.sqrt(max(aa, 0.0) * max(bb, 0.0))
        output.update(
            A_energy=float(aa),
            B_energy=float(bb),
            rna_split_agreement=(
                float(vrna / rna_denominator) if rna_denominator > NUMERIC_TOL else math.nan
            ),
        )
    return output
