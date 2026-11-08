#!/usr/bin/env python3
"""HEST Ridge fitting and the neighbouring-difference expression objective."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


TASK_SAMPLES = {
    "IDC": ("TENX99", "TENX95", "NCBI785", "NCBI783"),
    "PAAD": ("TENX116", "TENX140", "TENX126"),
    "LUNG": ("TENX141", "TENX118"),
}
OFFICIAL_PCA = 256
OFFICIAL_GENES = 50
PCA_RANDOM_STATE = 1
RIDGE_SOLVER = "lsqr"
RIDGE_MAX_ITER = 1000
EDGE_RADIUS_MULTIPLIER = 1.5
PRODUCTION_RHO = (0.0, 0.5)
TRUTH_TRANSFORM_ID = "log1p_raw_integer_counts_no_library_normalization"


class ContractError(RuntimeError):
    pass


def _decode(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    return np.asarray([
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in arr
    ], dtype=object)


def _read_sparse_or_dense(group: h5py.Group | h5py.Dataset):
    if isinstance(group, h5py.Dataset):
        return np.asarray(group)
    encoding = group.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    if encoding not in {"csr_matrix", "csc_matrix"}:
        raise ContractError(f"UNSUPPORTED_H5AD_X_ENCODING:{encoding}")
    shape = tuple(int(x) for x in group.attrs["shape"])
    constructor = sparse.csr_matrix if encoding == "csr_matrix" else sparse.csc_matrix
    return constructor(
        (np.asarray(group["data"]), np.asarray(group["indices"]), np.asarray(group["indptr"])),
        shape=shape,
    )


def _read_embedding(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not path.is_file():
        raise ContractError(f"MISSING_EMBEDDING:{path}")
    with h5py.File(path, "r") as handle:
        if "embeddings" not in handle or "barcodes" not in handle:
            raise ContractError(f"EMBEDDING_SCHEMA:{path}")
        # Preserve float32: StandardScaler and randomized PCA retain this dtype,
        # so an eager float64 cast would change the fitted PCA basis.
        x = np.asarray(handle["embeddings"], dtype=np.float32)
        barcodes = _decode(handle["barcodes"][...])
        coordinate_key = "coords" if "coords" in handle else "patch_coords" if "patch_coords" in handle else None
        patch_coords = np.asarray(handle[coordinate_key]) if coordinate_key else None
    if x.ndim != 2 or len(x) != len(barcodes) or not np.isfinite(x).all():
        raise ContractError(f"EMBEDDING_CONTENT:{path}:{x.shape}:{len(barcodes)}")
    if len(set(barcodes.tolist())) != len(barcodes):
        raise ContractError(f"DUPLICATE_EMBEDDING_BARCODE:{path}")
    return x, barcodes, patch_coords


def _read_gene_manifest(path: Path, task: str) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")
    required = {"task", "gene", "gene_order"}
    if not required.issubset(frame.columns):
        raise ContractError(f"GENE_MANIFEST_COLUMNS:{sorted(required)}")
    chosen = frame.loc[frame["task"].astype(str).eq(task)].sort_values("gene_order")
    genes = chosen["gene"].astype(str).to_numpy(dtype=object)
    if len(genes) != OFFICIAL_GENES or len(set(genes.tolist())) != OFFICIAL_GENES:
        raise ContractError(f"GENE_MANIFEST_COUNT:{task}:{len(genes)}")
    if chosen["gene_order"].astype(int).tolist() != list(range(OFFICIAL_GENES)):
        raise ContractError(f"GENE_MANIFEST_ORDER:{task}")
    return genes


def _read_official_split(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise ContractError(f"OFFICIAL_SPLIT_COLUMNS:{path}")
    samples = frame["sample_id"].astype(str).tolist()
    if len(samples) != len(set(samples)):
        raise ContractError(f"DUPLICATE_OFFICIAL_SPLIT_SAMPLE:{path}")
    return samples


def _matched_truth(
    dataset_root: Path,
    task: str,
    sample: str,
    barcodes: np.ndarray,
    genes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = dataset_root / task / "adata" / f"{sample}.h5ad"
    if not path.is_file():
        raise ContractError(f"MISSING_ADATA:{path}")
    with h5py.File(path, "r") as handle:
        counts = _read_sparse_or_dense(handle["X"])
        obs = _decode(handle["obs/_index"][...])
        var = _decode(handle["var/_index"][...])
        coords = np.asarray(handle["obsm/spatial"], dtype=np.float64)
    obs_lookup = {key: index for index, key in enumerate(obs.tolist())}
    var_lookup = {key: index for index, key in enumerate(var.tolist())}
    if len(obs_lookup) != len(obs) or len(var_lookup) != len(var):
        raise ContractError(f"NONUNIQUE_ADATA_KEYS:{task}:{sample}")
    if any(key not in obs_lookup for key in barcodes) or any(key not in var_lookup for key in genes):
        raise ContractError(f"EXPLICIT_KEY_JOIN_FAILED:{task}:{sample}")
    ridx = np.asarray([obs_lookup[key] for key in barcodes], dtype=np.int64)
    gidx = np.asarray([var_lookup[key] for key in genes], dtype=np.int64)
    if sparse.issparse(counts):
        raw = counts[ridx][:, gidx].toarray()
    else:
        raw = counts[np.ix_(ridx, gidx)]
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (len(barcodes), OFFICIAL_GENES):
        raise ContractError(f"TRUTH_SHAPE:{task}:{sample}:{raw.shape}")
    if raw.min(initial=0) < 0 or not np.allclose(raw, np.rint(raw)):
        raise ContractError(f"RAW_INTEGER_COUNTS_REQUIRED:{task}:{sample}")
    aligned_coords = coords[ridx]
    if aligned_coords.shape != (len(barcodes), 2) or not np.isfinite(aligned_coords).all():
        raise ContractError(f"COORDINATE_CONTRACT:{task}:{sample}")
    return raw, np.log1p(raw), aligned_coords


def build_spatial_edges(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return deterministic immediate-neighbour edges for one specimen."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2 or not np.isfinite(coords).all():
        raise ContractError(f"EDGE_COORDINATE_CONTRACT:{coords.shape}")
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nearest = distances[:, 1]
    positive = nearest[np.isfinite(nearest) & (nearest > 0)]
    if not len(positive):
        raise ContractError("EDGE_SPACING_NOT_ESTIMABLE")
    spacing = float(np.median(positive))
    radius = EDGE_RADIUS_MULTIPLIER * spacing
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if pairs.size == 0:
        raise ContractError("NO_SPATIAL_EDGES")
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    return pairs[:, 0], pairs[:, 1], spacing


def _pearson_columns(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    p = prediction - prediction.mean(axis=0, keepdims=True)
    t = truth - truth.mean(axis=0, keepdims=True)
    numerator = np.sum(p * t, axis=0)
    denominator = np.sqrt(np.sum(p * p, axis=0) * np.sum(t * t, axis=0))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def fit_frozen_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_coords: Sequence[np.ndarray],
    specimen_lengths: Sequence[int],
    test_x: np.ndarray,
    rho: float,
    pca_components: int = OFFICIAL_PCA,
    loss_edge_transform=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit baseline and intervention through one deterministic code path."""
    if not math.isclose(rho, 0.0) and not math.isclose(rho, 0.5):
        if os.environ.get("STAUDIT_TEST_MODE") != "1":
            raise ContractError(f"UNFROZEN_RHO:{rho}")
    if len(train_coords) != len(specimen_lengths) or sum(specimen_lengths) != len(train_x):
        raise ContractError("TRAIN_SPECIMEN_BOUNDARY_CONTRACT")
    if pca_components > min(train_x.shape):
        raise ContractError(f"PCA_COMPONENT_CONTRACT:{pca_components}:{train_x.shape}")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = scaler.transform(test_x)
    pca = PCA(n_components=pca_components, random_state=PCA_RANDOM_STATE)
    reduced = pca.fit_transform(train_scaled)
    test_reduced = pca.transform(test_scaled)

    edge_left: list[np.ndarray] = []
    edge_right: list[np.ndarray] = []
    graph_rows: list[dict[str, Any]] = []
    offset = 0
    for index, (coords, length) in enumerate(zip(train_coords, specimen_lengths)):
        if len(coords) != length:
            raise ContractError(f"TRAIN_COORDINATE_LENGTH:{index}:{len(coords)}:{length}")
        left, right, spacing = build_spatial_edges(coords)
        if loss_edge_transform is not None:
            left, right = loss_edge_transform(index, coords, left, right, spacing)
        edge_left.append(left + offset)
        edge_right.append(right + offset)
        graph_rows.append({"train_specimen_index": index, "spots": length, "edges": len(left), "spacing": spacing})
        offset += length
    left = np.concatenate(edge_left)
    right = np.concatenate(edge_right)
    n_spots = len(reduced)
    n_edges = len(left)
    if n_edges == 0:
        raise ContractError("NO_TRAINING_EDGES")

    if math.isclose(rho, 0.0):
        design = reduced
        response = train_y
    else:
        spot_weight = math.sqrt(1.0 - rho)
        edge_weight = math.sqrt(rho * n_spots / n_edges)
        design = np.vstack((spot_weight * reduced, edge_weight * (reduced[left] - reduced[right])))
        response = np.vstack((spot_weight * train_y, edge_weight * (train_y[left] - train_y[right])))

    alpha = 100.0 / (pca_components * train_y.shape[1])
    model = Ridge(
        alpha=alpha,
        solver=RIDGE_SOLVER,
        fit_intercept=False,
        max_iter=RIDGE_MAX_ITER,
    )
    model.fit(design, response)
    prediction = model.predict(test_reduced)
    metadata = {
        "rho": float(rho),
        "ordinary_rows": int(n_spots),
        "edge_rows": int(n_edges),
        "solver_rows": int(len(design)),
        "pca_components": int(pca_components),
        "pca_random_state": PCA_RANDOM_STATE,
        "ridge_alpha": float(alpha),
        "ridge_solver": RIDGE_SOLVER,
        "ridge_fit_intercept": False,
        "ridge_max_iter": RIDGE_MAX_ITER,
        "edge_radius_multiplier": EDGE_RADIUS_MULTIPLIER,
        "graph_rows": graph_rows,
    }
    return prediction, metadata


def fit_fold(args: argparse.Namespace) -> dict[str, Any]:
    samples = TASK_SAMPLES[args.task]
    if (args.fold is None) == (args.test_specimen is None):
        raise ContractError("EXACTLY_ONE_OF_FOLD_OR_TEST_SPECIMEN_REQUIRED")
    if args.test_specimen is not None and args.test_specimen not in samples:
        raise ContractError(f"TEST_SPECIMEN_NOT_IN_TASK:{args.task}:{args.test_specimen}")
    fold = args.fold if args.fold is not None else samples.index(args.test_specimen)
    if fold < 0 or fold >= len(samples):
        raise ContractError(f"FOLD_RANGE:{args.task}:{fold}")
    if args.pca_components != OFFICIAL_PCA and os.environ.get("STAUDIT_TEST_MODE") != "1":
        raise ContractError("PRODUCTION_PCA_MUST_EQUAL_256")
    genes = _read_gene_manifest(args.gene_manifest, args.task)
    embedding_map: dict[tuple[str, str], Path] = {}
    if args.embedding_manifest is not None:
        manifest = pd.read_csv(args.embedding_manifest, sep="\t")
        required = {"task", "specimen", "encoder", "embedding_path"}
        if not required.issubset(manifest.columns):
            raise ContractError(f"EMBEDDING_MANIFEST_COLUMNS:{sorted(required)}")
        selected = manifest.loc[manifest["encoder"].astype(str).eq(args.encoder)].copy()
        if selected.duplicated(["task", "specimen"]).any():
            raise ContractError("DUPLICATE_EMBEDDING_MANIFEST_KEY")
        embedding_map = {
            (str(row.task), str(row.specimen)): Path(str(row.embedding_path))
            for row in selected.itertuples(index=False)
        }

    def embedding_for(sample: str) -> Path:
        if args.embedding_manifest is not None:
            key = (args.task, sample)
            if key not in embedding_map:
                raise ContractError(f"MISSING_EMBEDDING_MANIFEST_KEY:{args.task}:{sample}:{args.encoder}")
            return embedding_map[key]
        return args.embedding_root / args.task / f"{sample}.{args.encoder}.h5"

    test_sample = samples[fold]
    split_root = args.dataset_root / args.task / "splits"
    train_samples = _read_official_split(split_root / f"train_{fold}.csv")
    test_samples = _read_official_split(split_root / f"test_{fold}.csv")
    if test_samples != [test_sample]:
        raise ContractError(f"OFFICIAL_TEST_SPLIT:{args.task}:{fold}:{test_samples}:{test_sample}")
    if test_sample in train_samples or set(train_samples) | {test_sample} != set(samples):
        raise ContractError(f"OFFICIAL_TRAIN_SPLIT:{args.task}:{fold}:{train_samples}")
    train_xs: list[np.ndarray] = []
    train_ys: list[np.ndarray] = []
    train_coords: list[np.ndarray] = []
    lengths: list[int] = []
    for sample in train_samples:
        embedding_path = embedding_for(sample)
        x, barcodes, _ = _read_embedding(embedding_path)
        _, y, coords = _matched_truth(args.dataset_root, args.task, sample, barcodes, genes)
        train_xs.append(x)
        train_ys.append(y)
        train_coords.append(coords)
        lengths.append(len(x))
    test_embedding = embedding_for(test_sample)
    test_x, test_barcodes, patch_coords = _read_embedding(test_embedding)
    raw, test_y, test_coords = _matched_truth(args.dataset_root, args.task, test_sample, test_barcodes, genes)
    prediction, metadata = fit_frozen_ridge(
        np.concatenate(train_xs),
        np.concatenate(train_ys),
        train_coords,
        lengths,
        test_x,
        args.rho,
        args.pca_components,
    )
    gene_pearson = _pearson_columns(prediction, test_y)
    tag = f"rho{args.rho:.1f}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"{args.task}.fold{fold}.{test_sample}.{args.encoder}.{tag}.predictions.h5"
    tmp = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    with h5py.File(tmp, "w") as handle:
        strings = h5py.string_dtype("utf-8")
        handle.create_dataset("barcodes", data=test_barcodes.astype(object), dtype=strings)
        handle.create_dataset("genes", data=genes.astype(object), dtype=strings)
        if patch_coords is not None:
            handle.create_dataset("patch_coords", data=patch_coords)
        handle.create_dataset("spatial_coords", data=test_coords)
        handle.create_dataset("raw_counts", data=np.rint(raw).astype(np.uint32), compression="lzf")
        handle.create_dataset("transformed_truth", data=test_y, compression="lzf")
        handle.create_dataset("prediction", data=prediction, compression="lzf")
        handle.create_dataset("gene_pearson", data=gene_pearson)
        handle.attrs.update({
            "status": "PASS",
            "experiment": "G0-054",
            "task": args.task,
            "fold": fold,
            "sample_id": test_sample,
            "encoder": args.encoder,
            "rho": args.rho,
            "truth_transform_id": TRUTH_TRANSFORM_ID,
            "pca_components": args.pca_components,
            "pca_random_state": PCA_RANDOM_STATE,
            "ridge_alpha": metadata["ridge_alpha"],
            "ridge_solver": RIDGE_SOLVER,
            "ridge_fit_intercept": False,
            "ridge_max_iter": RIDGE_MAX_ITER,
            "edge_rule": "radius_1.5x_median_positive_nearest_neighbor_within_specimen",
        })
        handle.flush()
    os.replace(tmp, output)
    score = {
        "status": "PASS",
        "experiment": "G0-054",
        "task": args.task,
        "fold": fold,
        "test_specimen": test_sample,
        "train_specimens": train_samples,
        "train_specimen_order_source": str(split_root / f"train_{fold}.csv"),
        "encoder": args.encoder,
        "rho": args.rho,
        "n_train_spots": int(sum(lengths)),
        "n_test_spots": int(len(test_x)),
        "genes": OFFICIAL_GENES,
        "ordinary_gene_wise_pcc": float(np.mean(gene_pearson)),
        "gene_pcc_std": float(np.std(gene_pearson)),
        "prediction_path": str(output),
        "truth_transform_id": TRUTH_TRANSFORM_ID,
        **metadata,
    }
    score_path = output.with_suffix(".score.json")
    score_path.write_text(json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-fold")
    fit.add_argument("--dataset-root", type=Path, required=True)
    embeddings = fit.add_mutually_exclusive_group(required=True)
    embeddings.add_argument("--embedding-root", type=Path)
    embeddings.add_argument("--embedding-manifest", type=Path)
    fit.add_argument("--gene-manifest", type=Path, required=True)
    fit.add_argument("--output-root", type=Path, required=True)
    fit.add_argument("--task", choices=tuple(TASK_SAMPLES), required=True)
    fit.add_argument("--fold", type=int)
    fit.add_argument("--test-specimen", choices=tuple(sample for values in TASK_SAMPLES.values() for sample in values))
    fit.add_argument("--encoder", default="phikon_v2")
    fit.add_argument("--rho", type=float, choices=PRODUCTION_RHO, required=True)
    fit.add_argument("--pca-components", type=int, default=OFFICIAL_PCA, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit-fold":
        result = fit_fold(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
