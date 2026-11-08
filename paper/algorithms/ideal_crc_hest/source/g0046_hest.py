#!/usr/bin/env python3
"""Official HEST-Benchmark ResNet50 extraction and fold scoring for G0-046.

The executable deliberately separates the official ordinary-PCC reproduction
from spectrum authorization.  A passing PCC gate does not authorize q/spectrum
analysis until a HEST-specific transform-by-sampling calibration passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np


HEST_COMMIT = "3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3"
TRIDENT_COMMIT = "78d7002f1f8087edb8f18a067be92897b4ac0e8e"
DATASET_REVISION = "b039b5965ed7092f280d4f336da7972ef9a10e1a"
RESNET_REPO = "timm/resnet50.tv_in1k"
RESNET_REVISION = "78f3ecfdb38e06d9b8397f662e7ab8fee96026fa"
HIBOU_REPO = "histai/hibou-L"
HIBOU_REVISION = "31a3b9fee"
TASK_SAMPLES = {
    "IDC": ("TENX99", "TENX95", "NCBI785", "NCBI783"),
    "PAAD": ("TENX116", "TENX140", "TENX126"),
    "LUNG": ("TENX141", "TENX118"),
}
OFFICIAL_PCC = {
    "resnet50": {"IDC": 0.4739, "PAAD": 0.3880, "LUNG": 0.4949},
    "hibou_large": {"IDC": 0.5701, "PAAD": 0.4674, "LUNG": 0.5762},
}
PCC_TOLERANCE = 0.03
OFFICIAL_GENES = 50
OFFICIAL_PCA = 256
OFFICIAL_ALPHA = 100.0 / (OFFICIAL_PCA * OFFICIAL_GENES)


class ContractError(RuntimeError):
    """An official-data or provenance contract failed."""


def _decode(values: Any) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    return np.asarray([
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
        for item in array
    ], dtype=object)


def _sha256(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_tsv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_matrix(group_or_dataset: Any) -> Any:
    if isinstance(group_or_dataset, h5py.Dataset):
        return np.asarray(group_or_dataset[...])
    group = group_or_dataset
    encoding = group.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding not in {"csr_matrix", "csc_matrix"}:
        raise ContractError(f"UNSUPPORTED_H5AD_X_ENCODING:{encoding!r}")
    from scipy import sparse

    shape = tuple(int(x) for x in group.attrs["shape"])
    constructor = sparse.csr_matrix if encoding == "csr_matrix" else sparse.csc_matrix
    return constructor(
        (np.asarray(group["data"]), np.asarray(group["indices"]), np.asarray(group["indptr"])),
        shape=shape,
    )


def read_h5ad(path: Path) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Read the small benchmark expression matrix without requiring anndata."""
    with h5py.File(path, "r") as handle:
        required = ("X", "obs/_index", "var/_index", "obsm/spatial")
        missing = [key for key in required if key not in handle]
        if missing:
            raise ContractError(f"H5AD_MISSING_KEYS:{path}:{','.join(missing)}")
        matrix = _read_matrix(handle["X"])
        barcodes = _decode(handle["obs/_index"][...])
        genes = _decode(handle["var/_index"][...])
        spatial = np.asarray(handle["obsm/spatial"][...], dtype=np.float64)
    if matrix.shape != (len(barcodes), len(genes)):
        raise ContractError(f"H5AD_SHAPE_MISMATCH:{path}:{matrix.shape}:{len(barcodes)}:{len(genes)}")
    if spatial.shape != (len(barcodes), 2):
        raise ContractError(f"H5AD_SPATIAL_SHAPE:{path}:{spatial.shape}")
    return matrix, barcodes, genes, spatial


def inspect_h5ad(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, str]:
    """Validate raw-count storage without forcing the whole matrix through CCI."""
    with h5py.File(path, "r") as handle:
        required = ("X", "obs/_index", "var/_index", "obsm/spatial")
        missing = [key for key in required if key not in handle]
        if missing:
            raise ContractError(f"H5AD_MISSING_KEYS:{path}:{','.join(missing)}")
        barcodes = _decode(handle["obs/_index"][...])
        genes = _decode(handle["var/_index"][...])
        spatial = np.asarray(handle["obsm/spatial"][...], dtype=np.float64)
        x = handle["X"]
        if isinstance(x, h5py.Dataset):
            shape = x.shape
            dtype = x.dtype
            values = x if dtype.kind not in "u" else None
        else:
            shape = tuple(int(v) for v in x.attrs["shape"])
            data = x["data"]
            dtype = data.dtype
            values = data if dtype.kind not in "u" else None
        if tuple(shape) != (len(barcodes), len(genes)):
            raise ContractError(f"H5AD_SHAPE_MISMATCH:{path}:{shape}:{len(barcodes)}:{len(genes)}")
        if spatial.shape != (len(barcodes), 2):
            raise ContractError(f"H5AD_SPATIAL_SHAPE:{path}:{spatial.shape}")
        if dtype.kind not in "uif":
            raise ContractError(f"RAW_COUNT_DTYPE:{path}:{dtype}")
        # The official files are uint16, so this takes no matrix read.  Non-unsigned
        # variants are scanned because dtype alone cannot prove nonnegative integer values.
        if values is not None:
            array = np.asarray(values[...])
            if not np.isfinite(array).all() or np.min(array) < 0 or not np.allclose(array, np.rint(array)):
                raise ContractError(f"RAW_INTEGER_COUNTS_REQUIRED:{path}")
    return len(barcodes), barcodes, genes, spatial, str(dtype)


def read_patch_keys(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], np.dtype]:
    with h5py.File(path, "r") as handle:
        required = ("img", "coords", "barcode")
        missing = [key for key in required if key not in handle]
        if missing:
            raise ContractError(f"PATCH_MISSING_KEYS:{path}:{','.join(missing)}")
        barcodes = _decode(handle["barcode"][...])
        coords = np.asarray(handle["coords"][...])
        image_shape = tuple(int(x) for x in handle["img"].shape)
        image_dtype = handle["img"].dtype
    return barcodes, coords, image_shape, image_dtype


def read_genes(task_root: Path) -> list[str]:
    with (task_root / "var_50genes.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    genes = [str(x) for x in payload["genes"]]
    if len(genes) != OFFICIAL_GENES or len(set(genes)) != OFFICIAL_GENES:
        raise ContractError(f"OFFICIAL_GENE_CONTRACT:{task_root}:{len(genes)}:{len(set(genes))}")
    return genes


def _read_split(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [str(row["sample_id"]) for row in csv.DictReader(handle)]


def folds_for_task(dataset_root: Path, task: str) -> list[dict[str, Any]]:
    task_root = dataset_root / task
    folds: list[dict[str, Any]] = []
    for fold, expected_test in enumerate(TASK_SAMPLES[task]):
        train = _read_split(task_root / "splits" / f"train_{fold}.csv")
        test = _read_split(task_root / "splits" / f"test_{fold}.csv")
        if test != [expected_test]:
            raise ContractError(f"OFFICIAL_TEST_SPLIT:{task}:{fold}:{test}:{expected_test}")
        if expected_test in train or set(train) | set(test) != set(TASK_SAMPLES[task]):
            raise ContractError(f"OFFICIAL_TRAIN_SPLIT:{task}:{fold}:{train}")
        folds.append({"task": task, "fold": fold, "train": train, "test": expected_test})
    return folds


def contract_check(dataset_root: Path, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    total_patch = 0
    total_h5ad = 0
    for task, samples in TASK_SAMPLES.items():
        task_root = dataset_root / task
        genes = read_genes(task_root)
        folds.extend(folds_for_task(dataset_root, task))
        for sample in samples:
            patch_path = task_root / "patches" / f"{sample}.h5"
            adata_path = task_root / "adata" / f"{sample}.h5ad"
            patch_barcodes, patch_coords, image_shape, image_dtype = read_patch_keys(patch_path)
            h5ad_rows, adata_barcodes, adata_genes, spatial, count_dtype = inspect_h5ad(adata_path)
            if image_shape != (len(patch_barcodes), 224, 224, 3) or image_dtype != np.dtype("uint8"):
                raise ContractError(f"PATCH_IMAGE_CONTRACT:{task}:{sample}:{image_shape}:{image_dtype}")
            if patch_coords.shape != (len(patch_barcodes), 2) or not np.isfinite(patch_coords).all():
                raise ContractError(f"PATCH_COORD_CONTRACT:{task}:{sample}:{patch_coords.shape}")
            if len(set(patch_barcodes.tolist())) != len(patch_barcodes):
                raise ContractError(f"DUPLICATE_PATCH_BARCODE:{task}:{sample}")
            if len(set(adata_barcodes.tolist())) != len(adata_barcodes):
                raise ContractError(f"DUPLICATE_H5AD_BARCODE:{task}:{sample}")
            lookup = {barcode: idx for idx, barcode in enumerate(adata_barcodes.tolist())}
            missing_patch = [barcode for barcode in patch_barcodes if barcode not in lookup]
            if missing_patch:
                raise ContractError(f"PATCH_BARCODE_NOT_IN_H5AD:{task}:{sample}:{len(missing_patch)}")
            if not set(genes).issubset(set(adata_genes.tolist())):
                missing_genes = sorted(set(genes) - set(adata_genes.tolist()))
                raise ContractError(f"MISSING_OFFICIAL_GENES:{task}:{sample}:{','.join(missing_genes)}")
            subset_spatial = spatial[[lookup[x] for x in patch_barcodes]]
            if not np.isfinite(subset_spatial).all() or np.ptp(subset_spatial, axis=0).min() <= 0:
                raise ContractError(f"SPATIAL_COORD_CONTRACT:{task}:{sample}")
            total_patch += len(patch_barcodes)
            total_h5ad += h5ad_rows
            rows.append({
                "task": task,
                "sample_id": sample,
                "patch_rows": len(patch_barcodes),
                "h5ad_rows": len(adata_barcodes),
                "patch_is_unique_h5ad_subset": "PASS",
                "genes": len(genes),
                "raw_integer_counts": "PASS",
                "count_dtype": count_dtype,
                "finite_nondegenerate_coordinates": "PASS",
                "patch_h5": str(patch_path),
                "h5ad": str(adata_path),
            })
    _atomic_tsv(out_dir / "HEST_INPUT_CONTRACT.tsv", rows, list(rows[0]))
    _atomic_tsv(
        out_dir / "HEST_OFFICIAL_FOLDS.tsv",
        [{**x, "train": ",".join(x["train"])} for x in folds],
        ("task", "fold", "train", "test"),
    )
    result = {
        "status": "PASS",
        "dataset_root": str(dataset_root),
        "dataset_revision": DATASET_REVISION,
        "tasks": list(TASK_SAMPLES),
        "specimens": len(rows),
        "patch_common_support_rows": total_patch,
        "whole_h5ad_rows": total_h5ad,
        "barcode_contract": "each unique patch barcode maps to exactly one h5ad row; whole-file row counts need not match",
        "official_genes_per_task": OFFICIAL_GENES,
        "folds": len(folds),
        "hest_commit": HEST_COMMIT,
        "trident_commit": TRIDENT_COMMIT,
        "resnet50_official_feature_stage": "out_indices=[3] (layer3), adaptive-average-pooled",
        "resnet50_embedding_dimensions": 1024,
        "resnet50_source": f"https://github.com/mahmoodlab/TRIDENT/blob/{TRIDENT_COMMIT}/trident/patch_encoder_models/load.py#L551-L611",
    }
    _atomic_json(out_dir / "HEST_INPUT_CONTRACT.json", result)
    return result


class PatchDataset:
    """Worker-safe lazy HDF5 patch reader."""

    def __init__(self, path: Path):
        self.path = str(path)
        with h5py.File(self.path, "r") as handle:
            self.length = int(handle["img"].shape[0])
        self._handle = None

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "length": self.length, "_handle": None}

    def __getitem__(self, index: int):
        import torch

        if self._handle is None:
            self._handle = h5py.File(self.path, "r", swmr=True)
        image = np.asarray(self._handle["img"][index], dtype=np.uint8)
        tensor = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        return (tensor - mean) / std


def _load_resnet50(cache_dir: Path | None, local_files_only: bool):
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weight_path = hf_hub_download(
        repo_id=RESNET_REPO,
        filename="model.safetensors",
        revision=RESNET_REVISION,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
    )
    model = timm.create_model(
        "resnet50",
        pretrained=False,
        features_only=True,
        out_indices=[3],
        num_classes=0,
    )
    state_dict = load_file(weight_path)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    # The frozen official feature extractor stops after layer3
    # (``features_only=True, out_indices=[3]``).  A full ImageNet checkpoint
    # therefore legitimately contains unused layer4 and classifier tensors.
    unexpected = [
        key for key in incompatible.unexpected_keys
        if not key.startswith(("layer4.", "fc."))
    ]
    if missing or unexpected:
        raise ContractError(f"RESNET_WEIGHT_CONTRACT:missing={missing}:unexpected={unexpected}")
    return model, Path(weight_path)


def embed_sample(
    dataset_root: Path,
    task: str,
    sample: str,
    output: Path,
    batch_size: int,
    workers: int,
    device: str,
    cache_dir: Path | None,
    local_files_only: bool,
) -> dict[str, Any]:
    if sample not in TASK_SAMPLES.get(task, ()):
        raise ContractError(f"UNFROZEN_TASK_SAMPLE:{task}:{sample}")
    import torch
    from torch.utils.data import DataLoader

    patch_path = dataset_root / task / "patches" / f"{sample}.h5"
    patch_barcodes, patch_coords, image_shape, _ = read_patch_keys(patch_path)
    if output.exists():
        with h5py.File(output, "r") as handle:
            if (
                handle.attrs.get("status", "") == "PASS"
                and handle.attrs.get("sample_id", "") == sample
                and int(handle["embeddings"].shape[0]) == len(patch_barcodes)
            ):
                return {"status": "PASS_REUSED", "output": str(output), "rows": len(patch_barcodes)}
        raise ContractError(f"EXISTING_EMBEDDING_CONTRACT_MISMATCH:{output}")
    model, weight_path = _load_resnet50(cache_dir, local_files_only)
    model.eval().to(device)
    loader = DataLoader(
        PatchDataset(patch_path),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    try:
        with h5py.File(tmp, "w") as handle:
            embeddings = None
            offset = 0
            with torch.inference_mode():
                for batch in loader:
                    result = model(batch.to(device, non_blocking=True))
                    if isinstance(result, (tuple, list)):
                        result = result[-1]
                    if result.ndim == 4:
                        result = result.mean(dim=(-2, -1))
                    values = result.detach().cpu().numpy().astype(np.float32, copy=False)
                    if values.ndim != 2 or values.shape[1] != 1024:
                        raise ContractError(f"OFFICIAL_RESNET_EMBEDDING_DIM:{values.shape}")
                    if embeddings is None:
                        embeddings = handle.create_dataset(
                            "embeddings",
                            shape=(len(patch_barcodes), values.shape[1]),
                            dtype="float32",
                            chunks=(min(1024, len(patch_barcodes)), values.shape[1]),
                            compression="lzf",
                        )
                    embeddings[offset : offset + len(values)] = values
                    offset += len(values)
            if embeddings is None or offset != len(patch_barcodes):
                raise ContractError(f"INCOMPLETE_EMBEDDING:{task}:{sample}:{offset}:{len(patch_barcodes)}")
            strings = h5py.string_dtype("utf-8")
            handle.create_dataset("barcodes", data=patch_barcodes.astype(object), dtype=strings)
            handle.create_dataset("coords", data=patch_coords)
            handle.attrs.update({
                "status": "PASS",
                "task": task,
                "sample_id": sample,
                "encoder": "resnet50",
                "encoder_repo": RESNET_REPO,
                "encoder_revision": RESNET_REVISION,
                "weights_path": str(weight_path),
                "weights_sha256": _sha256(weight_path),
                "hest_commit": HEST_COMMIT,
                "trident_commit": TRIDENT_COMMIT,
                "dataset_revision": DATASET_REVISION,
                "patch_rows": len(patch_barcodes),
                "patch_shape": json.dumps(image_shape),
            })
            handle.flush()
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"status": "PASS", "output": str(output), "rows": len(patch_barcodes)}


def _read_embedding(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("status", "") != "PASS":
            raise ContractError(f"EMBEDDING_NOT_PASS:{path}")
        embeddings = np.asarray(handle["embeddings"], dtype=np.float32)
        barcodes = _decode(handle["barcodes"][...])
        coords = np.asarray(handle["coords"])
    if len(embeddings) != len(barcodes) or coords.shape != (len(barcodes), 2):
        raise ContractError(f"EMBEDDING_KEY_SHAPE:{path}")
    return embeddings, barcodes, coords


def _matched_truth(dataset_root: Path, task: str, sample: str, barcodes: np.ndarray):
    counts, all_barcodes, all_genes, spatial = read_h5ad(dataset_root / task / "adata" / f"{sample}.h5ad")
    genes = read_genes(dataset_root / task)
    b_lookup = {x: i for i, x in enumerate(all_barcodes.tolist())}
    g_lookup = {x: i for i, x in enumerate(all_genes.tolist())}
    if len(b_lookup) != len(all_barcodes) or any(x not in b_lookup for x in barcodes):
        raise ContractError(f"TRUTH_BARCODE_JOIN:{task}:{sample}")
    if any(x not in g_lookup for x in genes):
        raise ContractError(f"TRUTH_GENE_JOIN:{task}:{sample}")
    ridx = np.asarray([b_lookup[x] for x in barcodes], dtype=np.int64)
    gidx = np.asarray([g_lookup[x] for x in genes], dtype=np.int64)
    raw = counts[np.ix_(ridx, gidx)]
    from scipy import sparse

    if sparse.issparse(raw):
        raw = raw.toarray()
    raw = np.asarray(raw)
    if np.min(raw) < 0 or not np.allclose(raw, np.rint(raw)):
        raise ContractError(f"RAW_INTEGER_COUNTS_REQUIRED:{task}:{sample}")
    transformed = np.log1p(raw.astype(np.float64))
    return raw, transformed, spatial[ridx], np.asarray(genes, dtype=object)


def _pearson_columns(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    p = prediction - prediction.mean(axis=0, keepdims=True)
    t = truth - truth.mean(axis=0, keepdims=True)
    numerator = np.sum(p * t, axis=0)
    denominator = np.sqrt(np.sum(p * p, axis=0) * np.sum(t * t, axis=0))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64), where=denominator > 0)


def fit_official_fold(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    pca_components: int = OFFICIAL_PCA,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if pca_components > min(train_x.shape[0], train_x.shape[1]):
        raise ContractError(f"PCA_COMPONENT_CONTRACT:{pca_components}:{train_x.shape}")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = scaler.transform(test_x)
    pca = PCA(n_components=pca_components, random_state=1)
    train_reduced = pca.fit_transform(train_scaled)
    test_reduced = pca.transform(test_scaled)
    alpha = 100.0 / (pca_components * train_y.shape[1])
    regressor = Ridge(alpha=alpha, solver="lsqr", fit_intercept=False, max_iter=1000)
    regressor.fit(train_reduced, train_y)
    prediction = regressor.predict(test_reduced)
    correlations = _pearson_columns(prediction, test_y)
    return prediction, correlations


def score_fold(
    dataset_root: Path,
    embedding_root: Path,
    output_root: Path,
    task: str,
    fold: int,
    encoder: str = "resnet50",
    pca_components: int = OFFICIAL_PCA,
) -> dict[str, Any]:
    if encoder != "resnet50":
        raise ContractError(f"ENCODER_NOT_IMPLEMENTED:{encoder}")
    folds = folds_for_task(dataset_root, task)
    if fold < 0 or fold >= len(folds):
        raise ContractError(f"FOLD_RANGE:{task}:{fold}")
    split = folds[fold]
    train_xs: list[np.ndarray] = []
    train_ys: list[np.ndarray] = []
    for sample in split["train"]:
        x, barcodes, _ = _read_embedding(embedding_root / task / f"{sample}.{encoder}.h5")
        _, y, _, _ = _matched_truth(dataset_root, task, sample, barcodes)
        train_xs.append(x)
        train_ys.append(y)
    test_sample = split["test"]
    test_x, barcodes, patch_coords = _read_embedding(embedding_root / task / f"{test_sample}.{encoder}.h5")
    raw, test_y, spatial_coords, genes = _matched_truth(dataset_root, task, test_sample, barcodes)
    train_x = np.concatenate(train_xs, axis=0)
    train_y = np.concatenate(train_ys, axis=0)
    prediction, correlations = fit_official_fold(train_x, train_y, test_x, test_y, pca_components)
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / f"{task}.fold{fold}.{test_sample}.{encoder}.predictions.h5"
    tmp = prediction_path.with_suffix(prediction_path.suffix + f".tmp.{os.getpid()}")
    try:
        with h5py.File(tmp, "w") as handle:
            strings = h5py.string_dtype("utf-8")
            handle.create_dataset("barcodes", data=barcodes.astype(object), dtype=strings)
            handle.create_dataset("genes", data=genes.astype(object), dtype=strings)
            handle.create_dataset("patch_coords", data=patch_coords)
            handle.create_dataset("spatial_coords", data=spatial_coords)
            handle.create_dataset("raw_counts", data=np.rint(raw).astype(np.uint32), compression="lzf")
            handle.create_dataset("transformed_truth", data=test_y, compression="lzf")
            handle.create_dataset("prediction", data=prediction, compression="lzf")
            handle.create_dataset("gene_pearson", data=correlations)
            handle.attrs.update({
                "status": "PASS",
                "task": task,
                "fold": fold,
                "sample_id": test_sample,
                "encoder": encoder,
                "truth_transform_id": "log1p_raw_integer_counts_no_library_normalization",
                "pca_components": pca_components,
                "pca_random_state": 1,
                "ridge_alpha": 100.0 / (pca_components * len(genes)),
                "ridge_solver": "lsqr",
                "ridge_fit_intercept": False,
                "ridge_max_iter": 1000,
                "dataset_revision": DATASET_REVISION,
            })
            handle.flush()
        os.replace(tmp, prediction_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    reference = OFFICIAL_PCC[encoder][task]
    score = float(np.mean(correlations))
    payload = {
        "status": "PASS" if abs(score - reference) <= PCC_TOLERANCE else "OFFICIAL_PCC_MISMATCH",
        "task": task,
        "fold": fold,
        "test_specimen": test_sample,
        "encoder": encoder,
        "n_train_spots": int(len(train_x)),
        "n_test_spots": int(len(test_x)),
        "genes": int(len(genes)),
        "ordinary_gene_wise_pcc": score,
        "gene_pcc_std": float(np.std(correlations)),
        "official_task_reference": reference,
        "note": "fold is reported against the task reference for visibility; the formal tolerance gate is applied after equal-specimen task aggregation",
        "prediction_path": str(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "truth_transform_id": "log1p_raw_integer_counts_no_library_normalization",
    }
    result_path = output_root / f"{task}.fold{fold}.{test_sample}.{encoder}.score.json"
    _atomic_json(result_path, payload)
    return payload


def aggregate_scores(score_root: Path, output: Path, encoder: str = "resnet50") -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for task, samples in TASK_SAMPLES.items():
        scores = []
        for fold, sample in enumerate(samples):
            path = score_root / f"{task}.fold{fold}.{sample}.{encoder}.score.json"
            if not path.is_file():
                raise ContractError(f"MISSING_FOLD_SCORE:{path}")
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            scores.append(float(payload["ordinary_gene_wise_pcc"]))
            rows.append({"task": task, "fold": fold, "specimen": sample, "pcc": scores[-1], "score_json": str(path)})
        observed = float(np.mean(scores))
        reference = OFFICIAL_PCC[encoder][task]
        difference = observed - reference
        status = "PASS" if abs(difference) <= PCC_TOLERANCE else "FAIL"
        tasks[task] = {
            "status": status,
            "observed_equal_specimen_pcc": observed,
            "official_reference_pcc": reference,
            "absolute_difference": abs(difference),
            "specimens": len(scores),
        }
    pcc_status = "PASS" if all(item["status"] == "PASS" for item in tasks.values()) else "FAIL"
    payload = {
        "status": pcc_status,
        "gate": "OFFICIAL_HEST_ORDINARY_PCC_REPRODUCTION",
        "encoder": encoder,
        "absolute_tolerance": PCC_TOLERANCE,
        "tasks": tasks,
        "spectrum_authorization": "NOT_CALIBRATED",
        "spectrum_reason": "HEST log1p transform requires its own transform-by-sampling equal-difficulty calibration; ordinary PCC reproduction alone cannot authorize q/spectrum",
    }
    _atomic_json(output, payload)
    _atomic_tsv(output.with_suffix(".folds.tsv"), rows, ("task", "fold", "specimen", "pcc", "score_json"))
    calibration = {
        "status": "NOT_CALIBRATED",
        "gate": "HEST_TRANSFORM_BY_SAMPLING_CALIBRATION",
        "ordinary_pcc_gate": pcc_status,
        "truth_transform_id": "log1p_raw_integer_counts_no_library_normalization",
        "required_before_spectrum": True,
        "reason": "G0-046 WP0 found transform-by-sampling behavior is not transferable automatically; HEST needs cohort-specific calibration",
    }
    _atomic_json(output.parent / "HEST_SPECTRUM_CALIBRATION_GATE.json", calibration)
    return payload


def write_hibou_receipt(output: Path) -> dict[str, Any]:
    payload = {
        "status": "ACCESS_BLOCKED",
        "encoder": "hibou_large",
        "repository": HIBOU_REPO,
        "requested_revision": HIBOU_REVISION,
        "observed_access": "HTTP_403_GATED_NO_LOCAL_CACHE",
        "action": "DO_NOT_SUBSTITUTE_MODEL__DO_NOT_WAIT_FOR_MANUAL_ACCESS",
        "scientific_interpretation": "NOT_ESTIMABLE_ENGINEERING_ACCESS_CONTRACT; not a model failure",
    }
    _atomic_json(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    contract = sub.add_parser("contract")
    contract.add_argument("--dataset-root", type=Path, required=True)
    contract.add_argument("--out-dir", type=Path, required=True)
    hibou = sub.add_parser("hibou-receipt")
    hibou.add_argument("--out", type=Path, required=True)
    embed = sub.add_parser("embed")
    embed.add_argument("--dataset-root", type=Path, required=True)
    embed.add_argument("--task", choices=tuple(TASK_SAMPLES), required=True)
    embed.add_argument("--sample", required=True)
    embed.add_argument("--out", type=Path, required=True)
    embed.add_argument("--batch-size", type=int, default=256)
    embed.add_argument("--workers", type=int, default=8)
    embed.add_argument("--device", default="cuda")
    embed.add_argument("--hf-cache", type=Path)
    embed.add_argument("--local-files-only", action="store_true")
    score = sub.add_parser("score-fold")
    score.add_argument("--dataset-root", type=Path, required=True)
    score.add_argument("--embedding-root", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    score.add_argument("--task", choices=tuple(TASK_SAMPLES), required=True)
    score.add_argument("--fold", type=int, required=True)
    score.add_argument("--encoder", default="resnet50", choices=("resnet50",))
    score.add_argument("--pca-components", type=int, default=OFFICIAL_PCA, help=argparse.SUPPRESS)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--score-root", type=Path, required=True)
    aggregate.add_argument("--out", type=Path, required=True)
    aggregate.add_argument("--encoder", default="resnet50", choices=("resnet50",))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "contract":
        result = contract_check(args.dataset_root, args.out_dir)
    elif args.command == "hibou-receipt":
        result = write_hibou_receipt(args.out)
    elif args.command == "embed":
        result = embed_sample(
            args.dataset_root, args.task, args.sample, args.out, args.batch_size,
            args.workers, args.device, args.hf_cache, args.local_files_only,
        )
    elif args.command == "score-fold":
        if args.pca_components != OFFICIAL_PCA and os.environ.get("STAUDIT_TEST_MODE") != "1":
            raise ContractError("PRODUCTION_PCA_MUST_EQUAL_256")
        result = score_fold(
            args.dataset_root, args.embedding_root, args.output_root, args.task,
            args.fold, args.encoder, args.pca_components,
        )
    elif args.command == "aggregate":
        result = aggregate_scores(args.score_root, args.out, args.encoder)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
