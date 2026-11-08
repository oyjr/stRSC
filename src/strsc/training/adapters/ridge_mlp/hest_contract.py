"""Load HEST fold inputs through the shared preprocessing implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ridge import RidgeAdapterError, load_g0054_module


@dataclass(frozen=True)
class HestFold:
    task: str
    fold: int
    train_specimens: tuple[str, ...]
    test_specimen: str
    genes: np.ndarray
    train_embeddings: tuple[np.ndarray, ...]
    train_barcodes: tuple[np.ndarray, ...]
    train_truth: tuple[np.ndarray, ...]
    train_coords: tuple[np.ndarray, ...]
    test_embeddings: np.ndarray
    test_barcodes: np.ndarray
    test_raw_counts: np.ndarray
    test_truth: np.ndarray
    test_coords: np.ndarray
    test_patch_coords: np.ndarray | None
    split_path: Path


def _embedding_map(path: Path, encoder: str) -> dict[tuple[str, str], Path]:
    frame = pd.read_csv(path, sep="\t")
    required = {"task", "specimen", "encoder", "embedding_path"}
    if not required.issubset(frame.columns):
        raise RidgeAdapterError(f"EMBEDDING_MANIFEST_COLUMNS:{sorted(required)}")
    selected = frame.loc[frame["encoder"].astype(str).eq(encoder)].copy()
    if selected.duplicated(["task", "specimen"]).any():
        raise RidgeAdapterError("DUPLICATE_EMBEDDING_MANIFEST_KEY")
    return {
        (str(row.task), str(row.specimen)): Path(str(row.embedding_path))
        for row in selected.itertuples(index=False)
    }


def load_hest_fold(
    *,
    dataset_root: Path,
    gene_manifest: Path,
    task: str,
    fold: int,
    encoder: str = "phikon_v2",
    embedding_root: Path | None = None,
    embedding_manifest: Path | None = None,
    g0054_module: Path | None = None,
) -> HestFold:
    """Load aligned embeddings, ordered genes, RNA and coordinates for a HEST fold."""

    if (embedding_root is None) == (embedding_manifest is None):
        raise RidgeAdapterError("EXACTLY_ONE_EMBEDDING_SOURCE_REQUIRED")
    frozen = load_g0054_module(g0054_module)
    if task not in frozen.TASK_SAMPLES:
        raise RidgeAdapterError(f"TASK:{task}")
    samples = tuple(frozen.TASK_SAMPLES[task])
    if fold < 0 or fold >= len(samples):
        raise RidgeAdapterError(f"FOLD_RANGE:{task}:{fold}")
    genes = frozen._read_gene_manifest(Path(gene_manifest), task)
    test_specimen = samples[fold]
    split_root = Path(dataset_root) / task / "splits"
    split_path = split_root / f"train_{fold}.csv"
    train_specimens = tuple(frozen._read_official_split(split_path))
    test_specimens = frozen._read_official_split(split_root / f"test_{fold}.csv")
    if test_specimens != [test_specimen]:
        raise RidgeAdapterError(
            f"OFFICIAL_TEST_SPLIT:{task}:{fold}:{test_specimens}:{test_specimen}"
        )
    if test_specimen in train_specimens or set(train_specimens) | {test_specimen} != set(samples):
        raise RidgeAdapterError(
            f"OFFICIAL_TRAIN_SPLIT:{task}:{fold}:{train_specimens}"
        )

    manifest = _embedding_map(Path(embedding_manifest), encoder) if embedding_manifest else {}

    def embedding_for(specimen: str) -> Path:
        if embedding_manifest is not None:
            key = (task, specimen)
            if key not in manifest:
                raise RidgeAdapterError(
                    f"MISSING_EMBEDDING_MANIFEST_KEY:{task}:{specimen}:{encoder}"
                )
            return manifest[key]
        assert embedding_root is not None
        return Path(embedding_root) / task / f"{specimen}.{encoder}.h5"

    train_x: list[np.ndarray] = []
    train_barcodes: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    train_coords: list[np.ndarray] = []
    for specimen in train_specimens:
        embeddings, barcodes, _ = frozen._read_embedding(embedding_for(specimen))
        _, truth, coords = frozen._matched_truth(
            Path(dataset_root), task, specimen, barcodes, genes
        )
        train_x.append(embeddings)
        train_barcodes.append(barcodes)
        train_y.append(truth)
        train_coords.append(coords)

    test_embeddings, test_barcodes, patch_coords = frozen._read_embedding(
        embedding_for(test_specimen)
    )
    raw, test_truth, test_coords = frozen._matched_truth(
        Path(dataset_root), task, test_specimen, test_barcodes, genes
    )
    return HestFold(
        task=task,
        fold=fold,
        train_specimens=train_specimens,
        test_specimen=test_specimen,
        genes=genes,
        train_embeddings=tuple(train_x),
        train_barcodes=tuple(train_barcodes),
        train_truth=tuple(train_y),
        train_coords=tuple(train_coords),
        test_embeddings=test_embeddings,
        test_barcodes=test_barcodes,
        test_raw_counts=raw,
        test_truth=test_truth,
        test_coords=test_coords,
        test_patch_coords=patch_coords,
        split_path=split_path,
    )
