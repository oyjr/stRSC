"""Create and reuse the TENX39 spot split for paired sCellST training."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import random_split


SPLIT_SCHEMA = "g0055_scellst_tenx39_split_v1"


@dataclass(frozen=True)
class FrozenSpotSplit:
    """Dataset indices in the exact order emitted by seeded ``random_split``."""

    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    manifest_path: Path
    receipt_path: Path
    seed: int
    frac_train: float


def _validate_inputs(
    spot_ids: Sequence[str], coords: np.ndarray, specimen_ids: Sequence[str]
) -> tuple[tuple[str, ...], np.ndarray, tuple[str, ...]]:
    ids = tuple(str(value) for value in spot_ids)
    specimens = tuple(str(value) for value in specimen_ids)
    xy = np.asarray(coords, dtype=np.float64)
    if len(ids) < 2:
        raise ValueError("At least two spots are required to freeze a train/val split")
    if len(set(ids)) != len(ids):
        raise ValueError("Spot IDs must be unique before freezing the split")
    if xy.shape != (len(ids), 2) or not np.isfinite(xy).all():
        raise ValueError(f"Expected finite coordinates with shape {(len(ids), 2)}, got {xy.shape}")
    if len(specimens) != len(ids):
        raise ValueError("specimen_ids must align one-to-one with spot_ids")
    return ids, xy, specimens


def _receipt_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(manifest_path.suffix + ".receipt.json")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def _create_split(n_spots: int, seed: int, frac_train: float) -> tuple[list[int], list[int]]:
    if not 0.0 < frac_train < 1.0:
        raise ValueError(f"frac_train must be between zero and one, got {frac_train}")
    n_train = int(np.floor(n_spots * float(frac_train)))
    n_train = min(max(n_train, 1), n_spots - 1)
    n_val = n_spots - n_train
    generator = torch.Generator().manual_seed(int(seed))
    train, val = random_split(range(n_spots), [n_train, n_val], generator=generator)
    train_indices = [int(index) for index in train.indices]
    val_indices = [int(index) for index in val.indices]
    if not train_indices or not val_indices:
        raise ValueError("Frozen split produced an empty train or validation partition")
    return train_indices, val_indices


def _serialize_manifest(
    ids: Sequence[str],
    coords: np.ndarray,
    specimens: Sequence[str],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
) -> str:
    train_order = {index: order for order, index in enumerate(train_indices)}
    val_order = {index: order for order, index in enumerate(val_indices)}
    rows: list[list[str]] = [[
        "dataset_index",
        "spot_id",
        "specimen_id",
        "x",
        "y",
        "partition",
        "partition_order",
    ]]
    for index, (spot_id, specimen_id, coord) in enumerate(zip(ids, specimens, coords)):
        if index in train_order:
            partition = "train"
            order = train_order[index]
        else:
            partition = "val"
            order = val_order[index]
        rows.append([
            str(index),
            str(spot_id),
            str(specimen_id),
            format(float(coord[0]), ".17g"),
            format(float(coord[1]), ".17g"),
            partition,
            str(order),
        ])

    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def _load_existing(
    manifest_path: Path,
    receipt_path: Path,
    ids: Sequence[str],
    coords: np.ndarray,
    specimens: Sequence[str],
    seed: int,
    frac_train: float,
) -> FrozenSpotSplit:
    if not receipt_path.is_file():
        raise RuntimeError(f"Frozen split receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_receipt = {
        "schema": SPLIT_SCHEMA,
        "seed": int(seed),
        "frac_train": float(frac_train),
        "n_spots": len(ids),
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise RuntimeError(
                f"Frozen split receipt mismatch for {key}: {receipt.get(key)!r} != {expected!r}"
            )

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != len(ids):
        raise RuntimeError(f"Frozen split has {len(rows)} rows; expected {len(ids)}")

    train_pairs: list[tuple[int, int]] = []
    val_pairs: list[tuple[int, int]] = []
    for expected_index, row in enumerate(rows):
        index = int(row["dataset_index"])
        if index != expected_index:
            raise RuntimeError("Frozen split dataset_index is not contiguous and ordered")
        if row["spot_id"] != ids[index] or row["specimen_id"] != specimens[index]:
            raise RuntimeError(f"Frozen split identity mismatch at dataset index {index}")
        stored_coord = np.asarray([float(row["x"]), float(row["y"])], dtype=np.float64)
        if not np.array_equal(stored_coord, coords[index]):
            raise RuntimeError(f"Frozen split coordinate mismatch at dataset index {index}")
        pair = (int(row["partition_order"]), index)
        if row["partition"] == "train":
            train_pairs.append(pair)
        elif row["partition"] == "val":
            val_pairs.append(pair)
        else:
            raise RuntimeError(f"Unknown frozen partition {row['partition']!r}")

    train_pairs.sort()
    val_pairs.sort()
    if [order for order, _ in train_pairs] != list(range(len(train_pairs))):
        raise RuntimeError("Frozen train partition_order is invalid")
    if [order for order, _ in val_pairs] != list(range(len(val_pairs))):
        raise RuntimeError("Frozen validation partition_order is invalid")
    train_indices = tuple(index for _, index in train_pairs)
    val_indices = tuple(index for _, index in val_pairs)
    if set(train_indices).intersection(val_indices) or set(train_indices).union(val_indices) != set(range(len(ids))):
        raise RuntimeError("Frozen split partitions are not disjoint and exhaustive")
    return FrozenSpotSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        seed=int(seed),
        frac_train=float(frac_train),
    )


def freeze_or_load_spot_split(
    *,
    spot_ids: Sequence[str],
    coords: np.ndarray,
    specimen_ids: Sequence[str],
    manifest_path: str | Path,
    seed: int = 0,
    frac_train: float = 0.8,
) -> FrozenSpotSplit:
    """Create one explicit split, or validate and reuse the already frozen split.

    The saved ``partition_order`` retains a seeded PyTorch ``random_split``
    ordering with explicit integer partition sizes.  Existing manifests are
    never silently regenerated when IDs, coordinates, seed, or fraction differ.
    """

    ids, xy, specimens = _validate_inputs(spot_ids, coords, specimen_ids)
    path = Path(manifest_path)
    receipt = _receipt_path(path)
    if path.exists():
        return _load_existing(path, receipt, ids, xy, specimens, seed, frac_train)
    if receipt.exists():
        raise RuntimeError(f"Split receipt exists without its manifest: {receipt}")

    train_indices, val_indices = _create_split(len(ids), int(seed), float(frac_train))
    manifest_text = _serialize_manifest(ids, xy, specimens, train_indices, val_indices)
    receipt_payload = {
        "schema": SPLIT_SCHEMA,
        "status": "FROZEN",
        "seed": int(seed),
        "frac_train": float(frac_train),
        "n_spots": len(ids),
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "specimens": sorted(set(specimens)),
        "manifest": str(path),
    }
    _atomic_write_text(path, manifest_text)
    _atomic_write_text(receipt, json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n")
    return FrozenSpotSplit(
        train_indices=tuple(train_indices),
        val_indices=tuple(val_indices),
        manifest_path=path,
        receipt_path=receipt,
        seed=int(seed),
        frac_train=float(frac_train),
    )
