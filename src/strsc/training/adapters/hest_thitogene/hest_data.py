"""Strict HEST 9x50 section and patch alignment for the THItoGene adapter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from strsc.training.adapters.ridge_mlp.ridge import load_g0054_module


MODEL_NAME = "THItoGene"
VALIDATION_PREFIX = "G0055_INNER_VAL_V1"
N_POSITION_BINS = 64
PATCH_PIXELS = 112
PHYSICAL_SUPPORT_UM = 112.0
PATCH_VALUE_SCALE = "raw_0_255"
UPSTREAM_EXPRESSION_TRANSFORM = "log10_1p_library_size_10000_on_ordered_50_gene_panel"


class HestTHItoGeneDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class PatchStore:
    task: str
    specimen: str
    path: Path
    image_key: str
    barcode_key: str
    layout: str
    physical_support_um: float
    value_scale: str


@dataclass(frozen=True)
class SectionMetadata:
    task: str
    specimen: str
    fold: int
    is_test: bool
    barcodes: np.ndarray
    genes: np.ndarray
    raw_counts: Optional[np.ndarray]
    target: Optional[np.ndarray]
    spatial_coords: np.ndarray
    patch_coords: np.ndarray | None
    validation_mask: np.ndarray | None
    patch_store: PatchStore


@dataclass(frozen=True)
class SectionBatch:
    metadata: SectionMetadata
    patches: torch.Tensor
    positions: torch.Tensor
    model_adjacency: torch.Tensor
    target: Optional[torch.Tensor]

    def to(self, device: torch.device) -> "SectionBatch":
        return SectionBatch(
            metadata=self.metadata,
            patches=self.patches.to(device),
            positions=self.positions.to(device),
            model_adjacency=self.model_adjacency.to(device),
            target=None if self.target is None else self.target.to(device),
        )


def _decode(values: Any) -> np.ndarray:
    result = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, (bytes, np.bytes_)):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return np.asarray(result, dtype=object)


def validation_mask(
    *, task: str, specimen: str, barcodes: np.ndarray
) -> np.ndarray:
    result = np.zeros(len(barcodes), dtype=bool)
    for index, barcode in enumerate(barcodes):
        token = (
            f"{VALIDATION_PREFIX}|{MODEL_NAME}|{task}|{specimen}|{str(barcode)}"
        )
        digest = hashlib.blake2b(token.encode("utf-8")).digest()
        result[index] = int.from_bytes(digest[:8], "big", signed=False) % 10 == 0
    if result.all() or not result.any():
        raise HestTHItoGeneDataError(
            f"VALIDATION_EMPTY_PARTITION:{task}:{specimen}:"
            f"train={int((~result).sum())}:validation={int(result.sum())}"
        )
    return result


def upstream_expression_transform(raw_counts: np.ndarray) -> np.ndarray:
    """Exact defaults used by upstream scprep normalize then transform.log."""

    raw = np.asarray(raw_counts, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 50 or raw.min(initial=0) < 0:
        raise HestTHItoGeneDataError(f"RAW_COUNT_CONTRACT:{raw.shape}")
    library = raw.sum(axis=1, keepdims=True)
    normalized = np.divide(
        raw,
        library,
        out=np.zeros_like(raw),
        where=library > 0,
    )
    normalized *= 10000.0
    transformed = np.log10(1.0 + normalized)
    if not np.isfinite(transformed).all():
        raise HestTHItoGeneDataError("TRANSFORM_NONFINITE")
    return transformed


def position_bins(coords: np.ndarray, n_bins: int = N_POSITION_BINS) -> np.ndarray:
    """Map a HEST section into the unchanged 64-entry upstream embeddings."""

    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise HestTHItoGeneDataError(f"POSITION_COORDINATES:{points.shape}")
    if n_bins < 2:
        raise HestTHItoGeneDataError(f"POSITION_BINS:{n_bins}")
    result = np.zeros(points.shape, dtype=np.int64)
    for axis in range(2):
        values = points[:, axis]
        extent = float(values.max() - values.min())
        if extent > 0:
            scaled = (values - values.min()) / extent * (n_bins - 1)
            result[:, axis] = np.rint(scaled).astype(np.int64)
    if result.min(initial=0) < 0 or result.max(initial=0) >= n_bins:
        raise HestTHItoGeneDataError("POSITION_BIN_RANGE")
    return result


def load_patch_manifest(path: Path) -> dict[tuple[str, str], PatchStore]:
    frame = pd.read_csv(path, sep="\t")
    required = {
        "task",
        "specimen",
        "patch_h5",
        "image_key",
        "barcode_key",
        "layout",
        "physical_support_um",
        "value_scale",
    }
    if not required.issubset(frame.columns):
        raise HestTHItoGeneDataError(
            f"PATCH_MANIFEST_COLUMNS:{sorted(required.difference(frame.columns))}"
        )
    if frame.duplicated(["task", "specimen"]).any():
        raise HestTHItoGeneDataError("PATCH_MANIFEST_DUPLICATE_KEY")
    result: dict[tuple[str, str], PatchStore] = {}
    for row in frame.itertuples(index=False):
        layout = str(row.layout).upper()
        support = float(row.physical_support_um)
        scale = str(row.value_scale)
        if layout not in {"NCHW", "NHWC"}:
            raise HestTHItoGeneDataError(f"PATCH_LAYOUT:{layout}")
        if not np.isclose(support, PHYSICAL_SUPPORT_UM):
            raise HestTHItoGeneDataError(
                f"PATCH_PHYSICAL_SUPPORT:{support}:{PHYSICAL_SUPPORT_UM}"
            )
        if scale != PATCH_VALUE_SCALE:
            raise HestTHItoGeneDataError(
                f"PATCH_VALUE_SCALE:{scale}:{PATCH_VALUE_SCALE}"
            )
        patch_path = Path(str(row.patch_h5))
        if not patch_path.is_absolute():
            patch_path = path.parent / patch_path
        item = PatchStore(
            task=str(row.task),
            specimen=str(row.specimen),
            path=patch_path.resolve(),
            image_key=str(row.image_key),
            barcode_key=str(row.barcode_key),
            layout=layout,
            physical_support_um=support,
            value_scale=scale,
        )
        result[(item.task, item.specimen)] = item
    return result


def _embedding_map(path: Path, encoder: str) -> dict[tuple[str, str], Path]:
    frame = pd.read_csv(path, sep="\t")
    required = {"task", "specimen", "encoder", "embedding_path"}
    if not required.issubset(frame.columns):
        raise HestTHItoGeneDataError(
            f"EMBEDDING_MANIFEST_COLUMNS:{sorted(required.difference(frame.columns))}"
        )
    chosen = frame.loc[frame["encoder"].astype(str).eq(encoder)].copy()
    if chosen.duplicated(["task", "specimen"]).any():
        raise HestTHItoGeneDataError("EMBEDDING_MANIFEST_DUPLICATE_KEY")
    result: dict[tuple[str, str], Path] = {}
    for row in chosen.itertuples(index=False):
        value = Path(str(row.embedding_path))
        if not value.is_absolute():
            value = path.parent / value
        result[(str(row.task), str(row.specimen))] = value.resolve()
    return result


def _coordinates_without_rna(
    *, dataset_root: Path, task: str, specimen: str, barcodes: np.ndarray
) -> np.ndarray:
    path = dataset_root / task / "adata" / f"{specimen}.h5ad"
    if not path.is_file():
        raise HestTHItoGeneDataError(f"MISSING_ADATA:{path}")
    with h5py.File(path, "r") as handle:
        obs = _decode(handle["obs/_index"][...])
        coords = np.asarray(handle["obsm/spatial"], dtype=np.float64)
    lookup = {barcode: index for index, barcode in enumerate(obs)}
    if len(lookup) != len(obs) or any(barcode not in lookup for barcode in barcodes):
        raise HestTHItoGeneDataError(
            f"HELD_OUT_BARCODE_JOIN:{task}:{specimen}"
        )
    order = np.asarray([lookup[barcode] for barcode in barcodes], dtype=np.int64)
    aligned = coords[order]
    if aligned.shape != (len(barcodes), 2) or not np.isfinite(aligned).all():
        raise HestTHItoGeneDataError(
            f"HELD_OUT_COORDINATES:{task}:{specimen}:{aligned.shape}"
        )
    return aligned


def load_fold_metadata(
    *,
    dataset_root: Path,
    gene_manifest: Path,
    embedding_manifest: Path,
    patch_manifest: Path,
    task: str,
    fold: int,
    include_test: bool,
    include_test_rna: bool = False,
    encoder: str = "phikon_v2",
    g0054_module: Path | None = None,
) -> tuple[list[SectionMetadata], SectionMetadata | None]:
    """Load training metadata first; held-out metadata only when explicitly requested."""

    frozen = load_g0054_module(g0054_module)
    if task not in frozen.TASK_SAMPLES:
        raise HestTHItoGeneDataError(f"TASK:{task}")
    samples = tuple(frozen.TASK_SAMPLES[task])
    if fold < 0 or fold >= len(samples):
        raise HestTHItoGeneDataError(f"FOLD:{task}:{fold}")
    genes = frozen._read_gene_manifest(gene_manifest, task)
    split_root = dataset_root / task / "splits"
    train_specimens = frozen._read_official_split(split_root / f"train_{fold}.csv")
    test_specimens = frozen._read_official_split(split_root / f"test_{fold}.csv")
    test_specimen = samples[fold]
    if test_specimens != [test_specimen]:
        raise HestTHItoGeneDataError(
            f"OFFICIAL_TEST_SPLIT:{task}:{fold}:{test_specimens}:{test_specimen}"
        )
    embeddings = _embedding_map(embedding_manifest, encoder)
    patches = load_patch_manifest(patch_manifest)

    def load_one(specimen: str, *, is_test: bool) -> SectionMetadata:
        key = (task, specimen)
        if key not in embeddings or key not in patches:
            raise HestTHItoGeneDataError(f"SECTION_MANIFEST_KEY:{task}:{specimen}")
        _, barcodes, patch_coords = frozen._read_embedding(embeddings[key])
        if is_test and not include_test_rna:
            raw = None
            target = None
            coords = _coordinates_without_rna(
                dataset_root=dataset_root,
                task=task,
                specimen=specimen,
                barcodes=barcodes,
            )
        else:
            raw, _, coords = frozen._matched_truth(
                dataset_root, task, specimen, barcodes, genes
            )
            target = upstream_expression_transform(raw)
        return SectionMetadata(
            task=task,
            specimen=specimen,
            fold=fold,
            is_test=is_test,
            barcodes=barcodes,
            genes=genes,
            raw_counts=raw,
            target=target,
            spatial_coords=coords,
            patch_coords=patch_coords,
            validation_mask=(
                None
                if is_test
                else validation_mask(task=task, specimen=specimen, barcodes=barcodes)
            ),
            patch_store=patches[key],
        )

    training = [load_one(specimen, is_test=False) for specimen in train_specimens]
    held_out = load_one(test_specimen, is_test=True) if include_test else None
    return training, held_out


def read_aligned_patches(metadata: SectionMetadata) -> torch.Tensor:
    store = metadata.patch_store
    if not store.path.is_file():
        raise HestTHItoGeneDataError(f"PATCH_STORE_MISSING:{store.path}")
    with h5py.File(store.path, "r") as handle:
        if store.image_key not in handle or store.barcode_key not in handle:
            raise HestTHItoGeneDataError(
                f"PATCH_STORE_SCHEMA:{store.path}:{store.image_key}:{store.barcode_key}"
            )
        stored_barcodes = _decode(handle[store.barcode_key][...])
        if len(set(stored_barcodes.tolist())) != len(stored_barcodes):
            raise HestTHItoGeneDataError(f"PATCH_BARCODE_DUPLICATE:{store.path}")
        lookup = {barcode: index for index, barcode in enumerate(stored_barcodes)}
        if any(barcode not in lookup for barcode in metadata.barcodes):
            raise HestTHItoGeneDataError(
                f"PATCH_BARCODE_JOIN:{metadata.task}:{metadata.specimen}"
            )
        images = np.asarray(handle[store.image_key][...])
    if images.shape[0] != len(stored_barcodes):
        raise HestTHItoGeneDataError(
            f"PATCH_IMAGE_ROWS:{images.shape[0]}:{len(stored_barcodes)}"
        )
    order = np.asarray([lookup[value] for value in metadata.barcodes], dtype=np.int64)
    images = images[order]
    if store.layout == "NHWC":
        if images.ndim != 4 or images.shape[-1] != 3:
            raise HestTHItoGeneDataError(f"PATCH_NHWC_SHAPE:{images.shape}")
        images = np.transpose(images, (0, 3, 1, 2))
    elif images.ndim != 4 or images.shape[1] != 3:
        raise HestTHItoGeneDataError(f"PATCH_NCHW_SHAPE:{images.shape}")
    if not np.issubdtype(images.dtype, np.number) or not np.isfinite(images).all():
        raise HestTHItoGeneDataError(f"PATCH_IMAGE_CONTENT:{images.dtype}")
    if images.min(initial=0) < 0 or images.max(initial=0) > 255:
        raise HestTHItoGeneDataError(
            f"PATCH_VALUE_RANGE:{images.min()}:{images.max()}"
        )
    tensor = torch.as_tensor(np.ascontiguousarray(images), dtype=torch.float32)
    if tensor.shape[-2:] != (PATCH_PIXELS, PATCH_PIXELS):
        chunks = []
        for start in range(0, len(tensor), 256):
            chunks.append(
                F.interpolate(
                    tensor[start : start + 256],
                    size=(PATCH_PIXELS, PATCH_PIXELS),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        tensor = torch.cat(chunks, dim=0)
    # Upstream DataLoader adds a section batch dimension of one.
    return tensor.unsqueeze(0)


def materialize_section(
    metadata: SectionMetadata,
    *,
    upstream_graph_module: Any,
) -> SectionBatch:
    patches = read_aligned_patches(metadata)
    positions = torch.as_tensor(
        position_bins(metadata.spatial_coords), dtype=torch.long
    ).unsqueeze(0)
    # Call the audited upstream 4-NN implementation rather than replacing the
    # model's own context graph with the intervention graph.
    adjacency = upstream_graph_module.calcADJ(
        coord=metadata.spatial_coords, k=4, pruneTag="NA"
    ).to(dtype=torch.float32)
    if adjacency.shape != (len(metadata.barcodes), len(metadata.barcodes)):
        raise HestTHItoGeneDataError(
            f"UPSTREAM_ADJACENCY_SHAPE:{tuple(adjacency.shape)}:{len(metadata.barcodes)}"
        )
    edge_index = torch.nonzero(adjacency > 0, as_tuple=False).transpose(0, 1).contiguous()
    return SectionBatch(
        metadata=metadata,
        patches=patches,
        positions=positions,
        model_adjacency=edge_index.to(dtype=torch.long),
        target=(
            None
            if metadata.target is None
            else torch.as_tensor(metadata.target, dtype=torch.float32)
        ),
    )
