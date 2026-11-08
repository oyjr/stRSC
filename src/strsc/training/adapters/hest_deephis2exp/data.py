"""Barcode-exact HEST 9x50 LOSO data contract for DeepHis2Exp models."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import h5py
import numpy as np
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import Dataset

try:  # Register optional HEST HDF5 compression filters when available.
    import hdf5plugin  # noqa: F401
except ImportError:  # Built-in gzip/lzf files do not require it.
    hdf5plugin = None


TASK_SPECIMENS: dict[str, tuple[str, ...]] = {
    "IDC": ("TENX99", "TENX95", "NCBI785", "NCBI783"),
    "PAAD": ("TENX116", "TENX140", "TENX126"),
    "LUNG": ("TENX141", "TENX118"),
}
TRUTH_TRANSFORM_ID = "log1p_raw_integer_counts_no_library_normalization"
PATCH_IMAGE_KEYS = ("img", "imgs", "patches", "images")
PATCH_BARCODE_KEYS = ("barcodes", "barcode")


class HestAdapterError(RuntimeError):
    """A frozen alignment or image-input contract was violated."""


def _decode(values: Any) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    return np.asarray(
        [
            item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in array
        ],
        dtype=object,
    )


def _one_existing_key(handle: h5py.File, candidates: Sequence[str], kind: str) -> str:
    found = [key for key in candidates if key in handle]
    if len(found) != 1:
        raise HestAdapterError(f"PATCH_{kind}_KEY:{handle.filename}:{found}")
    return found[0]


def _load_g0054_module(path: Path | None = None):
    source = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[2] / "hest_data.py"
    )
    if not source.is_file():
        raise HestAdapterError(f"G0054_MODULE_MISSING:{source}")
    spec = importlib.util.spec_from_file_location("g0054_hest_contract", source)
    if spec is None or spec.loader is None:
        raise HestAdapterError(f"G0054_MODULE_LOAD:{source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_index(path: Path) -> tuple[str, np.ndarray, tuple[int, ...]]:
    if not path.is_file():
        raise HestAdapterError(f"PATCH_FILE_MISSING:{path}")
    with h5py.File(path, "r") as handle:
        image_key = _one_existing_key(handle, PATCH_IMAGE_KEYS, "IMAGE")
        barcode_key = _one_existing_key(handle, PATCH_BARCODE_KEYS, "BARCODE")
        images = handle[image_key]
        barcodes = _decode(handle[barcode_key][...])
        shape = tuple(int(value) for value in images.shape)
    if len(shape) != 4 or shape[0] != len(barcodes):
        raise HestAdapterError(f"PATCH_SHAPE:{path}:{shape}:{len(barcodes)}")
    if len(set(barcodes.tolist())) != len(barcodes):
        raise HestAdapterError(f"PATCH_DUPLICATE_BARCODE:{path}")
    channel_first = shape[1] == 3
    channel_last = shape[-1] == 3
    if channel_first == channel_last:
        raise HestAdapterError(f"PATCH_CHANNEL_LAYOUT:{path}:{shape}")
    return image_key, barcodes, shape


def _dense_lattice_positions(coords: np.ndarray, n_pos: int = 256) -> np.ndarray:
    """Make deterministic non-negative position-embedding indices.

    Exact repeated coordinate levels are ranked when possible. If physical
    coordinates contain per-spot jitter or more than ``n_pos`` distinct levels,
    the axis is linearly quantized. The physical coordinates themselves remain
    unchanged for the frozen loss graph and exported provenance.
    """

    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise HestAdapterError(f"POSITION_COORDINATES:{points.shape}")
    if n_pos < 2:
        raise HestAdapterError(f"POSITION_EMBEDDING_SIZE:{n_pos}")

    result = np.zeros_like(points, dtype=np.int64)
    for axis in range(2):
        values = points[:, axis]
        levels, inverse = np.unique(values, return_inverse=True)
        if len(levels) <= n_pos:
            result[:, axis] = inverse
        else:
            span = float(values.max() - values.min())
            if span <= 0:
                result[:, axis] = 0
            else:
                scaled = (values - values.min()) / span * (n_pos - 1)
                result[:, axis] = np.rint(scaled).astype(np.int64)
    if result.min(initial=0) < 0 or result.max(initial=0) >= n_pos:
        raise HestAdapterError("POSITION_EMBEDDING_RANGE")
    return result


@dataclass(frozen=True)
class AlignedHestSpecimen:
    task: str
    specimen: str
    genes: np.ndarray
    barcodes: np.ndarray
    raw_counts: np.ndarray
    truth: np.ndarray
    spatial_coords: np.ndarray
    positions: np.ndarray
    patch_path: Path
    image_key: str
    image_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        rows = len(self.barcodes)
        if self.truth.shape != (rows, 50) or self.raw_counts.shape != (rows, 50):
            raise HestAdapterError(
                f"ALIGNED_EXPRESSION_SHAPE:{self.specimen}:{self.truth.shape}:{self.raw_counts.shape}"
            )
        if self.spatial_coords.shape != (rows, 2) or self.positions.shape != (rows, 2):
            raise HestAdapterError(f"ALIGNED_COORDINATE_SHAPE:{self.specimen}")
        if len(self.genes) != 50 or len(set(self.genes.tolist())) != 50:
            raise HestAdapterError(f"ALIGNED_GENE_COUNT:{self.specimen}:{len(self.genes)}")
        if len(set(self.barcodes.tolist())) != rows:
            raise HestAdapterError(f"ALIGNED_BARCODE_DUPLICATE:{self.specimen}")
        if not np.isfinite(self.truth).all() or not np.isfinite(self.spatial_coords).all():
            raise HestAdapterError(f"ALIGNED_NONFINITE:{self.specimen}")

    @property
    def size_factors(self) -> np.ndarray:
        totals = self.raw_counts.sum(axis=1, dtype=np.float64)
        positive = totals[totals > 0]
        median = float(np.median(positive)) if len(positive) else 1.0
        if median <= 0 or not np.isfinite(median):
            median = 1.0
        return (totals / median).astype(np.float32)


@dataclass(frozen=True)
class AlignedHestFold:
    task: str
    fold: int
    genes: np.ndarray
    train_specimens: tuple[AlignedHestSpecimen, ...]
    test_specimen_name: str
    test_specimen: AlignedHestSpecimen | None
    split_path: Path


def _aligned_specimen(
    *,
    frozen: Any,
    dataset_root: Path,
    task: str,
    specimen: str,
    genes: np.ndarray,
    n_pos: int,
) -> AlignedHestSpecimen:
    patch_path = dataset_root / task / "patches" / f"{specimen}.h5"
    image_key, barcodes, shape = _patch_index(patch_path)
    raw, truth, coords = frozen._matched_truth(
        dataset_root, task, specimen, barcodes, genes
    )
    if not np.array_equal(truth, np.log1p(raw)):
        raise HestAdapterError(f"TRUTH_TRANSFORM:{task}:{specimen}")
    return AlignedHestSpecimen(
        task=task,
        specimen=specimen,
        genes=np.asarray(genes, dtype=object),
        barcodes=barcodes,
        raw_counts=np.asarray(raw, dtype=np.float32),
        truth=np.asarray(truth, dtype=np.float32),
        spatial_coords=np.asarray(coords, dtype=np.float64),
        positions=_dense_lattice_positions(coords, n_pos=n_pos),
        patch_path=patch_path,
        image_key=image_key,
        image_shape=shape,
    )


def load_aligned_hest_fold(
    *,
    dataset_root: Path,
    gene_manifest: Path,
    task: str,
    fold: int,
    g0054_module: Path | None = None,
    n_pos: int = 256,
    include_test: bool = False,
) -> AlignedHestFold:
    """Load the official fold, preserving CSV specimen and patch-H5 row order.

    ``include_test`` defaults to false so training and engineering-gate code
    cannot load held-out RNA accidentally. Enable it only in the locked outer
    inference/export phase.
    """

    task = str(task)
    if task not in TASK_SPECIMENS:
        raise HestAdapterError(f"TASK:{task}")
    if fold < 0 or fold >= len(TASK_SPECIMENS[task]):
        raise HestAdapterError(f"FOLD_RANGE:{task}:{fold}")
    root = Path(dataset_root)
    frozen = _load_g0054_module(g0054_module)
    genes = frozen._read_gene_manifest(Path(gene_manifest), task)
    split_root = root / task / "splits"
    split_path = split_root / f"train_{fold}.csv"
    train_names = tuple(frozen._read_official_split(split_path))
    test_names = frozen._read_official_split(split_root / f"test_{fold}.csv")
    expected_test = TASK_SPECIMENS[task][fold]
    if test_names != [expected_test]:
        raise HestAdapterError(
            f"OFFICIAL_TEST_SPLIT:{task}:{fold}:{test_names}:{expected_test}"
        )
    if expected_test in train_names or set(train_names) | {expected_test} != set(
        TASK_SPECIMENS[task]
    ):
        raise HestAdapterError(f"OFFICIAL_TRAIN_SPLIT:{task}:{fold}:{train_names}")

    train = tuple(
        _aligned_specimen(
            frozen=frozen,
            dataset_root=root,
            task=task,
            specimen=name,
            genes=genes,
            n_pos=n_pos,
        )
        for name in train_names
    )
    test = (
        _aligned_specimen(
            frozen=frozen,
            dataset_root=root,
            task=task,
            specimen=expected_test,
            genes=genes,
            n_pos=n_pos,
        )
        if include_test
        else None
    )
    return AlignedHestFold(
        task=task,
        fold=int(fold),
        genes=np.asarray(genes, dtype=object),
        train_specimens=train,
        test_specimen_name=expected_test,
        test_specimen=test,
        split_path=split_path,
    )


def _image_to_chw_float(
    image: np.ndarray, *, expected_patch_pixels: int | None = 224
) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim != 3:
        raise HestAdapterError(f"PATCH_IMAGE_DIMENSION:{array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        chw = array
    elif array.shape[-1] == 3 and array.shape[0] != 3:
        chw = np.moveaxis(array, -1, 0)
    else:
        raise HestAdapterError(f"PATCH_IMAGE_LAYOUT:{array.shape}")
    if expected_patch_pixels is not None and tuple(chw.shape[1:]) != (
        expected_patch_pixels,
        expected_patch_pixels,
    ):
        raise HestAdapterError(
            f"PATCH_PIXEL_SHAPE:{tuple(chw.shape)}:expected={expected_patch_pixels}"
        )
    tensor = torch.as_tensor(np.ascontiguousarray(chw), dtype=torch.float32)
    if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 255:
        raise HestAdapterError("PATCH_PIXEL_RANGE:expected_raw_0_255")
    return tensor


class _PatchFiles:
    """Per-process lazy HDF5 handles; no handle is pickled into a worker."""

    def __init__(self) -> None:
        self._handles: dict[Path, h5py.File] = {}

    def image(self, specimen: AlignedHestSpecimen, row: int) -> np.ndarray:
        handle = self._handles.get(specimen.patch_path)
        if handle is None:
            handle = h5py.File(specimen.patch_path, "r")
            self._handles[specimen.patch_path] = handle
        return np.asarray(handle[specimen.image_key][row])

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        del state
        self._handles = {}

    def __del__(self) -> None:
        self.close()


class HestSpotDataset(Dataset):
    """One global spot order made by official specimen order then H5 row order."""

    def __init__(
        self,
        specimens: Sequence[AlignedHestSpecimen],
        *,
        expected_patch_pixels: int | None = 224,
    ) -> None:
        if not specimens:
            raise HestAdapterError("NO_SPECIMENS")
        self.specimens = tuple(specimens)
        self.expected_patch_pixels = expected_patch_pixels
        self._lookup: list[tuple[int, int]] = [
            (specimen_index, row)
            for specimen_index, specimen in enumerate(self.specimens)
            for row in range(len(specimen.barcodes))
        ]
        self._files = _PatchFiles()

    def __len__(self) -> int:
        return len(self._lookup)

    def __getitem__(self, index: int) -> dict[str, Any]:
        specimen_index, row = self._lookup[index]
        specimen = self.specimens[specimen_index]
        image = _image_to_chw_float(
            self._files.image(specimen, row),
            expected_patch_pixels=self.expected_patch_pixels,
        )
        return {
            "images": image,
            "target": torch.from_numpy(specimen.truth[row]).float(),
            "raw_counts": torch.from_numpy(specimen.raw_counts[row]).float(),
            "positions": torch.from_numpy(specimen.positions[row]).long(),
            "spatial_coords": torch.from_numpy(specimen.spatial_coords[row]).double(),
            "size_factors": torch.tensor(specimen.size_factors[row], dtype=torch.float32),
            "barcode": str(specimen.barcodes[row]),
            "specimen": specimen.specimen,
            "global_index": int(index),
            "specimen_index": int(specimen_index),
            "specimen_row": int(row),
        }

    def specimen_global_indices(self) -> tuple[np.ndarray, ...]:
        blocks: list[np.ndarray] = []
        offset = 0
        for specimen in self.specimens:
            block = np.arange(offset, offset + len(specimen.barcodes), dtype=np.int64)
            blocks.append(block)
            offset += len(specimen.barcodes)
        return tuple(blocks)


def build_context_adjacency(coords: np.ndarray, neighbors: int = 8) -> torch.Tensor:
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise HestAdapterError(f"CONTEXT_COORDINATES:{points.shape}")
    k = min(int(neighbors), len(points) - 1)
    if k < 1:
        raise HestAdapterError("CONTEXT_NEIGHBOR_COUNT")
    matrix = kneighbors_graph(
        points,
        n_neighbors=k,
        mode="connectivity",
        metric="euclidean",
        include_self=True,
    ).toarray()
    return torch.as_tensor(matrix, dtype=torch.float32)


def load_specimen_batch(
    specimen: AlignedHestSpecimen,
    *,
    rows: Iterable[int] | None = None,
    expected_patch_pixels: int | None = 224,
    context_neighbors: int = 8,
) -> dict[str, Any]:
    """Load one context batch without mixing specimens or changing row order."""

    selected = (
        np.arange(len(specimen.barcodes), dtype=np.int64)
        if rows is None
        else np.asarray(list(rows), dtype=np.int64)
    )
    if selected.ndim != 1 or len(selected) < 2:
        raise HestAdapterError(f"SPECIMEN_BATCH_ROWS:{selected.shape}")
    if selected.min() < 0 or selected.max() >= len(specimen.barcodes):
        raise HestAdapterError("SPECIMEN_BATCH_ROW_RANGE")
    if len(set(selected.tolist())) != len(selected):
        raise HestAdapterError("SPECIMEN_BATCH_ROW_DUPLICATE")

    reader = _PatchFiles()
    try:
        images = torch.stack(
            [
                _image_to_chw_float(
                    reader.image(specimen, int(row)),
                    expected_patch_pixels=expected_patch_pixels,
                )
                for row in selected
            ]
        )
    finally:
        reader.close()
    coords = specimen.spatial_coords[selected]
    return {
        "images": images,
        "target": torch.from_numpy(specimen.truth[selected]).float(),
        "raw_counts": torch.from_numpy(specimen.raw_counts[selected]).float(),
        "positions": torch.from_numpy(specimen.positions[selected]).long(),
        "spatial_coords": torch.from_numpy(coords).double(),
        "size_factors": torch.from_numpy(specimen.size_factors[selected]).float(),
        "context_adjacency": build_context_adjacency(coords, neighbors=context_neighbors),
        "barcodes": [str(value) for value in specimen.barcodes[selected]],
        "specimen": specimen.specimen,
        "specimen_rows": torch.from_numpy(selected).long(),
    }
