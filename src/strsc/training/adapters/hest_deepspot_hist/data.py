"""HEST-aligned expression and published feature contracts for DeepSpot/HiST."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import h5py
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import Dataset

from strsc.training.adapters.hest_deephis2exp.data import (  # Reuse the frozen barcode/gene join.
    AlignedHestFold,
    AlignedHestSpecimen,
    load_aligned_hest_fold,
)
from strsc.training.shared.adjacency import build_spatial_edges

from .contract import HestDeepSpotHiSTError, SpecimenInnerSplit


DEEPSPOT_CONTEXT = "spot_subspot_neighbors"
DEEPSPOT_MORPHOLOGY = "inception"
DEEPSPOT_SUBSPOTS = 9
DEEPSPOT_TARGET_SUM = 10_000.0
HIST_BACKBONE = "UNI"
HIST_FEATURE_DIM = 1024


def _decode(values: Any) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in np.asarray(values)
        ],
        dtype=object,
    )


def _h5ad_matrix(node: h5py.Dataset | h5py.Group):
    if isinstance(node, h5py.Dataset):
        return np.asarray(node)
    required = {"data", "indices", "indptr"}
    if not required.issubset(node.keys()) or "shape" not in node.attrs:
        raise HestDeepSpotHiSTError(f"H5AD_X_LAYOUT:{node.name}")
    shape = tuple(int(value) for value in node.attrs["shape"])
    encoding = str(node.attrs.get("encoding-type", "csr_matrix"))
    cls = sparse.csc_matrix if "csc" in encoding else sparse.csr_matrix
    return cls(
        (
            np.asarray(node["data"]),
            np.asarray(node["indices"]),
            np.asarray(node["indptr"]),
        ),
        shape=shape,
    )


def load_full_library_totals(
    dataset_root: str | Path,
    *,
    task: str,
    specimen: str,
    barcodes: Sequence[str],
) -> np.ndarray:
    """Load full-transcriptome library totals used by published DeepSpot."""

    path = Path(dataset_root) / str(task) / "adata" / f"{specimen}.h5ad"
    if not path.is_file():
        raise HestDeepSpotHiSTError(f"ADATA_MISSING:{path}")
    with h5py.File(path, "r") as handle:
        obs = _decode(handle["obs/_index"][...])
        matrix = _h5ad_matrix(handle["X"])
    lookup = {barcode: index for index, barcode in enumerate(obs.tolist())}
    requested = [str(value) for value in barcodes]
    if len(lookup) != len(obs) or any(value not in lookup for value in requested):
        raise HestDeepSpotHiSTError(f"ADATA_BARCODE_JOIN:{task}:{specimen}")
    rows = np.asarray([lookup[value] for value in requested], dtype=np.int64)
    totals = np.asarray(matrix[rows].sum(axis=1)).reshape(-1).astype(np.float64)
    if totals.shape != (len(requested),) or not np.isfinite(totals).all() or np.any(totals < 0):
        raise HestDeepSpotHiSTError(f"LIBRARY_TOTALS:{task}:{specimen}")
    return totals


def deepspot_native_truth(raw_counts: np.ndarray, library_totals: np.ndarray) -> np.ndarray:
    """Published DeepSpot log1p library-size normalization before scaling."""

    raw = np.asarray(raw_counts, dtype=np.float64)
    totals = np.asarray(library_totals, dtype=np.float64)
    if raw.ndim != 2 or totals.shape != (raw.shape[0],):
        raise HestDeepSpotHiSTError(f"DEEPSPOT_TARGET_SHAPE:{raw.shape}:{totals.shape}")
    if np.any(raw < 0) or np.any(totals < 0) or not np.isfinite(raw).all():
        raise HestDeepSpotHiSTError("DEEPSPOT_TARGET_VALUES")
    return np.log1p(raw / (totals[:, None] + 1e-12) * DEEPSPOT_TARGET_SUM)


@dataclass(frozen=True)
class FrozenStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if mean.shape != scale.shape or mean.ndim != 1:
            raise HestDeepSpotHiSTError("STANDARDIZER_SHAPE")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise HestDeepSpotHiSTError("STANDARDIZER_VALUES")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.mean):
            raise HestDeepSpotHiSTError("STANDARDIZER_TRANSFORM_SHAPE")
        return ((array - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.mean):
            raise HestDeepSpotHiSTError("STANDARDIZER_INVERSE_SHAPE")
        return (array * self.scale + self.mean).astype(np.float32)

    def sklearn_scaler(self):
        """Materialize the scaler interface saved by the upstream model."""

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.mean_ = self.mean.copy()
        scaler.scale_ = self.scale.copy()
        scaler.var_ = np.square(self.scale)
        scaler.n_features_in_ = len(self.mean)
        scaler.n_samples_seen_ = np.int64(1)
        return scaler


@dataclass(frozen=True)
class DeepSpotTargets:
    standardizer: FrozenStandardizer
    native_by_specimen: dict[str, np.ndarray]
    scaled_by_specimen: dict[str, np.ndarray]
    transform_id: str = "deepspot_log1p_full_library_10000_then_train_only_standard_scaler"


def fit_deepspot_targets(
    fold: AlignedHestFold,
    splits: Sequence[SpecimenInnerSplit],
    *,
    dataset_root: str | Path,
) -> DeepSpotTargets:
    """Fit published standard scaling on supervised outer-training rows only."""

    if len(fold.train_specimens) != len(splits):
        raise HestDeepSpotHiSTError("TARGET_SPLIT_COUNT")
    native: dict[str, np.ndarray] = {}
    fit_rows: list[np.ndarray] = []
    for specimen, split in zip(fold.train_specimens, splits, strict=True):
        if specimen.specimen != split.specimen:
            raise HestDeepSpotHiSTError("TARGET_SPLIT_ORDER")
        totals = load_full_library_totals(
            dataset_root,
            task=fold.task,
            specimen=specimen.specimen,
            barcodes=specimen.barcodes,
        )
        values = deepspot_native_truth(specimen.raw_counts, totals)
        native[specimen.specimen] = values
        fit_rows.append(values[split.train_mask])
    fitted = np.concatenate(fit_rows, axis=0).astype(np.float64)
    mean = fitted.mean(axis=0)
    scale = fitted.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    standardizer = FrozenStandardizer(mean, scale)
    scaled = {name: standardizer.transform(values) for name, values in native.items()}
    return DeepSpotTargets(standardizer, native, scaled)


class DeepSpotFeatureStore:
    """Strict official Inception cache: main spot followed by nine subspots."""

    def __init__(self, root: str | Path, *, feature_dim: int | None = None) -> None:
        self.root = Path(root)
        self.expected_feature_dim = feature_dim
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def load(self, specimen: str, barcode: str) -> np.ndarray:
        key = (str(specimen), str(barcode))
        if key not in self._cache:
            path = self.root / key[0] / f"{key[1]}.npy"
            if not path.is_file():
                raise HestDeepSpotHiSTError(f"DEEPSPOT_FEATURE_MISSING:{path}")
            values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
            expected_rows = DEEPSPOT_SUBSPOTS + 1
            if values.ndim != 2 or values.shape[0] != expected_rows:
                raise HestDeepSpotHiSTError(
                    f"DEEPSPOT_FEATURE_SHAPE:{path}:{values.shape}:expected_rows={expected_rows}"
                )
            if self.expected_feature_dim is not None and values.shape[1] != self.expected_feature_dim:
                raise HestDeepSpotHiSTError(
                    f"DEEPSPOT_FEATURE_DIM:{path}:{values.shape[1]}:{self.expected_feature_dim}"
                )
            if not np.isfinite(values).all():
                raise HestDeepSpotHiSTError(f"DEEPSPOT_FEATURE_NONFINITE:{path}")
            self._cache[key] = values
        return self._cache[key]


class DeepSpotSpecimenDataset(Dataset):
    """Official multiscale spot/subspot/neighbour tensors on selected HEST rows."""

    def __init__(
        self,
        specimen: AlignedHestSpecimen,
        *,
        rows: Sequence[int],
        scaled_targets: np.ndarray,
        feature_store: DeepSpotFeatureStore,
    ) -> None:
        self.specimen = specimen
        self.rows = np.asarray(rows, dtype=np.int64)
        self.targets = np.asarray(scaled_targets, dtype=np.float32)
        self.feature_store = feature_store
        if self.rows.ndim != 1 or len(self.rows) == 0 or len(set(self.rows.tolist())) != len(self.rows):
            raise HestDeepSpotHiSTError(f"DEEPSPOT_ROWS:{specimen.specimen}")
        if self.rows.min() < 0 or self.rows.max() >= len(specimen.barcodes):
            raise HestDeepSpotHiSTError(f"DEEPSPOT_ROW_RANGE:{specimen.specimen}")
        if self.targets.shape != specimen.truth.shape:
            raise HestDeepSpotHiSTError(f"DEEPSPOT_TARGET_ALIGNMENT:{self.targets.shape}")
        full_edges, _, _ = build_spatial_edges(
            specimen.spatial_coords, specimen_id=specimen.specimen
        )
        neighbors: list[list[int]] = [[] for _ in range(len(specimen.barcodes))]
        for left, right in full_edges.T:
            neighbors[int(left)].append(int(right))
            neighbors[int(right)].append(int(left))
        self.neighbors = tuple(tuple(sorted(values)) for values in neighbors)
        self.max_neighbors = max((len(values) for values in neighbors), default=0)
        if self.max_neighbors == 0:
            raise HestDeepSpotHiSTError(f"DEEPSPOT_CONTEXT_GRAPH_EMPTY:{specimen.specimen}")

    def __len__(self) -> int:
        return len(self.rows)

    def _context(self, row: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        barcode = str(self.specimen.barcodes[row])
        encoded = self.feature_store.load(self.specimen.specimen, barcode)
        main = encoded[:1]
        subspots = encoded[1:]
        neighbor_main = [
            self.feature_store.load(
                self.specimen.specimen, str(self.specimen.barcodes[neighbor])
            )[0]
            for neighbor in self.neighbors[row]
        ]
        padded = np.zeros((self.max_neighbors, encoded.shape[1]), dtype=np.float32)
        if neighbor_main:
            padded[: len(neighbor_main)] = np.stack(neighbor_main)
        return tuple(torch.from_numpy(value.copy()) for value in (main, subspots, padded))

    def __getitem__(self, index: int):
        row = int(self.rows[index])
        return {
            "x": self._context(row),
            "target": torch.from_numpy(self.targets[row].copy()),
            "barcode": str(self.specimen.barcodes[row]),
            "specimen": self.specimen.specimen,
            "source_row": row,
            "spatial_coords": torch.as_tensor(
                self.specimen.spatial_coords[row], dtype=torch.float64
            ),
        }


@dataclass(frozen=True)
class BlindHestSupport:
    """Outer-test image/coordinate support with no measured expression field."""

    task: str
    specimen: str
    genes: np.ndarray
    barcodes: np.ndarray
    spatial_coords: np.ndarray
    patch_path: Path

    def __post_init__(self) -> None:
        genes = np.asarray(self.genes, dtype=object)
        barcodes = np.asarray(self.barcodes, dtype=object)
        coords = np.asarray(self.spatial_coords, dtype=np.float64)
        if len(genes) != 50 or len(set(genes.tolist())) != 50:
            raise HestDeepSpotHiSTError(f"BLIND_GENE_COUNT:{len(genes)}")
        if barcodes.ndim != 1 or len(barcodes) == 0 or len(set(barcodes.tolist())) != len(barcodes):
            raise HestDeepSpotHiSTError(f"BLIND_BARCODES:{self.specimen}")
        if coords.shape != (len(barcodes), 2) or not np.isfinite(coords).all():
            raise HestDeepSpotHiSTError(
                f"BLIND_COORDINATES:{self.specimen}:{coords.shape}"
            )
        genes.setflags(write=False)
        barcodes.setflags(write=False)
        coords.setflags(write=False)
        object.__setattr__(self, "genes", genes)
        object.__setattr__(self, "barcodes", barcodes)
        object.__setattr__(self, "spatial_coords", coords)


def load_blind_hest_support(
    dataset_root: str | Path,
    *,
    task: str,
    specimen: str,
    genes: Sequence[str],
) -> BlindHestSupport:
    """Load held-out barcodes and coordinates without opening the RNA matrix.

    Barcode order follows the official HEST patch H5, as in the aligned
    training contract.  The h5ad handle is used only for ``obs/_index`` and
    ``obsm/spatial``; ``X`` is neither required nor accessed.
    """

    root = Path(dataset_root)
    patch_path = root / str(task) / "patches" / f"{specimen}.h5"
    if not patch_path.is_file():
        raise HestDeepSpotHiSTError(f"BLIND_PATCH_MISSING:{patch_path}")
    with h5py.File(patch_path, "r") as handle:
        barcode_keys = [key for key in ("barcodes", "barcode") if key in handle]
        if len(barcode_keys) != 1:
            raise HestDeepSpotHiSTError(
                f"BLIND_PATCH_BARCODE_KEY:{patch_path}:{barcode_keys}"
            )
        barcodes = _decode(
            np.asarray(handle[barcode_keys[0]][...]).reshape(-1)
        )

    adata_path = root / str(task) / "adata" / f"{specimen}.h5ad"
    if not adata_path.is_file():
        raise HestDeepSpotHiSTError(f"BLIND_ADATA_MISSING:{adata_path}")
    with h5py.File(adata_path, "r") as handle:
        obs = _decode(handle["obs/_index"][...])
        coords = np.asarray(handle["obsm/spatial"], dtype=np.float64)
    lookup = {barcode: index for index, barcode in enumerate(obs.tolist())}
    if len(lookup) != len(obs) or any(barcode not in lookup for barcode in barcodes):
        raise HestDeepSpotHiSTError(f"BLIND_BARCODE_JOIN:{task}:{specimen}")
    order = np.asarray([lookup[barcode] for barcode in barcodes], dtype=np.int64)
    return BlindHestSupport(
        task=str(task),
        specimen=str(specimen),
        genes=np.asarray([str(value) for value in genes], dtype=object),
        barcodes=barcodes,
        spatial_coords=coords[order],
        patch_path=patch_path,
    )


class DeepSpotBlindDataset(Dataset):
    """Published DeepSpot context for endpoint-blind outer-test inference."""

    def __init__(
        self,
        support: BlindHestSupport,
        *,
        feature_store: DeepSpotFeatureStore,
    ) -> None:
        self.support = support
        self.feature_store = feature_store
        full_edges, _, _ = build_spatial_edges(
            support.spatial_coords, specimen_id=support.specimen
        )
        neighbors: list[list[int]] = [[] for _ in range(len(support.barcodes))]
        for left, right in full_edges.T:
            neighbors[int(left)].append(int(right))
            neighbors[int(right)].append(int(left))
        self.neighbors = tuple(tuple(sorted(values)) for values in neighbors)
        self.max_neighbors = max((len(values) for values in neighbors), default=0)
        if self.max_neighbors == 0:
            raise HestDeepSpotHiSTError(
                f"DEEPSPOT_BLIND_CONTEXT_GRAPH_EMPTY:{support.specimen}"
            )

    def __len__(self) -> int:
        return len(self.support.barcodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        barcode = str(self.support.barcodes[index])
        encoded = self.feature_store.load(self.support.specimen, barcode)
        neighbor_main = [
            self.feature_store.load(
                self.support.specimen, str(self.support.barcodes[neighbor])
            )[0]
            for neighbor in self.neighbors[index]
        ]
        padded = np.zeros(
            (self.max_neighbors, encoded.shape[1]), dtype=np.float32
        )
        if neighbor_main:
            padded[: len(neighbor_main)] = np.stack(neighbor_main)
        return {
            "x": tuple(
                torch.from_numpy(value.copy())
                for value in (encoded[:1], encoded[1:], padded)
            ),
            "barcode": barcode,
            "source_row": int(index),
        }


class DeepSpotEdgePairDataset(Dataset):
    """Explicit training-only adjacent pairs over one specimen dataset."""

    def __init__(self, spots: DeepSpotSpecimenDataset, edge_index: np.ndarray) -> None:
        self.spots = spots
        edges = np.asarray(edge_index, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] == 0:
            raise HestDeepSpotHiSTError(f"DEEPSPOT_EDGES:{edges.shape}")
        if edges.min() < 0 or edges.max() >= len(spots):
            raise HestDeepSpotHiSTError("DEEPSPOT_EDGE_RANGE")
        self.edge_index = edges

    def __len__(self) -> int:
        return self.edge_index.shape[1]

    def __getitem__(self, index: int):
        return {
            "left": self.spots[int(self.edge_index[0, index])],
            "right": self.spots[int(self.edge_index[1, index])],
            "edge_number": int(index),
        }


def build_deepspot_train_dataset(
    specimen: AlignedHestSpecimen,
    split: SpecimenInnerSplit,
    *,
    scaled_targets: np.ndarray,
    feature_store: DeepSpotFeatureStore,
) -> tuple[DeepSpotSpecimenDataset, np.ndarray]:
    train_rows = np.flatnonzero(split.train_mask)
    spots = DeepSpotSpecimenDataset(
        specimen,
        rows=train_rows,
        scaled_targets=scaled_targets,
        feature_store=feature_store,
    )
    edge_index, _, _ = build_spatial_edges(
        specimen.spatial_coords[train_rows], specimen_id=specimen.specimen
    )
    return spots, edge_index


def _stack_deepspot_context(items: Sequence[dict[str, Any]]) -> list[torch.Tensor]:
    if not items:
        raise HestDeepSpotHiSTError("DEEPSPOT_CONTEXT_BATCH_EMPTY")
    main = torch.stack([item["x"][0] for item in items])
    subspots = torch.stack([item["x"][1] for item in items])
    neighbor_width = max(int(item["x"][2].shape[0]) for item in items)
    feature_dim = int(items[0]["x"][2].shape[1])
    neighbors = torch.zeros(
        (len(items), neighbor_width, feature_dim), dtype=items[0]["x"][2].dtype
    )
    for index, item in enumerate(items):
        value = item["x"][2]
        if value.shape[1] != feature_dim:
            raise HestDeepSpotHiSTError("DEEPSPOT_BATCH_FEATURE_DIM")
        neighbors[index, : value.shape[0]] = value
    return [main, subspots, neighbors]


def collate_deepspot_edge_pairs(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise HestDeepSpotHiSTError("DEEPSPOT_PAIR_BATCH_EMPTY")

    def endpoint(name: str):
        items = [sample[name] for sample in samples]
        target = torch.stack([item["target"] for item in items])
        return target, items

    left_target, left_items = endpoint("left")
    right_target, right_items = endpoint("right")
    x = _stack_deepspot_context(left_items + right_items)
    target = torch.cat((left_target, right_target), dim=0)
    pairs = len(samples)
    edge_index = torch.stack(
        (torch.arange(pairs, dtype=torch.long), torch.arange(pairs, 2 * pairs, dtype=torch.long))
    )
    return {
        "upstream_batch": (x, target),
        "edge_index": edge_index,
        "left_barcodes": [item["barcode"] for item in left_items],
        "right_barcodes": [item["barcode"] for item in right_items],
        "edge_numbers": torch.as_tensor([sample["edge_number"] for sample in samples]),
    }


def collate_deepspot_spots(samples: Sequence[dict[str, Any]]):
    """Return the exact ``(X, y)`` structure consumed by upstream DeepSpot."""

    if not samples:
        raise HestDeepSpotHiSTError("DEEPSPOT_SPOT_BATCH_EMPTY")
    x = _stack_deepspot_context(samples)
    target = torch.stack([sample["target"] for sample in samples])
    return x, target


def collate_deepspot_blind(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise HestDeepSpotHiSTError("DEEPSPOT_BLIND_BATCH_EMPTY")
    return {
        "x": _stack_deepspot_context(samples),
        "barcodes": [str(sample["barcode"]) for sample in samples],
        "source_rows": torch.as_tensor(
            [int(sample["source_row"]) for sample in samples], dtype=torch.long
        ),
    }


class HiSTFeatureStore:
    """Barcode-exact precomputed UNI feature files for the released HiST mode."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def aligned(self, specimen: AlignedHestSpecimen) -> np.ndarray:
        name = specimen.specimen
        if name not in self._cache:
            path = self.root / f"{name}.h5"
            if not path.is_file():
                raise HestDeepSpotHiSTError(f"HIST_FEATURE_MISSING:{path}")
            with h5py.File(path, "r") as handle:
                if "mor_feats" not in handle or "barcodes" not in handle:
                    raise HestDeepSpotHiSTError(
                        f"HIST_FEATURE_KEYS:{path}:required=mor_feats,barcodes"
                    )
                features = np.asarray(handle["mor_feats"], dtype=np.float32)
                barcodes = _decode(handle["barcodes"][...])
            if features.shape != (len(barcodes), HIST_FEATURE_DIM) or not np.isfinite(features).all():
                raise HestDeepSpotHiSTError(f"HIST_FEATURE_SHAPE:{path}:{features.shape}")
            if len(set(barcodes.tolist())) != len(barcodes):
                raise HestDeepSpotHiSTError(f"HIST_FEATURE_DUPLICATE_BARCODE:{path}")
            self._cache[name] = (barcodes, features)
        barcodes, features = self._cache[name]
        lookup = {value: index for index, value in enumerate(barcodes.tolist())}
        requested = [str(value) for value in specimen.barcodes]
        if any(value not in lookup for value in requested):
            raise HestDeepSpotHiSTError(f"HIST_FEATURE_BARCODE_JOIN:{name}")
        return features[np.asarray([lookup[value] for value in requested], dtype=np.int64)]


def add_hist_to_path(deps_root: str | Path) -> Path:
    root = Path(deps_root).resolve()
    if not (root / "tasks" / "gene_prediction" / "model.py").is_file():
        raise HestDeepSpotHiSTError(f"HIST_SOURCE_MISSING:{root}")
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


def _load_hist_spatial_utils(deps_root: str | Path):
    """Load released coordinate utilities without importing optional DGL layers."""

    root = add_hist_to_path(deps_root)
    path = root / "src" / "span" / "preprocessing" / "spatial_utils.py"
    spec = importlib.util.spec_from_file_location("g0055_hist_spatial_utils", path)
    if spec is None or spec.loader is None:
        raise HestDeepSpotHiSTError(f"HIST_SPATIAL_UTILS_LOAD:{path}")
    module = importlib.util.module_from_spec(spec)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return module


def hist_lattice_coordinates(coords: np.ndarray, *, deps_root: str | Path) -> torch.Tensor:
    """Use HiST's released collision-free integer mapper after spacing normalization."""

    spatial_utils = _load_hist_spatial_utils(deps_root)
    points = np.asarray(coords, dtype=np.float64)
    _, spacing, _ = build_spatial_edges(points, specimen_id="hist_lattice")
    continuous = (points - points.min(axis=0, keepdims=True)) / float(spacing)
    lattice = spatial_utils.map_to_integer_grid(
        torch.as_tensor(continuous, dtype=torch.float32)
    )
    if lattice.shape != (len(points), 2) or lattice.min().item() < 0:
        raise HestDeepSpotHiSTError(f"HIST_LATTICE:{tuple(lattice.shape)}")
    if len({tuple(value) for value in lattice.cpu().tolist()}) != len(points):
        raise HestDeepSpotHiSTError("HIST_LATTICE_COLLISION")
    return lattice.cpu().long()


class HiSTPartitionDataset(Dataset):
    """Whole-specimen HiST context restricted to one frozen RNA partition."""

    def __init__(
        self,
        specimens: Sequence[AlignedHestSpecimen],
        splits: Sequence[SpecimenInnerSplit],
        *,
        validation: bool,
        feature_store: HiSTFeatureStore,
        deps_root: str | Path,
        coordinate_transform: Callable[[torch.Tensor], Any] | None = None,
    ) -> None:
        if not specimens or len(specimens) != len(splits):
            raise HestDeepSpotHiSTError("HIST_PARTITION_COUNT")
        for specimen, split in zip(specimens, splits, strict=True):
            if specimen.specimen != split.specimen:
                raise HestDeepSpotHiSTError("HIST_PARTITION_ORDER")
        self.specimens = tuple(specimens)
        self.splits = tuple(splits)
        self.validation = bool(validation)
        self.feature_store = feature_store
        self.deps_root = Path(deps_root)
        self.coordinate_transform = coordinate_transform

    def __len__(self) -> int:
        return len(self.specimens)

    def set_aug_multiplier(self, multiplier: float) -> None:
        for transform in getattr(self.coordinate_transform, "transforms", ()):
            if hasattr(transform, "set_multiplier"):
                transform.set_multiplier(float(multiplier))

    def __getitem__(self, index: int) -> SimpleNamespace:
        specimen = self.specimens[index]
        split = self.splits[index]
        mask = split.validation_mask if self.validation else split.train_mask
        original_rows = np.flatnonzero(mask)
        features = self.feature_store.aligned(specimen)[original_rows]
        raw = np.asarray(specimen.raw_counts, dtype=np.float32)[original_rows]
        full_physical = np.asarray(specimen.spatial_coords, dtype=np.float64)
        physical = full_physical[original_rows]
        barcodes = np.asarray(specimen.barcodes, dtype=object)[original_rows]
        lattice = hist_lattice_coordinates(full_physical, deps_root=self.deps_root)[
            torch.as_tensor(original_rows, dtype=torch.long)
        ]

        keep = None
        if not self.validation and self.coordinate_transform is not None:
            transformed = self.coordinate_transform(lattice.float())
            if isinstance(transformed, tuple):
                lattice, keep = transformed
            else:
                lattice = transformed
            if keep is not None:
                keep_np = np.asarray(keep.detach().cpu(), dtype=bool)
                features = features[keep_np]
                raw = raw[keep_np]
                physical = physical[keep_np]
                barcodes = barcodes[keep_np]
                original_rows = original_rows[keep_np]

        spatial_utils = _load_hist_spatial_utils(self.deps_root)
        order = (
            spatial_utils.get_sorted_indices(lattice)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        lattice = lattice[torch.as_tensor(order, dtype=torch.long)].long()
        features = features[order]
        raw = raw[order]
        physical = physical[order]
        barcodes = barcodes[order]
        original_rows = original_rows[order]

        if self.validation:
            edge_index = np.empty((2, 0), dtype=np.int64)
        else:
            train_rows = np.flatnonzero(split.train_mask)
            frozen_edges, _, _ = build_spatial_edges(
                specimen.spatial_coords[train_rows], specimen_id=specimen.specimen
            )
            frozen_original = train_rows[frozen_edges]
            lookup = {int(row): local for local, row in enumerate(original_rows.tolist())}
            kept = [
                (lookup[int(left)], lookup[int(right)])
                for left, right in frozen_original.T
                if int(left) in lookup and int(right) in lookup
            ]
            if not kept:
                raise HestDeepSpotHiSTError(f"HIST_TRAIN_GRAPH_EMPTY:{specimen.specimen}")
            edge_index = np.asarray(kept, dtype=np.int64).T

        return SimpleNamespace(
            slide_name=specimen.specimen,
            barcodes=barcodes.tolist(),
            mor_feats=torch.from_numpy(features.copy()).float(),
            mol_feats=torch.from_numpy(raw.copy()).float(),
            cords=lattice,
            final_cords=lattice,
            orig_cords=torch.as_tensor(physical, dtype=torch.float64),
            g0055_edge_index=torch.from_numpy(edge_index).long(),
            g0055_original_rows=torch.as_tensor(original_rows, dtype=torch.long),
        )


def build_hist_blind_batch(
    support: BlindHestSupport,
    *,
    feature_store: HiSTFeatureStore,
    deps_root: str | Path,
) -> SimpleNamespace:
    """Build released whole-context HiST input with target-independent dummies."""

    features = feature_store.aligned(support)
    lattice = hist_lattice_coordinates(
        support.spatial_coords, deps_root=deps_root
    )
    spatial_utils = _load_hist_spatial_utils(deps_root)
    order = (
        spatial_utils.get_sorted_indices(lattice)
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    ordered = torch.as_tensor(order, dtype=torch.long)
    return SimpleNamespace(
        slide_name=support.specimen,
        barcodes=support.barcodes[order].tolist(),
        mor_feats=torch.from_numpy(features[order].copy()).float(),
        # Upstream PredictionModel computes an MSE even at inference.  These
        # zeros satisfy that signature; predictions do not depend on mol_feats.
        mol_feats=torch.zeros((len(order), 50), dtype=torch.float32),
        cords=lattice[ordered].long(),
        final_cords=lattice[ordered].long(),
        orig_cords=torch.as_tensor(
            support.spatial_coords[order], dtype=torch.float64
        ),
        g0055_original_rows=ordered,
        g0055_measured_rna_loaded=False,
    )
