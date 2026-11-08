"""Spatial metadata and explicit edge-pair loading for upstream sCellST bags."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .split import FrozenSpotSplit, freeze_or_load_spot_split


SPOT_ID_KEY = "g0055_spot_id"
SPOT_COORDS_KEY = "g0055_spot_coords"
SPECIMEN_ID_KEY = "g0055_specimen_id"


class SpatialBagDataset(Dataset):
    """Add aligned spot identity and coordinates without changing upstream items."""

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        spot_ids: Sequence[str],
        coords: np.ndarray,
        specimen_ids: Sequence[str],
    ) -> None:
        self.base_dataset = base_dataset
        self.spot_ids = np.asarray([str(value) for value in spot_ids], dtype=object)
        self.coords = np.asarray(coords, dtype=np.float64)
        self.specimen_ids = np.asarray([str(value) for value in specimen_ids], dtype=object)
        n_items = len(base_dataset)
        if len(self.spot_ids) != n_items or len(self.specimen_ids) != n_items:
            raise ValueError("Spatial metadata must align one-to-one with the base dataset")
        if self.coords.shape != (n_items, 2) or not np.isfinite(self.coords).all():
            raise ValueError(f"Expected finite coordinates with shape {(n_items, 2)}, got {self.coords.shape}")
        if len(set(self.spot_ids.tolist())) != n_items:
            raise ValueError("SpatialBagDataset requires unique spot IDs")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base_dataset[index])
        item[SPOT_ID_KEY] = str(self.spot_ids[index])
        item[SPOT_COORDS_KEY] = torch.as_tensor(self.coords[index], dtype=torch.float64)
        item[SPECIMEN_ID_KEY] = str(self.specimen_ids[index])
        return item


def collate_spatial_bags(
    batch: list[dict[str, Any]], base_collate: Callable[[list[dict[str, Any]]], dict[str, Any]]
) -> dict[str, Any]:
    """Run the unmodified sCellST collate, then restore spatial metadata."""

    if not batch:
        raise ValueError("Cannot collate an empty spatial bag batch")
    collated = base_collate(batch)
    collated[SPOT_ID_KEY] = [str(item[SPOT_ID_KEY]) for item in batch]
    collated[SPOT_COORDS_KEY] = torch.stack([item[SPOT_COORDS_KEY] for item in batch])
    collated[SPECIMEN_ID_KEY] = [str(item[SPECIMEN_ID_KEY]) for item in batch]
    return collated


def scellst_spatial_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pickle-safe production collate with a lazy upstream import."""

    from scellst.dataset.dataset_utils import custom_collate

    return collate_spatial_bags(batch, custom_collate)


class EdgePairDataset(Dataset):
    """Explicit adjacent-spot pairs over a training-only spatial dataset."""

    def __init__(self, train_dataset: Dataset, edge_index: np.ndarray) -> None:
        self.train_dataset = train_dataset
        edges = np.asarray(edge_index, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] == 0:
            raise ValueError(f"edge_index must have shape [2, E] with E > 0, got {edges.shape}")
        if edges.min() < 0 or edges.max() >= len(train_dataset):
            raise ValueError("edge_index contains an endpoint outside the training dataset")
        if np.any(edges[0] == edges[1]):
            raise ValueError("Self edges are not valid adjacent-spot pairs")
        canonical = np.sort(edges, axis=0)
        if len({(int(a), int(b)) for a, b in canonical.T}) != edges.shape[1]:
            raise ValueError("Duplicate undirected edges are not allowed")
        self.edge_index = canonical

    def __len__(self) -> int:
        return self.edge_index.shape[1]

    def __getitem__(self, index: int) -> dict[str, dict[str, Any]]:
        left = int(self.edge_index[0, index])
        right = int(self.edge_index[1, index])
        return {
            "left": self.train_dataset[left],
            "right": self.train_dataset[right],
        }


def collate_edge_pairs(
    batch: list[dict[str, dict[str, Any]]],
    base_collate: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not batch:
        raise ValueError("Cannot collate an empty edge-pair batch")

    def collate_endpoint(items: list[dict[str, Any]]) -> dict[str, Any]:
        collated = collate_spatial_bags(items, base_collate)
        # Upstream custom_collate derives the bag pointer with
        # unique_consecutive(original_spot_id).  An edge batch may contain the
        # same endpoint in consecutive pairs, where that operation merges two
        # legitimate pair occurrences and returns one prediction for two
        # targets.  Edge occurrences are the bags here, so assign a fresh local
        # bag index to every occurrence while retaining the upstream item data.
        if "ptr_idx_bag_inst" in collated:
            instance_counts = torch.as_tensor(
                [len(item["X"]) for item in items], dtype=torch.long
            )
            collated["ptr_idx_bag_inst"] = torch.repeat_interleave(
                torch.arange(len(items), dtype=torch.long), instance_counts
            )
        return collated

    return {
        "left": collate_endpoint([item["left"] for item in batch]),
        "right": collate_endpoint([item["right"] for item in batch]),
    }


def scellst_edge_pair_collate(
    batch: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Pickle-safe production edge collate with a lazy upstream import."""

    from scellst.dataset.dataset_utils import custom_collate

    return collate_edge_pairs(batch, custom_collate)


def _spatial_metadata(dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(dataset, SpatialBagDataset):
        indices = np.arange(len(dataset), dtype=np.int64)
        return dataset.spot_ids.copy(), dataset.coords.copy(), dataset.specimen_ids.copy()
    if isinstance(dataset, Subset):
        ids, coords, specimens = _spatial_metadata(dataset.dataset)
        indices = np.asarray(dataset.indices, dtype=np.int64)
        return ids[indices], coords[indices], specimens[indices]
    raise TypeError(
        "Spatial metadata is only available from SpatialBagDataset or a Subset wrapping it"
    )


def build_training_adjacency(train_dataset: Dataset, radius_multiplier: float = 1.5):
    """Build spatial adjacency using training rows only."""

    from strsc.training.shared.adjacency import build_within_specimen_adjacency

    _, coords, specimen_ids = _spatial_metadata(train_dataset)
    ordered_specimens = list(dict.fromkeys(specimen_ids.tolist()))
    coordinate_blocks: list[np.ndarray] = []
    concatenated_row_order: list[int] = []
    for specimen_id in ordered_specimens:
        rows = np.flatnonzero(specimen_ids == specimen_id)
        coordinate_blocks.append(coords[rows])
        concatenated_row_order.extend(rows.tolist())
    if concatenated_row_order != list(range(len(train_dataset))):
        raise ValueError(
            "Training rows from multiple specimens must be contiguous before building global edge indices"
        )
    return build_within_specimen_adjacency(
        coordinate_blocks,
        specimen_ids=ordered_specimens,
        radius_multiplier=radius_multiplier,
    )


@dataclass(frozen=True)
class PreparedSpatialData:
    train_dataset: Subset
    val_dataset: Subset
    split: FrozenSpotSplit
    spot_ids: tuple[str, ...]
    coords: np.ndarray
    specimen_ids: tuple[str, ...]


def freeze_scellst_tenx39_split(
    data_module: Any,
    *,
    manifest_path: str | Path,
    split_seed: int = 0,
    frac_train: float = 0.8,
) -> PreparedSpatialData:
    """Install the explicit TENX39 split on a prepared upstream STDataModule.

    Call this after ``data_module.prepare_data()`` and instead of upstream
    ``data_module.setup(stage="fit")``.  This adapter supports only
    TENX39 and the current ``no_scaling`` sCellST configuration.
    """

    training_ids = [str(value) for value in data_module.list_training_ids]
    if training_ids != ["TENX39"]:
        raise ValueError(f"G0-055 sCellST requires exactly TENX39, got {training_ids}")
    if getattr(data_module, "scale", "no_scaling") != "no_scaling":
        raise ValueError("G0-055 sCellST overlay currently requires the frozen no_scaling contract")
    if not hasattr(data_module, "adata_dict") or "TENX39" not in data_module.adata_dict:
        raise RuntimeError("Call data_module.prepare_data() before freezing the G0-055 split")

    adata = data_module.adata_dict["TENX39"]
    if "spatial" not in adata.obsm:
        raise RuntimeError("TENX39 AnnData is missing obsm['spatial']")
    base_dataset = data_module.dataset_handler.create_dataset(adata, data_module.data_dir)
    if len(base_dataset) != adata.n_obs:
        raise RuntimeError(
            f"sCellST dataset/AnnData row mismatch: {len(base_dataset)} != {adata.n_obs}"
        )

    spot_ids = tuple(str(value) for value in adata.obs_names)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    specimen_ids = tuple("TENX39" for _ in spot_ids)
    full_dataset = SpatialBagDataset(
        base_dataset,
        spot_ids=spot_ids,
        coords=coords,
        specimen_ids=specimen_ids,
    )
    split = freeze_or_load_spot_split(
        spot_ids=spot_ids,
        coords=coords,
        specimen_ids=specimen_ids,
        manifest_path=manifest_path,
        seed=split_seed,
        frac_train=frac_train,
    )
    train_dataset = Subset(full_dataset, list(split.train_indices))
    val_dataset = Subset(full_dataset, list(split.val_indices))

    # Install only the datasets/collate used by the existing train/val loader methods.
    data_module.train_dataset = train_dataset
    data_module.val_dataset = val_dataset
    data_module.custom_collate = scellst_spatial_collate
    data_module.adata = adata
    return PreparedSpatialData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        split=split,
        spot_ids=spot_ids,
        coords=coords,
        specimen_ids=specimen_ids,
    )


class PairedSpotEdgeLoader(Iterable[dict[str, Any]]):
    """Preserve the ordinary spot stream while cycling an explicit edge stream."""

    def __init__(self, spot_loader: DataLoader, edge_loader: DataLoader) -> None:
        if len(spot_loader) == 0 or len(edge_loader) == 0:
            raise ValueError("Both spot and edge loaders must be non-empty")
        self.spot_loader = spot_loader
        self.edge_loader = edge_loader

    def __len__(self) -> int:
        return len(self.spot_loader)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        edge_iterator = iter(self.edge_loader)
        for spot_batch in self.spot_loader:
            try:
                edge_batch = next(edge_iterator)
            except StopIteration:
                edge_iterator = iter(self.edge_loader)
                edge_batch = next(edge_iterator)
            yield {"spot": spot_batch, "edge": edge_batch}
