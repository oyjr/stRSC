"""Train-only edge pairs and context batches for the four HEST adapters."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from strsc.training.shared.adjacency import build_spatial_edges

from .data import (
    AlignedHestSpecimen,
    HestAdapterError,
    HestSpotDataset,
    load_specimen_batch,
)
from .split import SpecimenInnerSplit


@dataclass(frozen=True, order=True)
class TrainingEdge:
    specimen_index: int
    left_global: int
    right_global: int


@dataclass(frozen=True)
class FrozenTrainingEdges:
    edges: tuple[TrainingEdge, ...]
    supervised_global_indices: tuple[int, ...]
    spacing_by_specimen: tuple[float, ...]
    radius_by_specimen: tuple[float, ...]

    def __post_init__(self) -> None:
        supervised = set(self.supervised_global_indices)
        if not self.edges:
            raise HestAdapterError("TRAINING_GRAPH_EMPTY")
        for edge in self.edges:
            if edge.left_global >= edge.right_global:
                raise HestAdapterError("TRAINING_EDGE_ORDER")
            if edge.left_global not in supervised or edge.right_global not in supervised:
                raise HestAdapterError("TRAINING_EDGE_NOT_SUPERVISED")


def build_frozen_training_edges(
    dataset: HestSpotDataset,
    splits: Sequence[SpecimenInnerSplit],
) -> FrozenTrainingEdges:
    if len(dataset.specimens) != len(splits):
        raise HestAdapterError("INNER_SPLIT_SPECIMEN_COUNT")
    global_blocks = dataset.specimen_global_indices()
    edges: list[TrainingEdge] = []
    supervised: list[int] = []
    spacings: list[float] = []
    radii: list[float] = []
    for specimen_index, (specimen, split, block) in enumerate(
        zip(dataset.specimens, splits, global_blocks, strict=True)
    ):
        if specimen.specimen != split.specimen:
            raise HestAdapterError(
                f"INNER_SPLIT_SPECIMEN_ORDER:{specimen.specimen}:{split.specimen}"
            )
        train_rows = np.flatnonzero(split.train_mask)
        local_edge_index, spacing, radius = build_spatial_edges(
            specimen.spatial_coords[train_rows], specimen_id=specimen.specimen
        )
        train_globals = block[train_rows]
        supervised.extend(int(value) for value in train_globals)
        for column in range(local_edge_index.shape[1]):
            left = int(train_globals[local_edge_index[0, column]])
            right = int(train_globals[local_edge_index[1, column]])
            if left > right:
                left, right = right, left
            edges.append(TrainingEdge(specimen_index, left, right))
        spacings.append(float(spacing))
        radii.append(float(radius))
    return FrozenTrainingEdges(
        edges=tuple(sorted(edges)),
        supervised_global_indices=tuple(sorted(supervised)),
        spacing_by_specimen=tuple(spacings),
        radius_by_specimen=tuple(radii),
    )


class EdgePairDataset(Dataset):
    def __init__(self, spots: HestSpotDataset, graph: FrozenTrainingEdges) -> None:
        self.spots = spots
        self.graph = graph

    def __len__(self) -> int:
        return len(self.graph.edges)

    def __getitem__(self, index: int) -> dict[str, Any]:
        edge = self.graph.edges[index]
        return {
            "left": self.spots[edge.left_global],
            "right": self.spots[edge.right_global],
            "edge_number": int(index),
            "specimen_index": int(edge.specimen_index),
            "left_global": int(edge.left_global),
            "right_global": int(edge.right_global),
        }


class DualStreamPatchDataset(Dataset):
    """Attach an independently frozen edge pair to each ordinary spot.

    Dataset positions and length remain those of ``ordinary_dataset``.  A
    DataLoader using the same shuffle seed therefore emits the same ordinary
    spots as the baseline arm; the edge stream changes only the intervention
    objective and never degree-weights the ordinary MSE.
    """

    def __init__(
        self,
        ordinary_dataset: Dataset,
        edge_dataset: EdgePairDataset,
        *,
        edge_seed: int,
    ) -> None:
        if len(ordinary_dataset) < 1 or len(edge_dataset) < 1:
            raise HestAdapterError("DUAL_STREAM_EMPTY")
        self.ordinary_dataset = ordinary_dataset
        self.edge_dataset = edge_dataset
        generator = np.random.default_rng(int(edge_seed))
        permutation = generator.permutation(len(edge_dataset))
        repeats = int(np.ceil(len(ordinary_dataset) / len(permutation)))
        self.edge_assignment = np.tile(permutation, repeats)[: len(ordinary_dataset)]

    def __len__(self) -> int:
        return len(self.ordinary_dataset)

    def __getitem__(self, position: int) -> dict[str, Any]:
        return {
            "ordinary": self.ordinary_dataset[position],
            "edge": self.edge_dataset[int(self.edge_assignment[position])],
            "ordinary_position": int(position),
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_edge_pair_loader(
    spots: HestSpotDataset,
    graph: FrozenTrainingEdges,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        EdgePairDataset(spots, graph),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
        generator=generator,
        worker_init_fn=_seed_worker,
        collate_fn=collate_edge_pairs,
        persistent_workers=bool(num_workers),
    )


def make_partition_spot_loader(
    spots: HestSpotDataset,
    splits: Sequence[SpecimenInnerSplit],
    *,
    validation: bool,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    if len(spots.specimens) != len(splits):
        raise HestAdapterError("PARTITION_SPLIT_SPECIMEN_COUNT")
    indices: list[int] = []
    offset = 0
    for specimen, split in zip(spots.specimens, splits, strict=True):
        if specimen.specimen != split.specimen:
            raise HestAdapterError("PARTITION_SPLIT_SPECIMEN_ORDER")
        mask = split.validation_mask if validation else split.train_mask
        indices.extend((offset + np.flatnonzero(mask)).tolist())
        offset += len(specimen.barcodes)
    if not indices:
        raise HestAdapterError("PARTITION_SPOT_LOADER_EMPTY")
    return DataLoader(
        Subset(spots, indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
    )


def _partition_global_indices(
    spots: HestSpotDataset,
    splits: Sequence[SpecimenInnerSplit],
    *,
    validation: bool,
) -> tuple[int, ...]:
    if len(spots.specimens) != len(splits):
        raise HestAdapterError("PARTITION_SPLIT_SPECIMEN_COUNT")
    indices: list[int] = []
    offset = 0
    for specimen, split in zip(spots.specimens, splits, strict=True):
        if specimen.specimen != split.specimen:
            raise HestAdapterError("PARTITION_SPLIT_SPECIMEN_ORDER")
        mask = split.validation_mask if validation else split.train_mask
        indices.extend((offset + np.flatnonzero(mask)).tolist())
        offset += len(specimen.barcodes)
    if not indices:
        raise HestAdapterError("PARTITION_SPOT_LOADER_EMPTY")
    return tuple(int(value) for value in indices)


@dataclass(frozen=True)
class MatchedPatchTrainingLoaders:
    """Patch-model loaders with an identical ordinary stream across arms."""

    baseline: DataLoader
    intervention: DataLoader


def make_matched_patch_training_loaders(
    spots: HestSpotDataset,
    splits: Sequence[SpecimenInnerSplit],
    graph: FrozenTrainingEdges,
    *,
    batch_size: int,
    ordinary_seed: int,
    edge_seed: int,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = False,
) -> MatchedPatchTrainingLoaders:
    """Build formal ST-Net/DeepSpaCE streams without changing spot weights."""

    ordinary_indices = _partition_global_indices(
        spots, splits, validation=False
    )
    if tuple(sorted(ordinary_indices)) != graph.supervised_global_indices:
        raise HestAdapterError("PATCH_GRAPH_TRAIN_PARTITION_MISMATCH")
    ordinary = Subset(spots, ordinary_indices)
    dual = DualStreamPatchDataset(
        ordinary,
        EdgePairDataset(spots, graph),
        edge_seed=edge_seed,
    )
    resolved_pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)

    def loader(dataset: Dataset) -> DataLoader:
        generator = torch.Generator().manual_seed(int(ordinary_seed))
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=resolved_pin,
            drop_last=bool(drop_last),
            generator=generator,
            worker_init_fn=_seed_worker,
            persistent_workers=bool(num_workers),
        )

    return MatchedPatchTrainingLoaders(
        baseline=loader(ordinary),
        intervention=loader(dual),
    )


def _concat(left: Any, right: Any, key: str) -> Any:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.cat((left, right), dim=0)
    if isinstance(left, list) and isinstance(right, list):
        return left + right
    if isinstance(left, tuple) and isinstance(right, tuple):
        return left + right
    raise TypeError(f"PAIR_FIELD:{key}:{type(left).__name__}:{type(right).__name__}")


def flatten_edge_pair_batch(pair_batch: Mapping[str, Any]) -> dict[str, Any]:
    left = pair_batch.get("left")
    right = pair_batch.get("right")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise HestAdapterError("PAIR_ENDPOINT_MAPPING")
    if set(left) != set(right):
        raise HestAdapterError("PAIR_ENDPOINT_FIELDS")
    flat = {key: _concat(left[key], right[key], key) for key in left}
    pair_count = int(flat["target"].shape[0] // 2)
    flat["edge_index"] = torch.stack(
        (
            torch.arange(pair_count, dtype=torch.long),
            torch.arange(pair_count, 2 * pair_count, dtype=torch.long),
        )
    )
    flat["loss_mask"] = torch.ones(2 * pair_count, dtype=torch.bool)
    flat["context_adjacency"] = torch.eye(2 * pair_count, dtype=torch.float32)
    return flat


def collate_edge_pairs(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Top-level picklable collator used by formal multi-worker loaders."""

    return flatten_edge_pair_batch(default_collate(samples))


def build_context_partition_batch(
    specimen: AlignedHestSpecimen,
    split: SpecimenInnerSplit,
    *,
    validation: bool,
    rows: Sequence[int] | None = None,
    expected_patch_pixels: int | None = 224,
    context_neighbors: int = 8,
    include_local_edges: bool = True,
) -> dict[str, Any]:
    """Build one partition-only slide batch for an unchanged upstream loss.

    HisToGene and Hist2ST cannot mask locations in their published
    ``training_step``.  We therefore form the ordinary slide from inner-train
    rows only and the validation slide from inner-validation rows only.  This
    lets rho zero call upstream exactly and makes it structurally impossible
    for validation RNA to enter a training input or loss.
    """

    full_mask = split.validation_mask if validation else split.train_mask
    selected = (
        np.flatnonzero(full_mask).astype(np.int64)
        if rows is None
        else np.asarray(rows, dtype=np.int64)
    )
    if selected.ndim != 1 or len(selected) < 1:
        raise HestAdapterError(
            f"CONTEXT_PARTITION_EMPTY:{specimen.specimen}:validation={validation}"
        )
    if int(selected.min()) < 0 or int(selected.max()) >= len(full_mask):
        raise HestAdapterError("CONTEXT_PARTITION_ROW_RANGE")
    if not np.all(full_mask[selected]):
        raise HestAdapterError("CONTEXT_PARTITION_CROSSES_INNER_SPLIT")
    batch = load_specimen_batch(
        specimen,
        rows=selected,
        expected_patch_pixels=expected_patch_pixels,
        context_neighbors=context_neighbors,
    )
    batch["loss_mask"] = torch.ones(len(selected), dtype=torch.bool)

    if include_local_edges and not validation:
        local_edges, _, _ = build_spatial_edges(
            batch["spatial_coords"].numpy(),
            specimen_id=specimen.specimen,
        )
        batch["edge_index"] = torch.from_numpy(np.asarray(local_edges).copy()).long()
    return batch


def partition_context_rows(
    rows: Sequence[int] | np.ndarray,
    *,
    num_subsets: int,
    shuffle: bool,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Split one slide partition the same way Hist2ST's upstream loader does."""

    selected = np.asarray(rows, dtype=np.int64).copy()
    subsets = int(num_subsets)
    if selected.ndim != 1 or len(selected) < 2:
        raise HestAdapterError(f"CONTEXT_SUBSET_ROWS:{selected.shape}")
    if subsets < 1:
        raise HestAdapterError(f"CONTEXT_SUBSET_COUNT:{subsets}")
    if subsets == 1:
        return (selected,)
    if len(selected) // subsets < 2:
        raise HestAdapterError(
            f"CONTEXT_SUBSET_TOO_SMALL:rows={len(selected)}:subsets={subsets}"
        )
    if shuffle:
        np.random.RandomState(int(seed)).shuffle(selected)
    subset_size = len(selected) // subsets
    chunks = [
        selected[index * subset_size : (index + 1) * subset_size]
        for index in range(subsets - 1)
    ]
    chunks.append(selected[(subsets - 1) * subset_size :])
    return tuple(chunks)


class HestContextDataset(Dataset):
    """Partition-only slide items, optionally using Hist2ST's official subsets."""

    def __init__(
        self,
        specimens: Sequence[AlignedHestSpecimen],
        splits: Sequence[SpecimenInnerSplit],
        *,
        validation: bool,
        seed: int,
        num_subsets: int = 1,
        expected_patch_pixels: int | None = 224,
        context_neighbors: int = 8,
        include_local_edges: bool = True,
    ) -> None:
        if len(specimens) != len(splits) or not specimens:
            raise HestAdapterError("CONTEXT_DATASET_SPECIMEN_COUNT")
        for specimen, split in zip(specimens, splits, strict=True):
            if specimen.specimen != split.specimen:
                raise HestAdapterError("CONTEXT_DATASET_SPECIMEN_ORDER")
        blocks: list[np.ndarray] = []
        offset = 0
        for specimen in specimens:
            block = np.arange(offset, offset + len(specimen.barcodes), dtype=np.int64)
            blocks.append(block)
            offset += len(specimen.barcodes)
        self.specimen_global_indices = tuple(blocks)
        self.items: list[
            tuple[int, int, AlignedHestSpecimen, SpecimenInnerSplit, np.ndarray]
        ] = []
        for specimen_index, (specimen, split) in enumerate(
            zip(specimens, splits, strict=True)
        ):
            mask = split.validation_mask if validation else split.train_mask
            chunks = partition_context_rows(
                np.flatnonzero(mask),
                num_subsets=int(num_subsets),
                shuffle=not validation and int(num_subsets) > 1,
                seed=int(seed) + specimen_index,
            )
            self.items.extend(
                (specimen_index, subset_index, specimen, split, rows)
                for subset_index, rows in enumerate(chunks)
            )
        self.validation = bool(validation)
        self.expected_patch_pixels = expected_patch_pixels
        self.context_neighbors = int(context_neighbors)
        self.include_local_edges = bool(include_local_edges)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        _, _, specimen, split, rows = self.items[index]
        return build_context_partition_batch(
            specimen,
            split,
            validation=self.validation,
            rows=rows,
            expected_patch_pixels=self.expected_patch_pixels,
            context_neighbors=self.context_neighbors,
            include_local_edges=self.include_local_edges,
        )

    @property
    def item_keys(self) -> tuple[tuple[int, int], ...]:
        return tuple((item[0], item[1]) for item in self.items)


class ContextEdgePackDataset(Dataset):
    """All frozen train-only edges, packed by specimen for Hist2ST."""

    def __init__(
        self,
        ordinary_dataset: HestContextDataset,
        graph: FrozenTrainingEdges,
        *,
        edge_seed: int,
        edge_microbatch_max_spots: int = 1024,
    ) -> None:
        if int(edge_microbatch_max_spots) < 2:
            raise HestAdapterError(
                f"CONTEXT_EDGE_MICROBATCH_SPOTS:{edge_microbatch_max_spots}"
            )
        edges_by_specimen: dict[int, list[TrainingEdge]] = {}
        for edge in graph.edges:
            edges_by_specimen.setdefault(edge.specimen_index, []).append(edge)

        packs: dict[tuple[int, int], tuple[TrainingEdge, ...]] = {}
        for specimen_index in sorted(edges_by_specimen):
            keys = [
                key for key in ordinary_dataset.item_keys if key[0] == specimen_index
            ]
            values = edges_by_specimen[specimen_index]
            order = np.random.RandomState(int(edge_seed) + specimen_index).permutation(
                len(values)
            )
            shuffled = [values[int(index)] for index in order]
            edge_chunks = np.array_split(np.asarray(shuffled, dtype=object), len(keys))
            if any(len(chunk) < 1 for chunk in edge_chunks):
                raise HestAdapterError(
                    f"CONTEXT_EDGE_PACK_EMPTY:{specimen_index}:{len(values)}:{len(keys)}"
                )
            for key, chunk in zip(keys, edge_chunks, strict=True):
                packs[key] = tuple(chunk.tolist())

        if set(packs) != set(ordinary_dataset.item_keys):
            raise HestAdapterError("CONTEXT_EDGE_PACK_KEYS")
        self.ordinary_dataset = ordinary_dataset
        self.packs = tuple(packs[key] for key in ordinary_dataset.item_keys)
        self.edge_microbatch_max_spots = int(edge_microbatch_max_spots)

    def __len__(self) -> int:
        return len(self.packs)

    @staticmethod
    def _partition_edges(
        edges: Sequence[TrainingEdge], *, max_spots: int
    ) -> tuple[tuple[TrainingEdge, ...], ...]:
        chunks: list[tuple[TrainingEdge, ...]] = []
        current: list[TrainingEdge] = []
        current_spots: set[int] = set()
        for edge in edges:
            endpoints = {int(edge.left_global), int(edge.right_global)}
            if current and len(current_spots | endpoints) > int(max_spots):
                chunks.append(tuple(current))
                current = []
                current_spots = set()
            current.append(edge)
            current_spots.update(endpoints)
        if current:
            chunks.append(tuple(current))
        if not chunks:
            raise HestAdapterError("CONTEXT_EDGE_MICROBATCH_EMPTY")
        return tuple(chunks)

    def _load_edge_microbatch(
        self,
        *,
        specimen_index: int,
        specimen: AlignedHestSpecimen,
        edges: Sequence[TrainingEdge],
    ) -> dict[str, Any]:
        global_block = self.ordinary_dataset.specimen_global_indices[specimen_index]
        offset = int(global_block[0])
        global_rows = sorted(
            {value for edge in edges for value in (edge.left_global, edge.right_global)}
        )
        local_rows = np.asarray([value - offset for value in global_rows], dtype=np.int64)
        batch = load_specimen_batch(
            specimen,
            rows=local_rows,
            expected_patch_pixels=self.ordinary_dataset.expected_patch_pixels,
            context_neighbors=self.ordinary_dataset.context_neighbors,
        )
        lookup = {global_index: row for row, global_index in enumerate(global_rows)}
        batch["edge_index"] = torch.tensor(
            [
                [lookup[edge.left_global] for edge in edges],
                [lookup[edge.right_global] for edge in edges],
            ],
            dtype=torch.long,
        )
        batch["loss_mask"] = torch.ones(len(global_rows), dtype=torch.bool)
        return batch

    def __getitem__(self, index: int) -> dict[str, Any]:
        specimen_index, _, specimen, _, ordinary_rows = self.ordinary_dataset.items[
            index
        ]
        edges = self.packs[index]
        max_spots = min(self.edge_microbatch_max_spots, len(ordinary_rows))
        edge_chunks = self._partition_edges(edges, max_spots=max_spots)
        return {
            "microbatches": [
                self._load_edge_microbatch(
                    specimen_index=specimen_index,
                    specimen=specimen,
                    edges=chunk,
                )
                for chunk in edge_chunks
            ],
            "edge_count": int(len(edges)),
            "max_spots": int(max_spots),
        }


class DualStreamContextDataset(Dataset):
    """Pair each ordinary Hist2ST subset with one pack of frozen local edges."""

    def __init__(
        self, ordinary_dataset: HestContextDataset, edge_dataset: ContextEdgePackDataset
    ) -> None:
        if len(ordinary_dataset) != len(edge_dataset):
            raise HestAdapterError("CONTEXT_DUAL_STREAM_LENGTH")
        self.ordinary_dataset = ordinary_dataset
        self.edge_dataset = edge_dataset

    def __len__(self) -> int:
        return len(self.ordinary_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "ordinary": self.ordinary_dataset[index],
            "edge": self.edge_dataset[index],
        }


def make_context_loader(
    specimens: Sequence[AlignedHestSpecimen],
    splits: Sequence[SpecimenInnerSplit],
    *,
    validation: bool,
    seed: int,
    num_subsets: int = 1,
    num_workers: int = 0,
    expected_patch_pixels: int | None = 224,
    context_neighbors: int = 8,
    include_local_edges: bool | None = None,
) -> DataLoader:
    """Reproducible specimen stream; batch_size=None preserves variable slide size."""

    generator = torch.Generator().manual_seed(int(seed))
    resolved_edges = (not validation) if include_local_edges is None else bool(
        include_local_edges
    )
    if validation and resolved_edges:
        raise HestAdapterError("VALIDATION_CONTEXT_EDGES_FORBIDDEN")
    return DataLoader(
        HestContextDataset(
            specimens,
            splits,
            validation=validation,
            seed=int(seed),
            num_subsets=int(num_subsets),
            expected_patch_pixels=expected_patch_pixels,
            context_neighbors=context_neighbors,
            include_local_edges=resolved_edges,
        ),
        batch_size=None,
        shuffle=not validation,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=_seed_worker,
        persistent_workers=bool(num_workers),
    )


@dataclass(frozen=True)
class MatchedContextTrainingLoaders:
    """Context-model loaders with matching slides and isolated local edges."""

    baseline: DataLoader
    intervention: DataLoader


def make_matched_context_training_loaders(
    specimens: Sequence[AlignedHestSpecimen],
    splits: Sequence[SpecimenInnerSplit],
    *,
    ordinary_seed: int,
    num_subsets: int = 1,
    frozen_edges: FrozenTrainingEdges | None = None,
    edge_seed: int | None = None,
    edge_microbatch_max_spots: int = 1024,
    num_workers: int = 0,
    expected_patch_pixels: int | None = 224,
    context_neighbors: int = 8,
) -> MatchedContextTrainingLoaders:
    """Build separate baseline/no-edge and intervention/edge slide streams."""

    dataset_common = dict(
        specimens=specimens,
        splits=splits,
        validation=False,
        seed=int(ordinary_seed),
        num_subsets=int(num_subsets),
        expected_patch_pixels=expected_patch_pixels,
        context_neighbors=int(context_neighbors),
    )
    baseline_dataset = HestContextDataset(
        **dataset_common, include_local_edges=False
    )
    intervention_ordinary = HestContextDataset(
        **dataset_common, include_local_edges=frozen_edges is None
    )
    if frozen_edges is None:
        intervention_dataset: Dataset = intervention_ordinary
    else:
        if edge_seed is None:
            raise HestAdapterError("CONTEXT_EDGE_SEED_REQUIRED")
        intervention_dataset = DualStreamContextDataset(
            intervention_ordinary,
            ContextEdgePackDataset(
                intervention_ordinary,
                frozen_edges,
                edge_seed=int(edge_seed),
                edge_microbatch_max_spots=int(edge_microbatch_max_spots),
            ),
        )

    def loader(dataset: Dataset) -> DataLoader:
        generator = torch.Generator().manual_seed(int(ordinary_seed))
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=torch.cuda.is_available(),
            generator=generator,
            worker_init_fn=_seed_worker,
            persistent_workers=bool(num_workers),
        )

    return MatchedContextTrainingLoaders(
        baseline=loader(baseline_dataset),
        intervention=loader(intervention_dataset),
    )


def select_coordinate_local_rows(coords: np.ndarray, limit: int) -> np.ndarray:
    """Deterministic real-batch gate subset chosen from coordinates only."""

    points = np.asarray(coords, dtype=np.float64)
    if limit < 2:
        raise HestAdapterError("GATE_CONTEXT_LIMIT")
    if len(points) <= limit:
        return np.arange(len(points), dtype=np.int64)
    center = np.median(points, axis=0)
    squared = np.sum(np.square(points - center), axis=1)
    return np.sort(np.argsort(squared, kind="stable")[:limit]).astype(np.int64)
