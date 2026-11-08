"""Sample DeepSpot2Cell training pairs from a fixed spatial graph."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .graph import FrozenAdjacency


class FrozenEdgePairDataset(Dataset):
    """Return both endpoints of one precomputed training edge.

    No neighbour search happens in ``__getitem__`` and an ordinary random
    minibatch can never silently determine which local-difference terms exist.
    """

    def __init__(self, base_dataset: Dataset, adjacency: FrozenAdjacency) -> None:
        if not adjacency.edges:
            raise ValueError("Cannot pair-sample an empty adjacency")
        self.base_dataset = base_dataset
        self.adjacency = adjacency

    def __len__(self) -> int:
        return len(self.adjacency.edges)

    def __getitem__(self, edge_index: int) -> dict[str, Any]:
        edge = self.adjacency.edges[edge_index]
        return {
            "left": self.base_dataset[edge.left_index],
            "right": self.base_dataset[edge.right_index],
            "edge_index": int(edge_index),
            "left_index": int(edge.left_index),
            "right_index": int(edge.right_index),
            "edge_distance": float(edge.distance),
            "edge_specimen": edge.specimen,
        }


class DualStreamTrainingDataset(Dataset):
    """Attach an explicit edge pair to each ordinary training item.

    Dataset positions and length are exactly those of ``ordinary_dataset``.
    Consequently, using the same DataLoader shuffle generator yields the same
    ordinary spot stream as the rho-zero arm. The edge assignment is frozen by
    its own independent seed and is never constructed by the rho-zero loader.
    """

    def __init__(
        self,
        ordinary_dataset: Dataset,
        edge_dataset: FrozenEdgePairDataset,
        *,
        edge_seed: int,
    ) -> None:
        if len(ordinary_dataset) < 1 or len(edge_dataset) < 1:
            raise ValueError("Ordinary and edge datasets must both be non-empty")
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
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_edge_pair_loader(
    base_dataset: Dataset,
    adjacency: FrozenAdjacency,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """Create a reproducible edge-pair stream independent of model RNG.

    Construct this loader once per arm with the same ``seed``.  Its private
    generator keeps minibatch ordering independent from model initialization
    and dropout random-number consumption.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        FrozenEdgePairDataset(base_dataset, adjacency),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
        drop_last=bool(drop_last),
        generator=generator,
        worker_init_fn=_seed_worker,
        persistent_workers=bool(num_workers),
    )


@dataclass(frozen=True)
class MatchedTrainingLoaders:
    """Loaders whose ordinary minibatch stream is identical across arms."""

    baseline: DataLoader
    intervention: DataLoader


def make_matched_training_loaders(
    base_dataset: Dataset,
    train_indices: list[int] | tuple[int, ...],
    adjacency: FrozenAdjacency,
    *,
    batch_size: int,
    ordinary_seed: int,
    edge_seed: int,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    drop_last: bool = False,
) -> MatchedTrainingLoaders:
    """Build the formal rho-zero and rho-half training streams.

    The baseline loader contains only ordinary spots and therefore cannot read
    an edge. The intervention loader has the same length and shuffles the same
    ordinary dataset positions with a separate generator initialized to the
    same ordinary seed; its edge assignments were frozen independently.
    """

    normalized_train = tuple(sorted(int(index) for index in train_indices))
    if normalized_train != adjacency.train_indices:
        raise ValueError("Ordinary training indices do not match frozen adjacency")
    ordinary = Subset(base_dataset, normalized_train)
    edge_dataset = FrozenEdgePairDataset(base_dataset, adjacency)
    dual = DualStreamTrainingDataset(ordinary, edge_dataset, edge_seed=edge_seed)
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

    return MatchedTrainingLoaders(
        baseline=loader(ordinary),
        intervention=loader(dual),
    )


@dataclass(frozen=True)
class FlattenedPairBatch:
    batch: dict[str, Any]
    left_positions: torch.Tensor
    right_positions: torch.Tensor


def _concatenate_collated(left: Any, right: Any, key: str) -> Any:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.cat((left, right), dim=0)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.concatenate((left, right), axis=0)
    if isinstance(left, list) and isinstance(right, list):
        return left + right
    if isinstance(left, tuple) and isinstance(right, tuple):
        return left + right
    raise TypeError(
        f"Cannot concatenate collated field {key!r} of types "
        f"{type(left).__name__} and {type(right).__name__}"
    )


def flatten_pair_batch(pair_batch: Mapping[str, Any]) -> FlattenedPairBatch:
    """Flatten nested endpoints once, preserving exact left/right positions."""

    if "left" not in pair_batch or "right" not in pair_batch:
        raise KeyError("Paired batch must contain 'left' and 'right' endpoint mappings")
    left = pair_batch["left"]
    right = pair_batch["right"]
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise TypeError("Paired endpoint batches must be mappings")
    if set(left) != set(right):
        raise ValueError("Left and right endpoint fields do not match")

    flat = {
        key: _concatenate_collated(left[key], right[key], key)
        for key in left
    }
    first_tensor = next(
        (value for value in left.values() if isinstance(value, torch.Tensor)), None
    )
    if first_tensor is None or first_tensor.ndim == 0:
        raise ValueError("Cannot infer pair-batch size from endpoint fields")
    pair_count = int(first_tensor.shape[0])
    positions = torch.arange(pair_count, device=first_tensor.device, dtype=torch.long)
    return FlattenedPairBatch(
        batch=flat,
        left_positions=positions,
        right_positions=positions + pair_count,
    )
