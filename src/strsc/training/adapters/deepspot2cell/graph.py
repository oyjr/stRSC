"""Frozen, training-only adjacency construction for DeepSpot2Cell.

The graph is constructed independently within each specimen:
the radius is 1.5 times the median positive nearest-neighbour distance.  Only
indices explicitly supplied as training indices participate in either the
spacing estimate or the edge search.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True, order=True)
class FrozenEdge:
    """One undirected training edge, expressed in base-dataset indices."""

    specimen: str
    left_index: int
    right_index: int
    distance: float

    def __post_init__(self) -> None:
        if self.left_index >= self.right_index:
            raise ValueError("FrozenEdge endpoints must be stored in ascending order")
        if not np.isfinite(self.distance) or self.distance <= 0:
            raise ValueError("FrozenEdge distance must be finite and positive")


@dataclass(frozen=True)
class FrozenAdjacency:
    """Immutable adjacency plus the inputs needed to audit it."""

    train_indices: tuple[int, ...]
    edges: tuple[FrozenEdge, ...]
    spacing_by_specimen: Mapping[str, float]
    radius_multiplier: float = 1.5

    def validate(self, patch_barcodes: Sequence[tuple[str, str]]) -> None:
        train = set(self.train_indices)
        if not train:
            raise ValueError("Training index set is empty")
        if len(train) != len(self.train_indices):
            raise ValueError("Training indices contain duplicates")
        if min(train) < 0 or max(train) >= len(patch_barcodes):
            raise IndexError("Training index is outside the base dataset")
        if not self.edges:
            raise ValueError("Frozen training graph has no edges")

        seen: set[tuple[int, int]] = set()
        for edge in self.edges:
            pair = (edge.left_index, edge.right_index)
            if pair in seen:
                raise ValueError(f"Duplicate frozen edge: {pair}")
            seen.add(pair)
            if edge.left_index not in train or edge.right_index not in train:
                raise ValueError(f"Non-training endpoint found in frozen edge: {pair}")
            left_specimen = str(patch_barcodes[edge.left_index][0])
            right_specimen = str(patch_barcodes[edge.right_index][0])
            if left_specimen != right_specimen or left_specimen != edge.specimen:
                raise ValueError(f"Cross-specimen or mislabeled edge found: {pair}")


def parse_patch_barcode_coordinates(patch_barcode: str) -> tuple[float, float]:
    """Read the grid row and column from ``patch_<row>_<col>_...``."""

    parts = patch_barcode.split("_")
    if len(parts) < 3 or parts[0] != "patch":
        raise ValueError(
            f"Unsupported patch barcode {patch_barcode!r}; expected patch_<row>_<col>_..."
        )
    try:
        return float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise ValueError(f"Non-numeric coordinates in patch barcode {patch_barcode!r}") from exc


def _validated_train_indices(
    train_indices: Sequence[int], dataset_size: int
) -> tuple[int, ...]:
    indices = tuple(sorted(int(index) for index in train_indices))
    if not indices:
        raise ValueError("Training index set is empty")
    if len(set(indices)) != len(indices):
        raise ValueError("Training indices contain duplicates")
    if indices[0] < 0 or indices[-1] >= dataset_size:
        raise IndexError("Training index is outside the base dataset")
    return indices


def _resolve_coordinates(
    patch_barcodes: Sequence[tuple[str, str]],
    coordinates: Sequence[Sequence[float]] | Mapping[int, Sequence[float]] | None,
) -> dict[int, tuple[float, float]]:
    if coordinates is None:
        return {
            index: parse_patch_barcode_coordinates(barcode)
            for index, (_, barcode) in enumerate(patch_barcodes)
        }

    if isinstance(coordinates, Mapping):
        resolved = {int(index): tuple(map(float, value)) for index, value in coordinates.items()}
    else:
        if len(coordinates) != len(patch_barcodes):
            raise ValueError("Coordinate sequence must align one-to-one with patch_barcodes")
        resolved = {
            index: tuple(map(float, value)) for index, value in enumerate(coordinates)
        }

    for index, value in resolved.items():
        if len(value) != 2 or not np.isfinite(value).all():
            raise ValueError(f"Coordinate for index {index} must contain two finite values")
    return resolved


def build_frozen_train_adjacency(
    patch_barcodes: Sequence[tuple[str, str]],
    train_indices: Sequence[int],
    *,
    coordinates: Sequence[Sequence[float]] | Mapping[int, Sequence[float]] | None = None,
    radius_multiplier: float = 1.5,
) -> FrozenAdjacency:
    """Build the frozen graph using training coordinates and nothing else.

    ``patch_barcodes`` is the base DeepSpot2Cell dataset's ordered
    ``(specimen_id, patch_barcode)`` sequence.  Supplying all barcodes is safe:
    only ``train_indices`` are read while estimating spacing and querying pairs.
    """

    if not np.isfinite(radius_multiplier) or radius_multiplier <= 0:
        raise ValueError("radius_multiplier must be finite and positive")
    train = _validated_train_indices(train_indices, len(patch_barcodes))
    coords_by_index = _resolve_coordinates(patch_barcodes, coordinates)

    by_specimen: dict[str, list[int]] = {}
    for index in train:
        specimen = str(patch_barcodes[index][0])
        by_specimen.setdefault(specimen, []).append(index)

    spacings: dict[str, float] = {}
    edges: list[FrozenEdge] = []
    for specimen in sorted(by_specimen):
        specimen_indices = sorted(by_specimen[specimen])
        if len(specimen_indices) < 2:
            raise ValueError(f"Specimen {specimen!r} has fewer than two training locations")
        try:
            specimen_coords = np.asarray(
                [coords_by_index[index] for index in specimen_indices], dtype=np.float64
            )
        except KeyError as exc:
            raise ValueError(f"Missing coordinate for training index {exc.args[0]}") from exc

        if np.unique(specimen_coords, axis=0).shape[0] != len(specimen_coords):
            raise ValueError(f"Specimen {specimen!r} contains duplicate training coordinates")

        tree = cKDTree(specimen_coords)
        nearest = tree.query(specimen_coords, k=2)[0][:, 1]
        positive = nearest[np.isfinite(nearest) & (nearest > 0)]
        if positive.size == 0:
            raise ValueError(f"Specimen {specimen!r} has no positive nearest-neighbour distance")
        spacing = float(np.median(positive))
        spacings[specimen] = spacing
        radius = float(radius_multiplier * spacing)

        local_pairs = sorted(tree.query_pairs(radius, output_type="set"))
        for left_local, right_local in local_pairs:
            distance = float(
                np.linalg.norm(specimen_coords[left_local] - specimen_coords[right_local])
            )
            if distance <= 0:
                continue
            left = specimen_indices[left_local]
            right = specimen_indices[right_local]
            if left > right:
                left, right = right, left
            edges.append(FrozenEdge(specimen, left, right, distance))

    adjacency = FrozenAdjacency(
        train_indices=train,
        edges=tuple(sorted(edges)),
        spacing_by_specimen=spacings,
        radius_multiplier=float(radius_multiplier),
    )
    adjacency.validate(patch_barcodes)
    return adjacency


def write_frozen_adjacency(adjacency: FrozenAdjacency, path: str | Path) -> Path:
    """Write the exact graph used by both arms as an auditable JSON receipt."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "g0055.deepspot2cell.frozen_adjacency.v1",
        "radius_multiplier": adjacency.radius_multiplier,
        "train_indices": list(adjacency.train_indices),
        "spacing_by_specimen": dict(adjacency.spacing_by_specimen),
        "edges": [asdict(edge) for edge in adjacency.edges],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_frozen_adjacency(
    path: str | Path, patch_barcodes: Sequence[tuple[str, str]]
) -> FrozenAdjacency:
    """Reload and revalidate a graph receipt before constructing either arm."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "g0055.deepspot2cell.frozen_adjacency.v1":
        raise ValueError("Unsupported frozen adjacency schema")
    adjacency = FrozenAdjacency(
        train_indices=tuple(int(index) for index in payload["train_indices"]),
        edges=tuple(FrozenEdge(**edge) for edge in payload["edges"]),
        spacing_by_specimen={
            str(specimen): float(spacing)
            for specimen, spacing in payload["spacing_by_specimen"].items()
        },
        radius_multiplier=float(payload["radius_multiplier"]),
    )
    adjacency.validate(patch_barcodes)
    return adjacency
