"""Deterministic spatial adjacency within each training specimen.

Coordinates are grouped by specimen.  The radius is 1.5 times the median
positive nearest-neighbour distance, computed independently for each specimen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


class AdjacencyContractError(RuntimeError):
    """Raised when a training adjacency graph cannot satisfy the frozen contract."""


@dataclass(frozen=True)
class SpecimenGraph:
    specimen_id: str
    offset: int
    spots: int
    edges: int
    spacing: float
    radius: float


@dataclass(frozen=True)
class AdjacencyGraph:
    """A concatenated graph whose node order matches concatenated model rows."""

    edge_index: np.ndarray
    specimens: tuple[SpecimenGraph, ...]
    radius_multiplier: float

    @property
    def n_nodes(self) -> int:
        return sum(row.spots for row in self.specimens)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to_metadata(self) -> dict[str, object]:
        return {
            "edge_rule": "radius_1.5x_median_positive_nearest_neighbor_within_specimen",
            "radius_multiplier": self.radius_multiplier,
            "nodes": self.n_nodes,
            "edges": self.n_edges,
            "specimens": [
                {
                    "specimen_id": row.specimen_id,
                    "offset": row.offset,
                    "spots": row.spots,
                    "edges": row.edges,
                    "spacing": row.spacing,
                    "radius": row.radius,
                }
                for row in self.specimens
            ],
        }


def _validated_coordinates(coords: np.ndarray, specimen_id: str) -> np.ndarray:
    result = np.asarray(coords, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise AdjacencyContractError(
            f"EDGE_COORDINATE_SHAPE:{specimen_id}:{result.shape}"
        )
    if len(result) < 2:
        raise AdjacencyContractError(
            f"EDGE_COORDINATE_COUNT:{specimen_id}:{len(result)}"
        )
    if not np.isfinite(result).all():
        raise AdjacencyContractError(f"EDGE_COORDINATE_NONFINITE:{specimen_id}")
    return result


def build_spatial_edges(
    coords: np.ndarray,
    *,
    specimen_id: str = "specimen_0",
    radius_multiplier: float = 1.5,
) -> tuple[np.ndarray, float, float]:
    """Build sorted undirected edges for one specimen.

    Returns ``(edge_index, spacing, radius)``. ``edge_index`` has shape
    ``(2, n_edges)`` and every edge satisfies ``left < right``.
    """

    if not np.isfinite(radius_multiplier) or radius_multiplier <= 0:
        raise AdjacencyContractError(
            f"EDGE_RADIUS_MULTIPLIER:{radius_multiplier}"
        )
    points = _validated_coordinates(coords, specimen_id)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nearest = distances[:, 1]
    positive = nearest[np.isfinite(nearest) & (nearest > 0)]
    if not len(positive):
        raise AdjacencyContractError(f"EDGE_SPACING_NOT_ESTIMABLE:{specimen_id}")
    spacing = float(np.median(positive))
    radius = float(radius_multiplier * spacing)
    pairs = np.asarray(tree.query_pairs(radius, output_type="ndarray"), dtype=np.int64)
    if pairs.size == 0:
        raise AdjacencyContractError(f"NO_SPATIAL_EDGES:{specimen_id}")
    pairs = pairs.reshape(-1, 2)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    edge_index = np.ascontiguousarray(pairs.T, dtype=np.int64)
    edge_index.setflags(write=False)
    return edge_index, spacing, radius


def build_within_specimen_adjacency(
    coords_by_specimen: Sequence[np.ndarray],
    *,
    specimen_ids: Sequence[str] | None = None,
    radius_multiplier: float = 1.5,
) -> AdjacencyGraph:
    """Build one global graph without ever searching across specimens."""

    coordinates = list(coords_by_specimen)
    if not coordinates:
        raise AdjacencyContractError("NO_TRAINING_SPECIMENS")
    if specimen_ids is None:
        ids = [f"specimen_{index}" for index in range(len(coordinates))]
    else:
        ids = [str(value) for value in specimen_ids]
    if len(ids) != len(coordinates):
        raise AdjacencyContractError(
            f"SPECIMEN_ID_COUNT:{len(ids)}:{len(coordinates)}"
        )
    if len(set(ids)) != len(ids):
        raise AdjacencyContractError("DUPLICATE_SPECIMEN_ID")

    offset = 0
    edge_blocks: list[np.ndarray] = []
    metadata: list[SpecimenGraph] = []
    for specimen_id, coords in zip(ids, coordinates):
        points = _validated_coordinates(coords, specimen_id)
        local_edges, spacing, radius = build_spatial_edges(
            points,
            specimen_id=specimen_id,
            radius_multiplier=radius_multiplier,
        )
        global_edges = local_edges + offset
        edge_blocks.append(global_edges)
        metadata.append(
            SpecimenGraph(
                specimen_id=specimen_id,
                offset=offset,
                spots=len(points),
                edges=local_edges.shape[1],
                spacing=spacing,
                radius=radius,
            )
        )
        offset += len(points)

    edge_index = np.ascontiguousarray(np.concatenate(edge_blocks, axis=1), dtype=np.int64)
    edge_index.setflags(write=False)
    return AdjacencyGraph(
        edge_index=edge_index,
        specimens=tuple(metadata),
        radius_multiplier=float(radius_multiplier),
    )
