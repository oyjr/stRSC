"""Build publication-shaped EGN inputs without changing either architecture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from scipy.spatial import cKDTree

from .exemplars import ExemplarBank, retrieve_exemplars


class ModelInputContractError(RuntimeError):
    pass


def egn_v1_batch(
    *,
    model_patches: torch.Tensor,
    query_features: torch.Tensor,
    query_rows: Union[np.ndarray, torch.Tensor],
    query_specimen: str,
    bank: ExemplarBank,
) -> dict[str, torch.Tensor]:
    rows = torch.as_tensor(query_rows, dtype=torch.long)
    features = query_features[rows]
    exemplars = retrieve_exemplars(
        features, query_specimen=query_specimen, bank=bank
    )
    return {
        "img": model_patches[rows],
        "p_feature": features.unsqueeze(1),
        "op_feature": exemplars.features,
        "op_count": exemplars.targets,
    }


def egn_grid_positions(coords: np.ndarray) -> torch.Tensor:
    """Map physical HEST positions to the publication's integer spot grid."""

    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ModelInputContractError(f"GRID_COORDINATES:{points.shape}")
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    positive = distances[:, 1][distances[:, 1] > 0]
    if not len(positive):
        raise ModelInputContractError("GRID_SPACING_NOT_ESTIMABLE")
    spacing = float(np.median(positive))
    grid = np.rint((points - points.min(axis=0, keepdims=True)) / spacing)
    return torch.as_tensor(grid, dtype=torch.float32)


@dataclass(frozen=True)
class EGNv2GraphInput:
    x_dict: dict[str, torch.Tensor]
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    window_positions: torch.Tensor
    exemplar_sources: tuple[tuple[str, str], ...]

    def to(self, device: torch.device) -> "EGNv2GraphInput":
        return EGNv2GraphInput(
            x_dict={key: value.to(device) for key, value in self.x_dict.items()},
            edge_index_dict={
                key: value.to(device) for key, value in self.edge_index_dict.items()
            },
            window_positions=self.window_positions.to(device),
            exemplar_sources=self.exemplar_sources,
        )


def build_egn_v2_graph(
    *,
    query_features: torch.Tensor,
    query_coords: np.ndarray,
    query_barcodes: np.ndarray | None = None,
    query_specimen: str,
    bank: ExemplarBank,
    torch_geometric_module,
    allow_single_reference_fallback: bool = False,
) -> EGNv2GraphInput:
    """Reproduce generate_graph.py with a train-only exemplar candidate bank."""

    retrieved = retrieve_exemplars(
        query_features,
        query_specimen=query_specimen,
        bank=bank,
        query_barcodes=query_barcodes,
        query_coords=query_coords,
        allow_single_reference_fallback=allow_single_reference_fallback,
    )
    positions = egn_grid_positions(query_coords)
    radius_graph = torch_geometric_module.nn.radius_graph(
        positions,
        math.sqrt(2.0),
        None,
        False,
        max_num_neighbors=5,
        flow="source_to_target",
        num_workers=1,
    )

    n_windows, n_exemplars, _ = retrieved.features.shape
    source_windows = torch.repeat_interleave(
        torch.arange(n_windows, dtype=torch.long), n_exemplars
    )
    flat_features = retrieved.features.reshape(n_windows * n_exemplars, -1)
    flat_targets = retrieved.targets.reshape(n_windows * n_exemplars, -1)
    combined = torch.cat((flat_features, flat_targets), dim=1)
    unique_combined, inverse = torch.unique(combined, dim=0, return_inverse=True)
    feature_width = flat_features.shape[1]
    unique_features = unique_combined[:, :feature_width]
    unique_targets = unique_combined[:, feature_width:]
    cross_edge = torch.stack((inverse, source_windows), dim=0)
    close_edge = torch_geometric_module.nn.knn_graph(
        unique_combined, k=3, loop=False
    )

    source_pairs: list[tuple[str, str]] = []
    for specimen, barcode in zip(
        retrieved.source_specimens.reshape(-1), retrieved.source_barcodes.reshape(-1)
    ):
        source_pairs.append((str(specimen), str(barcode)))
    return EGNv2GraphInput(
        x_dict={
            "window": query_features,
            "example": torch.cat((unique_features, unique_targets), dim=1),
        },
        edge_index_dict={
            ("window", "near", "window"): radius_graph,
            ("example", "refer", "window"): cross_edge,
            ("example", "close", "example"): close_edge,
        },
        window_positions=positions,
        exemplar_sources=tuple(source_pairs),
    )
