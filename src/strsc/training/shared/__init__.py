"""Shared spatial adjacency, expression losses and numerical equivalence checks."""

from .adjacency import (
    AdjacencyContractError,
    AdjacencyGraph,
    SpecimenGraph,
    build_spatial_edges,
    build_within_specimen_adjacency,
)
from .losses import LocalDifferenceLossError, local_difference_mse, mixed_expression_mse
from .noop import NoOpCheckError, NoOpReceipt, check_rho_zero_noop

__all__ = [
    "AdjacencyContractError",
    "AdjacencyGraph",
    "SpecimenGraph",
    "build_spatial_edges",
    "build_within_specimen_adjacency",
    "LocalDifferenceLossError",
    "local_difference_mse",
    "mixed_expression_mse",
    "NoOpCheckError",
    "NoOpReceipt",
    "check_rho_zero_noop",
]
