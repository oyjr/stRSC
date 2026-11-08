"""Neighbouring-pair supervision and model-specific integration."""
from .shared.adjacency import build_spatial_edges, build_within_specimen_adjacency
from .shared.losses import local_difference_mse, mixed_expression_mse

__all__ = ["build_spatial_edges", "build_within_specimen_adjacency", "local_difference_mse", "mixed_expression_mse"]
