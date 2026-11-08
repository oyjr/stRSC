"""Unified, non-invasive HEST adapter for four DeepHis2Exp models."""

from .data import (
    AlignedHestFold,
    AlignedHestSpecimen,
    HestSpotDataset,
    load_aligned_hest_fold,
)
from .batching import (
    HestContextDataset,
    build_frozen_training_edges,
    make_context_loader,
    make_edge_pair_loader,
    make_matched_context_training_loaders,
    make_matched_patch_training_loaders,
    make_partition_spot_loader,
)
from .export import export_hest_prediction, predict_aligned_specimen
from .models import build_upstream_model, canonical_model_name
from .objective import (
    compute_context_training_objective,
    compute_patch_training_objective,
    compute_training_objective,
)
from .overlay import install_objective_overlay, remove_objective_overlay

__all__ = [
    "AlignedHestFold",
    "AlignedHestSpecimen",
    "HestSpotDataset",
    "HestContextDataset",
    "build_frozen_training_edges",
    "build_upstream_model",
    "canonical_model_name",
    "compute_training_objective",
    "compute_context_training_objective",
    "compute_patch_training_objective",
    "export_hest_prediction",
    "install_objective_overlay",
    "load_aligned_hest_fold",
    "make_context_loader",
    "make_edge_pair_loader",
    "make_matched_context_training_loaders",
    "make_matched_patch_training_loaders",
    "make_partition_spot_loader",
    "predict_aligned_specimen",
    "remove_objective_overlay",
]
