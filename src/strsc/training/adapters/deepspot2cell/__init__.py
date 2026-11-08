"""DeepSpot2Cell adapters for an existing model and dataset."""

from .graph import (
    FrozenAdjacency,
    FrozenEdge,
    build_frozen_train_adjacency,
    load_frozen_adjacency,
    parse_patch_barcode_coordinates,
    write_frozen_adjacency,
)
from .initialization import capture_initial_state, restore_initial_state
from .objective import LocalDifferenceLoss, compute_local_difference_loss
from .overlay import install_local_difference_overlay, remove_local_difference_overlay
from .paired_data import (
    DualStreamTrainingDataset,
    FrozenEdgePairDataset,
    MatchedTrainingLoaders,
    make_edge_pair_loader,
    make_matched_training_loaders,
)

__all__ = [
    "FrozenAdjacency",
    "FrozenEdge",
    "FrozenEdgePairDataset",
    "DualStreamTrainingDataset",
    "MatchedTrainingLoaders",
    "LocalDifferenceLoss",
    "build_frozen_train_adjacency",
    "capture_initial_state",
    "compute_local_difference_loss",
    "install_local_difference_overlay",
    "load_frozen_adjacency",
    "make_edge_pair_loader",
    "make_matched_training_loaders",
    "parse_patch_barcode_coordinates",
    "remove_local_difference_overlay",
    "restore_initial_state",
    "write_frozen_adjacency",
]
