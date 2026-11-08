"""sCellST training adapters and split utilities.

The Lightning subclass is in :mod:`lightning_overlay`.  Importing the split
utilities does not require the full sCellST dependency stack.
"""

from .data import (
    EdgePairDataset,
    PairedSpotEdgeLoader,
    PreparedSpatialData,
    SpatialBagDataset,
    build_training_adjacency,
    collate_edge_pairs,
    collate_spatial_bags,
    freeze_scellst_tenx39_split,
    scellst_edge_pair_collate,
    scellst_spatial_collate,
)
from .objective import (
    LossParts,
    mixed_spot_edge_mse,
    paired_local_difference_mse,
)
from .split import FrozenSpotSplit, freeze_or_load_spot_split

__all__ = [
    "EdgePairDataset",
    "FrozenSpotSplit",
    "LossParts",
    "PairedSpotEdgeLoader",
    "PreparedSpatialData",
    "SpatialBagDataset",
    "build_training_adjacency",
    "collate_edge_pairs",
    "collate_spatial_bags",
    "freeze_or_load_spot_split",
    "freeze_scellst_tenx39_split",
    "mixed_spot_edge_mse",
    "paired_local_difference_mse",
    "scellst_edge_pair_collate",
    "scellst_spatial_collate",
]
