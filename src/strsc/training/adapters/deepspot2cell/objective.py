"""Neighbouring-difference expression loss for DeepSpot2Cell."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .paired_data import flatten_pair_batch


@dataclass(frozen=True)
class LocalDifferenceLoss:
    total: torch.Tensor
    expression: torch.Tensor
    pointwise_mse: torch.Tensor
    local_difference_mse: torch.Tensor
    auxiliary: torch.Tensor
    predictions: torch.Tensor
    targets: torch.Tensor
    edge_predictions: torch.Tensor | None
    edge_targets: torch.Tensor | None


def assert_deepspot2cell_mse_contract(model: torch.nn.Module) -> None:
    """Fail before training if the adapter would no longer be a loss-only change."""

    loss_type = str(getattr(model, "loss_type", "mse")).lower()
    if loss_type != "mse":
        raise ValueError(
            f"G0-055 DeepSpot2Cell adapter requires original MSE, found {loss_type!r}"
        )
    if not callable(getattr(model, "forward", None)):
        raise TypeError("DeepSpot2Cell model must expose a callable forward method")


def _forward_spots(
    model: torch.nn.Module, flat_batch: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    required = {
        "cell_embeddings",
        "spot_expression",
        "cell_mask",
        "neighb_embeddings",
        "neighb_masks",
        "is_inside_spot",
    }
    missing = sorted(required.difference(flat_batch))
    if missing:
        raise KeyError(f"DeepSpot2Cell endpoint batch is missing: {', '.join(missing)}")

    cell_embeddings = flat_batch["cell_embeddings"].float()
    spot_targets = flat_batch["spot_expression"].float()
    mask = flat_batch["cell_mask"].float()
    context = flat_batch["neighb_embeddings"].float()
    context_mask = flat_batch["neighb_masks"].bool()
    is_inside = flat_batch["is_inside_spot"].float()
    mask_inside = mask * is_inside
    centroids = flat_batch.get("cell_centroids")
    if isinstance(centroids, torch.Tensor):
        centroids = centroids.float()

    auxiliary_weight = float(getattr(model, "aux_loss_weight", 0.0))
    if auxiliary_weight > 0:
        if not callable(getattr(model, "_forward_superspot", None)):
            raise TypeError("Auxiliary-loss model must expose _forward_superspot")
        predictions, cell_features = model._forward_superspot(
            cell_embeddings * is_inside.unsqueeze(-1),
            mask_inside,
            context,
            context_mask,
            cell_centroids=centroids,
            return_cell_features=True,
        )
    else:
        predictions = model(
            cell_embeddings * is_inside.unsqueeze(-1),
            mask_inside,
            context,
            context_mask,
            cell_centroids=centroids,
        )
        cell_features = None
    return predictions, spot_targets, cell_features, mask_inside


def _unchanged_auxiliary_loss(
    model: torch.nn.Module,
    cell_features: torch.Tensor | None,
    mask_inside: torch.Tensor,
    spot_targets: torch.Tensor,
) -> torch.Tensor:
    weight = float(getattr(model, "aux_loss_weight", 0.0))
    if weight == 0:
        return spot_targets.new_zeros(())
    if cell_features is None or not callable(getattr(model, "_apply_rho", None)):
        raise TypeError("Cannot preserve DeepSpot2Cell auxiliary consistency loss")

    batch, cells, feature_dim = cell_features.shape
    per_cell_counts = model._apply_rho(
        cell_features.reshape(batch * cells, feature_dim)
    ).reshape(batch, cells, -1)
    reconstructed = torch.log1p(
        (per_cell_counts * mask_inside.unsqueeze(-1)).sum(dim=1)
    )
    return F.mse_loss(reconstructed, spot_targets)


def _edge_forward_preserving_ordinary_stream(
    model: torch.nn.Module, edge_batch: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the edge-only forward without advancing ordinary stochastic state.

    The intervention necessarily needs an additional forward. Restoring Python,
    NumPy and Torch RNG states keeps every later ordinary dropout/augmentation
    draw paired to rho zero. BatchNorm running buffers are likewise restored so
    they continue to summarize ordinary spots rather than degree-sampled edge
    endpoints; affine BatchNorm parameters still receive the edge gradient.
    """

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    batch_norm_state = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            batch_norm_state.append((module, bool(module.track_running_stats)))
            # In training mode BatchNorm still uses edge-batch statistics, but
            # no running buffer is updated or captured by the autograd graph.
            module.track_running_stats = False
    try:
        predictions, targets, _, _ = _forward_spots(model, edge_batch)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for module, track_running_stats in batch_norm_state:
            module.track_running_stats = track_running_stats
    return predictions, targets


def compute_local_difference_loss(
    model: torch.nn.Module,
    ordinary_batch: Mapping[str, Any],
    *,
    rho: float,
    edge_pair_batch: Mapping[str, Any] | None = None,
) -> LocalDifferenceLoss:
    """Compute the dual-stream objective while preserving ordinary spot weights.

    The ordinary expression MSE and all published auxiliary terms are computed
    only on ``ordinary_batch``. At ``rho=0`` no edge batch is accepted or read.
    At ``rho=0.5`` a separate explicit edge batch supplies only the measured-
    difference term, so high-degree spots cannot change ordinary spot weights.
    """

    assert_deepspot2cell_mse_contract(model)
    rho = float(rho)
    if rho not in (0.0, 0.5):
        raise ValueError("Frozen G0-055 DeepSpot2Cell arms permit only rho=0.0 or rho=0.5")

    predictions, targets, cell_features, mask_inside = _forward_spots(
        model, ordinary_batch
    )
    pointwise = F.mse_loss(predictions, targets)
    auxiliary = _unchanged_auxiliary_loss(
        model, cell_features, mask_inside, targets
    )
    auxiliary_term = float(getattr(model, "aux_loss_weight", 0.0)) * auxiliary

    edge_predictions: torch.Tensor | None = None
    edge_targets: torch.Tensor | None = None
    if rho == 0.0:
        if edge_pair_batch is not None:
            raise ValueError("rho=0 must not receive or access an edge batch")
        local = pointwise.new_zeros(())
        expression = pointwise
    else:
        if edge_pair_batch is None:
            raise ValueError("rho=0.5 requires an explicit edge-pair batch")
        flattened = flatten_pair_batch(edge_pair_batch)
        edge_predictions, edge_targets = _edge_forward_preserving_ordinary_stream(
            model, flattened.batch
        )
        left_prediction = edge_predictions.index_select(0, flattened.left_positions)
        right_prediction = edge_predictions.index_select(0, flattened.right_positions)
        left_target = edge_targets.index_select(0, flattened.left_positions)
        right_target = edge_targets.index_select(0, flattened.right_positions)
        local = F.mse_loss(
            left_prediction - right_prediction,
            left_target - right_target,
        )
        expression = 0.5 * pointwise + 0.5 * local
    total = expression + auxiliary_term
    return LocalDifferenceLoss(
        total=total,
        expression=expression,
        pointwise_mse=pointwise,
        local_difference_mse=local,
        auxiliary=auxiliary,
        predictions=predictions,
        targets=targets,
        edge_predictions=edge_predictions,
        edge_targets=edge_targets,
    )
