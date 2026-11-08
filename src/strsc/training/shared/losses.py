"""Shared neighbouring-difference expression loss for model adapters."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class LocalDifferenceLossError(RuntimeError):
    """Raised when an expression-loss input violates the shared contract."""


def _validate_expression_tensors(
    prediction: torch.Tensor, target: torch.Tensor
) -> None:
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise LocalDifferenceLossError("PREDICTION_AND_TARGET_MUST_BE_TENSORS")
    if prediction.shape != target.shape:
        raise LocalDifferenceLossError(
            f"EXPRESSION_SHAPE_MISMATCH:{tuple(prediction.shape)}:{tuple(target.shape)}"
        )
    if prediction.ndim != 2 or prediction.shape[0] < 1 or prediction.shape[1] < 1:
        raise LocalDifferenceLossError(f"EXPRESSION_SHAPE:{tuple(prediction.shape)}")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise LocalDifferenceLossError("EXPRESSION_DTYPE_MUST_BE_FLOATING")
    if prediction.device != target.device:
        raise LocalDifferenceLossError(
            f"EXPRESSION_DEVICE_MISMATCH:{prediction.device}:{target.device}"
        )


def _edge_tensor(edge_index: Any, prediction: torch.Tensor) -> torch.Tensor:
    if isinstance(edge_index, torch.Tensor):
        edges = edge_index.to(device=prediction.device, dtype=torch.long)
    else:
        edges = torch.as_tensor(
            np.asarray(edge_index), device=prediction.device, dtype=torch.long
        )
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] < 1:
        raise LocalDifferenceLossError(f"EDGE_INDEX_SHAPE:{tuple(edges.shape)}")
    if int(edges.min().item()) < 0 or int(edges.max().item()) >= prediction.shape[0]:
        raise LocalDifferenceLossError(
            f"EDGE_INDEX_RANGE:{int(edges.min().item())}:{int(edges.max().item())}:"
            f"{prediction.shape[0]}"
        )
    if torch.any(edges[0] == edges[1]):
        raise LocalDifferenceLossError("EDGE_SELF_LOOP")
    return edges


def local_difference_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    edge_index: Any,
) -> torch.Tensor:
    """Mean squared error of measured versus predicted adjacent differences."""

    _validate_expression_tensors(prediction, target)
    edges = _edge_tensor(edge_index, prediction)
    residual = prediction - target
    difference_error = residual[edges[0]] - residual[edges[1]]
    return torch.mean(torch.square(difference_error))


def mixed_expression_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    edge_index: Any,
    *,
    rho: float,
) -> torch.Tensor:
    """Return ``(1-rho)*spot MSE + rho*edge-difference MSE``.

    The explicit ``rho == 0`` branch is load-bearing: the baseline calls the
    original PyTorch MSE implementation directly and never evaluates, samples,
    or differentiates through graph rows.
    """

    if not math.isfinite(float(rho)) or float(rho) < 0.0 or float(rho) > 1.0:
        raise LocalDifferenceLossError(f"RHO_RANGE:{rho}")
    _validate_expression_tensors(prediction, target)
    ordinary = F.mse_loss(prediction, target, reduction="mean")
    if math.isclose(float(rho), 0.0, rel_tol=0.0, abs_tol=0.0):
        return ordinary
    edge = local_difference_mse(prediction, target, edge_index)
    if math.isclose(float(rho), 1.0, rel_tol=0.0, abs_tol=0.0):
        return edge
    return (1.0 - float(rho)) * ordinary + float(rho) * edge
