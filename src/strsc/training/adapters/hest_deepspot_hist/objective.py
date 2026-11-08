"""Neighbouring-difference expression loss for HEST DeepSpot and HiST."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .contract import HestDeepSpotHiSTError, validate_rho


@dataclass(frozen=True)
class ExpressionLoss:
    total: torch.Tensor
    ordinary_mse: torch.Tensor
    local_difference_mse: torch.Tensor | None


def mixed_expression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    edge_index: torch.Tensor | None,
    *,
    rho: float,
) -> ExpressionLoss:
    """Mix MSE with residual-difference MSE; ``rho=0`` is a hard no-op."""

    value = validate_rho(rho)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise HestDeepSpotHiSTError(
            f"EXPRESSION_SHAPE:{tuple(prediction.shape)}:{tuple(target.shape)}"
        )
    ordinary = F.mse_loss(prediction, target)
    if value == 0.0:
        return ExpressionLoss(ordinary, ordinary, None)
    if edge_index is None:
        raise HestDeepSpotHiSTError("EDGE_INDEX_REQUIRED")

    from strsc.training.shared.losses import local_difference_mse

    local = local_difference_mse(prediction, target, edge_index)
    total = (1.0 - value) * ordinary + value * local
    return ExpressionLoss(total, ordinary, local)


def mixed_separate_expression_loss(
    ordinary_prediction: torch.Tensor,
    ordinary_target: torch.Tensor,
    *,
    rho: float,
    edge_prediction: torch.Tensor | None = None,
    edge_target: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
) -> ExpressionLoss:
    """Keep the ordinary spot stream separate from the explicit edge stream."""

    value = validate_rho(rho)
    if ordinary_prediction.shape != ordinary_target.shape or ordinary_prediction.ndim != 2:
        raise HestDeepSpotHiSTError("ORDINARY_EXPRESSION_SHAPE")
    ordinary = F.mse_loss(ordinary_prediction, ordinary_target)
    if value == 0.0:
        return ExpressionLoss(ordinary, ordinary, None)
    if edge_prediction is None or edge_target is None or edge_index is None:
        raise HestDeepSpotHiSTError("SEPARATE_EDGE_STREAM_REQUIRED")
    if edge_prediction.shape != edge_target.shape or edge_prediction.ndim != 2:
        raise HestDeepSpotHiSTError("EDGE_EXPRESSION_SHAPE")
    from strsc.training.shared.losses import local_difference_mse

    local = local_difference_mse(edge_prediction, edge_target, edge_index)
    return ExpressionLoss((1.0 - value) * ordinary + value * local, ordinary, local)


def named_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def require_finite_backward(loss: torch.Tensor, model: torch.nn.Module) -> None:
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise HestDeepSpotHiSTError("LOSS_NONFINITE")
    loss.backward()
    invalid = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    if invalid:
        raise HestDeepSpotHiSTError(f"GRADIENT_NONFINITE:{invalid[:5]}")
