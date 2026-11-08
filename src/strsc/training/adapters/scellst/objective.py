"""Neighbouring-difference expression loss for sCellST spot bags."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossParts:
    total: torch.Tensor
    ordinary_mse: torch.Tensor
    local_difference_mse: torch.Tensor | None


def _validate_pair_tensors(
    left_prediction: torch.Tensor,
    left_target: torch.Tensor,
    right_prediction: torch.Tensor,
    right_target: torch.Tensor,
) -> None:
    if left_prediction.shape != left_target.shape:
        raise ValueError("Left prediction and target shapes differ")
    if right_prediction.shape != right_target.shape:
        raise ValueError("Right prediction and target shapes differ")
    if left_prediction.shape != right_prediction.shape:
        raise ValueError("Left and right edge endpoint shapes differ")
    if left_prediction.ndim != 2 or left_prediction.shape[0] == 0:
        raise ValueError("Edge endpoint tensors must have non-empty shape [pairs, genes]")


def paired_local_difference_mse(
    left_prediction: torch.Tensor,
    left_target: torch.Tensor,
    right_prediction: torch.Tensor,
    right_target: torch.Tensor,
) -> torch.Tensor:
    """Use the shared incidence-loss implementation on explicit edge pairs."""

    _validate_pair_tensors(left_prediction, left_target, right_prediction, right_target)
    from strsc.training.shared.losses import local_difference_mse

    n_pairs = left_prediction.shape[0]
    prediction = torch.cat((left_prediction, right_prediction), dim=0)
    target = torch.cat((left_target, right_target), dim=0)
    left = torch.arange(n_pairs, device=prediction.device, dtype=torch.long)
    edge_index = torch.stack((left, left + n_pairs), dim=0)
    return local_difference_mse(prediction, target, edge_index)


def mixed_spot_edge_mse(
    spot_prediction: torch.Tensor,
    spot_target: torch.Tensor,
    *,
    rho: float,
    left_prediction: torch.Tensor | None = None,
    left_target: torch.Tensor | None = None,
    right_prediction: torch.Tensor | None = None,
    right_target: torch.Tensor | None = None,
) -> LossParts:
    """Mix ordinary spot MSE with adjacent-spot residual-difference MSE.

    ``rho=0`` returns ``F.mse_loss`` directly and deliberately does not inspect
    or evaluate edge tensors.  This is the load-bearing baseline no-op path.
    """

    value = float(rho)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"rho must lie in [0, 1], got {rho}")
    ordinary = F.mse_loss(spot_prediction, spot_target)
    if value == 0.0:
        return LossParts(total=ordinary, ordinary_mse=ordinary, local_difference_mse=None)
    if any(
        tensor is None
        for tensor in (left_prediction, left_target, right_prediction, right_target)
    ):
        raise ValueError("rho > 0 requires all four edge endpoint tensors")
    local = paired_local_difference_mse(
        left_prediction,
        left_target,
        right_prediction,
        right_target,
    )
    total = (1.0 - value) * ordinary + value * local
    return LossParts(total=total, ordinary_mse=ordinary, local_difference_mse=local)
