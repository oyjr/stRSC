"""Reversible instance-level training overlay for DeepSpot2Cell."""

from __future__ import annotations

from types import MethodType
from typing import Any, Mapping

import torch

from .objective import assert_deepspot2cell_mse_contract, compute_local_difference_loss


_ORIGINAL_STEP_ATTR = "_g0055_original_training_step"
_RHO_ATTR = "_g0055_local_difference_rho"


def _paired_training_step(
    model: torch.nn.Module, batch: Mapping[str, Any], batch_idx: int
) -> torch.Tensor:
    rho = float(getattr(model, _RHO_ATTR))
    if rho == 0.0:
        # Load-bearing no-op: use the bound upstream method directly. The
        # baseline arm receives an ordinary loader and never constructs or
        # indexes an edge stream.
        return getattr(model, _ORIGINAL_STEP_ATTR)(batch, batch_idx)
    if "ordinary" not in batch or "edge" not in batch:
        raise KeyError("rho=0.5 batch must contain independent 'ordinary' and 'edge' streams")
    result = compute_local_difference_loss(
        model,
        batch["ordinary"],
        rho=rho,
        edge_pair_batch=batch["edge"],
    )
    if callable(getattr(model, "log", None)):
        model.log("train_loss", result.total, on_epoch=True, prog_bar=True)
        model.log(
            "train_pointwise_mse", result.pointwise_mse, on_epoch=True, prog_bar=False
        )
        model.log(
            "train_local_difference_mse",
            result.local_difference_mse,
            on_epoch=True,
            prog_bar=False,
        )
    train_losses = getattr(model, "train_batch_losses", None)
    if isinstance(train_losses, list):
        train_losses.append(float(result.total.detach().cpu()))
    return result.total


def install_local_difference_overlay(
    model: torch.nn.Module, *, rho: float
) -> torch.nn.Module:
    """Patch only ``training_step``; validation and optimization stay upstream."""

    assert_deepspot2cell_mse_contract(model)
    rho = float(rho)
    if rho not in (0.0, 0.5):
        raise ValueError("Frozen G0-055 DeepSpot2Cell arms permit only rho=0.0 or rho=0.5")
    if hasattr(model, _ORIGINAL_STEP_ATTR):
        raise RuntimeError("G0-055 local-difference overlay is already installed")
    if not callable(getattr(model, "training_step", None)):
        raise TypeError("DeepSpot2Cell model does not expose training_step")

    setattr(model, _ORIGINAL_STEP_ATTR, model.training_step)
    setattr(model, _RHO_ATTR, rho)
    model.training_step = MethodType(_paired_training_step, model)
    return model


def remove_local_difference_overlay(model: torch.nn.Module) -> torch.nn.Module:
    """Restore the exact bound upstream training method."""

    if not hasattr(model, _ORIGINAL_STEP_ATTR):
        raise RuntimeError("G0-055 local-difference overlay is not installed")
    model.training_step = getattr(model, _ORIGINAL_STEP_ATTR)
    delattr(model, _ORIGINAL_STEP_ATTR)
    delattr(model, _RHO_ATTR)
    return model
