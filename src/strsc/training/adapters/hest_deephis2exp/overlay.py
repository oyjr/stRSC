"""Reversible Lightning-method overlay; optimizer and model stay upstream."""

from __future__ import annotations

from types import MethodType
from typing import Any, Mapping

import torch

from .models import canonical_model_name
from .objective import as_upstream_batch, compute_training_objective


_ORIGINAL_TRAIN = "_g0055_deephis2exp_original_training_step"
_ORIGINAL_VALIDATION = "_g0055_deephis2exp_original_validation_step"
_ORIGINAL_TRANSFER = "_g0055_deephis2exp_original_transfer_batch_to_device"
_MODEL_NAME = "_g0055_deephis2exp_model_name"
_RHO = "_g0055_deephis2exp_rho"


def _training_step(
    model: torch.nn.Module, batch: Mapping[str, Any], batch_idx: int
) -> torch.Tensor:
    rho = float(getattr(model, _RHO))
    if rho == 0.0:
        # Load-bearing no-op: the baseline loader contains only ordinary spots
        # or a partition-only slide. No edge object is received or inspected.
        return getattr(model, _ORIGINAL_TRAIN)(as_upstream_batch(batch), batch_idx)

    canonical = getattr(model, _MODEL_NAME)
    if canonical in {"ST-Net", "DeepSpaCE"}:
        if "ordinary" not in batch or "edge" not in batch:
            raise KeyError(
                "rho=0.5 batch requires independent 'ordinary' and 'edge' streams"
            )
        ordinary = batch["ordinary"]
        edge_pair = batch["edge"]
    elif canonical == "Hist2ST" and {"ordinary", "edge"}.issubset(batch):
        ordinary = batch["ordinary"]
        edge_pair = batch["edge"]
    else:
        ordinary = batch
        edge_pair = None
    result = compute_training_objective(
        canonical,
        model,
        ordinary,
        rho=rho,
        edge_pair_batch=edge_pair,
    )
    if callable(getattr(model, "log", None)):
        model.log("train_loss", result.total, on_epoch=True, prog_bar=True, logger=True)
        model.log(
            "train_pointwise_mse",
            result.pointwise_mse,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        model.log(
            "train_local_difference_mse",
            result.local_difference_mse,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        if result.auxiliary.detach().abs().item() > 0:
            model.log(
                "train_published_auxiliary",
                result.auxiliary,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
    return result.total


def _validation_step(
    model: torch.nn.Module, batch: Mapping[str, Any], batch_idx: int
) -> torch.Tensor:
    # Checkpoint selection is the untouched published validation step on the
    # inner-validation-only ordinary batch for both arms.
    return getattr(model, _ORIGINAL_VALIDATION)(as_upstream_batch(batch), batch_idx)


def _transfer_batch_to_device(
    model: torch.nn.Module,
    batch: Any,
    device: torch.device,
    dataloader_idx: int,
) -> Any:
    """Move HisToGene metadata and targets, but stream raw patches from CPU."""

    original = getattr(model, _ORIGINAL_TRANSFER)
    if not isinstance(batch, Mapping) or "images" not in batch:
        return original(batch, device, dataloader_idx)
    remainder = {key: value for key, value in batch.items() if key != "images"}
    transferred = original(remainder, device, dataloader_idx)
    return {
        key: batch["images"] if key == "images" else transferred[key]
        for key in batch
    }


def install_objective_overlay(
    model: torch.nn.Module, *, model_name: str, rho: float
) -> torch.nn.Module:
    canonical = canonical_model_name(model_name)
    rho = float(rho)
    if rho not in (0.0, 0.5):
        raise ValueError("Frozen G0-055 arms permit only rho=0.0 or rho=0.5")
    if hasattr(model, _ORIGINAL_TRAIN):
        raise RuntimeError("DeepHis2Exp objective overlay is already installed")
    if not callable(getattr(model, "training_step", None)) or not callable(
        getattr(model, "validation_step", None)
    ):
        raise TypeError("Upstream model must expose training_step and validation_step")
    setattr(model, _ORIGINAL_TRAIN, model.training_step)
    setattr(model, _ORIGINAL_VALIDATION, model.validation_step)
    setattr(model, _MODEL_NAME, canonical)
    setattr(model, _RHO, rho)
    model.training_step = MethodType(_training_step, model)
    model.validation_step = MethodType(_validation_step, model)
    transfer = getattr(model, "transfer_batch_to_device", None)
    if canonical == "HisToGene" and callable(transfer):
        setattr(model, _ORIGINAL_TRANSFER, transfer)
        model.transfer_batch_to_device = MethodType(_transfer_batch_to_device, model)
    return model


def remove_objective_overlay(model: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(model, _ORIGINAL_TRAIN):
        raise RuntimeError("DeepHis2Exp objective overlay is not installed")
    model.training_step = getattr(model, _ORIGINAL_TRAIN)
    model.validation_step = getattr(model, _ORIGINAL_VALIDATION)
    if hasattr(model, _ORIGINAL_TRANSFER):
        model.transfer_batch_to_device = getattr(model, _ORIGINAL_TRANSFER)
        delattr(model, _ORIGINAL_TRANSFER)
    for name in (_ORIGINAL_TRAIN, _ORIGINAL_VALIDATION, _MODEL_NAME, _RHO):
        delattr(model, name)
    return model
