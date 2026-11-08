"""Lightning model overlay; imports the upstream sCellST stack only at runtime."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from lightning.pytorch.utilities.types import STEP_OUTPUT

from scellst.constant import REGISTRY_KEYS
from scellst.lightning_model.gene_lightning_model import GeneLightningModel

from .objective import mixed_spot_edge_mse


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.random.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(copy.deepcopy(state["python"]))
    np.random.set_state(copy.deepcopy(state["numpy"]))
    torch.random.set_rng_state(state["torch"].clone())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.clone() for value in state["cuda"]])


@contextmanager
def isolated_auxiliary_forward(module: nn.Module):
    """Keep edge forwards from changing later ordinary RNG or BatchNorm state."""

    rng_state = _capture_rng_state()
    batch_norm_state = []
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            batch_norm_state.append(
                (
                    child,
                    None if child.running_mean is None else child.running_mean.detach().clone(),
                    None if child.running_var is None else child.running_var.detach().clone(),
                    None
                    if child.num_batches_tracked is None
                    else child.num_batches_tracked.detach().clone(),
                )
            )
    try:
        yield
    finally:
        for child, running_mean, running_var, batches in batch_norm_state:
            if running_mean is not None:
                child.running_mean.copy_(running_mean)
            if running_var is not None:
                child.running_var.copy_(running_var)
            if batches is not None:
                child.num_batches_tracked.copy_(batches)
        _restore_rng_state(rng_state)


class LocalDifferenceGeneLightningModel(GeneLightningModel):
    """sCellST with a training-only adjacent-spot expression objective."""

    def __init__(self, *args: Any, rho: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.g0055_rho = float(rho)
        if self.g0055_rho not in (0.0, 0.5):
            raise ValueError(f"Frozen G0-055 rho must be 0.0 or 0.5, got {rho}")

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        spot_batch = batch["spot"] if isinstance(batch, dict) and "spot" in batch else batch
        if self.g0055_rho == 0.0:
            # Exact upstream computation: no edge forward and no arithmetic wrapper.
            return super().training_step(spot_batch, batch_idx)
        if not isinstance(batch, dict) or "edge" not in batch:
            raise ValueError("rho=0.5 requires a paired {'spot', 'edge'} training batch")

        edge_batch = batch["edge"]
        spot_output, _ = self.model(spot_batch)
        # The ordinary forward above is the only stream allowed to advance
        # dropout RNG or persistent normalization statistics.  Edge gradients
        # still flow through the exact same train-mode model parameters.
        with isolated_auxiliary_forward(self.model):
            left_output, _ = self.model(edge_batch["left"])
            right_output, _ = self.model(edge_batch["right"])
        parts = mixed_spot_edge_mse(
            spot_output[REGISTRY_KEYS.OUTPUT_PREDICTION],
            spot_batch[REGISTRY_KEYS.Y_BAG_KEY],
            rho=self.g0055_rho,
            left_prediction=left_output[REGISTRY_KEYS.OUTPUT_PREDICTION],
            left_target=edge_batch["left"][REGISTRY_KEYS.Y_BAG_KEY],
            right_prediction=right_output[REGISTRY_KEYS.OUTPUT_PREDICTION],
            right_target=edge_batch["right"][REGISTRY_KEYS.Y_BAG_KEY],
        )
        batch_size = len(spot_batch[REGISTRY_KEYS.Y_BAG_KEY])
        self.log(
            "train_loss",
            parts.total,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_expression_mse",
            parts.ordinary_mse,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train_local_difference_mse",
            parts.local_difference_mse,
            batch_size=len(edge_batch["left"][REGISTRY_KEYS.Y_BAG_KEY]),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return parts.total


def prepare_local_difference_model(config: Any, *, rho: float) -> LocalDifferenceGeneLightningModel:
    """Mirror upstream ``prepare_model`` without modifying the repository."""

    if "rn18" in config.data.embedding_tag:
        config.predictor.input_dim = 512
    elif "one-hot-celltype" in config.data.embedding_tag:
        config.predictor.input_dim = 6
    else:
        config.predictor.input_dim = 2048
    if config.data.dataset_handler == "supervised":
        raise ValueError("G0-055 sCellST overlay requires the MIL data handler")
    return LocalDifferenceGeneLightningModel(
        predictor_config=config.predictor,
        rho=rho,
        **config.model,
    )
