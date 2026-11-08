"""Model-agnostic loss overlay applied to unmodified THItoGene predictions."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from strsc.training.shared import (
    AdjacencyGraph,
    build_within_specimen_adjacency,
    check_rho_zero_noop,
    mixed_expression_mse,
)


class THItoGeneOverlayError(RuntimeError):
    pass


@dataclass(frozen=True)
class THItoGeneSupervision:
    train_mask: np.ndarray
    validation_mask: np.ndarray
    train_graph: AdjacencyGraph
    loss_edge_index: np.ndarray

    @property
    def n_spots(self) -> int:
        return len(self.train_mask)

    @property
    def n_train(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_validation(self) -> int:
        return int(self.validation_mask.sum())

    def to_metadata(self) -> dict[str, object]:
        return {
            "spots": self.n_spots,
            "supervised_training_spots": self.n_train,
            "validation_spots": self.n_validation,
            "train_graph": self.train_graph.to_metadata(),
        }


def build_supervision(
    *,
    coords: np.ndarray,
    validation_mask: np.ndarray,
    specimen_id: str,
) -> THItoGeneSupervision:
    points = np.asarray(coords, dtype=np.float64)
    validation = np.asarray(validation_mask, dtype=bool)
    if points.ndim != 2 or points.shape != (len(validation), 2):
        raise THItoGeneOverlayError(
            f"SUPERVISION_COORDINATES:{points.shape}:{len(validation)}"
        )
    training = ~validation
    if not training.any() or not validation.any():
        raise THItoGeneOverlayError(
            f"SUPERVISION_EMPTY_PARTITION:{training.sum()}:{validation.sum()}"
        )
    graph = build_within_specimen_adjacency(
        [points[training]], specimen_ids=[specimen_id], radius_multiplier=1.5
    )
    return THItoGeneSupervision(
        train_mask=training,
        validation_mask=validation,
        train_graph=graph,
        # The shared graph is intentionally immutable. PyTorch warns when a
        # read-only NumPy view is wrapped, so retain one writable identical
        # copy for repeated differentiable loss calls.
        loss_edge_index=np.array(graph.edge_index, dtype=np.int64, copy=True),
    )


def _aligned_prediction_target(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = prediction
    truth = target
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if truth.ndim == 3 and truth.shape[0] == 1:
        truth = truth[0]
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise THItoGeneOverlayError(
            f"PREDICTION_TARGET_ALIGNMENT:{tuple(pred.shape)}:{tuple(truth.shape)}"
        )
    return pred, truth


def published_training_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervision: THItoGeneSupervision,
) -> torch.Tensor:
    pred, truth = _aligned_prediction_target(prediction, target)
    mask = torch.as_tensor(supervision.train_mask, device=pred.device)
    return F.mse_loss(pred[mask], truth[mask], reduction="mean")


def published_validation_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervision: THItoGeneSupervision,
) -> torch.Tensor:
    pred, truth = _aligned_prediction_target(prediction, target)
    mask = torch.as_tensor(supervision.validation_mask, device=pred.device)
    return F.mse_loss(pred[mask], truth[mask], reduction="mean")


def expression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervision: THItoGeneSupervision,
    *,
    rho: float,
) -> torch.Tensor:
    """Overlay the shared objective on the published supervised MSE support."""

    pred, truth = _aligned_prediction_target(prediction, target)
    mask = torch.as_tensor(supervision.train_mask, device=pred.device)
    return mixed_expression_mse(
        pred[mask],
        truth[mask],
        supervision.loss_edge_index,
        rho=rho,
    )


@dataclass
class _RngState:
    python: object
    numpy: tuple[object, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None


def _capture_rng() -> _RngState:
    return _RngState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state().clone(),
        torch_cuda=(
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    )


def _restore_rng(state: _RngState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(state.torch_cuda)


def _grads(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def forward_section(
    model: torch.nn.Module,
    *,
    patches: torch.Tensor,
    positions: torch.Tensor,
    model_adjacency: torch.Tensor,
) -> torch.Tensor:
    prediction = model(patches, positions, model_adjacency)
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction.squeeze(0)
    if prediction.ndim != 2:
        raise THItoGeneOverlayError(
            f"UPSTREAM_PREDICTION_SHAPE:{tuple(prediction.shape)}"
        )
    return prediction


def run_rho_zero_noop_gate(
    model: torch.nn.Module,
    *,
    patches: torch.Tensor,
    positions: torch.Tensor,
    model_adjacency: torch.Tensor,
    target: torch.Tensor,
    supervision: THItoGeneSupervision,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, object]:
    """Compare two stochastic upstream forwards from identical state and RNG."""

    initial_state = copy.deepcopy(model.state_dict())
    initial_rng = _capture_rng()
    original_mode = model.training
    model.train()

    model.zero_grad(set_to_none=True)
    original_prediction = forward_section(
        model,
        patches=patches,
        positions=positions,
        model_adjacency=model_adjacency,
    )
    original_loss = published_training_mse(
        original_prediction, target, supervision
    )
    original_loss.backward()
    original_gradients = _grads(model)

    model.load_state_dict(initial_state)
    _restore_rng(initial_rng)
    model.zero_grad(set_to_none=True)
    rho_zero_prediction = forward_section(
        model,
        patches=patches,
        positions=positions,
        model_adjacency=model_adjacency,
    )
    rho_zero_loss = expression_loss(
        rho_zero_prediction, target, supervision, rho=0.0
    )
    rho_zero_loss.backward()
    rho_zero_gradients = _grads(model)
    receipt = check_rho_zero_noop(
        original_loss=original_loss,
        rho_zero_loss=rho_zero_loss,
        original_predictions=original_prediction,
        rho_zero_predictions=rho_zero_prediction,
        original_gradients=original_gradients,
        rho_zero_gradients=rho_zero_gradients,
        atol=atol,
        rtol=rtol,
    )

    model.load_state_dict(initial_state)
    _restore_rng(initial_rng)
    model.train(original_mode)
    result = receipt.to_dict()
    result.update(
        {
            "published_loss": "torch.nn.functional.mse_loss(reduction=mean)",
            "overlay_rho": 0.0,
            "model_state_restored": True,
            "rng_state_restored": True,
        }
    )
    return result
