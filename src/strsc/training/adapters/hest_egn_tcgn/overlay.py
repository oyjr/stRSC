"""Add neighbouring-difference loss to EGN and TCGN training."""

from __future__ import annotations

import copy
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from strsc.training.shared import (
    AdjacencyGraph,
    build_within_specimen_adjacency,
    check_rho_zero_noop,
    local_difference_mse,
    mixed_expression_mse,
)


class OverlayContractError(RuntimeError):
    pass


@contextmanager
def preserve_model_buffers(model: torch.nn.Module):
    """Prevent edge-only forwards from adding non-objective state updates.

    Gradients through edge predictions are retained.  BatchNorm running means,
    variances and all other non-parameter buffers are restored afterwards, so
    both paired arms update persistent buffers from the ordinary stream only.
    """

    snapshots = {
        name: value.detach().clone() for name, value in model.named_buffers()
    }
    try:
        yield
    finally:
        with torch.no_grad():
            current = dict(model.named_buffers())
            if set(current) != set(snapshots):
                raise OverlayContractError("MODEL_BUFFER_KEYS_CHANGED_DURING_EDGE_FORWARD")
            for name, value in current.items():
                value.copy_(snapshots[name])


@dataclass(frozen=True)
class SpotSupervision:
    specimen_id: str
    train_mask: np.ndarray
    validation_mask: np.ndarray
    train_rows: np.ndarray
    train_graph: AdjacencyGraph
    loss_edge_index: np.ndarray

    @property
    def n_spots(self) -> int:
        return len(self.train_mask)

    @property
    def n_train(self) -> int:
        return len(self.train_rows)

    @property
    def n_validation(self) -> int:
        return int(self.validation_mask.sum())

    def to_metadata(self) -> dict[str, object]:
        return {
            "specimen": self.specimen_id,
            "spots": self.n_spots,
            "supervised_training_spots": self.n_train,
            "validation_spots": self.n_validation,
            "train_graph": self.train_graph.to_metadata(),
            "validation_rna_in_training_loss": False,
            "validation_rna_in_local_graph": False,
        }


def build_supervision(
    *, coords: np.ndarray, validation_mask: np.ndarray, specimen_id: str
) -> SpotSupervision:
    points = np.asarray(coords, dtype=np.float64)
    validation = np.asarray(validation_mask, dtype=bool)
    if points.ndim != 2 or points.shape != (len(validation), 2):
        raise OverlayContractError(
            f"SUPERVISION_COORDINATES:{points.shape}:{len(validation)}"
        )
    training = ~validation
    if not training.any() or not validation.any():
        raise OverlayContractError(
            f"SUPERVISION_EMPTY_PARTITION:{training.sum()}:{validation.sum()}"
        )
    train_rows = np.flatnonzero(training).astype(np.int64)
    graph = build_within_specimen_adjacency(
        [points[training]], specimen_ids=[specimen_id], radius_multiplier=1.5
    )
    return SpotSupervision(
        specimen_id=str(specimen_id),
        train_mask=training,
        validation_mask=validation,
        train_rows=train_rows,
        train_graph=graph,
        loss_edge_index=np.array(graph.edge_index, dtype=np.int64, copy=True),
    )


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact differentiable EGN ``correlationMetric`` over genes."""

    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise OverlayContractError(
            f"CORRELATION_SHAPE:{tuple(prediction.shape)}:{tuple(target.shape)}"
        )
    correlation = prediction.new_zeros(())
    for index in range(prediction.size(1)):
        x = prediction[:, index]
        y = target[:, index]
        centered_x = x - torch.mean(x)
        centered_y = y - torch.mean(y)
        numerator = centered_x.dot(centered_y)
        denominator = torch.norm(centered_x, 2) * torch.norm(centered_y, 2)
        value = numerator / (denominator + 1e-8)
        value = torch.nan_to_num(value, nan=-1.0)
        correlation = correlation + value
    correlation = correlation / prediction.size(1)
    return (1.0 - correlation).mean()


def _validate_rho(rho: float) -> float:
    value = float(rho)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise OverlayContractError(f"RHO_RANGE:{rho}")
    return value


def full_section_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervision: SpotSupervision,
    *,
    rho: float,
    correlation_weight: float,
) -> torch.Tensor:
    """EGNv2 loss on a full section, with validation RNA masked out."""

    value = _validate_rho(rho)
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise OverlayContractError(
            f"FULL_SECTION_ALIGNMENT:{tuple(prediction.shape)}:{tuple(target.shape)}"
        )
    mask = torch.as_tensor(supervision.train_mask, dtype=torch.bool, device=prediction.device)
    pred = prediction[mask]
    truth = target[mask]
    expression = mixed_expression_mse(
        pred, truth, supervision.loss_edge_index, rho=value
    )
    if float(correlation_weight) == 0.0:
        return expression
    return expression + float(correlation_weight) * correlation_loss(pred, truth)


def published_full_section_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervision: SpotSupervision,
    *,
    correlation_weight: float,
) -> torch.Tensor:
    mask = torch.as_tensor(supervision.train_mask, dtype=torch.bool, device=prediction.device)
    pred = prediction[mask]
    truth = target[mask]
    result = F.mse_loss(pred, truth, reduction="mean")
    if float(correlation_weight) != 0.0:
        result = result + float(correlation_weight) * correlation_loss(pred, truth)
    return result


def validation_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supervision: SpotSupervision,
) -> torch.Tensor:
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise OverlayContractError(
            f"VALIDATION_ALIGNMENT:{tuple(prediction.shape)}:{tuple(target.shape)}"
        )
    mask = torch.as_tensor(
        supervision.validation_mask, dtype=torch.bool, device=prediction.device
    )
    return F.mse_loss(prediction[mask], target[mask], reduction="mean")


def minibatch_loss(
    ordinary_prediction: torch.Tensor,
    ordinary_target: torch.Tensor,
    *,
    rho: float,
    correlation_weight: float,
    edge_prediction: Optional[torch.Tensor] = None,
    edge_target: Optional[torch.Tensor] = None,
    edge_index: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> torch.Tensor:
    """Spot-batch loss for EGNv1/TCGN with an explicit edge endpoint batch.

    ``rho=0`` is a load-bearing direct branch: no edge tensor is required or
    touched.  At positive rho the ordinary batch remains the exact upstream MSE
    and EGN auxiliary-correlation support; a second, explicit spatial-edge batch
    supplies only the intervention term.
    """

    value = _validate_rho(rho)
    if ordinary_prediction.ndim != 2 or ordinary_prediction.shape != ordinary_target.shape:
        raise OverlayContractError(
            f"MINIBATCH_ALIGNMENT:{tuple(ordinary_prediction.shape)}:"
            f"{tuple(ordinary_target.shape)}"
        )
    ordinary = F.mse_loss(ordinary_prediction, ordinary_target, reduction="mean")
    auxiliary = (
        ordinary.new_zeros(())
        if float(correlation_weight) == 0.0
        else float(correlation_weight)
        * correlation_loss(ordinary_prediction, ordinary_target)
    )
    if value == 0.0:
        return ordinary + auxiliary
    if edge_prediction is None or edge_target is None or edge_index is None:
        raise OverlayContractError("POSITIVE_RHO_REQUIRES_EDGE_ENDPOINT_BATCH")
    local = local_difference_mse(edge_prediction, edge_target, edge_index)
    expression = local if value == 1.0 else (1.0 - value) * ordinary + value * local
    return expression + auxiliary


def published_minibatch_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    correlation_weight: float,
) -> torch.Tensor:
    result = F.mse_loss(prediction, target, reduction="mean")
    if float(correlation_weight) != 0.0:
        result = result + float(correlation_weight) * correlation_loss(prediction, target)
    return result


def edge_endpoint_batch(
    supervision: SpotSupervision,
    *,
    maximum_edges: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full-section rows and a remapped edge index for one edge batch."""

    edges = supervision.loss_edge_index
    if maximum_edges < 1:
        raise OverlayContractError(f"EDGE_BATCH_SIZE:{maximum_edges}")
    count = min(int(maximum_edges), edges.shape[1])
    chosen = np.sort(generator.choice(edges.shape[1], size=count, replace=False))
    selected = edges[:, chosen]
    train_nodes = np.unique(selected.reshape(-1))
    remap = np.full(supervision.n_train, -1, dtype=np.int64)
    remap[train_nodes] = np.arange(len(train_nodes), dtype=np.int64)
    local_edges = remap[selected]
    full_rows = supervision.train_rows[train_nodes]
    if (local_edges < 0).any():
        raise OverlayContractError("EDGE_BATCH_REMAP")
    return full_rows, local_edges


def edge_pair_batch(
    supervision: SpotSupervision,
    *,
    batch_size: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a separate explicit edge stream, never incidental batch neighbours.

    Returns two full-section row arrays, one for each endpoint.  Repeated nodes
    remain repeated so every sampled edge has equal weight in the difference
    term, independently of ordinary-spot weights.
    """

    if batch_size < 1:
        raise OverlayContractError(f"EDGE_PAIR_BATCH_SIZE:{batch_size}")
    edges = supervision.loss_edge_index
    replace = edges.shape[1] < batch_size
    chosen = generator.choice(edges.shape[1], size=batch_size, replace=replace)
    left_rows = supervision.train_rows[edges[0, chosen]]
    right_rows = supervision.train_rows[edges[1, chosen]]
    if left_rows.shape != (batch_size,) or right_rows.shape != (batch_size,):
        raise OverlayContractError("EDGE_PAIR_BATCH_CONTRACT")
    return left_rows, right_rows


@dataclass
class _RngState:
    python: object
    numpy: tuple[object, ...]
    torch_cpu: torch.Tensor
    torch_cuda: Optional[list[torch.Tensor]]


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


def _gradients(model: torch.nn.Module) -> dict[str, Optional[torch.Tensor]]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def run_rho_zero_noop_gate(
    model: torch.nn.Module,
    *,
    forward: Callable[[], torch.Tensor],
    published_loss: Callable[[torch.Tensor], torch.Tensor],
    rho_zero_loss: Callable[[torch.Tensor], torch.Tensor],
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, object]:
    """Compare loss, prediction and every named gradient from identical RNG."""

    initial_state = copy.deepcopy(model.state_dict())
    initial_rng = _capture_rng()
    original_mode = model.training
    model.train()

    model.zero_grad(set_to_none=True)
    original_prediction = forward()
    original_loss = published_loss(original_prediction)
    original_loss.backward()
    original_gradients = _gradients(model)

    model.load_state_dict(initial_state)
    _restore_rng(initial_rng)
    model.zero_grad(set_to_none=True)
    overlay_prediction = forward()
    overlay_loss = rho_zero_loss(overlay_prediction)
    overlay_loss.backward()
    overlay_gradients = _gradients(model)

    receipt = check_rho_zero_noop(
        original_loss=original_loss,
        rho_zero_loss=overlay_loss,
        original_predictions=original_prediction,
        rho_zero_predictions=overlay_prediction,
        original_gradients=original_gradients,
        rho_zero_gradients=overlay_gradients,
        atol=atol,
        rtol=rtol,
    )
    model.load_state_dict(initial_state)
    _restore_rng(initial_rng)
    model.train(original_mode)
    result = receipt.to_dict()
    result.update(
        {
            "rho": 0.0,
            "all_named_parameters_compared": True,
            "model_state_restored": True,
            "rng_state_restored": True,
        }
    )
    return result
