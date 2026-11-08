"""Model-specific forwards with one shared local-difference expression term."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import random
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .batching import flatten_edge_pair_batch
from .data import HestAdapterError
from .models import canonical_model_name


@dataclass(frozen=True)
class TrainingObjective:
    total: torch.Tensor
    expression: torch.Tensor
    pointwise_mse: torch.Tensor
    local_difference_mse: torch.Tensor
    auxiliary: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    edge_prediction: torch.Tensor | None = None
    edge_target: torch.Tensor | None = None


def _required(batch: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in batch]
    if missing:
        raise HestAdapterError("BATCH_FIELDS:" + ",".join(missing))


def _mask(batch: Mapping[str, Any], rows: int, device: torch.device) -> torch.Tensor:
    value = batch.get("loss_mask")
    if value is None:
        return torch.ones(rows, dtype=torch.bool, device=device)
    result = torch.as_tensor(value, dtype=torch.bool, device=device)
    if result.shape != (rows,) or not result.any():
        raise HestAdapterError(f"LOSS_MASK:{tuple(result.shape)}:{rows}")
    return result


def _edge_index_for_masked_rows(
    batch: Mapping[str, Any], mask: torch.Tensor
) -> torch.Tensor:
    edges = torch.as_tensor(batch["edge_index"], dtype=torch.long, device=mask.device)
    if edges.ndim == 3 and edges.shape[0] == 1:
        edges = edges.squeeze(0)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise HestAdapterError(f"EDGE_INDEX_SHAPE:{tuple(edges.shape)}")
    if edges.shape[1] < 1:
        raise HestAdapterError("TRAINING_EDGE_INDEX_EMPTY")
    if int(edges.min()) < 0 or int(edges.max()) >= len(mask):
        raise HestAdapterError("TRAINING_EDGE_INDEX_RANGE")
    if not torch.all(mask[edges]):
        raise HestAdapterError("TRAINING_EDGE_TOUCHES_NONSUPERVISED_ROW")
    old_to_new = torch.full((len(mask),), -1, dtype=torch.long, device=mask.device)
    old_to_new[mask] = torch.arange(int(mask.sum()), device=mask.device)
    return old_to_new[edges]


def _forward_expression(
    model_name: str,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    training: bool,
) -> tuple[torch.Tensor, Any, Any]:
    canonical = canonical_model_name(model_name)
    _required(batch, "images", "target", "positions")
    images = batch["images"].float()
    positions = batch["positions"].long()

    if canonical == "ST-Net":
        if training:
            images = model.aug(images)
        return model(images), None, None
    if canonical == "DeepSpaCE":
        return model(images), None, None
    if canonical == "HisToGene":
        prediction = model(images.unsqueeze(0).flatten(2), positions.unsqueeze(0))
        return prediction.squeeze(0), None, None
    if canonical == "Hist2ST":
        _required(batch, "context_adjacency")
        adjacency = batch["context_adjacency"].float()
        if adjacency.ndim == 3 and adjacency.shape[0] == 1:
            adjacency = adjacency.squeeze(0)
        prediction, extra, hidden = model(
            images.unsqueeze(0), positions.unsqueeze(0), adjacency
        )
        return prediction.squeeze(0), extra, hidden
    raise AssertionError(canonical)


def _hist2st_auxiliary(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    prediction: torch.Tensor,
    extra: Any,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bake_term = prediction.new_zeros(())
    zinb_term = prediction.new_zeros(())
    bake_weight = float(getattr(model, "lamb", 0.0))
    bake_count = int(getattr(model, "bake", 0))
    if bake_count > 0:
        images = batch["images"].float().unsqueeze(0)
        positions = batch["positions"].long().unsqueeze(0)
        adjacency = batch["context_adjacency"].float()
        if adjacency.ndim == 3 and adjacency.shape[0] == 1:
            adjacency = adjacency.squeeze(0)
        augmented = model.aug(images, positions, adjacency)
        distilled = model.distillation(augmented)
        # This published image-only consistency term is unchanged and is
        # evaluated exactly once on the ordinary inner-training slide.
        bake_loss = F.mse_loss(distilled, prediction)
        bake_term = bake_weight * bake_loss

    zinb_weight = float(getattr(model, "zinb", 0.0))
    if zinb_weight > 0:
        _required(batch, "raw_counts", "size_factors")
        if extra is None:
            raise HestAdapterError("HIST2ST_ZINB_OUTPUT_MISSING")
        from Models.Hist2ST.NB_module import NB_loss, ZINB_loss

        raw = batch["raw_counts"].float()[mask]
        size_factors = batch["size_factors"].float()[mask]
        if bool(getattr(model, "nb", False)):
            r, p = extra
            distribution_loss = NB_loss(raw, r[mask], p[mask])
        else:
            mean, dispersion, probability = extra
            distribution_loss = ZINB_loss(
                raw,
                mean[mask],
                dispersion[mask],
                probability[mask],
                size_factors,
            )
        zinb_term = zinb_weight * distribution_loss
    # Upstream computes bake first, then ZINB, and finally evaluates
    # ``mse + zinb_term + bake_term``. Returning the terms separately lets the
    # caller preserve that floating-point addition order for the rho-zero gate.
    return zinb_term, bake_term


def _pointwise_and_auxiliary(
    model_name: str,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the ordinary batch once and retain published auxiliary weights."""

    _required(batch, "target")
    prediction, extra, _ = _forward_expression(
        model_name, model, batch, training=True
    )
    target = batch["target"].float().to(prediction.device)
    if prediction.shape != target.shape:
        raise HestAdapterError(
            f"OUTPUT_ALIGNMENT:{tuple(prediction.shape)}:{tuple(target.shape)}"
        )
    mask = _mask(batch, len(prediction), prediction.device)
    pointwise = F.mse_loss(prediction[mask], target[mask])

    canonical = canonical_model_name(model_name)
    auxiliary = prediction.new_zeros(())
    if canonical == "Hist2ST":
        zinb_term, bake_term = _hist2st_auxiliary(
            model, batch, prediction, extra, mask
        )
        auxiliary = zinb_term + bake_term
    return pointwise, auxiliary, prediction, target


def _local_difference_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    residual = prediction - target
    return F.mse_loss(residual[edge_index[0]], residual[edge_index[1]])


@contextmanager
def _restore_rng_after_extra_stream():
    """Keep intervention-only forwards from advancing the ordinary RNG stream."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def _batchnorm_without_running_stat_updates(model: torch.nn.Module):
    """Use training batch statistics without mutating ordinary-stream state."""

    batch_norm_state = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            batch_norm_state.append((module, bool(module.track_running_stats)))
            module.track_running_stats = False
    try:
        yield
    finally:
        for module, track_running_stats in batch_norm_state:
            module.track_running_stats = track_running_stats


def _patch_edge_forward_preserving_ordinary_stream(
    model_name: str,
    model: torch.nn.Module,
    edge_batch: Mapping[str, Any],
) -> torch.Tensor:
    """Isolate the extra edge forward from later ordinary stochastic draws."""

    with _restore_rng_after_extra_stream(), _batchnorm_without_running_stat_updates(
        model
    ):
        prediction, _, _ = _forward_expression(
            model_name, model, edge_batch, training=True
        )
    return prediction


def _checkpointed_context_edge_forward(
    model_name: str,
    model: torch.nn.Module,
    edge_batch: Mapping[str, Any],
) -> torch.Tensor:
    """Checkpoint one Hist2ST edge microbatch, including its backward replay."""

    _required(edge_batch, "images", "positions", "context_adjacency")
    fixed_batch = dict(edge_batch)

    def forward(
        images: torch.Tensor,
        positions: torch.Tensor,
        context_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        current = dict(fixed_batch)
        current["images"] = images
        current["positions"] = positions
        current["context_adjacency"] = context_adjacency
        with _batchnorm_without_running_stat_updates(model):
            prediction, _, _ = _forward_expression(
                model_name, model, current, training=True
            )
        return prediction

    return checkpoint(
        forward,
        edge_batch["images"],
        edge_batch["positions"],
        edge_batch["context_adjacency"],
        use_reentrant=False,
        preserve_rng_state=True,
    )


def compute_patch_training_objective(
    model_name: str,
    model: torch.nn.Module,
    ordinary_batch: Mapping[str, Any],
    edge_pair_batch: Mapping[str, Any],
) -> TrainingObjective:
    """Patch-model rho-half loss with independent ordinary and edge streams."""

    canonical = canonical_model_name(model_name)
    if canonical not in {"ST-Net", "DeepSpaCE"}:
        raise HestAdapterError(f"PATCH_OBJECTIVE_MODEL:{canonical}")
    pointwise, auxiliary, prediction, target = _pointwise_and_auxiliary(
        canonical, model, ordinary_batch
    )
    edge_batch = flatten_edge_pair_batch(edge_pair_batch)
    edge_prediction = _patch_edge_forward_preserving_ordinary_stream(
        canonical, model, edge_batch
    )
    edge_target = edge_batch["target"].float().to(edge_prediction.device)
    if edge_prediction.shape != edge_target.shape:
        raise HestAdapterError("EDGE_OUTPUT_ALIGNMENT")
    edges = _edge_index_for_masked_rows(
        edge_batch,
        torch.ones(len(edge_prediction), dtype=torch.bool, device=edge_prediction.device),
    )
    local = _local_difference_mse(edge_prediction, edge_target, edges)
    expression = 0.5 * pointwise + 0.5 * local
    total = expression + auxiliary
    return TrainingObjective(
        total=total,
        expression=expression,
        pointwise_mse=pointwise,
        local_difference_mse=local,
        auxiliary=auxiliary,
        prediction=prediction,
        target=target,
        edge_prediction=edge_prediction,
        edge_target=edge_target,
    )


def compute_context_training_objective(
    model_name: str,
    model: torch.nn.Module,
    ordinary_batch: Mapping[str, Any],
) -> TrainingObjective:
    """Context-model rho-half loss on a train-only slide and train-only edges."""

    canonical = canonical_model_name(model_name)
    if canonical not in {"HisToGene", "Hist2ST"}:
        raise HestAdapterError(f"CONTEXT_OBJECTIVE_MODEL:{canonical}")
    _required(ordinary_batch, "edge_index")
    pointwise, auxiliary, prediction, target = _pointwise_and_auxiliary(
        canonical, model, ordinary_batch
    )
    mask = _mask(ordinary_batch, len(prediction), prediction.device)
    edges = _edge_index_for_masked_rows(ordinary_batch, mask)
    local = _local_difference_mse(prediction[mask], target[mask], edges)
    expression = 0.5 * pointwise + 0.5 * local
    total = expression + auxiliary
    return TrainingObjective(
        total=total,
        expression=expression,
        pointwise_mse=pointwise,
        local_difference_mse=local,
        auxiliary=auxiliary,
        prediction=prediction,
        target=target,
    )


def compute_context_edge_stream_training_objective(
    model_name: str,
    model: torch.nn.Module,
    ordinary_batch: Mapping[str, Any],
    edge_stream: Mapping[str, Any],
) -> TrainingObjective:
    """Hist2ST rho-half loss using every frozen edge in checkpointed microbatches."""

    canonical = canonical_model_name(model_name)
    if canonical != "Hist2ST":
        raise HestAdapterError(f"CONTEXT_EDGE_STREAM_MODEL:{canonical}")
    pointwise, auxiliary, prediction, target = _pointwise_and_auxiliary(
        canonical, model, ordinary_batch
    )
    microbatches = edge_stream.get("microbatches")
    if not isinstance(microbatches, (list, tuple)) or not microbatches:
        raise HestAdapterError("CONTEXT_EDGE_MICROBATCH_STREAM")

    local_sse = prediction.new_zeros(())
    local_elements = 0
    observed_edges = 0
    edge_predictions: list[torch.Tensor] = []
    edge_targets: list[torch.Tensor] = []
    with _restore_rng_after_extra_stream():
        for edge_batch in microbatches:
            if not isinstance(edge_batch, Mapping):
                raise HestAdapterError("CONTEXT_EDGE_MICROBATCH_MAPPING")
            edge_prediction = _checkpointed_context_edge_forward(
                canonical, model, edge_batch
            )
            edge_target = edge_batch["target"].float().to(edge_prediction.device)
            if edge_prediction.shape != edge_target.shape:
                raise HestAdapterError("CONTEXT_EDGE_OUTPUT_ALIGNMENT")
            mask = _mask(edge_batch, len(edge_prediction), edge_prediction.device)
            edges = _edge_index_for_masked_rows(edge_batch, mask)
            residual = edge_prediction[mask] - edge_target[mask]
            left_residual = residual[edges[0]]
            right_residual = residual[edges[1]]
            local_sse = local_sse + F.mse_loss(
                left_residual, right_residual, reduction="sum"
            )
            local_elements += int(left_residual.numel())
            observed_edges += int(edges.shape[1])
            edge_predictions.append(edge_prediction)
            edge_targets.append(edge_target)

    declared_edges = int(edge_stream.get("edge_count", observed_edges))
    if observed_edges != declared_edges or local_elements < 1:
        raise HestAdapterError(
            f"CONTEXT_EDGE_MICROBATCH_COVERAGE:{observed_edges}:{declared_edges}"
        )
    local = local_sse / local_elements
    expression = 0.5 * pointwise + 0.5 * local
    total = expression + auxiliary
    return TrainingObjective(
        total=total,
        expression=expression,
        pointwise_mse=pointwise,
        local_difference_mse=local,
        auxiliary=auxiliary,
        prediction=prediction,
        target=target,
        edge_prediction=torch.cat(edge_predictions, dim=0),
        edge_target=torch.cat(edge_targets, dim=0),
    )


def compute_training_objective(
    model_name: str,
    model: torch.nn.Module,
    ordinary_batch: Mapping[str, Any],
    *,
    rho: float,
    edge_pair_batch: Mapping[str, Any] | None = None,
) -> TrainingObjective:
    """Compatibility dispatcher for the only custom arm, ``rho=0.5``.

    Rho zero is deliberately unavailable here: the overlay must call the saved
    upstream ``training_step`` so the engineering no-op is structural rather
    than an assumed algebraic equivalence.
    """

    if float(rho) != 0.5:
        raise ValueError("Custom HEST objective is defined only for rho=0.5")
    canonical = canonical_model_name(model_name)
    if canonical in {"ST-Net", "DeepSpaCE"}:
        if edge_pair_batch is None:
            raise HestAdapterError("PATCH_EDGE_STREAM_REQUIRED")
        return compute_patch_training_objective(
            canonical, model, ordinary_batch, edge_pair_batch
        )
    if canonical == "Hist2ST" and edge_pair_batch is not None:
        return compute_context_edge_stream_training_objective(
            canonical, model, ordinary_batch, edge_pair_batch
        )
    if edge_pair_batch is not None:
        raise HestAdapterError("CONTEXT_EDGE_PAIR_STREAM_FORBIDDEN")
    return compute_context_training_objective(canonical, model, ordinary_batch)


def compute_validation_loss(
    model_name: str,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ordinary published expression MSE on the frozen validation partition."""

    prediction, extra, _ = _forward_expression(model_name, model, batch, training=False)
    target = batch["target"].float().to(prediction.device)
    if prediction.shape != target.shape:
        raise HestAdapterError("VALIDATION_OUTPUT_ALIGNMENT")
    mask = _mask(batch, len(prediction), prediction.device)
    ordinary = F.mse_loss(prediction[mask], target[mask])
    if canonical_model_name(model_name) == "Hist2ST":
        zinb_term, bake_term = _hist2st_auxiliary(
            model, batch, prediction, extra, mask
        )
        ordinary = ordinary + zinb_term + bake_term
    return ordinary, prediction


def predict_expression(
    model_name: str, model: torch.nn.Module, batch: Mapping[str, Any]
) -> torch.Tensor:
    prediction, _, _ = _forward_expression(model_name, model, batch, training=False)
    return prediction


def as_upstream_batch(batch: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
    """Convert an adapter batch to DeepHis2Exp's six-element slide tuple."""

    images = batch["images"].float().unsqueeze(0)
    positions = batch["positions"].long().unsqueeze(0)
    target = batch["target"].float().unsqueeze(0)
    adjacency = batch.get("context_adjacency")
    if adjacency is None:
        adjacency = torch.eye(len(batch["target"]), dtype=torch.float32)
    adjacency = torch.as_tensor(adjacency).float().unsqueeze(0)
    raw = batch["raw_counts"].float().unsqueeze(0)
    size_factors = batch["size_factors"].float().unsqueeze(0)
    return images, positions, target, adjacency, raw, size_factors
