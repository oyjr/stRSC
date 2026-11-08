"""Publication-budget paired training loops used by the formal runner."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .contracts import MODEL_CONFIGS, canonical_model_name
from .exemplars import (
    ExemplarBank,
    RetrievedExemplars,
    build_training_bank,
    retrieve_exemplars,
)
from .hest_data import FoldData, SectionMetadata, normalize_model_patches
from .model_inputs import EGNv2GraphInput, build_egn_v2_graph, egn_v1_batch
from .overlay import (
    SpotSupervision,
    build_supervision,
    full_section_loss,
    minibatch_loss,
    preserve_model_buffers,
    validation_mse,
)
from .pairing import edge_generator, epoch_order, seed_everything, step_rng_seed


class PairedTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaterializedTraining:
    model: str
    fold_data: FoldData
    raw_patches: tuple[torch.Tensor, ...]
    model_patches: Optional[tuple[torch.Tensor, ...]]
    features: Optional[tuple[torch.Tensor, ...]]
    exemplar_bank: Optional[ExemplarBank]
    v1_exemplars: Optional[tuple[RetrievedExemplars, ...]]
    graphs: Optional[tuple[EGNv2GraphInput, ...]]
    supervisions: tuple[SpotSupervision, ...]


@dataclass(frozen=True)
class ArmCheckpoint:
    rho: float
    best_epoch: int
    best_validation_mse: float
    final_training_loss: float
    state_dict: Mapping[str, torch.Tensor]
    termination: str = "epoch_budget_complete"
    termination_epoch: Optional[int] = None
    termination_step: Optional[int] = None
    epochs_completed: Optional[int] = None


def build_materialized_training(
    fold_data: FoldData,
    *,
    raw_patches: Sequence[torch.Tensor],
    features: Optional[Sequence[torch.Tensor]],
    torch_geometric_module=None,
) -> MaterializedTraining:
    name = canonical_model_name(fold_data.model)
    if len(raw_patches) != len(fold_data.training):
        raise PairedTrainingError("RAW_PATCH_SECTION_COUNT")
    supervisions = tuple(
        build_supervision(
            coords=section.spatial_coords,
            validation_mask=section.validation_mask,
            specimen_id=section.specimen,
        )
        for section in fold_data.training
    )
    model_patches = None
    bank = None
    v1_exemplars = None
    graphs = None
    feature_tuple = None if features is None else tuple(features)
    if name == "TCGN":
        if features is not None:
            raise PairedTrainingError("TCGN_MUST_NOT_RECEIVE_EXEMPLAR_FEATURES")
    else:
        if features is None or len(features) != len(fold_data.training):
            raise PairedTrainingError("EGN_FEATURE_SECTION_COUNT")
        bank = build_training_bank(name, fold_data.training, features)
        single_reference = len(fold_data.training) == 1
        if name == "EGNv1":
            cached: list[RetrievedExemplars] = []
            for section, feature_block in zip(fold_data.training, features):
                feature_chunks: list[torch.Tensor] = []
                target_chunks: list[torch.Tensor] = []
                specimen_chunks: list[np.ndarray] = []
                barcode_chunks: list[np.ndarray] = []
                for start in range(0, len(feature_block), 256):
                    chunk = retrieve_exemplars(
                        feature_block[start : start + 256],
                        query_specimen=section.specimen,
                        bank=bank,
                        query_barcodes=section.barcodes[start : start + 256],
                        query_coords=section.spatial_coords[start : start + 256],
                        allow_single_reference_fallback=single_reference,
                    )
                    feature_chunks.append(chunk.features)
                    target_chunks.append(chunk.targets)
                    specimen_chunks.append(chunk.source_specimens)
                    barcode_chunks.append(chunk.source_barcodes)
                cached.append(
                    RetrievedExemplars(
                        features=torch.cat(feature_chunks, dim=0),
                        targets=torch.cat(target_chunks, dim=0),
                        source_specimens=np.concatenate(specimen_chunks, axis=0),
                        source_barcodes=np.concatenate(barcode_chunks, axis=0),
                    )
                )
            v1_exemplars = tuple(cached)
        else:
            if torch_geometric_module is None:
                raise PairedTrainingError("EGNV2_REQUIRES_TORCH_GEOMETRIC")
            graphs = tuple(
                build_egn_v2_graph(
                    query_features=feature,
                    query_coords=section.spatial_coords,
                    query_barcodes=section.barcodes,
                    query_specimen=section.specimen,
                    bank=bank,
                    torch_geometric_module=torch_geometric_module,
                    allow_single_reference_fallback=single_reference,
                )
                for section, feature in zip(fold_data.training, features)
            )
    return MaterializedTraining(
        model=name,
        fold_data=fold_data,
        # EGNv2 trains entirely on its published frozen ResNet18 graph inputs.
        raw_patches=(tuple() if name == "EGNv2" else tuple(raw_patches)),
        model_patches=model_patches,
        features=feature_tuple,
        exemplar_bank=bank,
        v1_exemplars=v1_exemplars,
        graphs=graphs,
        supervisions=supervisions,
    )


def _global_training_rows(
    materialized: MaterializedTraining,
) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    for section_index, supervision in enumerate(materialized.supervisions):
        rows.extend((section_index, int(row)) for row in supervision.train_rows)
    return np.asarray(rows, dtype=np.int64)


def _global_training_edges(
    materialized: MaterializedTraining,
) -> np.ndarray:
    rows: list[tuple[int, int, int]] = []
    for section_index, supervision in enumerate(materialized.supervisions):
        for left, right in supervision.loss_edge_index.T:
            rows.append(
                (
                    section_index,
                    int(supervision.train_rows[int(left)]),
                    int(supervision.train_rows[int(right)]),
                )
            )
    if not rows:
        raise PairedTrainingError("NO_TRAINING_EDGES")
    return np.asarray(rows, dtype=np.int64)


def _targets(
    sections: Sequence[SectionMetadata], pairs: np.ndarray, *, device: torch.device
) -> torch.Tensor:
    values = [sections[int(s)].target[int(row)] for s, row in pairs]
    return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)


def _egn_v1_mixed_inputs(
    materialized: MaterializedTraining,
    pairs: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if (
        materialized.features is None
        or materialized.exemplar_bank is None
        or materialized.v1_exemplars is None
    ):
        raise PairedTrainingError("EGNV1_INPUTS_NOT_MATERIALIZED")
    result: dict[str, torch.Tensor] = {}
    for section_index in np.unique(pairs[:, 0]):
        positions = np.flatnonzero(pairs[:, 0] == section_index)
        section_rows = pairs[positions, 1]
        rows_tensor = torch.as_tensor(section_rows, dtype=torch.long)
        cached = materialized.v1_exemplars[int(section_index)]
        block = {
            "p_feature": materialized.features[int(section_index)][
                rows_tensor
            ].unsqueeze(1),
            "op_feature": cached.features[rows_tensor],
            "op_count": cached.targets[rows_tensor],
        }
        for key, value in block.items():
            if key not in result:
                result[key] = torch.empty(
                    (len(pairs), *value.shape[1:]), dtype=value.dtype
                )
            result[key][torch.as_tensor(positions)] = value
    result["img"] = torch.cat(
        [
            normalize_model_patches(
                "EGNv1",
                materialized.raw_patches[int(section_index)][int(row)]
                .unsqueeze(0)
                .to(device),
                training=False,
            )
            for section_index, row in pairs
        ],
        dim=0,
    )
    return {key: value.to(device) for key, value in result.items()}


def _tcgn_mixed_patches(
    materialized: MaterializedTraining,
    pairs: np.ndarray,
    *,
    device: torch.device,
    training: bool,
) -> torch.Tensor:
    return torch.cat(
        [
            normalize_model_patches(
                "TCGN",
                materialized.raw_patches[int(section_index)][int(row)]
                .unsqueeze(0)
                .to(device),
                training=training,
            )
            for section_index, row in pairs
        ],
        dim=0,
    )


def _forward_spot_pairs(
    model: torch.nn.Module,
    materialized: MaterializedTraining,
    pairs: np.ndarray,
    *,
    device: torch.device,
    training: bool,
) -> torch.Tensor:
    if materialized.model == "EGNv1":
        return model(_egn_v1_mixed_inputs(materialized, pairs, device=device))
    return model(
        _tcgn_mixed_patches(
            materialized, pairs, device=device, training=training
        )
    )


def _spot_validation_mse(
    model: torch.nn.Module,
    materialized: MaterializedTraining,
    *,
    device: torch.device,
) -> float:
    config = MODEL_CONFIGS[materialized.model]
    total_squared_error = 0.0
    total_values = 0
    validation_pairs: list[tuple[int, int]] = []
    for section_index, supervision in enumerate(materialized.supervisions):
        validation_pairs.extend(
            (section_index, int(row))
            for row in np.flatnonzero(supervision.validation_mask)
        )
    pairs = np.asarray(validation_pairs, dtype=np.int64)
    usable = len(pairs) // config.batch_size * config.batch_size
    if usable < config.batch_size:
        raise PairedTrainingError(
            f"PUBLISHED_VALIDATION_BATCH_EMPTY:{len(pairs)}:{config.batch_size}"
        )
    model.eval()
    with torch.no_grad():
        for start in range(0, usable, config.batch_size):
            batch_pairs = pairs[start : start + config.batch_size]
            prediction = _forward_spot_pairs(
                model,
                materialized,
                batch_pairs,
                device=device,
                training=False,
            )
            target = _targets(
                materialized.fold_data.training, batch_pairs, device=device
            )
            total_squared_error += float(
                F.mse_loss(prediction, target, reduction="sum").cpu()
            )
            total_values += int(target.numel())
    if total_values < 1:
        raise PairedTrainingError("NO_VALIDATION_VALUES")
    return total_squared_error / total_values


def _egn_v1_scheduler(
    optimizer: torch.optim.Optimizer, *, steps_per_epoch: int, learning_rate: float
):
    try:
        from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
    except Exception as error:
        raise PairedTrainingError(
            f"EGNV1_PUBLISHED_SCHEDULER_UNAVAILABLE:{error}"
        ) from error
    first_cycle_steps = steps_per_epoch // 2
    if first_cycle_steps < 2:
        raise PairedTrainingError(
            f"EGNV1_SCHEDULER_CYCLE_TOO_SHORT:{steps_per_epoch}"
        )
    return CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=1.0,
        max_lr=learning_rate,
        min_lr=learning_rate / 5,
        warmup_steps=min(100, first_cycle_steps - 1),
    )


def _copy_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_spot_arm(
    *,
    model_factory: Callable[[], torch.nn.Module],
    initial_state: Mapping[str, torch.Tensor],
    materialized: MaterializedTraining,
    rho: float,
    seed: int,
    device: torch.device,
    progress_callback=None,
) -> ArmCheckpoint:
    if materialized.model not in {"EGNv1", "TCGN"} or rho not in {0.0, 0.5}:
        raise PairedTrainingError(f"SPOT_ARM_CONTRACT:{materialized.model}:{rho}")
    config = MODEL_CONFIGS[materialized.model]
    rows = _global_training_rows(materialized)
    # The baseline arm never enumerates or samples the edge stream.
    edges = _global_training_edges(materialized) if rho > 0.0 else None
    steps = len(rows) // config.batch_size
    if steps < 1:
        raise PairedTrainingError("NO_COMPLETE_ORDINARY_TRAINING_BATCH")
    seed_everything(seed)
    model = model_factory().to(device)
    model.load_state_dict(copy.deepcopy(initial_state), strict=True)
    if materialized.model == "TCGN":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
        scheduler = _egn_v1_scheduler(
            optimizer,
            steps_per_epoch=steps,
            learning_rate=config.learning_rate,
        )
    best_validation = math.inf
    best_epoch = -1
    best_state = None
    final_training_loss = math.nan
    termination = "epoch_budget_complete"
    termination_epoch = None
    termination_step = None
    epochs_completed = 0
    if progress_callback is not None:
        progress_callback({'event': 'TRAINING_STARTED', 'epochs': config.epochs, 'steps_per_epoch': steps})
    for epoch in range(config.epochs):
        model.train()
        ordered = (
            rows[epoch_order(len(rows), seed=seed, epoch=epoch)]
            if config.ordinary_shuffle
            else rows
        )
        losses: list[float] = []
        terminate_arm = False
        for step in range(steps):
            ordinary_pairs = ordered[
                step * config.batch_size : (step + 1) * config.batch_size
            ]
            ordinary_target = _targets(
                materialized.fold_data.training, ordinary_pairs, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            # This stream is identical in both arms.  Extra edge forwards may
            # consume RNG, but cannot shift the next ordinary augmentation or
            # dropout draw because every ordinary step has its own frozen seed.
            seed_everything(
                step_rng_seed(
                    seed=seed, epoch=epoch, step=step, stream="ordinary"
                )
            )
            ordinary_prediction = _forward_spot_pairs(
                model,
                materialized,
                ordinary_pairs,
                device=device,
                training=True,
            )
            if rho == 0.0:
                loss = minibatch_loss(
                    ordinary_prediction,
                    ordinary_target,
                    rho=0.0,
                    correlation_weight=config.correlation_auxiliary_weight,
                )
                if not torch.isfinite(loss):
                    if materialized.model == "TCGN" and best_state is not None:
                        termination = "nonfinite_after_finite_checkpoint"
                        termination_epoch = epoch
                        termination_step = step
                        terminate_arm = True
                        break
                    raise PairedTrainingError(
                        f"TRAINING_LOSS_NONFINITE:{materialized.model}:"
                        f"{epoch}:{step}"
                    )
                loss.backward()
            else:
                if edges is None:
                    raise PairedTrainingError("INTERVENTION_EDGE_STREAM_MISSING")
                generator = edge_generator(seed=seed, epoch=epoch, step=step)
                chosen = generator.choice(
                    len(edges), size=config.batch_size, replace=len(edges) < config.batch_size
                )
                edge_rows = edges[chosen]
                left_pairs = edge_rows[:, [0, 1]]
                right_pairs = edge_rows[:, [0, 2]]
                edge_target = torch.cat(
                    (
                        _targets(
                            materialized.fold_data.training,
                            left_pairs,
                            device=device,
                        ),
                        _targets(
                            materialized.fold_data.training,
                            right_pairs,
                            device=device,
                        ),
                    ),
                    dim=0,
                )
                edge_index = torch.stack(
                    (
                        torch.arange(config.batch_size, device=device),
                        torch.arange(
                            config.batch_size,
                            2 * config.batch_size,
                            device=device,
                        ),
                    )
                )
                # Keep this context open through backward.  Restoring a
                # BatchNorm buffer before backward would invalidate autograd's
                # saved tensor version; after backward it is safe and leaves
                # persistent state owned by the ordinary stream only.
                with preserve_model_buffers(model):
                    seed_everything(
                        step_rng_seed(
                            seed=seed, epoch=epoch, step=step, stream="edge_left"
                        )
                    )
                    left_prediction = _forward_spot_pairs(
                        model,
                        materialized,
                        left_pairs,
                        device=device,
                        training=True,
                    )
                    seed_everything(
                        step_rng_seed(
                            seed=seed,
                            epoch=epoch,
                            step=step,
                            stream="edge_right",
                        )
                    )
                    right_prediction = _forward_spot_pairs(
                        model,
                        materialized,
                        right_pairs,
                        device=device,
                        training=True,
                    )
                    edge_prediction = torch.cat(
                        (left_prediction, right_prediction), dim=0
                    )
                    loss = minibatch_loss(
                        ordinary_prediction,
                        ordinary_target,
                        rho=rho,
                        correlation_weight=config.correlation_auxiliary_weight,
                        edge_prediction=edge_prediction,
                        edge_target=edge_target,
                        edge_index=edge_index,
                    )
                    if not torch.isfinite(loss):
                        if materialized.model == "TCGN" and best_state is not None:
                            termination = "nonfinite_after_finite_checkpoint"
                            termination_epoch = epoch
                            termination_step = step
                            terminate_arm = True
                            break
                        raise PairedTrainingError(
                            f"TRAINING_LOSS_NONFINITE:{materialized.model}:"
                            f"{epoch}:{step}"
                        )
                    loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if progress_callback is not None and (step == 0 or (step + 1) % 100 == 0 or step + 1 == steps):
                progress_callback({'event': 'TRAINING_PROGRESS', 'epoch': epoch + 1,
                                   'step': step + 1, 'epochs': config.epochs, 'steps_per_epoch': steps})
        if terminate_arm:
            break
        final_training_loss = float(np.mean(losses))
        validation = _spot_validation_mse(model, materialized, device=device)
        epochs_completed = epoch + 1
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch
            best_state = _copy_state(model)
    if best_state is None:
        raise PairedTrainingError("NO_SPOT_CHECKPOINT")
    return ArmCheckpoint(
        rho=float(rho),
        best_epoch=best_epoch,
        best_validation_mse=best_validation,
        final_training_loss=final_training_loss,
        state_dict=best_state,
        termination=termination,
        termination_epoch=termination_epoch,
        termination_step=termination_step,
        epochs_completed=epochs_completed,
    )


def _graph_validation_mse(
    model: torch.nn.Module,
    materialized: MaterializedTraining,
    *,
    device: torch.device,
) -> float:
    if materialized.graphs is None:
        raise PairedTrainingError("EGNV2_GRAPHS_NOT_MATERIALIZED")
    total = 0.0
    values = 0
    model.eval()
    with torch.no_grad():
        for graph_cpu, section, supervision in zip(
            materialized.graphs,
            materialized.fold_data.training,
            materialized.supervisions,
        ):
            graph = graph_cpu.to(device)
            prediction = model(graph.x_dict, graph.edge_index_dict)
            target = torch.as_tensor(section.target, dtype=torch.float32, device=device)
            loss = validation_mse(prediction, target, supervision)
            count = supervision.n_validation * 50
            total += float(loss.cpu()) * count
            values += count
    if values < 1:
        raise PairedTrainingError("NO_GRAPH_VALIDATION_VALUES")
    return total / values


def train_graph_arm(
    *,
    model_factory: Callable[[], torch.nn.Module],
    initial_state: Mapping[str, torch.Tensor],
    materialized: MaterializedTraining,
    rho: float,
    seed: int,
    device: torch.device,
) -> ArmCheckpoint:
    if materialized.model != "EGNv2" or materialized.graphs is None:
        raise PairedTrainingError("GRAPH_ARM_MODEL")
    if rho not in {0.0, 0.5}:
        raise PairedTrainingError(f"GRAPH_ARM_RHO:{rho}")
    config = MODEL_CONFIGS["EGNv2"]
    seed_everything(seed)
    model = model_factory().to(device)
    model.load_state_dict(copy.deepcopy(initial_state), strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )
    best_validation = math.inf
    best_epoch = -1
    best_state = None
    final_training_loss = math.nan
    # Published graph DataLoader has shuffle=False: official section order is fixed.
    for epoch in range(config.epochs):
        model.train()
        losses: list[float] = []
        for graph_cpu, section, supervision in zip(
            materialized.graphs,
            materialized.fold_data.training,
            materialized.supervisions,
        ):
            graph = graph_cpu.to(device)
            target = torch.as_tensor(section.target, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            seed_everything(
                step_rng_seed(
                    seed=seed,
                    epoch=epoch,
                    step=len(losses),
                    stream="ordinary",
                )
            )
            prediction = model(graph.x_dict, graph.edge_index_dict)
            loss = full_section_loss(
                prediction,
                target,
                supervision,
                rho=rho,
                correlation_weight=config.correlation_auxiliary_weight,
            )
            if not torch.isfinite(loss):
                raise PairedTrainingError(
                    f"TRAINING_LOSS_NONFINITE:EGNv2:{epoch}:{section.specimen}"
                )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_training_loss = float(np.mean(losses))
        validation = _graph_validation_mse(model, materialized, device=device)
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch
            best_state = _copy_state(model)
    if best_state is None:
        raise PairedTrainingError("NO_GRAPH_CHECKPOINT")
    return ArmCheckpoint(
        rho=float(rho),
        best_epoch=best_epoch,
        best_validation_mse=best_validation,
        final_training_loss=final_training_loss,
        state_dict=best_state,
    )


def predict_spot_section(
    model: torch.nn.Module,
    *,
    model_name: str,
    section: SectionMetadata,
    raw_patches: torch.Tensor,
    device: torch.device,
    query_features: Optional[torch.Tensor] = None,
    exemplar_bank: Optional[ExemplarBank] = None,
) -> np.ndarray:
    name = canonical_model_name(model_name)
    if name not in {"EGNv1", "TCGN"}:
        raise PairedTrainingError(f"SPOT_INFERENCE_MODEL:{name}")
    config = MODEL_CONFIGS[name]
    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(section.barcodes), config.batch_size):
            rows = np.arange(
                start, min(start + config.batch_size, len(section.barcodes))
            )
            if name == "EGNv1":
                local_rows = np.arange(len(rows), dtype=np.int64)
                data = egn_v1_batch(
                    model_patches=normalize_model_patches(
                        "EGNv1",
                        raw_patches[torch.as_tensor(rows)].to(device),
                        training=False,
                    ),
                    query_features=query_features[torch.as_tensor(rows)],
                    query_rows=local_rows,
                    query_specimen=section.specimen,
                    bank=exemplar_bank,
                )
                prediction = model(
                    {key: value.to(device) for key, value in data.items()}
                )
            else:
                batch = normalize_model_patches(
                    "TCGN", raw_patches[torch.as_tensor(rows)].to(device), training=False
                )
                prediction = model(batch)
            outputs.append(prediction.detach().cpu())
    result = torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
    if result.shape != (len(section.barcodes), 50):
        raise PairedTrainingError(
            f"SPOT_INFERENCE_SHAPE:{result.shape}:{len(section.barcodes)}"
        )
    return result


def predict_graph_section(
    model: torch.nn.Module,
    *,
    graph: EGNv2GraphInput,
    n_spots: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    moved = graph.to(device)
    with torch.no_grad():
        result = model(moved.x_dict, moved.edge_index_dict).detach().cpu().numpy()
    result = np.asarray(result, dtype=np.float32)
    if result.shape != (n_spots, 50):
        raise PairedTrainingError(
            f"GRAPH_INFERENCE_SHAPE:{result.shape}:{n_spots}"
        )
    return result
