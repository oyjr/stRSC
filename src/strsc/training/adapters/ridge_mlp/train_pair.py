#!/usr/bin/env python3
"""Train the paired two-layer MLP arms on one frozen HEST outer fold.

The command writes predictions and engineering receipts but deliberately does
not calculate or print held-out Pearson or STAudit band endpoints.  Those are
opened only after the complete formal prediction manifest is locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from strsc.training.shared import (  # noqa: E402
    AdjacencyGraph,
    build_within_specimen_adjacency,
    check_rho_zero_noop,
    local_difference_mse,
    mixed_expression_mse,
)

from strsc.training.adapters.ridge_mlp.hest_contract import HestFold, load_hest_fold  # noqa: E402
from strsc.training.adapters.ridge_mlp.model import TwoLayerMLP  # noqa: E402


RHO_ARMS = (0.0, 0.5)
FORMAL_SEEDS = (17, 23, 41)
VALIDATION_HASH_PREFIX = "G0055_MLP_VAL_V1"
VALIDATION_HASH_BYTEORDER = "big"
PCA_COMPONENTS = 256
PCA_RANDOM_STATE = 1
HIDDEN_DIM = 512
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EDGE_RADIUS_MULTIPLIER = 1.5
TRUTH_TRANSFORM_ID = "log1p_raw_integer_counts_no_library_normalization"


class MLPAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainConfig:
    input_dim: int = PCA_COMPONENTS
    hidden_dim: int = HIDDEN_DIM
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY


@dataclass(frozen=True)
class PreparedTraining:
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    graph: AdjacencyGraph
    split_rows: tuple[dict[str, Any], ...]
    scaler_mean: np.ndarray
    pca_explained_variance_ratio: np.ndarray


@dataclass(frozen=True)
class ArmResult:
    rho: float
    prediction: np.ndarray
    best_epoch: int
    best_validation_mse: float
    final_training_loss: float


def _validation_mask(
    *, task: str, specimen: str, barcodes: np.ndarray
) -> np.ndarray:
    """Assign validation rows using BLAKE2b hashes modulo 10."""

    result = np.zeros(len(barcodes), dtype=bool)
    for index, barcode in enumerate(barcodes):
        token = f"{VALIDATION_HASH_PREFIX}|{task}|{specimen}|{str(barcode)}"
        digest = hashlib.blake2b(token.encode("utf-8")).digest()
        value = int.from_bytes(
            digest[:8], byteorder=VALIDATION_HASH_BYTEORDER, signed=False
        )
        result[index] = value % 10 == 0
    if result.all() or not result.any():
        raise MLPAdapterError(
            f"VALIDATION_SPLIT_EMPTY_PARTITION:{task}:{specimen}:"
            f"train={int((~result).sum())}:validation={int(result.sum())}"
        )
    return result


def prepare_training(fold: HestFold, config: TrainConfig) -> PreparedTraining:
    """Fit HEST preprocessing and exclude validation rows before graph creation."""

    all_train_x = np.concatenate(fold.train_embeddings, axis=0)
    if config.input_dim > min(all_train_x.shape):
        raise MLPAdapterError(
            f"PCA_COMPONENT_CONTRACT:{config.input_dim}:{all_train_x.shape}"
        )
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_train_x)
    pca = PCA(n_components=config.input_dim, random_state=PCA_RANDOM_STATE)
    all_reduced = pca.fit_transform(all_scaled)
    test_reduced = pca.transform(scaler.transform(fold.test_embeddings))

    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    val_x: list[np.ndarray] = []
    val_y: list[np.ndarray] = []
    train_coords: list[np.ndarray] = []
    split_rows: list[dict[str, Any]] = []
    offset = 0
    for specimen, barcodes, truth, coords, embeddings in zip(
        fold.train_specimens,
        fold.train_barcodes,
        fold.train_truth,
        fold.train_coords,
        fold.train_embeddings,
    ):
        length = len(embeddings)
        reduced = all_reduced[offset : offset + length]
        offset += length
        validation = _validation_mask(
            task=fold.task, specimen=specimen, barcodes=barcodes
        )
        training = ~validation
        train_x.append(reduced[training])
        train_y.append(truth[training])
        val_x.append(reduced[validation])
        val_y.append(truth[validation])
        train_coords.append(coords[training])
        split_rows.append(
            {
                "specimen": specimen,
                "total_spots": int(length),
                "training_spots": int(training.sum()),
                "validation_spots": int(validation.sum()),
            }
        )
    if offset != len(all_reduced):
        raise MLPAdapterError(f"TRAIN_CONCATENATION_OFFSET:{offset}:{len(all_reduced)}")
    graph = build_within_specimen_adjacency(
        train_coords,
        specimen_ids=fold.train_specimens,
        radius_multiplier=EDGE_RADIUS_MULTIPLIER,
    )
    result = PreparedTraining(
        train_x=np.ascontiguousarray(np.concatenate(train_x), dtype=np.float32),
        train_y=np.ascontiguousarray(np.concatenate(train_y), dtype=np.float32),
        validation_x=np.ascontiguousarray(np.concatenate(val_x), dtype=np.float32),
        validation_y=np.ascontiguousarray(np.concatenate(val_y), dtype=np.float32),
        test_x=np.ascontiguousarray(test_reduced, dtype=np.float32),
        graph=graph,
        split_rows=tuple(split_rows),
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        pca_explained_variance_ratio=np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64
        ),
    )
    if result.graph.n_nodes != len(result.train_x):
        raise MLPAdapterError(
            f"GRAPH_NODE_ALIGNMENT:{result.graph.n_nodes}:{len(result.train_x)}"
        )
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _initial_state(
    *, config: TrainConfig, output_dim: int, seed: int
) -> Mapping[str, torch.Tensor]:
    _seed_everything(seed)
    model = TwoLayerMLP(config.input_dim, config.hidden_dim, output_dim)
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _gradients(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def run_mlp_noop_gate(
    *,
    config: TrainConfig,
    output_dim: int,
    initial_state: Mapping[str, torch.Tensor],
    train_x: np.ndarray,
    train_y: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    """Compare original MSE and the shared rho-zero overlay on one fixed batch."""

    count = min(config.batch_size, len(train_x))
    inputs = torch.as_tensor(train_x[:count], device=device)
    targets = torch.as_tensor(train_y[:count], device=device)
    model = TwoLayerMLP(config.input_dim, config.hidden_dim, output_dim).to(device)
    model.load_state_dict(initial_state)
    model.train()

    model.zero_grad(set_to_none=True)
    original_prediction = model(inputs)
    original_loss = F.mse_loss(original_prediction, targets, reduction="mean")
    original_loss.backward()
    original_gradients = _gradients(model)

    model.zero_grad(set_to_none=True)
    rho_zero_prediction = model(inputs)
    # Deliberately invalid graph sentinel: the rho-zero branch must not touch it.
    rho_zero_loss = mixed_expression_mse(
        rho_zero_prediction, targets, edge_index=None, rho=0.0
    )
    rho_zero_loss.backward()
    rho_zero_gradients = _gradients(model)
    receipt = check_rho_zero_noop(
        original_loss=original_loss,
        rho_zero_loss=rho_zero_loss,
        original_predictions=original_prediction,
        rho_zero_predictions=rho_zero_prediction,
        original_gradients=original_gradients,
        rho_zero_gradients=rho_zero_gradients,
        atol=0.0,
        rtol=0.0,
    )
    return receipt.to_dict()


def _epoch_schedule(
    *, n_spots: int, n_edges: int, batch_size: int, seed: int, epoch: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use every training spot and edge once in a deterministic paired epoch."""

    if min(n_spots, n_edges, batch_size) < 1:
        raise MLPAdapterError(
            f"SCHEDULE_COUNTS:{n_spots}:{n_edges}:{batch_size}"
        )
    steps = int(math.ceil(n_spots / batch_size))
    seed_sequence = np.random.SeedSequence([seed, epoch, 0x47503055])
    spot_seed, edge_seed = seed_sequence.spawn(2)
    spots = np.random.default_rng(spot_seed).permutation(n_spots)
    edges = np.random.default_rng(edge_seed).permutation(n_edges)
    spot_blocks = [block for block in np.array_split(spots, steps) if len(block)]
    edge_blocks = np.array_split(edges, len(spot_blocks))
    if any(len(block) == 0 for block in edge_blocks):
        raise MLPAdapterError(
            f"EDGE_BATCH_EMPTY:{n_edges}:{len(spot_blocks)}"
        )
    return list(zip(spot_blocks, edge_blocks))


def _validation_mse(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    squared_error = 0.0
    values = 0
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            stop = min(start + batch_size, len(inputs))
            x = torch.as_tensor(inputs[start:stop], device=device)
            y = torch.as_tensor(targets[start:stop], device=device)
            error = model(x) - y
            squared_error += float(torch.sum(torch.square(error)).cpu())
            values += int(error.numel())
    if values == 0:
        raise MLPAdapterError("EMPTY_VALIDATION_VALUES")
    return squared_error / values


def _predict(
    model: torch.nn.Module,
    inputs: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            prediction = model(
                torch.as_tensor(inputs[start : start + batch_size], device=device)
            )
            rows.append(prediction.detach().cpu().numpy())
    result = np.ascontiguousarray(np.concatenate(rows), dtype=np.float32)
    if result.shape[0] != len(inputs) or not np.isfinite(result).all():
        raise MLPAdapterError(f"PREDICTION_CONTENT:{result.shape}:{len(inputs)}")
    return result


def train_arm(
    *,
    prepared: PreparedTraining,
    output_dim: int,
    rho: float,
    seed: int,
    config: TrainConfig,
    initial_state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> ArmResult:
    if rho not in RHO_ARMS:
        raise MLPAdapterError(f"UNFROZEN_RHO:{rho}")
    _seed_everything(seed)
    model = TwoLayerMLP(config.input_dim, config.hidden_dim, output_dim).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    edge_index = prepared.graph.edge_index
    best_validation = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    final_training_loss = math.nan

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        schedule = _epoch_schedule(
            n_spots=len(prepared.train_x),
            n_edges=prepared.graph.n_edges,
            batch_size=config.batch_size,
            seed=seed,
            epoch=epoch,
        )
        for spot_rows, edge_rows in schedule:
            x = torch.as_tensor(prepared.train_x[spot_rows], device=device)
            y = torch.as_tensor(prepared.train_y[spot_rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            spot_prediction = model(x)
            ordinary = F.mse_loss(spot_prediction, y, reduction="mean")
            if rho == 0.0:
                loss = ordinary
            else:
                chosen = edge_index[:, edge_rows]
                left = chosen[0]
                right = chosen[1]
                pair_rows = np.concatenate((left, right))
                pair_prediction = model(
                    torch.as_tensor(prepared.train_x[pair_rows], device=device)
                )
                pair_target = torch.as_tensor(
                    prepared.train_y[pair_rows], device=device
                )
                pair_count = len(left)
                local_edges = np.vstack(
                    (
                        np.arange(pair_count, dtype=np.int64),
                        np.arange(pair_count, dtype=np.int64) + pair_count,
                    )
                )
                edge_loss = local_difference_mse(
                    pair_prediction, pair_target, local_edges
                )
                loss = (1.0 - rho) * ordinary + rho * edge_loss
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_steps += 1
        final_training_loss = epoch_loss / epoch_steps
        validation = _validation_mse(
            model,
            prepared.validation_x,
            prepared.validation_y,
            device=device,
            batch_size=config.batch_size,
        )
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None or best_epoch < 0 or not math.isfinite(best_validation):
        raise MLPAdapterError("NO_VALID_CHECKPOINT")
    model.load_state_dict(best_state)
    prediction = _predict(
        model,
        prepared.test_x,
        device=device,
        batch_size=config.batch_size,
    )
    return ArmResult(
        rho=float(rho),
        prediction=prediction,
        best_epoch=best_epoch,
        best_validation_mse=float(best_validation),
        final_training_loss=float(final_training_loss),
    )


def train_pair(
    *,
    prepared: PreparedTraining,
    output_dim: int,
    seed: int,
    config: TrainConfig,
    device: torch.device,
) -> tuple[tuple[ArmResult, ArmResult], dict[str, object]]:
    initial_state = _initial_state(config=config, output_dim=output_dim, seed=seed)
    noop = {"status": "NOT_RUN_MINIMAL_VALIDATION_POLICY"}
    results = tuple(
        train_arm(
            prepared=prepared,
            output_dim=output_dim,
            rho=rho,
            seed=seed,
            config=config,
            initial_state=initial_state,
            device=device,
        )
        for rho in RHO_ARMS
    )
    return (results[0], results[1]), noop


def _write_prediction(
    *, fold: HestFold, result: ArmResult, seed: int, output_root: Path
) -> Path:
    tag = f"rho{result.rho:.1f}"
    name = (
        f"MLP.{fold.task}.fold{fold.fold}.{fold.test_specimen}.phikon_v2."
        f"seed{seed}.{tag}.predictions.h5"
    )
    output = output_root / "predictions" / name
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    with h5py.File(temporary, "w") as handle:
        strings = h5py.string_dtype("utf-8")
        handle.create_dataset(
            "barcodes", data=fold.test_barcodes.astype(object), dtype=strings
        )
        handle.create_dataset("genes", data=fold.genes.astype(object), dtype=strings)
        if fold.test_patch_coords is not None:
            handle.create_dataset("patch_coords", data=fold.test_patch_coords)
        handle.create_dataset("spatial_coords", data=fold.test_coords)
        handle.create_dataset(
            "raw_counts",
            data=np.rint(fold.test_raw_counts).astype(np.uint32),
            compression="lzf",
        )
        handle.create_dataset(
            "transformed_truth", data=fold.test_truth, compression="lzf"
        )
        handle.create_dataset("prediction", data=result.prediction, compression="lzf")
        handle.attrs.update(
            {
                "status": "PREDICTION_COMPLETE_ENDPOINTS_UNOPENED",
                "experiment": "G0-055",
                "track": "HEST",
                "model": "two_layer_mlp",
                "task": fold.task,
                "fold": fold.fold,
                "sample_id": fold.test_specimen,
                "encoder": "phikon_v2",
                "seed": seed,
                "rho": result.rho,
                "truth_transform_id": TRUTH_TRANSFORM_ID,
                "pca_components": PCA_COMPONENTS,
                "pca_random_state": PCA_RANDOM_STATE,
                "validation_hash_prefix": VALIDATION_HASH_PREFIX,
                "validation_hash_byteorder": VALIDATION_HASH_BYTEORDER,
                "edge_rule": "radius_1.5x_median_positive_nearest_neighbor_within_specimen",
            }
        )
        handle.flush()
    os.replace(temporary, output)
    return output


def run_fold(args: argparse.Namespace) -> dict[str, object]:
    if args.seed not in FORMAL_SEEDS:
        raise MLPAdapterError(f"UNFROZEN_SEED:{args.seed}")
    fold = load_hest_fold(
        dataset_root=args.dataset_root,
        gene_manifest=args.gene_manifest,
        task=args.task,
        fold=args.fold,
        encoder="phikon_v2",
        embedding_root=args.embedding_root,
        embedding_manifest=args.embedding_manifest,
        g0054_module=args.g0054_module,
    )
    config = TrainConfig()
    prepared = prepare_training(fold, config)
    device = torch.device(args.device)
    results, noop = train_pair(
        prepared=prepared,
        output_dim=len(fold.genes),
        seed=args.seed,
        config=config,
        device=device,
    )
    prediction_rows = []
    for result in results:
        path = _write_prediction(
            fold=fold, result=result, seed=args.seed, output_root=args.output_root
        )
        prediction_rows.append(
            {
                "rho": result.rho,
                "prediction_path": str(path),
                "rows": len(result.prediction),
                "genes": result.prediction.shape[1],
                "best_epoch": result.best_epoch,
                "best_validation_mse": result.best_validation_mse,
                "final_training_loss": result.final_training_loss,
            }
        )
    receipt = {
        "status": "PASS_ENDPOINTS_UNOPENED",
        "experiment": "G0-055",
        "track": "HEST",
        "model": "two_layer_mlp",
        "task": fold.task,
        "fold": fold.fold,
        "test_specimen": fold.test_specimen,
        "train_specimens": list(fold.train_specimens),
        "seed": args.seed,
        "rho": list(RHO_ARMS),
        "config": asdict(config),
        "validation_hash_prefix": VALIDATION_HASH_PREFIX,
        "validation_hash_byteorder": VALIDATION_HASH_BYTEORDER,
        "split_rows": list(prepared.split_rows),
        "graph": prepared.graph.to_metadata(),
        "noop": noop,
        "predictions": prediction_rows,
        "held_out_pcc_calculated": False,
        "staudit_bands_calculated": False,
    }
    receipt_path = (
        args.output_root
        / "receipts"
        / f"MLP.{fold.task}.fold{fold.fold}.{fold.test_specimen}.seed{args.seed}.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(receipt_path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, receipt_path)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--embedding-root", type=Path)
    sources.add_argument("--embedding-manifest", type=Path)
    parser.add_argument("--gene-manifest", type=Path, required=True)
    parser.add_argument("--g0054-module", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", choices=("IDC", "PAAD", "LUNG"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_fold(args)
    # Deliberately expose only engineering state, never validation values or endpoints.
    print(
        json.dumps(
            {
                "status": result["status"],
                "model": result["model"],
                "task": result["task"],
                "fold": result["fold"],
                "seed": result["seed"],
                "prediction_files": len(result["predictions"]),
                "held_out_pcc_calculated": False,
                "staudit_bands_calculated": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
