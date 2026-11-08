#!/usr/bin/env python3
"""Formal paired prepare/train-predict runner for native DeepSpot and HiST.

The command never imports an STAudit scorer.  ``prepare`` freezes one native
initial state from outer-training data only.  ``train-predict`` restores that
same state for rho 0.0 and 0.5, selects each checkpoint by ordinary inner-
validation MSE, and then opens image/coordinate-only held-out support.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

from .contract import (
    GENES,
    RHOS,
    SEEDS,
    SOURCE_REVISIONS,
    TASK_SPECIMENS,
    HestDeepSpotHiSTError,
    build_fold_inner_splits,
    canonical_model_name,
    expected_outer_fold,
)
from .data import (
    DeepSpotBlindDataset,
    DeepSpotEdgePairDataset,
    DeepSpotFeatureStore,
    DeepSpotSpecimenDataset,
    FrozenStandardizer,
    HiSTFeatureStore,
    HiSTPartitionDataset,
    build_deepspot_train_dataset,
    build_hist_blind_batch,
    collate_deepspot_blind,
    collate_deepspot_edge_pairs,
    collate_deepspot_spots,
    fit_deepspot_targets,
    load_aligned_hest_fold,
    load_blind_hest_support,
)
from .deepspot import (
    build_upstream_deepspot,
    install_deepspot_overlay,
    remove_deepspot_overlay,
)
from .engineering_gate import classify_failure
from .hist import (
    build_hist_optimizer,
    build_upstream_hist,
    compose_hist_config,
    install_hist_overlay,
    remove_hist_overlay,
)
from .outputs import write_pair_manifest, write_prediction_h5
from .paired import seed_all


PREPARED_SCHEMA = "g0055_hest_deepspot_hist_prepared_state_v1"
CHECKPOINT_SCHEMA = "g0055_hest_deepspot_hist_checkpoint_v1"
DEEPSPOT_EPOCHS = 10
DEEPSPOT_BATCH_SIZE = 1024
DEEPSPOT_EDGE_PAIR_BATCH_SIZE = 512


class FormalRunnerError(RuntimeError):
    pass


class EngineeringNotEstimable(FormalRunnerError):
    def __init__(self, status: str, detail: str) -> None:
        super().__init__(f"{status}:{detail}")
        self.status = str(status)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise FormalRunnerError(f"PREPARED_PAYLOAD_TYPE:{path}")
    return payload


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _verify_source_revision(path: Path, model: str) -> str:
    import subprocess

    expected = SOURCE_REVISIONS[model]
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FormalRunnerError(f"SOURCE_REVISION_UNREADABLE:{path}:{exc}") from exc
    actual = result.stdout.strip()
    if actual != expected:
        raise FormalRunnerError(f"SOURCE_REVISION:{model}:{actual}:{expected}")
    return actual


def require_engineering_receipt(
    path: Path, *, model: str, task: str, fold: int
) -> dict[str, Any]:
    # The engineering gate is deliberately model-level: one representative
    # real HEST forward/backward is reused by every formal fold.  ``task`` and
    # ``fold`` remain in the signature so callers record their formal target,
    # but they are not used to repeat the gate nine times.
    del task, fold
    if not path.is_file():
        raise FormalRunnerError(f"ENGINEERING_RECEIPT_MISSING:{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalRunnerError(f"ENGINEERING_RECEIPT_INVALID:{path}:{exc}") from exc
    status = str(payload.get("status", ""))
    if status.startswith("ENGINEERING_NOT_ESTIMABLE"):
        raise EngineeringNotEstimable(status, str(payload.get("error", "gate")))
    expected = {
        "status": "PASS",
        "model": model,
        "source_revision": SOURCE_REVISIONS[model],
        "scientific_results_opened": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise FormalRunnerError(
                f"ENGINEERING_RECEIPT_NOT_PASSING:{key}:{payload.get(key)!r}:{value!r}"
            )
    return payload


def _initialize_hist_bias(model: torch.nn.Module, fold, splits) -> None:
    raw = np.concatenate(
        [
            np.asarray(specimen.raw_counts, dtype=np.float32)[split.train_mask]
            for specimen, split in zip(fold.train_specimens, splits, strict=True)
        ],
        axis=0,
    )
    gene_mean = torch.log1p(torch.from_numpy(raw)).mean(dim=0)
    try:
        final_linear = (
            model.decoder.blocks[-1]
            .modules_dict["convs_0"][0]
            .conv_block.linear[-1]
        )
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise HestDeepSpotHiSTError(
            f"HIST_PUBLISHED_BIAS_PATH:{type(exc).__name__}:{exc}"
        ) from exc
    if final_linear.bias is None or final_linear.bias.shape != gene_mean.shape:
        raise HestDeepSpotHiSTError("HIST_PUBLISHED_BIAS_SHAPE")
    final_linear.bias.data.copy_(gene_mean.to(final_linear.bias))


def prepare_run(args: argparse.Namespace) -> dict[str, Any]:
    model_name = canonical_model_name(args.model)
    if int(args.seed) not in SEEDS:
        raise FormalRunnerError(f"UNFROZEN_SEED:{args.seed}:{SEEDS}")
    expected_outer_fold(args.task, args.fold)
    require_engineering_receipt(
        Path(args.engineering_receipt),
        model=model_name,
        task=args.task,
        fold=args.fold,
    )
    source_revision = _verify_source_revision(Path(args.source_root), model_name)
    output = Path(args.prepared_state)
    if output.exists():
        raise FileExistsError(f"PREPARED_STATE_EXISTS:{output}")

    fold = load_aligned_hest_fold(
        dataset_root=Path(args.dataset_root),
        gene_manifest=Path(args.gene_manifest),
        task=args.task,
        fold=args.fold,
        g0054_module=args.g0054_module,
        include_test=False,
    )
    splits = build_fold_inner_splits(fold, model=model_name)
    seed_all(args.seed)
    model_payload: dict[str, Any]
    if model_name == "DeepSpot":
        targets = fit_deepspot_targets(
            fold, splits, dataset_root=Path(args.dataset_root)
        )
        feature_store = DeepSpotFeatureStore(args.feature_root)
        first = fold.train_specimens[0]
        first_train_row = int(np.flatnonzero(splits[0].train_mask)[0])
        input_size = int(
            feature_store.load(
                first.specimen, str(first.barcodes[first_train_row])
            ).shape[1]
        )
        model = build_upstream_deepspot(
            deps_root=args.source_root,
            input_size=input_size,
            seed=args.seed,
            scaler=targets.standardizer.sklearn_scaler(),
        )
        model_payload = {
            "input_size": input_size,
            "standardizer_mean": torch.from_numpy(
                targets.standardizer.mean.copy()
            ),
            "standardizer_scale": torch.from_numpy(
                targets.standardizer.scale.copy()
            ),
            "expression_scale": "log1p_full_library_count_per_10000",
        }
    else:
        cfg = compose_hist_config(args.source_root)
        feature_store = HiSTFeatureStore(args.feature_root)
        for specimen in fold.train_specimens:
            feature_store.aligned(specimen)
        model = build_upstream_hist(
            deps_root=args.source_root, cfg=cfg, num_genes=GENES
        )
        if bool(cfg.training.get("init_bias_with_mean", False)):
            _initialize_hist_bias(model, fold, splits)
        model_payload = {
            "input_size": 1024,
            "expression_scale": "raw_nonnegative_counts",
            "training_epochs": int(cfg.training.epochs),
        }

    payload = {
        "schema": PREPARED_SCHEMA,
        "model": model_name,
        "source_revision": source_revision,
        "task": fold.task,
        "fold": int(fold.fold),
        "outer_test_specimen": fold.test_specimen_name,
        "seed": int(args.seed),
        "genes": [str(value) for value in np.asarray(fold.genes)],
        "train_specimens": [specimen.specimen for specimen in fold.train_specimens],
        "inner_split_source": splits[0].source,
        "inner_training_rows": int(sum(split.train_mask.sum() for split in splits)),
        "inner_validation_rows": int(
            sum(split.validation_mask.sum() for split in splits)
        ),
        "rho_arms": list(RHOS),
        "model_payload": model_payload,
        "state_dict": _state_cpu(model),
        "scientific_results_opened": False,
        "held_out_rna_loaded": False,
    }
    _atomic_torch_save(output, payload)
    receipt = {
        "status": "PREPARED_PAIRED_INITIAL_STATE",
        "model": model_name,
        "task": fold.task,
        "fold": int(fold.fold),
        "seed": int(args.seed),
        "outer_test_specimen": fold.test_specimen_name,
        "prepared_state": str(output.resolve()),
        "rho_arms": list(RHOS),
        "same_initial_state": True,
        "held_out_rna_loaded": False,
        "staudit_endpoints_calculated": False,
    }
    return receipt


def _load_prepared(
    path: Path, *, model: str, task: str, fold: int, seed: int
) -> dict[str, Any]:
    if not path.is_file():
        raise FormalRunnerError(f"PREPARED_STATE_MISSING:{path}")
    payload = _torch_load(path)
    expected = {
        "schema": PREPARED_SCHEMA,
        "model": model,
        "source_revision": SOURCE_REVISIONS[model],
        "task": str(task),
        "fold": int(fold),
        "outer_test_specimen": expected_outer_fold(task, fold),
        "seed": int(seed),
        "rho_arms": list(RHOS),
        "scientific_results_opened": False,
        "held_out_rna_loaded": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise FormalRunnerError(
                f"PREPARED_STATE_METADATA:{key}:{payload.get(key)!r}:{value!r}"
            )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise FormalRunnerError("PREPARED_STATE_EMPTY")
    if len(payload.get("genes", ())) != GENES:
        raise FormalRunnerError("PREPARED_GENES")
    return payload


def _move_deepspot_upstream(batch, device: torch.device):
    x, target = batch
    return [value.to(device) for value in x], target.to(device)


def _move_deepspot_pair(ordinary, edge, device: torch.device):
    edge_x, edge_target = edge["upstream_batch"]
    return {
        "ordinary": _move_deepspot_upstream(ordinary, device),
        "edge": {
            **edge,
            "upstream_batch": (
                [value.to(device) for value in edge_x],
                edge_target.to(device),
            ),
            "edge_index": edge["edge_index"].to(device),
        },
    }


def _move_namespace(batch: SimpleNamespace, device: torch.device) -> SimpleNamespace:
    moved = SimpleNamespace()
    for key, value in vars(batch).items():
        setattr(
            moved,
            key,
            value.to(device) if isinstance(value, torch.Tensor) else value,
        )
    return moved


def _one_item(items):
    if len(items) != 1:
        raise FormalRunnerError(f"HIST_BATCH_SIZE:{len(items)}")
    return items[0]


def _deep_spot_partitions(fold, splits, targets, feature_store):
    train_datasets = []
    edge_datasets = []
    validation_datasets = []
    edge_count = 0
    for specimen, split in zip(fold.train_specimens, splits, strict=True):
        spots, edges = build_deepspot_train_dataset(
            specimen,
            split,
            scaled_targets=targets.scaled_by_specimen[specimen.specimen],
            feature_store=feature_store,
        )
        train_datasets.append(spots)
        edge_datasets.append(DeepSpotEdgePairDataset(spots, edges))
        edge_count += int(edges.shape[1])
        validation_datasets.append(
            DeepSpotSpecimenDataset(
                specimen,
                rows=np.flatnonzero(split.validation_mask),
                scaled_targets=targets.scaled_by_specimen[specimen.specimen],
                feature_store=feature_store,
            )
        )
    return (
        ConcatDataset(train_datasets),
        ConcatDataset(edge_datasets),
        ConcatDataset(validation_datasets),
        edge_count,
    )


def _deepspot_validation_mse(
    model: torch.nn.Module,
    dataset,
    *,
    device: torch.device,
    num_workers: int,
) -> float:
    loader = DataLoader(
        dataset,
        batch_size=DEEPSPOT_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_deepspot_spots,
    )
    squared_error = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, target = _move_deepspot_upstream(batch, device)
            prediction = model(x)
            squared_error += float(F.mse_loss(prediction, target, reduction="sum"))
            count += int(target.numel())
    if count == 0:
        raise FormalRunnerError("DEEPSPOT_VALIDATION_EMPTY")
    return squared_error / count


def _train_deepspot_arm(
    *,
    rho: float,
    seed: int,
    factory,
    initial_state: dict[str, torch.Tensor],
    training_dataset,
    edge_dataset,
    validation_dataset,
    device: torch.device,
    num_workers: int,
) -> dict[str, Any]:
    seed_all(seed)
    model = factory().to(device)
    model.load_state_dict(initial_state, strict=True)
    install_deepspot_overlay(model, rho=rho)
    optimizer = model.configure_optimizers()["optimizer"]
    best_mse = float("inf")
    best_epoch = -1
    best_state = None
    final_loss = float("nan")
    try:
        for epoch in range(1, DEEPSPOT_EPOCHS + 1):
            ordinary_loader = DataLoader(
                training_dataset,
                batch_size=DEEPSPOT_BATCH_SIZE,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + epoch),
                num_workers=num_workers,
                collate_fn=collate_deepspot_spots,
            )
            edge_loader = DataLoader(
                edge_dataset,
                batch_size=DEEPSPOT_EDGE_PAIR_BATCH_SIZE,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + 100_003 + epoch),
                num_workers=num_workers,
                collate_fn=collate_deepspot_edge_pairs,
            )
            edge_iterator = iter(edge_loader)
            model.train()
            for batch_idx, ordinary in enumerate(ordinary_loader):
                try:
                    edge = next(edge_iterator)
                except StopIteration:
                    edge_iterator = iter(edge_loader)
                    edge = next(edge_iterator)
                pair = _move_deepspot_pair(ordinary, edge, device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.training_step(pair, batch_idx)
                if not torch.isfinite(loss):
                    raise FormalRunnerError("DEEPSPOT_TRAIN_LOSS_NONFINITE")
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu())
            validation_mse = _deepspot_validation_mse(
                model,
                validation_dataset,
                device=device,
                num_workers=num_workers,
            )
            if validation_mse < best_mse:
                best_mse = validation_mse
                best_epoch = epoch
                best_state = _state_cpu(model)
    finally:
        remove_deepspot_overlay(model)
    if best_state is None:
        raise FormalRunnerError("DEEPSPOT_BEST_CHECKPOINT_MISSING")
    return {
        "rho": float(rho),
        "best_epoch": int(best_epoch),
        "best_validation_mse": float(best_mse),
        "final_training_loss": final_loss,
        "state_dict": best_state,
    }


def _hist_validation_mse(
    model: torch.nn.Module,
    dataset: HiSTPartitionDataset,
    *,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = _move_namespace(dataset[index], device)
            _, loss = model(batch)
            losses.append(float(loss.detach().cpu()))
    if not losses or not np.isfinite(losses).all():
        raise FormalRunnerError("HIST_VALIDATION_LOSS_NONFINITE")
    return float(np.mean(losses))


def _train_hist_arm(
    *,
    rho: float,
    seed: int,
    factory,
    initial_state: dict[str, torch.Tensor],
    training_dataset: HiSTPartitionDataset,
    validation_dataset: HiSTPartitionDataset,
    cfg,
    device: torch.device,
    num_workers: int,
) -> dict[str, Any]:
    from lib.utils.coord_aug import compute_coord_aug_multiplier

    seed_all(seed)
    model = factory().to(device)
    model.load_state_dict(initial_state, strict=True)
    install_hist_overlay(model, rho=rho)
    optimizer = build_hist_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.training.epochs),
        eta_min=float(cfg.training.lr)
        * float(cfg.training.get("eta_min_factor", 0.0)),
    )
    best_mse = float("inf")
    best_epoch = -1
    best_state = None
    final_loss = float("nan")
    eval_frequency = cfg.training.get("eval_metric_freq", None)
    try:
        for epoch in range(1, int(cfg.training.epochs) + 1):
            schedule = cfg.training.coord_aug.get("schedule", {})
            multiplier = (
                compute_coord_aug_multiplier(
                    schedule, epoch, int(cfg.training.epochs)
                )
                if cfg.training.coord_aug.get("enabled", False)
                and schedule.get("enabled", False)
                else 1.0
            )
            training_dataset.set_aug_multiplier(multiplier)
            loader = DataLoader(
                training_dataset,
                batch_size=1,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + epoch),
                num_workers=num_workers,
                collate_fn=_one_item,
            )
            model.train()
            for batch in loader:
                moved = _move_namespace(batch, device)
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(moved)
                if not torch.isfinite(loss):
                    raise FormalRunnerError("HIST_TRAIN_LOSS_NONFINITE")
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach().cpu())
            should_validate = (
                epoch == int(cfg.training.epochs)
                if eval_frequency is None
                else epoch % int(eval_frequency) == 0
                or epoch == int(cfg.training.epochs)
            )
            if should_validate:
                validation_mse = _hist_validation_mse(
                    model, validation_dataset, device=device
                )
                if validation_mse < best_mse:
                    best_mse = validation_mse
                    best_epoch = epoch
                    best_state = _state_cpu(model)
            scheduler.step()
    finally:
        remove_hist_overlay(model)
    if best_state is None:
        raise FormalRunnerError("HIST_BEST_CHECKPOINT_MISSING")
    return {
        "rho": float(rho),
        "best_epoch": int(best_epoch),
        "best_validation_mse": float(best_mse),
        "final_training_loss": final_loss,
        "state_dict": best_state,
    }


def _save_checkpoint(
    path: Path,
    *,
    model: str,
    task: str,
    fold: int,
    seed: int,
    arm: dict[str, Any],
) -> None:
    _atomic_torch_save(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "model": model,
            "task": task,
            "fold": int(fold),
            "seed": int(seed),
            "rho": float(arm["rho"]),
            "best_epoch": int(arm["best_epoch"]),
            "best_validation_mse": float(arm["best_validation_mse"]),
            "state_dict": arm["state_dict"],
            "contains_endpoint_scores": False,
        },
    )


def _predict_deepspot(
    model: torch.nn.Module,
    support,
    *,
    feature_store: DeepSpotFeatureStore,
    standardizer: FrozenStandardizer,
    device: torch.device,
    num_workers: int,
) -> np.ndarray:
    dataset = DeepSpotBlindDataset(support, feature_store=feature_store)
    loader = DataLoader(
        dataset,
        batch_size=DEEPSPOT_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_deepspot_blind,
    )
    scaled = np.empty((len(support.barcodes), GENES), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = [value.to(device) for value in batch["x"]]
            values = model(x).detach().cpu().numpy()
            scaled[batch["source_rows"].numpy()] = values
    return standardizer.inverse_transform(scaled)


def _predict_hist(
    model: torch.nn.Module,
    support,
    *,
    feature_store: HiSTFeatureStore,
    source_root: Path,
    device: torch.device,
) -> np.ndarray:
    batch = build_hist_blind_batch(
        support, feature_store=feature_store, deps_root=source_root
    )
    original_rows = batch.g0055_original_rows.numpy()
    moved = _move_namespace(batch, device)
    model.eval()
    with torch.no_grad():
        values, _ = model(moved)
    prediction = np.empty((len(support.barcodes), GENES), dtype=np.float32)
    prediction[original_rows] = values.detach().cpu().numpy()
    return prediction


def train_predict_run(args: argparse.Namespace) -> dict[str, Any]:
    model_name = canonical_model_name(args.model)
    prepared = _load_prepared(
        Path(args.prepared_state),
        model=model_name,
        task=args.task,
        fold=args.fold,
        seed=args.seed,
    )
    fold = load_aligned_hest_fold(
        dataset_root=Path(args.dataset_root),
        gene_manifest=Path(args.gene_manifest),
        task=args.task,
        fold=args.fold,
        g0054_module=args.g0054_module,
        include_test=False,
    )
    splits = build_fold_inner_splits(fold, model=model_name)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FormalRunnerError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    initial_state = prepared["state_dict"]
    feature_root = Path(args.feature_root)
    source_root = Path(args.source_root)
    num_workers = int(args.num_workers)

    if model_name == "DeepSpot":
        targets = fit_deepspot_targets(
            fold, splits, dataset_root=Path(args.dataset_root)
        )
        model_payload = prepared["model_payload"]
        standardizer = FrozenStandardizer(
            np.asarray(model_payload["standardizer_mean"], dtype=np.float64),
            np.asarray(model_payload["standardizer_scale"], dtype=np.float64),
        )
        feature_store = DeepSpotFeatureStore(
            feature_root, feature_dim=int(model_payload["input_size"])
        )
        training, edges, validation, edge_count = _deep_spot_partitions(
            fold, splits, targets, feature_store
        )

        def factory():
            return build_upstream_deepspot(
                deps_root=source_root,
                input_size=int(model_payload["input_size"]),
                seed=args.seed,
                scaler=standardizer.sklearn_scaler(),
            )

        arms = [
            _train_deepspot_arm(
                rho=rho,
                seed=args.seed,
                factory=factory,
                initial_state=initial_state,
                training_dataset=training,
                edge_dataset=edges,
                validation_dataset=validation,
                device=device,
                num_workers=num_workers,
            )
            for rho in RHOS
        ]
        expression_scale = str(model_payload["expression_scale"])
        training_edges = edge_count
        cfg = None
    else:
        cfg = compose_hist_config(source_root)
        from lib.utils.coord_aug import build_coord_aug_transform

        feature_store = HiSTFeatureStore(feature_root)
        training = HiSTPartitionDataset(
            fold.train_specimens,
            splits,
            validation=False,
            feature_store=feature_store,
            deps_root=source_root,
            coordinate_transform=build_coord_aug_transform(cfg),
        )
        validation = HiSTPartitionDataset(
            fold.train_specimens,
            splits,
            validation=True,
            feature_store=feature_store,
            deps_root=source_root,
            coordinate_transform=None,
        )

        def factory():
            return build_upstream_hist(
                deps_root=source_root, cfg=cfg, num_genes=GENES
            )

        arms = [
            _train_hist_arm(
                rho=rho,
                seed=args.seed,
                factory=factory,
                initial_state=initial_state,
                training_dataset=training,
                validation_dataset=validation,
                cfg=cfg,
                device=device,
                num_workers=num_workers,
            )
            for rho in RHOS
        ]
        expression_scale = str(prepared["model_payload"]["expression_scale"])
        training_edges = int(
            sum(training[index].g0055_edge_index.shape[1] for index in range(len(training)))
        )

    output_root = Path(args.output_root)
    checkpoint_paths = []
    for arm in arms:
        rho_tag = str(arm["rho"]).replace(".", "p")
        checkpoint_path = (
            output_root
            / "checkpoints"
            / model_name
            / args.task
            / f"fold{args.fold}.seed{args.seed}.rho{rho_tag}.pt"
        )
        _save_checkpoint(
            checkpoint_path,
            model=model_name,
            task=args.task,
            fold=args.fold,
            seed=args.seed,
            arm=arm,
        )
        checkpoint_paths.append(str(checkpoint_path.resolve()))

    # Training and ordinary validation are complete before held-out support is
    # opened.  This loader succeeds without an h5ad X matrix.
    support = load_blind_hest_support(
        args.dataset_root,
        task=args.task,
        specimen=fold.test_specimen_name,
        genes=prepared["genes"],
    )
    records = []
    for arm in arms:
        model = factory().to(device)
        model.load_state_dict(arm["state_dict"], strict=True)
        if model_name == "DeepSpot":
            prediction = _predict_deepspot(
                model,
                support,
                feature_store=feature_store,
                standardizer=standardizer,
                device=device,
                num_workers=num_workers,
            )
        else:
            prediction = _predict_hist(
                model,
                support,
                feature_store=feature_store,
                source_root=source_root,
                device=device,
            )
        rho_tag = str(arm["rho"]).replace(".", "p")
        prediction_path = (
            output_root
            / "predictions"
            / model_name
            / args.task
            / (
                f"fold{args.fold}.{support.specimen}.seed{args.seed}."
                f"rho{rho_tag}.h5"
            )
        )
        records.append(
            write_prediction_h5(
                prediction_path,
                model=model_name,
                task=args.task,
                fold=args.fold,
                specimen=support.specimen,
                seed=args.seed,
                rho=arm["rho"],
                predictions=prediction,
                barcodes=support.barcodes,
                genes=support.genes,
                expression_scale=expression_scale,
            )
        )

    manifest_stem = (
        output_root
        / "manifests"
        / model_name
        / args.task
        / f"fold{args.fold}.seed{args.seed}.tsv"
    )
    manifest_tsv, manifest_json = write_pair_manifest(
        manifest_stem,
        records,
        provenance={
            "source_revision": SOURCE_REVISIONS[model_name],
            "prepared_state": str(Path(args.prepared_state).resolve()),
            "inner_split_source": splits[0].source,
            "rho_arms": list(RHOS),
            "same_initial_state": True,
            "same_seed": True,
            "same_ordinary_order": True,
            "checkpoint_selection": "ordinary_inner_validation_mse",
            "training_edges": int(training_edges),
            "whole_context": model_name == "HiST",
            "measured_held_out_rna_loaded": False,
        },
    )
    return {
        "status": "PASS_PAIRED_PREDICTIONS_WRITTEN",
        "model": model_name,
        "task": args.task,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "outer_test_specimen": support.specimen,
        "arms": [
            {
                key: arm[key]
                for key in (
                    "rho",
                    "best_epoch",
                    "best_validation_mse",
                    "final_training_loss",
                )
            }
            for arm in arms
        ],
        "checkpoint_paths": checkpoint_paths,
        "prediction_manifest_tsv": str(manifest_tsv),
        "prediction_manifest_json": str(manifest_json),
        "prediction_files": len(records),
        "held_out_rna_loaded": False,
        "held_out_pcc_calculated": False,
        "staudit_endpoints_calculated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "train-predict"), required=True
    )
    parser.add_argument("--model", choices=("DeepSpot", "HiST"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gene-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--engineering-receipt", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASK_SPECIMENS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--prepared-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--g0054-module", type=Path)
    return parser


def _failure_status(model: str, exc: BaseException) -> str:
    if isinstance(exc, EngineeringNotEstimable):
        return exc.status
    if model == "HiST":
        return classify_failure(model, exc)
    return "FORMAL_RUNNER_FAILED_BEFORE_ENDPOINT_SCORING"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "train-predict" and args.output_root is None:
        parser.error("train-predict requires --output-root")
    try:
        result = prepare_run(args) if args.mode == "prepare" else train_predict_run(args)
        exit_code = 0
    except BaseException as exc:
        model_name = canonical_model_name(args.model)
        result = {
            "status": _failure_status(model_name, exc),
            "model": model_name,
            "task": args.task,
            "fold": int(args.fold),
            "seed": int(args.seed),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "held_out_rna_loaded": False,
            "held_out_pcc_calculated": False,
            "staudit_endpoints_calculated": False,
        }
        exit_code = 2
    _atomic_json(Path(args.receipt), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "model": result["model"],
                "held_out_rna_loaded": False,
                "staudit_endpoints_calculated": False,
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
