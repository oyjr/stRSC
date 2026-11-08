#!/usr/bin/env python3
"""Formal paired-arm runner for four unmodified DeepHis2Exp models.

The runner trains/predicts but never imports or calls a STAudit scorer.  Its
``prepare`` mode freezes one model initialization before either rho arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

from strsc.training.shared.adjacency import build_spatial_edges

from .batching import (
    build_frozen_training_edges,
    make_context_loader,
    make_matched_context_training_loaders,
    make_matched_patch_training_loaders,
    make_partition_spot_loader,
)
from .data import HestSpotDataset, TASK_SPECIMENS, load_aligned_hest_fold
from .export import export_hest_prediction, predict_aligned_specimen
from .models import build_upstream_model, canonical_model_name
from .overlay import install_objective_overlay, remove_objective_overlay
from .split import build_fold_inner_splits


INIT_SCHEMA = "g0055.hest_deephis2exp.initial_state.v1"
FROZEN_SEEDS = (17, 23, 41)
HIST2ST_EDGE_MICROBATCH_MAX_SPOTS = 1024
PUBLISHED_MAX_EPOCHS = {
    "ST-Net": 50,
    "DeepSpaCE": 100,
    "HisToGene": 100,
    "Hist2ST": 350,
}


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _capture_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _load_initial_state(
    path: Path, *, model: str, task: str, fold: int, seed: int
) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    expected = {
        "schema": INIT_SCHEMA,
        "model": model,
        "task": task,
        "fold": int(fold),
        "seed": int(seed),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"INITIAL_STATE_METADATA:{key}:{payload.get(key)!r}:{value!r}")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("INITIAL_STATE_EMPTY")
    return state


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    canonical = canonical_model_name(args.model)
    if int(args.seed) not in FROZEN_SEEDS:
        raise ValueError(
            f"G0-055 formal seed must be one of {FROZEN_SEEDS}, found {args.seed}"
        )
    if int(args.fold) < 0 or int(args.fold) >= len(TASK_SPECIMENS[str(args.task)]):
        raise ValueError(f"HEST fold is out of range for task {args.task}: {args.fold}")
    initial_path = Path(args.initial_state)
    if initial_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen initial state: {initial_path}")
    _seed_everything(args.seed)
    # No fold data or held-out RNA is needed to freeze architecture state.
    model = build_upstream_model(canonical, deps_root=args.deps_root)
    payload = {
        "schema": INIT_SCHEMA,
        "model": canonical,
        "task": str(args.task),
        "fold": int(args.fold),
        "seed": int(args.seed),
        "state_dict": _capture_state(model),
    }
    initial_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, initial_path)
    receipt = {
        "status": "PREPARED",
        "model": canonical,
        "task": str(args.task),
        "fold": int(args.fold),
        "seed": int(args.seed),
        "rho_arms": [0.0, 0.5],
        "scientific_results_opened": False,
    }
    _write_json(args.receipt, receipt)
    return receipt


def _training_loaders(args: argparse.Namespace, fold, splits, canonical: str):
    if canonical in {"ST-Net", "DeepSpaCE"}:
        spots = HestSpotDataset(fold.train_specimens)
        graph = build_frozen_training_edges(spots, splits)
        pair = make_matched_patch_training_loaders(
            spots,
            splits,
            graph,
            batch_size=int(args.batch_size),
            ordinary_seed=int(args.seed),
            edge_seed=int(args.seed) + 100_003,
            num_workers=int(args.num_workers),
        )
        validation = make_partition_spot_loader(
            spots,
            splits,
            validation=True,
            batch_size=int(args.validation_batch_size),
            num_workers=max(0, int(args.num_workers) // 2),
        )
        return pair, validation, len(graph.edges), len(graph.supervised_global_indices)

    context_subsets = int(args.hist2st_num_subsets) if canonical == "Hist2ST" else 1
    context_graph = None
    if canonical == "Hist2ST":
        context_graph = build_frozen_training_edges(
            HestSpotDataset(fold.train_specimens), splits
        )
    pair = make_matched_context_training_loaders(
        fold.train_specimens,
        splits,
        ordinary_seed=int(args.seed),
        num_subsets=context_subsets,
        frozen_edges=context_graph,
        edge_seed=int(args.seed) + 100_003,
        edge_microbatch_max_spots=int(args.hist2st_edge_microbatch_max_spots),
        num_workers=int(args.num_workers),
    )
    validation = make_context_loader(
        fold.train_specimens,
        splits,
        validation=True,
        seed=int(args.seed),
        num_subsets=context_subsets,
        num_workers=max(0, int(args.num_workers) // 2),
        include_local_edges=False,
    )
    edge_count = len(context_graph.edges) if context_graph is not None else 0
    training_rows = 0
    for specimen, split in zip(fold.train_specimens, splits, strict=True):
        rows = np.flatnonzero(split.train_mask)
        training_rows += len(rows)
        if context_graph is None:
            local_edges, _, _ = build_spatial_edges(
                specimen.spatial_coords[rows], specimen_id=specimen.specimen
            )
            edge_count += int(local_edges.shape[1])
    return pair, validation, edge_count, training_rows


def _build_trainer(
    args: argparse.Namespace,
    canonical: str,
    output_dir: Path,
    checkpoint_dir: Path,
):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-val",
        save_top_k=1,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        auto_insert_metric_name=False,
    )
    early = EarlyStopping(
        monitor="val_loss", mode="min", patience=int(args.patience)
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=int(args.devices),
        max_epochs=(
            int(args.max_epochs)
            if args.max_epochs is not None
            else PUBLISHED_MAX_EPOCHS[canonical]
        ),
        callbacks=[early, checkpoint],
        logger=CSVLogger(save_dir=output_dir, name="logs"),
        default_root_dir=str(output_dir),
        precision=args.precision,
        deterministic=bool(args.deterministic),
        log_every_n_steps=int(args.log_every_n_steps),
    )
    return trainer, checkpoint


def train_and_predict(args: argparse.Namespace, *, training_loader_factory=None) -> dict[str, Any]:
    canonical = canonical_model_name(args.model)
    if int(args.seed) not in FROZEN_SEEDS:
        raise ValueError(
            f"G0-055 formal seed must be one of {FROZEN_SEEDS}, found {args.seed}"
        )
    rho = float(args.rho)
    if rho not in (0.0, 0.5):
        raise ValueError("Frozen HEST arms permit only rho=0.0 or rho=0.5")
    _seed_everything(args.seed)
    fold = load_aligned_hest_fold(
        dataset_root=args.dataset_root,
        gene_manifest=args.gene_manifest,
        task=args.task,
        fold=args.fold,
        g0054_module=args.g0054_module,
        include_test=False,
    )
    splits = build_fold_inner_splits(fold, model=canonical)
    loader_factory = _training_loaders if training_loader_factory is None else training_loader_factory
    pair, validation_loader, edge_count, train_rows = loader_factory(
        args, fold, splits, canonical
    )
    training_loader = pair.baseline if rho == 0.0 else pair.intervention

    model = build_upstream_model(canonical, deps_root=args.deps_root)
    initial = _load_initial_state(
        args.initial_state,
        model=canonical,
        task=fold.task,
        fold=fold.fold,
        seed=args.seed,
    )
    model.load_state_dict(initial, strict=True)
    install_objective_overlay(model, model_name=canonical, rho=rho)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = (
        f"g0055-{canonical.lower().replace(' ', '-')}-{fold.task}-f{fold.fold}-"
        f"s{int(args.seed)}-rho{rho:g}-"
    )
    with tempfile.TemporaryDirectory(
        prefix=checkpoint_prefix,
        dir="/dev/shm",
    ) as checkpoint_tmp:
        trainer, checkpoint = _build_trainer(
            args,
            canonical,
            output_dir,
            Path(checkpoint_tmp),
        )
        trainer.fit(
            model,
            train_dataloaders=training_loader,
            val_dataloaders=validation_loader,
        )
        if not checkpoint.best_model_path:
            raise RuntimeError("BEST_VALIDATION_CHECKPOINT_MISSING")
        checkpoint_selection_path = str(checkpoint.best_model_path)
        checkpoint_selection_val_loss = (
            float(checkpoint.best_model_score.detach().cpu().item())
            if checkpoint.best_model_score is not None
            else None
        )
        checkpoint_payload = torch.load(
            checkpoint.best_model_path,
            map_location="cpu",
        )
        checkpoint_selection_epoch = checkpoint_payload.get("epoch")
        if checkpoint_selection_epoch is not None:
            checkpoint_selection_epoch = int(checkpoint_selection_epoch)
        model.load_state_dict(checkpoint_payload["state_dict"], strict=True)
        del checkpoint_payload
    remove_objective_overlay(model)

    # Only now, after ordinary validation selected the checkpoint, load the
    # held-out outer specimen for prediction. No score is calculated here.
    inference_fold = load_aligned_hest_fold(
        dataset_root=args.dataset_root,
        gene_manifest=args.gene_manifest,
        task=args.task,
        fold=args.fold,
        g0054_module=args.g0054_module,
        include_test=True,
    )
    specimen = inference_fold.test_specimen
    if specimen is None:
        raise RuntimeError("OUTER_TEST_SPECIMEN_MISSING_AT_INFERENCE")
    device = torch.device(args.inference_device)
    prediction = predict_aligned_specimen(
        canonical,
        model,
        specimen,
        device=device,
        patch_batch_size=int(args.inference_batch_size),
        context_num_subsets=(
            int(args.hist2st_num_subsets) if canonical == "Hist2ST" else 1
        ),
        num_workers=max(0, int(args.num_workers) // 2),
    )
    prediction_path = export_hest_prediction(
        args.prediction,
        specimen=specimen,
        prediction=prediction,
        model_name=canonical,
        fold=fold.fold,
        seed=args.seed,
        rho=rho,
        provenance={
            "inner_split_source": splits[0].source,
            "checkpoint_metric": "upstream_val_loss",
            "training_edges": int(edge_count),
            "training_precision": str(args.precision),
            "hist2st_num_subsets": (
                int(args.hist2st_num_subsets) if canonical == "Hist2ST" else 1
            ),
            "hist2st_edge_stream": (
                "all_frozen_train_edges_once_per_epoch"
                if canonical == "Hist2ST"
                else "ordinary_context"
            ),
            "hist2st_edge_microbatch_max_spots": (
                int(args.hist2st_edge_microbatch_max_spots)
                if canonical == "Hist2ST"
                else 0
            ),
            "hist2st_edge_microbatch_checkpointed": canonical == "Hist2ST",
            "histogene_attention_implementation": (
                "fp32_full_key_value_query_chunk_checkpoint"
                if canonical == "HisToGene"
                else "not_applicable"
            ),
            "histogene_attention_query_chunk_size": (
                512 if canonical == "HisToGene" else 0
            ),
            "histogene_patch_embedding_implementation": (
                "fp32_cpu_stream_spot_chunk_checkpoint"
                if canonical == "HisToGene"
                else "not_applicable"
            ),
            "histogene_patch_embedding_chunk_size": (
                int(getattr(model, "patch_embedding_chunk_size"))
                if canonical == "HisToGene"
                else 0
            ),
        },
    )
    receipt = {
        "status": "PREDICTION_EXPORTED",
        "model": canonical,
        "task": fold.task,
        "fold": fold.fold,
        "outer_test_specimen": specimen.specimen,
        "seed": int(args.seed),
        "rho": rho,
        "training_rows": int(train_rows),
        "validation_rows": int(sum(split.validation_mask.sum() for split in splits)),
        "training_edges": int(edge_count),
        "checkpoint_metric": "upstream_val_loss",
        "training_precision": str(args.precision),
        "hist2st_num_subsets": (
            int(args.hist2st_num_subsets) if canonical == "Hist2ST" else 1
        ),
        "hist2st_edge_stream": (
            "all_frozen_train_edges_once_per_epoch"
            if canonical == "Hist2ST"
            else "ordinary_context"
        ),
        "hist2st_edge_microbatch_max_spots": (
            int(args.hist2st_edge_microbatch_max_spots)
            if canonical == "Hist2ST"
            else 0
        ),
        "hist2st_edge_microbatch_checkpointed": canonical == "Hist2ST",
        "histogene_attention_implementation": (
            "fp32_full_key_value_query_chunk_checkpoint"
            if canonical == "HisToGene"
            else "not_applicable"
        ),
        "histogene_attention_query_chunk_size": (
            512 if canonical == "HisToGene" else 0
        ),
        "histogene_patch_embedding_implementation": (
            "fp32_cpu_stream_spot_chunk_checkpoint"
            if canonical == "HisToGene"
            else "not_applicable"
        ),
        "histogene_patch_embedding_chunk_size": (
            int(getattr(model, "patch_embedding_chunk_size"))
            if canonical == "HisToGene"
            else 0
        ),
        "checkpoint_path": checkpoint_selection_path,
        "checkpoint_storage": "node_local_tmpfs",
        "checkpoint_retained": False,
        "checkpoint_selection_path": checkpoint_selection_path,
        "checkpoint_selection_epoch": checkpoint_selection_epoch,
        "checkpoint_selection_val_loss": checkpoint_selection_val_loss,
        "prediction_path": str(prediction_path),
        "scientific_endpoint_opened": False,
    }
    _write_json(args.receipt, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "train-predict"), required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--gene-manifest", type=Path)
    parser.add_argument("--deps-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=("IDC", "PAAD", "LUNG"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default=32)
    parser.add_argument("--hist2st-num-subsets", type=int, default=15)
    parser.add_argument(
        "--hist2st-edge-microbatch-max-spots",
        type=int,
        default=HIST2ST_EDGE_MICROBATCH_MAX_SPOTS,
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--inference-device", default="cuda:0")
    parser.add_argument("--g0054-module", type=Path)
    args = parser.parse_args()

    if args.mode == "train-predict":
        required = (args.dataset_root, args.gene_manifest, args.output_dir, args.prediction)
        if any(value is None for value in required) or args.rho is None:
            parser.error(
                "train-predict requires --dataset-root, --gene-manifest, --output-dir, "
                "--prediction and --rho"
            )
        result = train_and_predict(args)
    else:
        result = prepare(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
