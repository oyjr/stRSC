#!/usr/bin/env python3
"""Prepare and execute paired sCellST training and prediction arms."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch

from .runtime import (
    TEST_SPECIMENS,
    SCellSTRunnerError,
    atomic_json,
    atomic_torch_save,
    bootstrap_upstream,
    build_model,
    initial_state_payload,
    load_frozen_config,
    load_initial_state,
    make_loaders,
    move_to_device,
    prediction_filename,
    prepare_training_data,
    restore_rng_state,
    set_run_seed,
    validate_initial_state,
    validate_rho,
    validate_seed,
)


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SCellSTRunnerError(f"DEVICE_UNAVAILABLE:{device}")
    if device.type not in {"cpu", "cuda"}:
        raise SCellSTRunnerError(f"DEVICE_UNSUPPORTED:{device}")
    return device


def _run_tag(seed: int, rho: float) -> tuple[str, str]:
    rho_tag = str(validate_rho(rho)).replace(".", "p")
    return (
        "g0055_scellst_tenx39",
        (
            "embedding_tag=imagenet-rn50_train;genes=xenium_Breast;"
            f"train_slide=TENX39;seed={validate_seed(seed)};rho={rho_tag}"
        ),
    )


def _prepare_trainer(config: Any, save_dir: Path, device: torch.device):
    """Mirror the released trainer while binding its one device explicitly."""

    import lightning as L
    from lightning.pytorch.loggers import WandbLogger
    from scellst.trainer import prepare_callbacks

    if save_dir.exists() and any(save_dir.iterdir()):
        raise SCellSTRunnerError(f"RUN_DIRECTORY_NOT_EMPTY:{save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(
        project="sCellST",
        save_dir=save_dir,
        name=save_dir.name,
        offline=True,
    )
    callbacks = prepare_callbacks(save_dir, int(config.trainer.patience))
    accelerator = "cpu" if device.type == "cpu" else "gpu"
    devices: Any = 1 if device.type == "cpu" else [0 if device.index is None else device.index]
    return L.Trainer(
        max_epochs=int(config.trainer.max_epoch),
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=10,
    )


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    _, revision = bootstrap_upstream(args.upstream_root)
    seed = validate_seed(args.seed)
    config = load_frozen_config(args.config)
    set_run_seed(seed)
    _, prepared = prepare_training_data(config, split_manifest=args.split_manifest)
    model = build_model(config, rho=0.0)
    payload = initial_state_payload(
        model,
        config,
        prepared,
        seed=seed,
        source_revision=revision,
    )
    atomic_torch_save(args.initial_state, payload)
    return {
        "schema": "g0055_scellst_prepare_receipt_v1",
        "status": "INITIAL_STATE_FROZEN",
        "seed": seed,
        "source_revision": revision,
        "initial_state": str(Path(args.initial_state).resolve()),
        "split_manifest": str(prepared.split.manifest_path.resolve()),
        "training_spots": len(prepared.train_dataset),
        "validation_spots": len(prepared.val_dataset),
        "genes": len(config.model.gene_names),
        "scientific_endpoint_opened": False,
    }


def _load_upstream_inference_model(
    config: Any,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    from scellst.lightning_model.gene_lightning_model import GeneLightningModel

    model = GeneLightningModel(predictor_config=config.predictor, **config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise SCellSTRunnerError(f"CHECKPOINT_STATE:{checkpoint_path}")
    model.load_state_dict(state, strict=True)
    model.set_test_mode("instance")
    return model.to(device).eval()


def _predict_one_specimen(
    model: torch.nn.Module,
    config: Any,
    *,
    specimen: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    from scellst.dataset.data_module import prepare_data_module
    from scellst.predict import add_information_cell_adata, format_predictions

    predict_config = copy.deepcopy(config)
    predict_config.data.predict_id = specimen
    predict_config.data.dataset_handler = "xenium"
    predict_config.data.shape_name = "xenium_nucleus"
    predict_config.data.batch_size = int(batch_size)
    predict_config.data.num_workers = int(num_workers)
    predict_config.data.genes = list(config.model.gene_names)
    data_module = prepare_data_module(
        predict_config.data,
        stage="inference",
        task_type=predict_config.model.task_type,
    )
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(data_module.predict_dataloader()):
            moved = move_to_device(batch, device)
            prediction = model.predict_step(moved, batch_index)
            if not isinstance(prediction, torch.Tensor) or not torch.isfinite(
                prediction
            ).all():
                raise SCellSTRunnerError(f"PREDICTION_NONFINITE:{specimen}:{batch_index}")
            predictions.append(prediction.detach().cpu())
    if not predictions:
        raise SCellSTRunnerError(f"PREDICTION_EMPTY:{specimen}")
    predicted = format_predictions(predictions, data_module)
    predicted = add_information_cell_adata(predicted)
    return predicted


def _atomic_write_h5ad(adata: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SCellSTRunnerError(f"PREDICTION_EXISTS:{output}")
    temporary = output.with_name(f"{output.stem}.tmp.{os.getpid()}.h5ad")
    adata.uns["cell_embedding_path"] = str(adata.uns["cell_embedding_path"])
    adata.write_h5ad(temporary)
    os.replace(temporary, output)


def run_train_predict(args: argparse.Namespace) -> dict[str, Any]:
    _, revision = bootstrap_upstream(args.upstream_root)
    seed = validate_seed(args.seed)
    rho = validate_rho(args.rho)
    device = _resolve_device(args.device)
    config = load_frozen_config(args.config)
    set_run_seed(seed)
    _, prepared = prepare_training_data(config, split_manifest=args.split_manifest)
    model = build_model(config, rho=rho)
    initial = load_initial_state(args.initial_state)
    validate_initial_state(
        initial,
        config,
        prepared,
        seed=seed,
        source_revision=revision,
    )
    model.load_state_dict(initial["model_state"], strict=True)
    restore_rng_state(initial["rng_state"], seed=seed)

    loaders = make_loaders(prepared, config, seed=seed, rho=rho)
    save_dir_tag, exp_tag = _run_tag(seed, rho)
    config.save_dir_tag = save_dir_tag
    config.exp_tag = exp_tag
    run_root = Path(args.run_root).resolve()
    save_dir = run_root / "models" / "mil" / save_dir_tag / exp_tag
    trainer = _prepare_trainer(config, save_dir, device)
    from omegaconf import OmegaConf

    OmegaConf.save(config=config, f=save_dir / "config.yaml")
    trainer.fit(
        model,
        train_dataloaders=loaders.train,
        val_dataloaders=loaders.validation,
    )
    checkpoint = save_dir / "best_model.ckpt"
    if not checkpoint.is_file():
        raise SCellSTRunnerError(f"CHECKPOINT_MISSING:{checkpoint}")

    inference_model = _load_upstream_inference_model(config, checkpoint, device)
    prediction_root = Path(args.prediction_root).resolve()
    prediction_rows = []
    for specimen in TEST_SPECIMENS:
        predicted = _predict_one_specimen(
            inference_model,
            config,
            specimen=specimen,
            device=device,
            batch_size=args.prediction_batch_size,
            num_workers=args.prediction_workers,
        )
        output = prediction_root / prediction_filename(specimen, seed=seed, rho=rho)
        _atomic_write_h5ad(predicted, output)
        prediction_rows.append(
            {
                "specimen": specimen,
                "path": str(output),
                "cells": int(predicted.n_obs),
                "genes": int(predicted.n_vars),
            }
        )

    return {
        "schema": "g0055_scellst_train_predict_receipt_v1",
        "status": "COMPLETE",
        "seed": seed,
        "rho": rho,
        "source_revision": revision,
        "initial_state": str(Path(args.initial_state).resolve()),
        "split_manifest": str(prepared.split.manifest_path.resolve()),
        "ordinary_loader_seed": seed,
        "edge_loader_seed": None if rho == 0.0 else seed + 100_003,
        "baseline_edge_loader_constructed": False if rho == 0.0 else None,
        "training_graph": loaders.graph_metadata,
        "checkpoint_monitor": "val_loss",
        "checkpoint": str(checkpoint),
        "maximum_epochs": int(config.trainer.max_epoch),
        "early_stopping_patience": int(config.trainer.patience),
        "optimizer": "upstream_AdamW",
        "predictions": prediction_rows,
        "scoring_run": False,
        "scientific_endpoint_opened_during_training": False,
    }


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(17, 23, 41), required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    prepare = subparsers.add_parser("prepare")
    _add_common(prepare)
    train = subparsers.add_parser("train-predict")
    _add_common(train)
    train.add_argument("--rho", type=float, choices=(0.0, 0.5), required=True)
    train.add_argument("--device", required=True)
    train.add_argument("--run-root", type=Path, required=True)
    train.add_argument("--prediction-root", type=Path, required=True)
    train.add_argument("--prediction-batch-size", type=int, default=2048)
    train.add_argument("--prediction-workers", type=int, default=4)
    args = parser.parse_args()
    try:
        payload = run_prepare(args) if args.mode == "prepare" else run_train_predict(args)
        exit_code = 0
    except Exception as exc:
        payload = {
            "schema": "g0055_scellst_runner_failure_v1",
            "status": "FAILED",
            "mode": args.mode,
            "error": f"{type(exc).__name__}:{exc}",
            "scoring_run": False,
            "scientific_endpoint_opened_during_training": False,
        }
        exit_code = 2
    atomic_json(args.receipt, payload)
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
