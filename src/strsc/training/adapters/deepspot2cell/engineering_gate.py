#!/usr/bin/env python3
"""Real one-batch DeepSpot2Cell no-op gate without scientific scoring."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, default_collate

from strsc.training.shared.noop import check_rho_zero_noop

from .formal_runner import (
    _add_upstream,
    _build_g0052_dataset,
    _load_config,
    spatial_holdout_indices,
)
from .graph import build_frozen_train_adjacency
from .initialization import capture_initial_state, restore_initial_state
from .overlay import install_local_difference_overlay, remove_local_difference_overlay
from .paired_data import FrozenEdgePairDataset


@dataclass
class RngState:
    python: object
    numpy: tuple
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None


def capture_rng() -> RngState:
    return RngState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng(state: RngState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(state.torch_cuda)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _gradients(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _run_and_capture_prediction(
    model: torch.nn.Module, call: Callable[[], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture the main superspot prediction without changing its return path."""

    original = model._forward_superspot
    predictions: list[torch.Tensor] = []

    def wrapped(*args, **kwargs):
        output = original(*args, **kwargs)
        prediction = output[0] if isinstance(output, tuple) else output
        predictions.append(prediction.detach().cpu().clone())
        return output

    model._forward_superspot = wrapped
    try:
        loss = call()
    finally:
        model._forward_superspot = original
    if not predictions:
        raise RuntimeError("DEEPSPOT2CELL_PREDICTION_CAPTURE_EMPTY")
    return loss, predictions[0]


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    _add_upstream(args.upstream_root)
    from deepspot2cell.utils.utils import fix_seed
    from scripts.train import build_model

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    cfg = _load_config(args.config)
    seed = int(cfg["experiment"]["random_seed"])
    fix_seed(seed)
    dataset = _build_g0052_dataset(cfg)
    train_idx, val_idx, cutoff = spatial_holdout_indices(
        dataset.patch_barcodes, val_fraction=args.val_fraction
    )
    batch_size = min(int(args.spots), len(train_idx))
    if batch_size < 1:
        raise RuntimeError("REAL_GATE_ORDINARY_BATCH_EMPTY")
    ordinary_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=False, num_workers=0
    )
    validation_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=min(batch_size, len(val_idx)), shuffle=False
    )
    ordinary_batch = _to_device(next(iter(ordinary_loader)), device)
    validation_batch = _to_device(next(iter(validation_loader)), device)

    model = build_model(dataset, cfg).to(device)
    model.train()
    initial_state = capture_initial_state(model)
    initial_rng = capture_rng()

    restore_rng(initial_rng)
    original_loss, original_prediction = _run_and_capture_prediction(
        model, lambda: model.training_step(ordinary_batch, 0)
    )
    original_loss.backward()
    original_gradients = _gradients(model)

    restore_initial_state(model, initial_state)
    model.zero_grad(set_to_none=True)
    install_local_difference_overlay(model, rho=0.0)
    restore_rng(initial_rng)
    rho_zero_loss, rho_zero_prediction = _run_and_capture_prediction(
        model, lambda: model.training_step(ordinary_batch, 0)
    )
    rho_zero_loss.backward()
    rho_zero_gradients = _gradients(model)
    noop = check_rho_zero_noop(
        original_loss=original_loss,
        rho_zero_loss=rho_zero_loss,
        original_predictions=original_prediction,
        rho_zero_predictions=rho_zero_prediction,
        original_gradients=original_gradients,
        rho_zero_gradients=rho_zero_gradients,
        atol=args.atol,
        rtol=args.rtol,
    )
    model.eval()
    with torch.no_grad():
        validation_loss = model.validation_step(validation_batch, 0)
    if not torch.isfinite(validation_loss):
        raise RuntimeError("ORDINARY_VALIDATION_LOSS_NONFINITE")
    remove_local_difference_overlay(model)

    # The edge graph and edge batch are deliberately constructed only after
    # the rho-zero loss/prediction/all-gradient audit has completed.
    graph = build_frozen_train_adjacency(dataset.patch_barcodes, train_idx)
    edge_dataset = FrozenEdgePairDataset(dataset, graph)
    edge_count = min(int(args.edges), len(edge_dataset))
    if edge_count < 1:
        raise RuntimeError("REAL_GATE_EDGE_BATCH_EMPTY")
    edge_batch = default_collate([edge_dataset[index] for index in range(edge_count)])
    intervention_batch = _to_device(
        {"ordinary": ordinary_batch, "edge": edge_batch}, device
    )

    restore_initial_state(model, initial_state)
    model.zero_grad(set_to_none=True)
    model.train()
    install_local_difference_overlay(model, rho=0.5)
    restore_rng(initial_rng)
    intervention_loss = model.training_step(intervention_batch, 0)
    intervention_loss.backward()
    if not torch.isfinite(intervention_loss):
        raise RuntimeError("INTERVENTION_LOSS_NONFINITE")
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise RuntimeError("INTERVENTION_GRADIENT_NONFINITE")

    return {
        "status": "PASS",
        "seed": seed,
        "ordinary_spots": batch_size,
        "edge_pairs": edge_count,
        "train_spots": len(train_idx),
        "validation_spots": len(val_idx),
        "column_cutoff": cutoff,
        "rho_arms": [0.0, 0.5],
        "ordinary_validation_loss_finite": True,
        "intervention_backward_finite": True,
        "noop": noop.to_dict(),
        "scientific_results_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spots", type=int, default=4)
    parser.add_argument("--edges", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    try:
        result = run_gate(args)
    except Exception as exc:
        result = {
            "status": "ENGINEERING_NOT_ESTIMABLE",
            "error": f"{type(exc).__name__}:{exc}",
            "scientific_results_opened": False,
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        raise
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
