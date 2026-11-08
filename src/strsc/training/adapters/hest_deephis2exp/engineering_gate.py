#!/usr/bin/env python3
"""One real training minibatch gate; never calculates a STAudit endpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch


if __package__ in {None, ""}:
    from strsc.training.adapters.hest_deephis2exp.batching import (
        EdgePairDataset,
        build_frozen_training_edges,
        select_coordinate_local_rows,
    )
    from strsc.training.adapters.hest_deephis2exp.data import (
        HestSpotDataset,
        build_context_adjacency,
        load_aligned_hest_fold,
        load_specimen_batch,
    )
    from strsc.training.adapters.hest_deephis2exp.models import (
        build_upstream_model,
        canonical_model_name,
    )
    from strsc.training.adapters.hest_deephis2exp.objective import predict_expression
    from strsc.training.adapters.hest_deephis2exp.overlay import (
        install_objective_overlay,
        remove_objective_overlay,
    )
    from strsc.training.adapters.hest_deephis2exp.split import build_fold_inner_splits
else:
    from .batching import (
        EdgePairDataset,
        build_frozen_training_edges,
        select_coordinate_local_rows,
    )
    from .data import (
        HestSpotDataset,
        build_context_adjacency,
        load_aligned_hest_fold,
        load_specimen_batch,
    )
    from .models import build_upstream_model, canonical_model_name
    from .objective import predict_expression
    from .overlay import install_objective_overlay, remove_objective_overlay
    from .split import build_fold_inner_splits

from strsc.training.shared.adjacency import build_spatial_edges
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


def _initial_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _prediction_hook(storage: list[torch.Tensor]):
    def hook(module, inputs, output):
        del module, inputs
        value = output[0] if isinstance(output, tuple) else output
        storage.append(value.detach().cpu().clone())

    return hook


def _ordinary_gate_batch(
    fold, split, model_name: str, edge_pairs: int, context_spots: int
):
    """Load an ordinary inner-training batch without constructing an edge."""

    canonical = canonical_model_name(model_name)
    specimen = fold.train_specimens[0]
    train_rows = np.flatnonzero(split.train_mask)
    if canonical in {"ST-Net", "DeepSpaCE"}:
        spots = HestSpotDataset((specimen,))
        count = min(max(2, 2 * int(edge_pairs)), len(train_rows))
        if count < 2:
            raise RuntimeError("REAL_GATE_NO_ORDINARY_SPOTS")
        samples = [spots[int(row)] for row in train_rows[:count]]
        return torch.utils.data.default_collate(samples), specimen.specimen, count

    local_choice = select_coordinate_local_rows(
        specimen.spatial_coords[train_rows], int(context_spots)
    )
    rows = train_rows[local_choice]
    batch = load_specimen_batch(specimen, rows=rows)
    batch["loss_mask"] = torch.ones(len(rows), dtype=torch.bool)
    batch["context_adjacency"] = build_context_adjacency(
        batch["spatial_coords"].numpy(), neighbors=8
    )
    return batch, specimen.specimen, len(rows)


def _validation_gate_batch(fold, split, model_name: str, context_spots: int):
    """Load RNA only from the frozen inner-validation partition."""

    canonical = canonical_model_name(model_name)
    specimen = fold.train_specimens[0]
    validation_rows = np.flatnonzero(split.validation_mask)
    if canonical in {"ST-Net", "DeepSpaCE"}:
        spots = HestSpotDataset((specimen,))
        count = min(max(1, int(context_spots)), len(validation_rows))
        samples = [spots[int(row)] for row in validation_rows[:count]]
        return torch.utils.data.default_collate(samples)

    if len(validation_rows) < 2:
        raise RuntimeError("REAL_GATE_CONTEXT_VALIDATION_REQUIRES_TWO_SPOTS")
    local_choice = select_coordinate_local_rows(
        specimen.spatial_coords[validation_rows], int(context_spots)
    )
    rows = validation_rows[local_choice]
    batch = load_specimen_batch(specimen, rows=rows)
    batch["loss_mask"] = torch.ones(len(rows), dtype=torch.bool)
    return batch


def _intervention_gate_batch(
    fold,
    split,
    model_name: str,
    ordinary_batch: Mapping[str, Any],
    edge_pairs: int,
):
    """Construct the explicit local stream only after the rho-zero audit."""

    canonical = canonical_model_name(model_name)
    specimen = fold.train_specimens[0]
    if canonical in {"ST-Net", "DeepSpaCE"}:
        spots = HestSpotDataset((specimen,))
        graph = build_frozen_training_edges(spots, (split,))
        pair_dataset = EdgePairDataset(spots, graph)
        count = min(int(edge_pairs), len(pair_dataset))
        if count < 1:
            raise RuntimeError("REAL_GATE_NO_EDGE_PAIRS")
        samples = [pair_dataset[index] for index in range(count)]
        edge_batch = torch.utils.data.default_collate(samples)
        return {"ordinary": ordinary_batch, "edge": edge_batch}, count

    local_edges, _, _ = build_spatial_edges(
        ordinary_batch["spatial_coords"].detach().cpu().numpy(),
        specimen_id=specimen.specimen,
    )
    intervention = dict(ordinary_batch)
    intervention["edge_index"] = torch.from_numpy(
        np.asarray(local_edges).copy()
    ).long()
    return intervention, int(local_edges.shape[1])


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    canonical = canonical_model_name(args.model)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    fold = load_aligned_hest_fold(
        dataset_root=args.dataset_root,
        gene_manifest=args.gene_manifest,
        task=args.task,
        fold=args.fold,
        g0054_module=args.g0054_module,
        include_test=False,
    )
    splits = build_fold_inner_splits(fold, model=canonical)
    batch, gate_specimen, rows = _ordinary_gate_batch(
        fold, splits[0], canonical, args.edge_pairs, args.context_spots
    )
    batch = _to_device(batch, device)

    model = build_upstream_model(canonical, deps_root=args.deps_root).to(device)
    model.train()
    intervention_batch, intervention_edges = _intervention_gate_batch(
        fold, splits[0], canonical, batch, args.edge_pairs
    )
    intervention_batch = _to_device(intervention_batch, device)
    install_objective_overlay(model, model_name=canonical, rho=0.5)
    intervention_loss = model.training_step(intervention_batch, 0)
    intervention_loss.backward()
    if not torch.isfinite(intervention_loss):
        raise RuntimeError("INTERVENTION_LOSS_NONFINITE")
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise RuntimeError("INTERVENTION_GRADIENT_NONFINITE")
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("INTERVENTION_GRADIENT_MISSING")

    model.eval()
    with torch.no_grad():
        inference = predict_expression(canonical, model, batch)
    if inference.ndim < 2 or inference.shape[-1] != 50:
        raise RuntimeError(f"INFERENCE_SHAPE:{tuple(inference.shape)}")
    if not torch.isfinite(inference).all():
        raise RuntimeError("INFERENCE_NONFINITE")
    remove_objective_overlay(model)

    return {
        "status": "PASS",
        "model": canonical,
        "task": fold.task,
        "fold": fold.fold,
        "gate_training_specimen": gate_specimen,
        "gate_rows": int(rows),
        "gate_edges": int(intervention_edges),
        "genes": 50,
        "inner_split_source": splits[0].source,
        "inner_training_rows": int(splits[0].train_mask.sum()),
        "inner_validation_rows": int(splits[0].validation_mask.sum()),
        "rho_arms": [0.5],
        "intervention_backward_finite": True,
        "inference_shape": list(inference.shape),
        "gate_scope": "one_rho_half_forward_backward_and_one_inference",
        "scientific_results_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gene-manifest", type=Path, required=True)
    parser.add_argument("--deps-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=("IDC", "PAAD", "LUNG"), default="IDC")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--edge-pairs", type=int, default=4)
    parser.add_argument("--context-spots", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--g0054-module", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    try:
        receipt = run_gate(args)
    except Exception as exc:
        receipt = {
            "status": "ENGINEERING_NOT_ESTIMABLE",
            "model": str(args.model),
            "error": f"{type(exc).__name__}:{exc}",
            "scientific_results_opened": False,
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, sort_keys=True))
        raise
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
