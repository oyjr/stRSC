#!/usr/bin/env python3
"""Check sCellST inputs, predictions and gradients on one batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .runtime import (
    SCellSTRunnerError,
    atomic_json,
    bootstrap_upstream,
    build_model,
    load_frozen_config,
    make_loaders,
    move_to_device,
    prepare_training_data,
    require_finite_backward,
    set_run_seed,
    validate_seed,
)


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SCellSTRunnerError(f"DEVICE_UNAVAILABLE:{device}")
    return device


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    _, revision = bootstrap_upstream(args.upstream_root)
    seed = validate_seed(args.seed)
    config = load_frozen_config(args.config)
    set_run_seed(seed)
    _, prepared = prepare_training_data(config, split_manifest=args.split_manifest)
    loaders = make_loaders(
        prepared,
        config,
        seed=seed,
        rho=0.5,
        batch_size=min(int(config.data.batch_size), int(args.gate_batch_size)),
        num_workers=0,
    )
    ordinary_batch = next(iter(loaders.ordinary))
    if loaders.edge is None:
        raise SCellSTRunnerError("ENGINEERING_EDGE_LOADER_MISSING")
    edge_batch = next(iter(loaders.edge))
    validation_batch = next(iter(loaders.validation))

    device = _resolve_device(args.device)
    ordinary_batch = move_to_device(ordinary_batch, device)
    edge_batch = move_to_device(edge_batch, device)
    validation_batch = move_to_device(validation_batch, device)
    model = build_model(config, rho=0.5).to(device)
    model.train()
    intervention_loss = model.training_step(
        {"spot": ordinary_batch, "edge": edge_batch},
        0,
    )
    require_finite_backward(intervention_loss, model)

    # The gate's held-out inference is the frozen TENX39 validation partition;
    # it does not open any Xenium scientific endpoint.
    model.eval()
    with torch.no_grad():
        inference_prediction = model.predict_step(validation_batch, 0)
    if not isinstance(inference_prediction, torch.Tensor) or not torch.isfinite(
        inference_prediction
    ).all():
        raise SCellSTRunnerError("VALIDATION_INFERENCE_NONFINITE")
    train_indices = set(prepared.split.train_indices)
    validation_indices = set(prepared.split.val_indices)
    if train_indices & validation_indices:
        raise SCellSTRunnerError("SPLIT_NOT_DISJOINT")
    if loaders.graph_metadata is None or int(loaders.graph_metadata["nodes"]) != len(
        prepared.train_dataset
    ):
        raise SCellSTRunnerError("TRAIN_ONLY_GRAPH_NODE_COUNT")

    return {
        "schema": "g0055_scellst_engineering_gate_v1",
        "status": "PASS",
        "eligible": True,
        "source_revision": revision,
        "seed": seed,
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "split_seed": prepared.split.seed,
        "training_spots": len(prepared.train_dataset),
        "validation_spots": len(prepared.val_dataset),
        "training_edges": int(loaders.graph_metadata["edges"]),
        "ordinary_batch_spots": len(ordinary_batch["Y_bag"]),
        "edge_batch_pairs": len(edge_batch["left"]["Y_bag"]),
        "validation_inference_shape": list(inference_prediction.shape),
        "rho_half_backward_finite": True,
        "validation_inference_finite": True,
        "train_only_split_and_graph": True,
        "gate_scope": "one_rho_half_forward_backward_and_one_inference",
        "scientific_endpoint_opened": False,
    }


def classify_failure(exc: BaseException) -> str:
    text = f"{type(exc).__name__}:{exc}".lower()
    if "upstream_revision" in text:
        return "ENGINEERING_NOT_ESTIMABLE_SOURCE_REVISION"
    if "device_unavailable" in text or "out of memory" in text:
        return "ENGINEERING_NOT_ESTIMABLE_DEVICE"
    if any(token in text for token in ("missing", "no such file", "does not exist")):
        return "ENGINEERING_NOT_ESTIMABLE_MISSING_INPUT"
    if any(token in text for token in ("modulenotfounderror", "importerror")):
        return "ENGINEERING_NOT_ESTIMABLE_PINNED_ENVIRONMENT"
    return "ENGINEERING_NOT_ESTIMABLE_RUNTIME"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(17, 23, 41), default=17)
    parser.add_argument("--gate-batch-size", type=int, default=8)
    args = parser.parse_args()
    try:
        payload = run_gate(args)
        exit_code = 0
    except Exception as exc:
        payload = {
            "schema": "g0055_scellst_engineering_gate_v1",
            "status": classify_failure(exc),
            "eligible": False,
            "error": f"{type(exc).__name__}:{exc}",
            "scientific_endpoint_opened": False,
        }
        exit_code = 2
    atomic_json(args.receipt, payload)
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
