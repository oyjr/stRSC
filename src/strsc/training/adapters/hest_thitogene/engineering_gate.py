#!/usr/bin/env python3
"""Check THItoGene inputs, predictions and gradients on one full-section batch."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Sequence

import torch


from strsc.training.adapters.hest_thitogene.hest_data import (  # noqa: E402
    MODEL_NAME,
    load_fold_metadata,
    materialize_section,
)
from strsc.training.adapters.hest_thitogene.overlay import (  # noqa: E402
    build_supervision,
    expression_loss,
    forward_section,
    run_rho_zero_noop_gate,
)
from strsc.training.adapters.hest_thitogene.upstream import (  # noqa: E402
    UpstreamContractError,
    assert_pinned_environment,
    instantiate_upstream_model,
    load_upstream_modules,
    verify_upstream_source,
)


class EngineeringGateError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _gpu_receipt(device: torch.device) -> dict[str, object]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"device": str(device), "cuda_available": torch.cuda.is_available()}
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": str(device),
        "cuda_available": True,
        "device_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def run_gate(args: argparse.Namespace) -> dict[str, object]:
    upstream_source = verify_upstream_source(args.upstream_root)
    packages = assert_pinned_environment()
    if args.preflight_only:
        return {
            "status": "PREFLIGHT_PASS_REAL_MINIBATCH_NOT_RUN",
            "eligible": False,
            "model": MODEL_NAME,
            "upstream": upstream_source.to_dict(),
            "packages": packages,
            "reason": "preflight_only_is_not_the_engineering_gate",
            "held_out_pcc_calculated": False,
            "staudit_bands_calculated": False,
        }

    vis_model, graph_module, upstream = load_upstream_modules(args.upstream_root)
    training, held_out = load_fold_metadata(
        dataset_root=args.dataset_root,
        gene_manifest=args.gene_manifest,
        embedding_manifest=args.embedding_manifest,
        patch_manifest=args.patch_manifest,
        task=args.task,
        fold=args.fold,
        include_test=False,
        g0054_module=args.g0054_module,
    )
    if held_out is not None:
        raise EngineeringGateError("HELD_OUT_LOADED_DURING_GATE")
    if args.specimen is None:
        metadata = training[0]
    else:
        matches = [row for row in training if row.specimen == args.specimen]
        if len(matches) != 1:
            raise EngineeringGateError(
                f"GATE_SPECIMEN_NOT_IN_TRAINING_FOLD:{args.specimen}"
            )
        metadata = matches[0]
    if metadata.validation_mask is None:
        raise EngineeringGateError("GATE_VALIDATION_MASK_MISSING")

    # No slicing, tiling, node sampling or reduced architecture is allowed here.
    cpu_batch = materialize_section(metadata, upstream_graph_module=graph_module)
    supervision = build_supervision(
        coords=metadata.spatial_coords,
        validation_mask=metadata.validation_mask,
        specimen_id=metadata.specimen,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EngineeringGateError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    batch = cpu_batch.to(device)
    model = instantiate_upstream_model(vis_model, n_genes=50).to(device)
    noop = run_rho_zero_noop_gate(
        model,
        patches=batch.patches,
        positions=batch.positions,
        model_adjacency=batch.model_adjacency,
        target=batch.target,
        supervision=supervision,
    )

    # One genuine intervention forward/backward and published Adam update.
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    prediction = forward_section(
        model,
        patches=batch.patches,
        positions=batch.positions,
        model_adjacency=batch.model_adjacency,
    )
    intervention_loss = expression_loss(
        prediction, batch.target, supervision, rho=0.5
    )
    if not torch.isfinite(intervention_loss):
        raise EngineeringGateError("INTERVENTION_LOSS_NONFINITE")
    intervention_loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise EngineeringGateError("INTERVENTION_GRADIENT_NONFINITE")
    optimizer.step()

    # Inference checks shape/finite values only. No RNA-derived endpoint is computed.
    model.eval()
    with torch.no_grad():
        inference = forward_section(
            model,
            patches=batch.patches,
            positions=batch.positions,
            model_adjacency=batch.model_adjacency,
        )
    expected = (len(metadata.barcodes), len(metadata.genes))
    if tuple(inference.shape) != expected or not torch.isfinite(inference).all():
        raise EngineeringGateError(
            f"INFERENCE_CONTRACT:{tuple(inference.shape)}:{expected}"
        )
    return {
        "status": "PASS_REAL_FULL_SECTION_MINIBATCH",
        "eligible": True,
        "model": MODEL_NAME,
        "task": metadata.task,
        "fold": metadata.fold,
        "specimen": metadata.specimen,
        "spots": len(metadata.barcodes),
        "genes": len(metadata.genes),
        "full_section": True,
        "tiling_or_node_sampling": False,
        "training_forward_backward": "PASS",
        "optimizer_step": "PASS_PUBLISHED_ADAM_LR_1E-5",
        "inference_shape": list(inference.shape),
        "prediction_finite": True,
        "gradient_finite": True,
        "noop": noop,
        "supervision": supervision.to_metadata(),
        "upstream": upstream.to_dict(),
        "packages": packages,
        "gpu": _gpu_receipt(device),
        "held_out_rna_loaded": False,
        "held_out_pcc_calculated": False,
        "staudit_bands_calculated": False,
    }


def classify_failure(error: BaseException) -> str:
    message = str(error).lower()
    if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return "ENGINEERING_NOT_ESTIMABLE_GPU_OOM"
    if isinstance(error, UpstreamContractError):
        return "ENGINEERING_NOT_ESTIMABLE_UPSTREAM_OR_ENVIRONMENT"
    return "ENGINEERING_NOT_ESTIMABLE_REAL_MINIBATCH_FAILED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--gene-manifest", type=Path)
    parser.add_argument("--embedding-manifest", type=Path)
    parser.add_argument("--patch-manifest", type=Path)
    parser.add_argument("--g0054-module", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--task", choices=("IDC", "PAAD", "LUNG"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--specimen")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _require_real_args(args: argparse.Namespace) -> None:
    if args.preflight_only:
        return
    required = (
        "dataset_root",
        "gene_manifest",
        "embedding_manifest",
        "patch_manifest",
        "task",
        "fold",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise EngineeringGateError(f"REAL_GATE_ARGUMENTS_MISSING:{missing}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _require_real_args(args)
        result = run_gate(args)
        exit_code = 0
    except BaseException as error:  # write auditable evidence for OOM/import/data failures
        result = {
            "status": classify_failure(error),
            "eligible": False,
            "model": MODEL_NAME,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "no_architecture_substitution": True,
            "no_tiling_or_node_sampling": True,
            "held_out_pcc_calculated": False,
            "staudit_bands_calculated": False,
        }
        exit_code = 2
    _atomic_json(args.receipt, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "eligible": result["eligible"],
                "model": MODEL_NAME,
                "held_out_pcc_calculated": False,
                "staudit_bands_calculated": False,
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
