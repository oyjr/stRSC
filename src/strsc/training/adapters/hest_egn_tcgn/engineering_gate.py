#!/usr/bin/env python3
"""Check EGN/TCGN inputs, predictions and gradients on one minibatch."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch


from strsc.training.adapters.hest_egn_tcgn.contracts import (  # noqa: E402
    MODEL_CONFIGS,
    canonical_model_name,
)
from strsc.training.adapters.hest_egn_tcgn.exemplars import (  # noqa: E402
    build_training_bank,
    extract_published_features,
    instantiate_published_encoder,
)
from strsc.training.adapters.hest_egn_tcgn.hest_data import (  # noqa: E402
    load_fold_data,
    normalize_model_patches,
    read_aligned_raw_patches,
)
from strsc.training.adapters.hest_egn_tcgn.model_inputs import (  # noqa: E402
    build_egn_v2_graph,
    egn_v1_batch,
)
from strsc.training.adapters.hest_egn_tcgn.overlay import (  # noqa: E402
    build_supervision,
    edge_pair_batch,
    full_section_loss,
    minibatch_loss,
    preserve_model_buffers,
    published_full_section_loss,
    published_minibatch_loss,
    run_rho_zero_noop_gate,
)
from strsc.training.adapters.hest_egn_tcgn.pairing import edge_generator, epoch_order  # noqa: E402
from strsc.training.adapters.hest_egn_tcgn.upstream import (  # noqa: E402
    UpstreamContractError,
    assert_pinned_environment,
    instantiate_upstream_model,
    load_tcgn_pretraining,
    load_upstream_model_module,
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


def _published_optimizer(
    model_name: str, model: torch.nn.Module
) -> torch.optim.Optimizer:
    config = MODEL_CONFIGS[model_name]
    if model_name == "TCGN":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )


def _published_egn_v1_scheduler(
    optimizer: torch.optim.Optimizer, *, steps_per_epoch: int
):
    try:
        from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
    except Exception as error:
        raise UpstreamContractError(
            f"EGNV1_PUBLISHED_SCHEDULER_UNAVAILABLE:{error}"
        ) from error
    first_cycle_steps = steps_per_epoch // 2
    if first_cycle_steps < 2:
        raise UpstreamContractError(
            f"EGNV1_SCHEDULER_CYCLE_TOO_SHORT:{steps_per_epoch}"
        )
    return CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=first_cycle_steps,
        cycle_mult=1.0,
        max_lr=MODEL_CONFIGS["EGNv1"].learning_rate,
        min_lr=MODEL_CONFIGS["EGNv1"].learning_rate / 5,
        warmup_steps=min(100, first_cycle_steps - 1),
    )


def _finite_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    forward: Callable[[], torch.Tensor],
    loss_from_prediction: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = forward()
    loss = loss_from_prediction(prediction)
    if not torch.isfinite(loss):
        raise EngineeringGateError("INTERVENTION_LOSS_NONFINITE")
    loss.backward()
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise EngineeringGateError("INTERVENTION_GRADIENT_NONFINITE")
    optimizer.step()
    return prediction, loss


def _prepare_egn_features(
    model_name: str,
    sections,
    *,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.nn.Module]:
    import torchvision

    encoder = instantiate_published_encoder(model_name, torchvision)
    patches: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    for section in sections:
        raw = read_aligned_raw_patches(section)
        patches.append(raw)
        features.append(
            extract_published_features(encoder, raw, device=device, batch_size=64)
        )
    return patches, features, encoder


def _run_held_out_inference_gate(
    *,
    model_name: str,
    model: torch.nn.Module,
    section,
    raw_patches: torch.Tensor,
    device: torch.device,
    feature_encoder: Optional[torch.nn.Module] = None,
    bank=None,
) -> dict[str, object]:
    """Shape-only real outer-held-out inference; measured RNA is absent."""

    model.eval()
    with torch.no_grad():
        if model_name == "TCGN":
            count = min(MODEL_CONFIGS[model_name].batch_size, len(raw_patches))
            prediction = model(
                normalize_model_patches(
                    "TCGN", raw_patches[:count].to(device), training=False
                )
            )
            expected = (count, 50)
        else:
            if feature_encoder is None or bank is None:
                raise EngineeringGateError("HELD_OUT_EGN_CONTEXT_MISSING")
            query_features = extract_published_features(
                feature_encoder, raw_patches, device=device, batch_size=64
            )
            if model_name == "EGNv1":
                count = min(
                    MODEL_CONFIGS[model_name].batch_size, len(raw_patches)
                )
                rows = np.arange(count, dtype=np.int64)
                inputs = egn_v1_batch(
                    model_patches=normalize_model_patches(
                        "EGNv1", raw_patches[:count], training=False
                    ),
                    query_features=query_features[:count],
                    query_rows=rows,
                    query_specimen=section.specimen,
                    bank=bank,
                )
                prediction = model(
                    {key: value.to(device) for key, value in inputs.items()}
                )
                expected = (count, 50)
            else:
                import torch_geometric

                graph = build_egn_v2_graph(
                    query_features=query_features,
                    query_coords=section.spatial_coords,
                    query_specimen=section.specimen,
                    bank=bank,
                    torch_geometric_module=torch_geometric,
                ).to(device)
                prediction = model(graph.x_dict, graph.edge_index_dict)
                expected = (len(section.barcodes), 50)
    if tuple(prediction.shape) != expected or not torch.isfinite(prediction).all():
        raise EngineeringGateError(
            f"HELD_OUT_INFERENCE_CONTRACT:{tuple(prediction.shape)}:{expected}"
        )
    return {
        "specimen": section.specimen,
        "shape": list(prediction.shape),
        "prediction_finite": True,
        "measured_rna_loaded": False,
        "pcc_calculated": False,
        "staudit_endpoints_calculated": False,
    }


def _run_spot_model_gate(
    model_name: str,
    model: torch.nn.Module,
    section,
    raw_patches: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
    steps_per_epoch: int,
    query_features: Optional[torch.Tensor] = None,
    bank=None,
) -> dict[str, object]:
    config = MODEL_CONFIGS[model_name]
    supervision = build_supervision(
        coords=section.spatial_coords,
        validation_mask=section.validation_mask,
        specimen_id=section.specimen,
    )
    ordinary_order = (
        epoch_order(supervision.n_train, seed=seed, epoch=0)
        if config.ordinary_shuffle
        else np.arange(supervision.n_train, dtype=np.int64)
    )
    if len(ordinary_order) < config.batch_size:
        raise EngineeringGateError(
            f"PUBLISHED_BATCH_NOT_AVAILABLE:{supervision.n_train}:{config.batch_size}"
        )
    ordinary_rows = supervision.train_rows[ordinary_order[: config.batch_size]]
    left_rows, right_rows = edge_pair_batch(
        supervision,
        batch_size=config.batch_size,
        generator=edge_generator(seed=seed, epoch=0, step=0),
    )
    edge_rows = np.concatenate((left_rows, right_rows))
    target = torch.as_tensor(
        section.target[ordinary_rows], dtype=torch.float32, device=device
    )
    edge_target = torch.as_tensor(
        section.target[edge_rows], dtype=torch.float32, device=device
    )
    local_edges = torch.stack(
        (
            torch.arange(config.batch_size, dtype=torch.long, device=device),
            torch.arange(
                config.batch_size,
                2 * config.batch_size,
                dtype=torch.long,
                device=device,
            ),
        )
    )
    if model_name == "EGNv1":
        def make_inputs(rows):
            rows_tensor = torch.as_tensor(rows, dtype=torch.long)
            count = len(rows_tensor)
            values = egn_v1_batch(
                model_patches=normalize_model_patches(
                    model_name, raw_patches[rows_tensor], training=False
                ),
                query_features=query_features[rows_tensor],
                query_rows=np.arange(count, dtype=np.int64),
                query_specimen=section.specimen,
                bank=bank,
            )
            return {key: value.to(device) for key, value in values.items()}

        ordinary_inputs = make_inputs(ordinary_rows)
        left_inputs = make_inputs(left_rows)
        right_inputs = make_inputs(right_rows)

        def forward() -> torch.Tensor:
            return model(ordinary_inputs)

        def forward_edges() -> torch.Tensor:
            return torch.cat((model(left_inputs), model(right_inputs)), dim=0)

    else:
        ordinary_raw = raw_patches[torch.as_tensor(ordinary_rows)].to(device)
        left_raw = raw_patches[torch.as_tensor(left_rows)].to(device)
        right_raw = raw_patches[torch.as_tensor(right_rows)].to(device)

        def forward() -> torch.Tensor:
            # Upstream applies augmentation after its basic transform.  The
            # helper performs the same stochastic operations from restored RNG.
            return model(normalize_model_patches("TCGN", ordinary_raw, training=True))

        def forward_edges() -> torch.Tensor:
            return torch.cat(
                (
                    model(normalize_model_patches("TCGN", left_raw, training=True)),
                    model(normalize_model_patches("TCGN", right_raw, training=True)),
                ),
                dim=0,
            )

    published = lambda pred: published_minibatch_loss(
        pred, target, correlation_weight=config.correlation_auxiliary_weight
    )
    rho_zero = lambda pred: minibatch_loss(
        pred,
        target,
        rho=0.0,
        correlation_weight=config.correlation_auxiliary_weight,
    )
    noop = run_rho_zero_noop_gate(
        model, forward=forward, published_loss=published, rho_zero_loss=rho_zero
    )
    optimizer = _published_optimizer(model_name, model)
    scheduler = (
        _published_egn_v1_scheduler(
            optimizer, steps_per_epoch=steps_per_epoch
        )
        if model_name == "EGNv1"
        else None
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    intervention_prediction = forward()
    with preserve_model_buffers(model):
        edge_prediction = forward_edges()
        intervention = minibatch_loss(
            intervention_prediction,
            target,
            rho=0.5,
            correlation_weight=config.correlation_auxiliary_weight,
            edge_prediction=edge_prediction,
            edge_target=edge_target,
            edge_index=local_edges,
        )
        if not torch.isfinite(intervention):
            raise EngineeringGateError("INTERVENTION_LOSS_NONFINITE")
        intervention.backward()
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise EngineeringGateError("INTERVENTION_GRADIENT_NONFINITE")
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    model.eval()
    with torch.no_grad():
        if model_name == "TCGN":
            inference = model(
                normalize_model_patches("TCGN", ordinary_raw, training=False)
            )
        else:
            inference = model(ordinary_inputs)
    if inference.shape != target.shape or not torch.isfinite(inference).all():
        raise EngineeringGateError(
            f"INFERENCE_CONTRACT:{tuple(inference.shape)}:{tuple(target.shape)}"
        )
    return {
        "batch_size": config.batch_size,
        "ordinary_spot_stream_unchanged_by_edge_sampling": True,
        "edge_stream_separate_and_explicit": True,
        "edge_pairs": config.batch_size,
        "edge_endpoint_forwards": "two_published_size_batches_left_then_right",
        "published_auxiliary_evaluated_on_ordinary_batch_only": True,
        "published_optimizer_step": "PASS",
        "published_scheduler_step": "PASS" if scheduler is not None else "NOT_APPLICABLE",
        "noop": noop,
        "intervention_loss_finite": bool(torch.isfinite(intervention).item()),
        "inference_shape": list(inference.shape),
        "supervision": supervision.to_metadata(),
    }


def _run_egn_v2_gate(
    model: torch.nn.Module,
    section,
    query_features: torch.Tensor,
    *,
    bank,
    device: torch.device,
) -> dict[str, object]:
    import torch_geometric

    config = MODEL_CONFIGS["EGNv2"]
    supervision = build_supervision(
        coords=section.spatial_coords,
        validation_mask=section.validation_mask,
        specimen_id=section.specimen,
    )
    graph = build_egn_v2_graph(
        query_features=query_features,
        query_coords=section.spatial_coords,
        query_specimen=section.specimen,
        bank=bank,
        torch_geometric_module=torch_geometric,
    ).to(device)
    target = torch.as_tensor(section.target, dtype=torch.float32, device=device)

    def forward() -> torch.Tensor:
        return model(graph.x_dict, graph.edge_index_dict)

    published = lambda pred: published_full_section_loss(
        pred,
        target,
        supervision,
        correlation_weight=config.correlation_auxiliary_weight,
    )
    rho_zero = lambda pred: full_section_loss(
        pred,
        target,
        supervision,
        rho=0.0,
        correlation_weight=config.correlation_auxiliary_weight,
    )
    noop = run_rho_zero_noop_gate(
        model, forward=forward, published_loss=published, rho_zero_loss=rho_zero
    )
    optimizer = _published_optimizer("EGNv2", model)
    _, intervention = _finite_step(
        model,
        optimizer,
        forward,
        lambda pred: full_section_loss(
            pred,
            target,
            supervision,
            rho=0.5,
            correlation_weight=config.correlation_auxiliary_weight,
        ),
    )
    model.eval()
    with torch.no_grad():
        inference = forward()
    expected = (len(section.barcodes), 50)
    if tuple(inference.shape) != expected or not torch.isfinite(inference).all():
        raise EngineeringGateError(
            f"INFERENCE_CONTRACT:{tuple(inference.shape)}:{expected}"
        )
    return {
        "full_section": True,
        "tiling_or_node_sampling": False,
        "window_nodes": len(section.barcodes),
        "example_nodes": int(graph.x_dict["example"].shape[0]),
        "all_example_rna_outer_train_supervised_only": True,
        "noop": noop,
        "intervention_loss_finite": bool(torch.isfinite(intervention).item()),
        "inference_shape": list(inference.shape),
        "supervision": supervision.to_metadata(),
    }


def run_gate(args: argparse.Namespace) -> dict[str, object]:
    model_name = canonical_model_name(args.model)
    upstream_source = verify_upstream_source(model_name, args.upstream_root)
    packages = assert_pinned_environment(model_name)
    if args.preflight_only:
        return {
            "status": "PREFLIGHT_PASS_REAL_MINIBATCH_NOT_RUN",
            "eligible": False,
            "model": model_name,
            "upstream": upstream_source.to_dict(),
            "packages": packages,
            "reason": "preflight_only_is_not_the_engineering_gate",
            "held_out_rna_loaded": False,
            "held_out_pcc_calculated": False,
            "staudit_endpoints_calculated": False,
        }

    module, upstream = load_upstream_model_module(
        model_name, args.upstream_root, require_pinned_environment=True
    )
    fold_data = load_fold_data(
        model=model_name,
        dataset_root=args.dataset_root,
        gene_manifest=args.gene_manifest,
        embedding_manifest=args.embedding_manifest,
        patch_manifest=args.patch_manifest,
        task=args.task,
        fold=args.fold,
        include_test=False,
        g0054_module=args.g0054_module,
    )
    if fold_data.held_out is not None:
        raise EngineeringGateError("HELD_OUT_LOADED_DURING_GATE")
    if args.specimen is None:
        section_index = 0
    else:
        matches = [
            index
            for index, section in enumerate(fold_data.training)
            if section.specimen == args.specimen
        ]
        if len(matches) != 1:
            raise EngineeringGateError(
                f"GATE_SPECIMEN_NOT_IN_TRAINING_FOLD:{args.specimen}"
            )
        section_index = matches[0]
    section = fold_data.training[section_index]
    total_supervised_spots = sum(
        int((~candidate.validation_mask).sum())
        for candidate in fold_data.training
    )
    steps_per_epoch = total_supervised_spots // MODEL_CONFIGS[model_name].batch_size
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EngineeringGateError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    if model_name.startswith("EGN"):
        torch.backends.cudnn.benchmark = True
    model = instantiate_upstream_model(model_name, module, n_genes=50).to(device)
    feature_encoder = None
    bank = None
    pretraining_receipt = None
    if model_name == "TCGN":
        pretraining = args.tcgn_pretrained or (
            Path(upstream.root) / "pretrained" / "cmt_tiny.pth"
        )
        pretraining_receipt = load_tcgn_pretraining(
            model, path=pretraining, map_location=device
        )
        pretraining_receipt["path"] = str(pretraining.resolve())
        raw = read_aligned_raw_patches(section)
        details = _run_spot_model_gate(
            model_name,
            model,
            section,
            raw,
            device=device,
            seed=args.seed,
            steps_per_epoch=steps_per_epoch,
        )
    else:
        raw_blocks, feature_blocks, feature_encoder = _prepare_egn_features(
            model_name, fold_data.training, device=device
        )
        bank = build_training_bank(model_name, fold_data.training, feature_blocks)
        if model_name == "EGNv1":
            details = _run_spot_model_gate(
                model_name,
                model,
                section,
                raw_blocks[section_index],
                device=device,
                seed=args.seed,
                steps_per_epoch=steps_per_epoch,
                query_features=feature_blocks[section_index],
                bank=bank,
            )
        else:
            details = _run_egn_v2_gate(
                model,
                section,
                feature_blocks[section_index],
                bank=bank,
                device=device,
            )
    # Only after the training/no-op/intervention checks finish, open outer-test
    # image support.  include_test_rna=False keeps its expression matrix closed.
    inference_fold = load_fold_data(
        model=model_name,
        dataset_root=args.dataset_root,
        gene_manifest=args.gene_manifest,
        embedding_manifest=args.embedding_manifest,
        patch_manifest=args.patch_manifest,
        task=args.task,
        fold=args.fold,
        include_test=True,
        include_test_rna=False,
        g0054_module=args.g0054_module,
    )
    held_out = inference_fold.held_out
    if held_out is None or held_out.raw_counts is not None or held_out.target is not None:
        raise EngineeringGateError("HELD_OUT_RNA_WAS_LOADED_DURING_INFERENCE_GATE")
    held_out_inference = _run_held_out_inference_gate(
        model_name=model_name,
        model=model,
        section=held_out,
        raw_patches=read_aligned_raw_patches(held_out),
        device=device,
        feature_encoder=feature_encoder,
        bank=bank,
    )
    return {
        "status": "PASS_REAL_PUBLISHED_MINIBATCH",
        "eligible": True,
        "model": model_name,
        "task": args.task,
        "fold": args.fold,
        "specimen": section.specimen,
        "genes": 50,
        "rho_zero_and_intervention_checked": True,
        "published_architecture_substituted": False,
        "upstream": upstream.to_dict(),
        "packages": packages,
        "training_contract": MODEL_CONFIGS[model_name].to_dict(),
        "published_pretraining": pretraining_receipt,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "details": details,
        "real_held_out_inference": held_out_inference,
        "gpu": _gpu_receipt(device),
        "held_out_rna_loaded": False,
        "held_out_pcc_calculated": False,
        "staudit_endpoints_calculated": False,
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
    parser.add_argument("--model", required=True, choices=("EGNv1", "EGNv2", "TCGN"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--gene-manifest", type=Path)
    parser.add_argument("--embedding-manifest", type=Path)
    parser.add_argument("--patch-manifest", type=Path)
    parser.add_argument("--g0054-module", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--tcgn-pretrained", type=Path)
    parser.add_argument("--task", choices=("IDC", "PAAD", "LUNG"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--specimen")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _require_real_args(args)
        result = run_gate(args)
        exit_code = 0
    except BaseException as error:
        result = {
            "status": classify_failure(error),
            "eligible": False,
            "model": args.model,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "clean_pinned_attempt_path_preserved": True,
            "attempt_stage": "source_and_environment_preflight_or_real_gate",
            "no_architecture_substitution": True,
            "no_tiling_or_node_sampling": True,
            "held_out_rna_loaded": False,
            "held_out_pcc_calculated": False,
            "staudit_endpoints_calculated": False,
        }
        exit_code = 2
    _atomic_json(args.receipt, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "eligible": result["eligible"],
                "model": result["model"],
                "held_out_pcc_calculated": False,
                "staudit_endpoints_calculated": False,
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
