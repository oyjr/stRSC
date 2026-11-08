#!/usr/bin/env python3
"""Train one paired EGN/TCGN HEST fold and emit aligned predictions only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


from strsc.training.adapters.hest_egn_tcgn.contracts import (  # noqa: E402
    MODEL_CONFIGS,
    canonical_model_name,
)
from strsc.training.adapters.hest_egn_tcgn.exemplars import (  # noqa: E402
    extract_published_features,
    instantiate_published_encoder,
)
from strsc.training.adapters.hest_egn_tcgn.hest_data import (  # noqa: E402
    load_fold_data,
    read_aligned_raw_patches,
)
from strsc.training.adapters.hest_egn_tcgn.model_inputs import build_egn_v2_graph  # noqa: E402
from strsc.training.adapters.hest_egn_tcgn.outputs import (  # noqa: E402
    write_prediction_h5,
    write_prediction_manifest,
)
from strsc.training.adapters.hest_egn_tcgn.pairing import seed_everything  # noqa: E402
from strsc.training.adapters.hest_egn_tcgn.training import (  # noqa: E402
    build_materialized_training,
    predict_graph_section,
    predict_spot_section,
    train_graph_arm,
    train_spot_arm,
)
from strsc.training.adapters.hest_egn_tcgn.upstream import (  # noqa: E402
    instantiate_upstream_model,
    load_tcgn_pretraining,
    load_upstream_model_module,
)


FORMAL_SEEDS = (17, 23, 41)
RHO_ARMS = (0.0, 0.5)


class PairRunnerError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def require_engineering_receipt(
    path: Path, *, model: str, task: str, fold: int
) -> dict[str, object]:
    if not path.is_file():
        raise PairRunnerError(f"ENGINEERING_RECEIPT_MISSING:{path}")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairRunnerError(f"ENGINEERING_RECEIPT_INVALID:{path}:{error}") from error
    upstream = receipt.get("upstream", {})
    held_out_inference = receipt.get("real_held_out_inference", {})
    legacy_pass = (
        receipt.get("status") == "PASS_REAL_PUBLISHED_MINIBATCH"
        and receipt.get("eligible") is True
        and receipt.get("model") == model
        and receipt.get("task") == task
        and int(receipt.get("fold", -1)) == fold
        and isinstance(upstream, dict)
        and upstream.get("revision") == MODEL_CONFIGS[model].revision
        and isinstance(held_out_inference, dict)
        and held_out_inference.get("prediction_finite") is True
        and isinstance(held_out_inference.get("shape"), list)
        and held_out_inference.get("measured_rna_loaded") is False
        and held_out_inference.get("pcc_calculated") is False
        and held_out_inference.get("staudit_endpoints_calculated") is False
        and receipt.get("held_out_rna_loaded") is False
        and receipt.get("held_out_pcc_calculated") is False
        and receipt.get("staudit_endpoints_calculated") is False
    )
    proportional_pass = (
        receipt.get("status") == "PASS"
        and receipt.get("eligible") is True
        and receipt.get("model") == model
        and isinstance(receipt.get("minimal_gate_policy"), str)
        and isinstance(upstream, dict)
        and upstream.get("revision") == MODEL_CONFIGS[model].revision
        and receipt.get("scientific_results_opened") is False
    )
    if not (legacy_pass or proportional_pass):
        raise PairRunnerError(
            f"ENGINEERING_RECEIPT_NOT_PASSING:{path}:{receipt.get('status')}"
        )
    return receipt


def _extract_training_inputs(model_name: str, sections, *, device: torch.device):
    raw = tuple(read_aligned_raw_patches(section) for section in sections)
    if model_name == "TCGN":
        return raw, None, None
    import torchvision

    encoder = instantiate_published_encoder(model_name, torchvision).to(device)
    features = tuple(
        extract_published_features(encoder, patches, device=device, batch_size=64)
        for patches in raw
    )
    return raw, features, encoder


def _state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _save_checkpoint(path: Path, checkpoint, *, model: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(
        {
            "model": model,
            "seed": int(seed),
            "rho": float(checkpoint.rho),
            "best_epoch": int(checkpoint.best_epoch),
            "best_validation_mse": float(checkpoint.best_validation_mse),
            "termination": checkpoint.termination,
            "termination_epoch": checkpoint.termination_epoch,
            "termination_step": checkpoint.termination_step,
            "epochs_completed": checkpoint.epochs_completed,
            "state_dict": checkpoint.state_dict,
            "contains_endpoint_scores": False,
        },
        temporary,
    )
    os.replace(temporary, path)


def run_pair(args: argparse.Namespace) -> dict[str, object]:
    model_name = canonical_model_name(args.model)
    if args.seed not in FORMAL_SEEDS:
        raise PairRunnerError(f"UNFROZEN_SEED:{args.seed}:{FORMAL_SEEDS}")
    engineering_receipt = require_engineering_receipt(
        args.engineering_receipt,
        model=model_name,
        task=args.task,
        fold=args.fold,
    )
    proportional_environment = engineering_receipt.get("status") == "PASS"
    module, upstream = load_upstream_model_module(
        model_name,
        args.upstream_root,
        require_pinned_environment=not proportional_environment,
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
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PairRunnerError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    if model_name.startswith("EGN"):
        torch.backends.cudnn.benchmark = True
    raw, features, feature_encoder = _extract_training_inputs(
        model_name, fold_data.training, device=device
    )
    torch_geometric_module = None
    if model_name == "EGNv2":
        import torch_geometric

        torch_geometric_module = torch_geometric
    materialized = build_materialized_training(
        fold_data,
        raw_patches=raw,
        features=features,
        torch_geometric_module=torch_geometric_module,
    )
    if model_name == "EGNv2":
        # Graphs/features now own every training input needed by EGNv2.
        del raw
    tcgn_pretrained = None
    if model_name == "TCGN":
        tcgn_pretrained = args.tcgn_pretrained or (
            Path(upstream.root) / "pretrained" / "cmt_tiny.pth"
        )

    def factory() -> torch.nn.Module:
        model = instantiate_upstream_model(model_name, module, n_genes=50)
        if model_name == "TCGN":
            load_tcgn_pretraining(
                model, path=tcgn_pretrained, map_location=torch.device("cpu")
            )
        return model

    seed_everything(args.seed)
    initial_model = factory()
    initial_state = _state_dict_cpu(initial_model)
    del initial_model
    checkpoints = []
    for rho in RHO_ARMS:
        if model_name == "EGNv2":
            checkpoint = train_graph_arm(
                model_factory=factory,
                initial_state=initial_state,
                materialized=materialized,
                rho=rho,
                seed=args.seed,
                device=device,
            )
        else:
            checkpoint = train_spot_arm(
                model_factory=factory,
                initial_state=initial_state,
                materialized=materialized,
                rho=rho,
                seed=args.seed,
                device=device,
            )
        checkpoints.append(checkpoint)

    # Outer-test support is opened only after both training/checkpoint paths are done.
    with_test = load_fold_data(
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
    held_out = with_test.held_out
    if held_out is None or held_out.raw_counts is not None or held_out.target is not None:
        raise PairRunnerError("HELD_OUT_RNA_WAS_LOADED")
    held_out_raw = read_aligned_raw_patches(held_out)
    held_out_features = None
    held_out_graph = None
    if model_name.startswith("EGN"):
        if feature_encoder is None:
            raise PairRunnerError("EGN_FEATURE_ENCODER_MISSING")
        held_out_features = extract_published_features(
            feature_encoder, held_out_raw, device=device, batch_size=64
        )
        if model_name == "EGNv2":
            held_out_graph = build_egn_v2_graph(
                query_features=held_out_features,
                query_coords=held_out.spatial_coords,
                query_specimen=held_out.specimen,
                bank=materialized.exemplar_bank,
                torch_geometric_module=torch_geometric_module,
            )

    records = []
    checkpoint_paths = []
    for checkpoint in checkpoints:
        checkpoint_path = (
            args.output_root
            / "checkpoints"
            / f"{model_name}.{args.task}.fold{args.fold}.seed{args.seed}."
            f"rho{checkpoint.rho:.1f}.pt"
        )
        _save_checkpoint(
            checkpoint_path, checkpoint, model=model_name, seed=args.seed
        )
        checkpoint_paths.append(str(checkpoint_path.resolve()))
        model = factory().to(device)
        model.load_state_dict(checkpoint.state_dict, strict=True)
        if model_name == "EGNv2":
            prediction = predict_graph_section(
                model,
                graph=held_out_graph,
                n_spots=len(held_out.barcodes),
                device=device,
            )
        else:
            prediction = predict_spot_section(
                model,
                model_name=model_name,
                section=held_out,
                raw_patches=held_out_raw,
                device=device,
                query_features=held_out_features,
                exemplar_bank=materialized.exemplar_bank,
            )
        prediction_path = (
            args.output_root
            / "predictions"
            / f"{model_name}.{args.task}.{held_out.specimen}.fold{args.fold}."
            f"seed{args.seed}.rho{checkpoint.rho:.1f}.h5"
        )
        records.append(
            write_prediction_h5(
                prediction_path,
                model=model_name,
                task=args.task,
                fold=args.fold,
                rho=checkpoint.rho,
                seed=args.seed,
                specimen=held_out.specimen,
                predictions=prediction,
                barcodes=held_out.barcodes,
                genes=held_out.genes,
                expression_scale=MODEL_CONFIGS[model_name].expression_transform,
            )
        )
        del model
    manifest_stem = (
        args.output_root
        / "manifests"
        / f"{model_name}.{args.task}.fold{args.fold}.seed{args.seed}.tsv"
    )
    manifest_tsv, manifest_json = write_prediction_manifest(
        manifest_stem,
        records,
        provenance={
            "upstream": upstream.to_dict(),
            "compatibility_environment": engineering_receipt.get(
                "compatibility_environment"
            ),
            "training_contract": MODEL_CONFIGS[model_name].to_dict(),
            "tcgn_pretraining_path": (
                str(tcgn_pretrained.resolve())
                if tcgn_pretrained is not None
                else None
            ),
            "paired_rho": list(RHO_ARMS),
            "same_initial_state": True,
            "same_ordinary_spot_order": True,
            "separate_fixed_edge_stream": model_name != "EGNv2",
            "train_only_exemplar_bank": model_name.startswith("EGN"),
            "single_reference_fold_exemplar_adaptation": (
                model_name.startswith("EGN") and len(fold_data.training) == 1
            ),
            "single_reference_exclusion": (
                "same_barcode_and_1.5x_median_nearest_neighbor_spacing"
                if model_name.startswith("EGN") and len(fold_data.training) == 1
                else None
            ),
            "checkpoint_selection": "ordinary_inner_validation_mse",
            "nonfinite_training_rule": (
                "TCGN_only_use_previous_finite_inner_validation_best_checkpoint"
                "_if_late_training_loss_becomes_nonfinite"
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "checkpoint_paths": checkpoint_paths,
        },
    )
    return {
        "status": "PASS_PAIRED_PREDICTIONS_WRITTEN",
        "model": model_name,
        "task": args.task,
        "fold": args.fold,
        "seed": args.seed,
        "held_out_specimen": held_out.specimen,
        "arms": [
            {
                "rho": checkpoint.rho,
                "best_epoch": checkpoint.best_epoch,
                "best_validation_mse": checkpoint.best_validation_mse,
                "final_training_loss": checkpoint.final_training_loss,
                "termination": checkpoint.termination,
                "termination_epoch": checkpoint.termination_epoch,
                "termination_step": checkpoint.termination_step,
                "epochs_completed": checkpoint.epochs_completed,
            }
            for checkpoint in checkpoints
        ],
        "prediction_manifest_tsv": str(manifest_tsv),
        "prediction_manifest_json": str(manifest_json),
        "prediction_files": len(records),
        "checkpoint_files": checkpoint_paths,
        "held_out_rna_loaded": False,
        "held_out_pcc_calculated": False,
        "staudit_endpoints_calculated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("EGNv1", "EGNv2", "TCGN"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gene-manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--engineering-receipt", type=Path, required=True)
    parser.add_argument("--g0054-module", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--tcgn-pretrained", type=Path)
    parser.add_argument("--task", choices=("IDC", "PAAD", "LUNG"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_pair(args)
        exit_code = 0
    except BaseException as error:
        result = {
            "status": "PAIR_FAILED_BEFORE_ENDPOINT_SCORING",
            "model": args.model,
            "task": args.task,
            "fold": args.fold,
            "seed": args.seed,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "held_out_pcc_calculated": False,
            "staudit_endpoints_calculated": False,
        }
        exit_code = 2
    _atomic_json(args.receipt, result)
    print(
        json.dumps(
            {
                "status": result["status"],
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
