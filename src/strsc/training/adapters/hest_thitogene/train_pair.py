#!/usr/bin/env python3
"""Paired HEST LOSO training runner for unmodified upstream THItoGene.

This runner never tiles a section or replaces the dense full-section context.
If that published architecture cannot fit a real HEST section on an 80-GB
H100, the separate engineering gate must mark it not estimable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch


from strsc.training.adapters.hest_thitogene.hest_data import (  # noqa: E402
    MODEL_NAME,
    PATCH_PIXELS,
    PHYSICAL_SUPPORT_UM,
    UPSTREAM_EXPRESSION_TRANSFORM,
    VALIDATION_PREFIX,
    SectionBatch,
    load_fold_metadata,
    materialize_section,
)
from strsc.training.adapters.hest_thitogene.overlay import (  # noqa: E402
    THItoGeneSupervision,
    build_supervision,
    expression_loss,
    forward_section,
    published_validation_mse,
)
from strsc.training.adapters.hest_thitogene.upstream import (  # noqa: E402
    ATTENTION_BACKEND,
    EXPECTED_REVISION,
    instantiate_upstream_model,
    load_upstream_modules,
)


RHO_ARMS = (0.0, 0.5)
FORMAL_SEEDS = (17, 23, 41)
FORMAL_EPOCHS = 200
PUBLISHED_LEARNING_RATE = 1e-5


class THItoGeneTrainError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = FORMAL_EPOCHS
    learning_rate: float = PUBLISHED_LEARNING_RATE


@dataclass(frozen=True)
class ArmCheckpoint:
    rho: float
    best_epoch: int
    best_validation_mse: float
    final_training_loss: float
    state_dict: Mapping[str, torch.Tensor]


def require_engineering_receipt(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise THItoGeneTrainError(f"ENGINEERING_RECEIPT_MISSING:{path}")
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise THItoGeneTrainError(
            f"ENGINEERING_RECEIPT_INVALID:{path}:{error}"
        ) from error
    upstream = receipt.get("upstream", {})
    legacy_pass = (
        receipt.get("status") == "PASS_REAL_FULL_SECTION_MINIBATCH"
        and receipt.get("eligible") is True
        and receipt.get("model") == MODEL_NAME
        and isinstance(upstream, dict)
        and upstream.get("revision") == EXPECTED_REVISION
        and receipt.get("held_out_pcc_calculated") is False
        and receipt.get("staudit_bands_calculated") is False
    )
    proportional_pass = (
        receipt.get("status") == "PASS"
        and receipt.get("eligible") is True
        and receipt.get("model") == MODEL_NAME
        and isinstance(receipt.get("minimal_gate_policy"), str)
        and isinstance(upstream, dict)
        and upstream.get("revision") == EXPECTED_REVISION
        and receipt.get("scientific_results_opened") is False
    )
    if not (legacy_pass or proportional_pass):
        raise THItoGeneTrainError(
            f"ENGINEERING_RECEIPT_NOT_PASSING:{path}:{receipt.get('status')}"
        )
    return receipt


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _section_order(count: int, *, seed: int, epoch: int) -> np.ndarray:
    if count < 1:
        raise THItoGeneTrainError("NO_TRAINING_SECTIONS")
    sequence = np.random.SeedSequence([seed, epoch, 0x54484954])
    return np.random.default_rng(sequence).permutation(count)


def build_supervisions(
    batches: Sequence[SectionBatch],
) -> tuple[THItoGeneSupervision, ...]:
    rows = []
    for batch in batches:
        mask = batch.metadata.validation_mask
        if mask is None:
            raise THItoGeneTrainError(
                f"TRAINING_SECTION_WITHOUT_VALIDATION:{batch.metadata.specimen}"
            )
        rows.append(
            build_supervision(
                coords=batch.metadata.spatial_coords,
                validation_mask=mask,
                specimen_id=batch.metadata.specimen,
            )
        )
    return tuple(rows)


def _weighted_validation_mse(
    model: torch.nn.Module,
    batches: Sequence[SectionBatch],
    supervisions: Sequence[THItoGeneSupervision],
    *,
    device: torch.device,
) -> float:
    model.eval()
    squared_error = 0.0
    values = 0
    with torch.no_grad():
        for cpu_batch, supervision in zip(batches, supervisions):
            batch = cpu_batch.to(device)
            prediction = forward_section(
                model,
                patches=batch.patches,
                positions=batch.positions,
                model_adjacency=batch.model_adjacency,
            )
            loss = published_validation_mse(
                prediction, batch.target, supervision
            )
            count = supervision.n_validation * batch.target.shape[-1]
            squared_error += float(loss.detach().cpu()) * count
            values += count
            del batch, prediction, loss
    if values < 1:
        raise THItoGeneTrainError("NO_VALIDATION_VALUES")
    return squared_error / values


def train_arm(
    *,
    model_factory: Callable[[], torch.nn.Module],
    initial_state: Mapping[str, torch.Tensor],
    batches: Sequence[SectionBatch],
    supervisions: Sequence[THItoGeneSupervision],
    rho: float,
    seed: int,
    config: TrainConfig,
    device: torch.device,
) -> ArmCheckpoint:
    if rho not in RHO_ARMS:
        raise THItoGeneTrainError(f"UNFROZEN_RHO:{rho}")
    if len(batches) != len(supervisions):
        raise THItoGeneTrainError(
            f"BATCH_SUPERVISION_COUNT:{len(batches)}:{len(supervisions)}"
        )
    _seed_everything(seed)
    model = model_factory().to(device)
    model.load_state_dict(initial_state)
    # Exact optimizer and learning rate from the published HER2ST training path.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_validation = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    final_training_loss = math.nan

    for epoch in range(config.epochs):
        model.train()
        losses = []
        for index in _section_order(len(batches), seed=seed, epoch=epoch):
            batch = batches[int(index)].to(device)
            supervision = supervisions[int(index)]
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_section(
                model,
                patches=batch.patches,
                positions=batch.positions,
                model_adjacency=batch.model_adjacency,
            )
            loss = expression_loss(
                prediction, batch.target, supervision, rho=rho
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            del batch, prediction, loss
        final_training_loss = float(np.mean(losses))
        validation = _weighted_validation_mse(
            model, batches, supervisions, device=device
        )
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None or best_epoch < 0 or not math.isfinite(best_validation):
        raise THItoGeneTrainError("NO_VALID_CHECKPOINT")
    return ArmCheckpoint(
        rho=float(rho),
        best_epoch=best_epoch,
        best_validation_mse=float(best_validation),
        final_training_loss=final_training_loss,
        state_dict=best_state,
    )


def train_pair(
    *,
    model_factory: Callable[[], torch.nn.Module],
    batches: Sequence[SectionBatch],
    seed: int,
    config: TrainConfig,
    device: torch.device,
) -> tuple[ArmCheckpoint, ArmCheckpoint]:
    _seed_everything(seed)
    initial_model = model_factory()
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in initial_model.state_dict().items()
    }
    supervisions = build_supervisions(batches)
    checkpoints = tuple(
        train_arm(
            model_factory=model_factory,
            initial_state=initial_state,
            batches=batches,
            supervisions=supervisions,
            rho=rho,
            seed=seed,
            config=config,
            device=device,
        )
        for rho in RHO_ARMS
    )
    return checkpoints[0], checkpoints[1]


def predict_checkpoint(
    *,
    model_factory: Callable[[], torch.nn.Module],
    checkpoint: ArmCheckpoint,
    held_out: SectionBatch,
    device: torch.device,
) -> np.ndarray:
    model = model_factory().to(device)
    model.load_state_dict(checkpoint.state_dict)
    model.eval()
    with torch.no_grad():
        batch = held_out.to(device)
        prediction = forward_section(
            model,
            patches=batch.patches,
            positions=batch.positions,
            model_adjacency=batch.model_adjacency,
        )
    result = np.asarray(prediction.detach().cpu(), dtype=np.float32)
    expected = (len(held_out.metadata.barcodes), len(held_out.metadata.genes))
    if result.shape != expected or not np.isfinite(result).all():
        raise THItoGeneTrainError(f"HELD_OUT_PREDICTION:{result.shape}:{expected}")
    return result


def _write_prediction(
    *,
    held_out: SectionBatch,
    prediction: np.ndarray,
    checkpoint: ArmCheckpoint,
    seed: int,
    output_root: Path,
) -> Path:
    metadata = held_out.metadata
    if metadata.raw_counts is not None or metadata.target is not None:
        raise THItoGeneTrainError("HELD_OUT_RNA_WAS_LOADED")
    filename = (
        f"THItoGene.{metadata.task}.fold{metadata.fold}.{metadata.specimen}."
        f"seed{seed}.rho{checkpoint.rho:.1f}.predictions.h5"
    )
    output = output_root / "predictions" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    with h5py.File(temporary, "w") as handle:
        strings = h5py.string_dtype("utf-8")
        handle.create_dataset(
            "barcodes", data=metadata.barcodes.astype(object), dtype=strings
        )
        handle.create_dataset(
            "genes", data=metadata.genes.astype(object), dtype=strings
        )
        handle.create_dataset("spatial_coords", data=metadata.spatial_coords)
        if metadata.patch_coords is not None:
            handle.create_dataset("patch_coords", data=metadata.patch_coords)
        handle.create_dataset("prediction", data=prediction, compression="lzf")
        handle.attrs.update(
            {
                "status": "PREDICTION_COMPLETE_ENDPOINTS_UNOPENED",
                "experiment": "G0-055",
                "track": "HEST",
                "model": MODEL_NAME,
                "upstream_revision": EXPECTED_REVISION,
                "attention_backend": ATTENTION_BACKEND,
                "task": metadata.task,
                "fold": metadata.fold,
                "specimen": metadata.specimen,
                "seed": seed,
                "rho": checkpoint.rho,
                "genes": len(metadata.genes),
                "patch_pixels": PATCH_PIXELS,
                "physical_support_um": PHYSICAL_SUPPORT_UM,
                "expression_transform": UPSTREAM_EXPRESSION_TRANSFORM,
                "validation_prefix": VALIDATION_PREFIX,
                "contains_measured_held_out_rna": False,
                "contains_scientific_endpoint_scores": False,
            }
        )
        handle.flush()
    os.replace(temporary, output)
    return output


def _write_manifest(
    *,
    rows: list[dict[str, object]],
    output_root: Path,
    task: str,
    fold: int,
    seed: int,
) -> tuple[Path, Path]:
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    stem = f"THItoGene.{task}.fold{fold}.seed{seed}.prediction_manifest"
    tsv = manifest_root / f"{stem}.tsv"
    js = manifest_root / f"{stem}.json"
    frame = pd.DataFrame(rows)
    key = ["model", "task", "specimen", "fold", "seed", "rho"]
    if len(frame) != 2 or frame.duplicated(key).any():
        raise THItoGeneTrainError(f"PREDICTION_MANIFEST_KEY:{len(frame)}")
    frame.to_csv(tsv, sep="\t", index=False)
    js.write_text(
        json.dumps(
            {
                "status": "LOCKABLE_ENDPOINTS_UNOPENED",
                "rows": rows,
                "unique_key": key,
                "barcode_alignment": "explicit_exact_order",
                "gene_alignment": "official_task_50_exact_order",
                "held_out_pcc_calculated": False,
                "staudit_bands_calculated": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return tsv, js


def run_fold(args: argparse.Namespace) -> dict[str, object]:
    if args.seed not in FORMAL_SEEDS:
        raise THItoGeneTrainError(f"UNFROZEN_SEED:{args.seed}")
    engineering_receipt = require_engineering_receipt(args.engineering_receipt)
    proportional_environment = engineering_receipt.get("status") == "PASS"
    vis_model, graph_module, upstream = load_upstream_modules(
        args.upstream_root,
        require_pinned_environment=not proportional_environment,
    )
    training_metadata, unexpected_test = load_fold_metadata(
        dataset_root=args.dataset_root,
        gene_manifest=args.gene_manifest,
        embedding_manifest=args.embedding_manifest,
        patch_manifest=args.patch_manifest,
        task=args.task,
        fold=args.fold,
        include_test=False,
        g0054_module=args.g0054_module,
    )
    if unexpected_test is not None:
        raise THItoGeneTrainError("HELD_OUT_LOADED_BEFORE_TRAINING")
    training_batches = [
        materialize_section(row, upstream_graph_module=graph_module)
        for row in training_metadata
    ]
    factory = lambda: instantiate_upstream_model(vis_model, n_genes=50)
    config = TrainConfig()
    checkpoints = train_pair(
        model_factory=factory,
        batches=training_batches,
        seed=args.seed,
        config=config,
        device=torch.device(args.device),
    )

    # Only now may the held-out images/support be loaded for inference.
    _, held_out_metadata = load_fold_metadata(
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
    if held_out_metadata is None:
        raise THItoGeneTrainError("HELD_OUT_METADATA_MISSING")
    if held_out_metadata.raw_counts is not None or held_out_metadata.target is not None:
        raise THItoGeneTrainError("HELD_OUT_RNA_WAS_LOADED")
    held_out = materialize_section(
        held_out_metadata, upstream_graph_module=graph_module
    )
    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        prediction = predict_checkpoint(
            model_factory=factory,
            checkpoint=checkpoint,
            held_out=held_out,
            device=torch.device(args.device),
        )
        path = _write_prediction(
            held_out=held_out,
            prediction=prediction,
            checkpoint=checkpoint,
            seed=args.seed,
            output_root=args.output_root,
        )
        rows.append(
            {
                "experiment": "G0-055",
                "track": "HEST",
                "model": MODEL_NAME,
                "upstream_revision": upstream.revision,
                "task": args.task,
                "specimen": held_out.metadata.specimen,
                "fold": args.fold,
                "seed": args.seed,
                "rho": checkpoint.rho,
                "barcodes": len(held_out.metadata.barcodes),
                "genes": len(held_out.metadata.genes),
                "barcode_alignment": "PASS_EXPLICIT_EXACT_ORDER",
                "gene_alignment": "PASS_OFFICIAL_50_EXACT_ORDER",
                "prediction_path": str(path),
                "best_epoch": checkpoint.best_epoch,
                "best_validation_mse": checkpoint.best_validation_mse,
                "held_out_pcc_calculated": False,
                "staudit_bands_calculated": False,
            }
        )
    tsv, js = _write_manifest(
        rows=rows,
        output_root=args.output_root,
        task=args.task,
        fold=args.fold,
        seed=args.seed,
    )
    return {
        "status": "PASS_ENDPOINTS_UNOPENED",
        "model": MODEL_NAME,
        "task": args.task,
        "fold": args.fold,
        "seed": args.seed,
        "config": asdict(config),
        "attention_backend": ATTENTION_BACKEND,
        "compatibility_environment": engineering_receipt.get(
            "compatibility_environment"
        ),
        "prediction_manifest_tsv": str(tsv),
        "prediction_manifest_json": str(js),
        "prediction_files": 2,
        "held_out_pcc_calculated": False,
        "staudit_bands_calculated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gene-manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--engineering-receipt", type=Path, required=True)
    parser.add_argument("--g0054-module", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", choices=("IDC", "PAAD", "LUNG"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_fold(args)
        exit_code = 0
    except BaseException as error:
        result = {
            "status": "PAIR_FAILED_BEFORE_ENDPOINT_SCORING",
            "model": MODEL_NAME,
            "task": args.task,
            "fold": args.fold,
            "seed": args.seed,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "held_out_pcc_calculated": False,
            "staudit_bands_calculated": False,
        }
        exit_code = 2
    _atomic_json(args.receipt, result)
    # Do not print validation losses or any held-out scientific endpoint.
    print(
        json.dumps(
            {
                "status": result["status"],
                "model": result["model"],
                "task": result["task"],
                "fold": result["fold"],
                "seed": result["seed"],
                "prediction_files": result.get("prediction_files", 0),
                "held_out_pcc_calculated": False,
                "staudit_bands_calculated": False,
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
