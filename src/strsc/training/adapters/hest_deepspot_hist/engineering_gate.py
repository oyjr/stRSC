#!/usr/bin/env python3
"""Real outer-training single-batch gates; no held-out RNA or scorer is opened."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


if __package__ in {None, ""}:
    from strsc.training.adapters.hest_deepspot_hist.contract import (  # noqa: E402
        SOURCE_REVISIONS,
        build_fold_inner_splits,
        canonical_model_name,
    )
    from strsc.training.adapters.hest_deepspot_hist.data import (  # noqa: E402
        DeepSpotEdgePairDataset,
        DeepSpotFeatureStore,
        DeepSpotSpecimenDataset,
        HiSTFeatureStore,
        HiSTPartitionDataset,
        build_deepspot_train_dataset,
        collate_deepspot_edge_pairs,
        collate_deepspot_spots,
        fit_deepspot_targets,
        load_aligned_hest_fold,
    )
    from strsc.training.adapters.hest_deepspot_hist.deepspot import (  # noqa: E402
        build_upstream_deepspot,
        install_deepspot_overlay,
        remove_deepspot_overlay,
    )
    from strsc.training.adapters.hest_deepspot_hist.hist import (  # noqa: E402
        build_upstream_hist,
        compose_hist_config,
        install_hist_overlay,
        remove_hist_overlay,
    )
    from strsc.training.adapters.hest_deepspot_hist.objective import (  # noqa: E402
        named_gradients,
        require_finite_backward,
    )
else:
    from .contract import SOURCE_REVISIONS, build_fold_inner_splits, canonical_model_name
    from .data import (
        DeepSpotEdgePairDataset,
        DeepSpotFeatureStore,
        DeepSpotSpecimenDataset,
        HiSTFeatureStore,
        HiSTPartitionDataset,
        build_deepspot_train_dataset,
        collate_deepspot_edge_pairs,
        collate_deepspot_spots,
        fit_deepspot_targets,
        load_aligned_hest_fold,
    )
    from .deepspot import (
        build_upstream_deepspot,
        install_deepspot_overlay,
        remove_deepspot_overlay,
    )
    from .hist import (
        build_upstream_hist,
        compose_hist_config,
        install_hist_overlay,
        remove_hist_overlay,
    )
    from .objective import named_gradients, require_finite_backward

from strsc.training.shared.noop import check_rho_zero_noop


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


def _frozen_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _verify_revision(path: Path, model: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip() if result.returncode == 0 else ""
    expected = SOURCE_REVISIONS[model]
    if revision != expected:
        raise RuntimeError(f"SOURCE_REVISION:{model}:{revision or 'UNKNOWN'}:{expected}")
    return revision


def _move_deepspot_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    ordinary_x, ordinary_target = batch["ordinary"]
    edge_x, edge_target = batch["edge"]["upstream_batch"]
    return {
        **batch,
        "ordinary": (
            [value.to(device) for value in ordinary_x],
            ordinary_target.to(device),
        ),
        "edge": {
            **batch["edge"],
            "upstream_batch": (
                [value.to(device) for value in edge_x],
                edge_target.to(device),
            ),
            "edge_index": batch["edge"]["edge_index"].to(device),
        },
    }


def _move_upstream_deepspot_batch(batch: Any, device: torch.device):
    x, target = batch
    return [value.to(device) for value in x], target.to(device)


def _move_namespace(batch: SimpleNamespace, device: torch.device) -> SimpleNamespace:
    result = SimpleNamespace()
    for key, value in vars(batch).items():
        setattr(result, key, value.to(device) if isinstance(value, torch.Tensor) else value)
    return result


def _limit_hist_batch(batch: SimpleNamespace, limit: int) -> SimpleNamespace:
    total = len(batch.mol_feats)
    if limit <= 0 or total <= limit:
        return batch
    edges = batch.g0055_edge_index.detach().cpu().numpy()
    selected: list[int] = []
    seen: set[int] = set()
    for left, right in edges.T:
        for value in (int(left), int(right)):
            if value not in seen and len(selected) < limit:
                selected.append(value)
                seen.add(value)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        points = batch.orig_cords.detach().cpu().numpy()
        center = np.median(points, axis=0)
        order = np.argsort(np.sum(np.square(points - center), axis=1), kind="stable")
        for value in order.tolist():
            if value not in seen:
                selected.append(int(value))
                seen.add(int(value))
                if len(selected) == limit:
                    break
    selected = sorted(selected)
    lookup = {old: new for new, old in enumerate(selected)}
    retained = [
        (lookup[int(left)], lookup[int(right)])
        for left, right in edges.T
        if int(left) in lookup and int(right) in lookup
    ]
    if not retained:
        raise RuntimeError("HIST_GATE_GRAPH_EMPTY")
    index = torch.as_tensor(selected, dtype=torch.long)
    result = SimpleNamespace()
    for key, value in vars(batch).items():
        if key == "g0055_edge_index":
            setattr(result, key, torch.as_tensor(retained, dtype=torch.long).T.contiguous())
        elif isinstance(value, torch.Tensor) and value.ndim >= 1 and len(value) == total:
            setattr(result, key, value[index])
        elif isinstance(value, list) and len(value) == total:
            setattr(result, key, [value[position] for position in selected])
        else:
            setattr(result, key, value)
    return result


def _capture_forward(storage: list[torch.Tensor]):
    def hook(module, inputs, output):
        del module, inputs
        value = output[0] if isinstance(output, tuple) else output
        storage.append(value.detach().cpu().clone())

    return hook


def _run_deepspot_gate(args, fold, splits, source_revision: str) -> dict[str, Any]:
    if args.feature_root is None:
        raise RuntimeError("DEEPSPOT_FEATURE_ROOT_REQUIRED")
    targets = fit_deepspot_targets(fold, splits, dataset_root=args.dataset_root)
    specimen = fold.train_specimens[0]
    split = splits[0]
    feature_store = DeepSpotFeatureStore(args.feature_root)
    train_spots, train_edges = build_deepspot_train_dataset(
        specimen,
        split,
        scaled_targets=targets.scaled_by_specimen[specimen.specimen],
        feature_store=feature_store,
    )
    pair_dataset = DeepSpotEdgePairDataset(train_spots, train_edges)
    count = min(int(args.edge_pairs), len(pair_dataset))
    edge_batch = collate_deepspot_edge_pairs([pair_dataset[index] for index in range(count)])
    ordinary_count = min(2 * count, len(train_spots))
    ordinary_batch = collate_deepspot_spots(
        [train_spots[index] for index in range(ordinary_count)]
    )
    pair_batch = {"ordinary": ordinary_batch, "edge": edge_batch}
    validation_rows = np.flatnonzero(split.validation_mask)
    validation_spots = DeepSpotSpecimenDataset(
        specimen,
        rows=validation_rows,
        scaled_targets=targets.scaled_by_specimen[specimen.specimen],
        feature_store=feature_store,
    )
    val_count = min(2 * count, len(validation_spots))
    validation_batch = collate_deepspot_spots(
        [validation_spots[index] for index in range(val_count)]
    )

    device = torch.device(args.device)
    pair_batch = _move_deepspot_batch(pair_batch, device)
    validation_batch = _move_upstream_deepspot_batch(validation_batch, device)
    input_size = int(pair_batch["ordinary"][0][0].shape[-1])
    model = build_upstream_deepspot(
        deps_root=args.source_root,
        input_size=input_size,
        seed=args.seed,
        scaler=targets.standardizer.sklearn_scaler(),
    ).to(device)
    model.train()
    state = _frozen_state(model)
    rng = capture_rng()

    original_predictions: list[torch.Tensor] = []
    hook = model.register_forward_hook(_capture_forward(original_predictions))
    original_loss = model.training_step(pair_batch["ordinary"], 0)
    require_finite_backward(original_loss, model)
    hook.remove()
    original_gradients = named_gradients(model)

    model.load_state_dict(state, strict=True)
    model.zero_grad(set_to_none=True)
    install_deepspot_overlay(model, rho=0.0)
    restore_rng(rng)
    rho_zero_predictions: list[torch.Tensor] = []
    hook = model.register_forward_hook(_capture_forward(rho_zero_predictions))
    rho_zero_loss = model.training_step(pair_batch, 0)
    require_finite_backward(rho_zero_loss, model)
    hook.remove()
    noop = check_rho_zero_noop(
        original_loss=original_loss.detach().cpu(),
        rho_zero_loss=rho_zero_loss.detach().cpu(),
        original_predictions=original_predictions[-1],
        rho_zero_predictions=rho_zero_predictions[-1],
        original_gradients=original_gradients,
        rho_zero_gradients=named_gradients(model),
        atol=args.atol,
        rtol=args.rtol,
    )
    model.eval()
    validation_loss = model.validation_step(validation_batch, 0)
    if not torch.isfinite(validation_loss):
        raise RuntimeError("DEEPSPOT_VALIDATION_LOSS_NONFINITE")
    remove_deepspot_overlay(model)

    model.load_state_dict(state, strict=True)
    model.zero_grad(set_to_none=True)
    install_deepspot_overlay(model, rho=0.5)
    restore_rng(rng)
    model.train()
    intervention_loss = model.training_step(pair_batch, 0)
    require_finite_backward(intervention_loss, model)
    remove_deepspot_overlay(model)
    return {
        "status": "PASS",
        "model": "DeepSpot",
        "source_revision": source_revision,
        "task": fold.task,
        "fold": fold.fold,
        "gate_training_specimen": specimen.specimen,
        "gate_edge_pairs": count,
        "input_feature": "published_inception_main_plus_9_subspots_with_neighbor_context",
        "expression_transform": targets.transform_id,
        "inner_training_rows": int(split.train_mask.sum()),
        "inner_validation_rows": int(split.validation_mask.sum()),
        "ordinary_validation_loss_finite": True,
        "intervention_backward_finite": True,
        "noop": noop.to_dict(),
        "scientific_results_opened": False,
    }


def _run_hist_gate(args, fold, splits, source_revision: str) -> dict[str, Any]:
    if args.feature_root is None:
        raise RuntimeError("HIST_FEATURE_ROOT_REQUIRED")
    cfg = compose_hist_config(args.source_root)
    feature_store = HiSTFeatureStore(args.feature_root)
    train_dataset = HiSTPartitionDataset(
        (fold.train_specimens[0],),
        (splits[0],),
        validation=False,
        feature_store=feature_store,
        deps_root=args.source_root,
        coordinate_transform=None,
    )
    validation_dataset = HiSTPartitionDataset(
        (fold.train_specimens[0],),
        (splits[0],),
        validation=True,
        feature_store=feature_store,
        deps_root=args.source_root,
        coordinate_transform=None,
    )
    train_batch = _limit_hist_batch(train_dataset[0], int(args.hist_context_spots))
    validation_batch = validation_dataset[0]
    if args.hist_context_spots > 0 and len(validation_batch.mol_feats) > args.hist_context_spots:
        # Validation has no edge graph; deterministic leading sorted context is sufficient for the gate.
        limit = int(args.hist_context_spots)
        validation_total = len(validation_batch.mol_feats)
        for key, value in list(vars(validation_batch).items()):
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and len(value) == validation_total:
                setattr(validation_batch, key, value[:limit])
            elif isinstance(value, list) and len(value) == validation_total:
                setattr(validation_batch, key, value[:limit])

    device = torch.device(args.device)
    train_batch = _move_namespace(train_batch, device)
    validation_batch = _move_namespace(validation_batch, device)
    model = build_upstream_hist(deps_root=args.source_root, cfg=cfg, num_genes=50).to(device)
    model.train()
    state = _frozen_state(model)
    rng = capture_rng()

    original_prediction, original_loss = model(train_batch)
    require_finite_backward(original_loss, model)
    original_gradients = named_gradients(model)

    model.load_state_dict(state, strict=True)
    model.zero_grad(set_to_none=True)
    install_hist_overlay(model, rho=0.0)
    restore_rng(rng)
    model.train()
    rho_zero_prediction, rho_zero_loss = model(train_batch)
    require_finite_backward(rho_zero_loss, model)
    noop = check_rho_zero_noop(
        original_loss=original_loss.detach().cpu(),
        rho_zero_loss=rho_zero_loss.detach().cpu(),
        original_predictions=original_prediction.detach().cpu(),
        rho_zero_predictions=rho_zero_prediction.detach().cpu(),
        original_gradients=original_gradients,
        rho_zero_gradients=named_gradients(model),
        atol=args.atol,
        rtol=args.rtol,
    )
    model.eval()
    _, validation_loss = model(validation_batch)
    if not torch.isfinite(validation_loss):
        raise RuntimeError("HIST_VALIDATION_LOSS_NONFINITE")
    remove_hist_overlay(model)

    model.load_state_dict(state, strict=True)
    model.zero_grad(set_to_none=True)
    install_hist_overlay(model, rho=0.5)
    restore_rng(rng)
    model.train()
    _, intervention_loss = model(train_batch)
    require_finite_backward(intervention_loss, model)
    remove_hist_overlay(model)
    return {
        "status": "PASS",
        "model": "HiST",
        "source_revision": source_revision,
        "task": fold.task,
        "fold": fold.fold,
        "gate_training_specimen": fold.train_specimens[0].specimen,
        "gate_rows": len(train_batch.mol_feats),
        "input_feature": "published_precomputed_UNI_1024",
        "expression_transform": "upstream_log1p_raw_counts",
        "inner_training_rows": int(splits[0].train_mask.sum()),
        "inner_validation_rows": int(splits[0].validation_mask.sum()),
        "ordinary_validation_loss_finite": True,
        "intervention_backward_finite": True,
        "noop": noop.to_dict(),
        "scientific_results_opened": False,
    }


def classify_failure(model: str, exc: BaseException) -> str:
    text = f"{type(exc).__name__}:{exc}".lower()
    if "feature_missing" in text or "feature_root_required" in text or "no such file" in text:
        return f"ENGINEERING_NOT_ESTIMABLE_MISSING_{model.upper()}_FEATURES"
    if "out of memory" in text:
        return "ENGINEERING_NOT_ESTIMABLE_GPU_OOM"
    if model == "HiST" and any(
        token in text
        for token in ("dgl", "graphbolt", "undefined symbol", "pinned_env", "pinned_model_build")
    ):
        return "ENGINEERING_NOT_ESTIMABLE_PINNED_ENVIRONMENT"
    if "source_revision" in text:
        return "ENGINEERING_NOT_ESTIMABLE_SOURCE_REVISION"
    return "ENGINEERING_NOT_ESTIMABLE_RUNTIME"


def run_gate(args) -> dict[str, Any]:
    model = canonical_model_name(args.model)
    source_revision = _verify_revision(Path(args.source_root), model)
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
    splits = build_fold_inner_splits(fold, model=model)
    if model == "DeepSpot":
        return _run_deepspot_gate(args, fold, splits, source_revision)
    return _run_hist_gate(args, fold, splits, source_revision)


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("DeepSpot", "HiST"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gene-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--task", choices=("IDC", "PAAD", "LUNG"), default="IDC")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, choices=(17, 23, 41), default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--edge-pairs", type=int, default=4)
    parser.add_argument("--hist-context-spots", type=int, default=128)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--g0054-module", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = run_gate(args)
        exit_code = 0
    except Exception as exc:
        model = canonical_model_name(args.model)
        payload = {
            "status": classify_failure(model, exc),
            "eligible": False,
            "model": model,
            "error": f"{type(exc).__name__}:{exc}",
            "scientific_results_opened": False,
        }
        exit_code = 2
    _write_receipt(args.receipt, payload)
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
