#!/usr/bin/env python3
"""Prepare or train one DeepSpot2Cell arm with a shared initialization.

The runner uses the upstream dataset, spatial split, validation and trainer,
and selects either the native or neighbouring-difference expression objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from .graph import (
    build_frozen_train_adjacency,
    load_frozen_adjacency,
    write_frozen_adjacency,
)
from .initialization import capture_initial_state, restore_initial_state
from .overlay import install_local_difference_overlay
from .paired_data import make_matched_training_loaders


INIT_SCHEMA = "g0055.deepspot2cell.initial_state.v1"
FROZEN_SEEDS = (17, 23, 41)


def spatial_holdout_indices(
    patch_barcodes: Sequence[tuple[str, str]], *, val_fraction: float = 0.20
) -> tuple[list[int], list[int], float]:
    """Select the rightmost spatial column for validation."""

    fraction = float(val_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    coordinates = []
    for _, barcode in patch_barcodes:
        parts = str(barcode).split("_")
        if len(parts) < 3:
            raise ValueError(f"Unsupported G0-052 patch barcode: {barcode!r}")
        coordinates.append((int(parts[1]), int(parts[2])))
    coords = np.asarray(coordinates, dtype=np.int64)
    cutoff = float(np.quantile(coords[:, 1], 1.0 - fraction))
    validation = np.flatnonzero(coords[:, 1] >= cutoff).tolist()
    training = np.flatnonzero(coords[:, 1] < cutoff).tolist()
    if not training or not validation:
        raise RuntimeError("Spatial train/validation split produced an empty partition")
    return training, validation, cutoff


def _add_upstream(root: Path) -> Path:
    resolved = Path(root).resolve()
    if not (resolved / "deepspot2cell").is_dir() or not (
        resolved / "scripts" / "train.py"
    ).is_file():
        raise FileNotFoundError(f"Incomplete DeepSpot2Cell checkout: {resolved}")
    text = str(resolved)
    if text not in sys.path:
        sys.path.insert(0, text)
    return resolved


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DeepSpot2Cell config must be a mapping")
    return payload


def _build_g0052_dataset(cfg: dict[str, Any]):
    from deepspot2cell import DS2CDataset

    data_cfg = cfg["data"]
    norm_cfg = cfg["normalization"]
    model_cfg = cfg["model"]
    train_ids = data_cfg["split"]["train"]["ids"]
    if len(train_ids) != 1:
        raise ValueError("Frozen G0-052 spatial holdout requires exactly one training ID")
    return DS2CDataset(
        ids_list=train_ids,
        cell_gt_available=False,
        dataset_variant=data_cfg["dataset_variant"],
        data_path=data_cfg["data_folder"],
        model_name=data_cfg["model_name"],
        standard_scaling=norm_cfg["standard_scaling"],
        normalize=norm_cfg["normalize"],
        minmax=norm_cfg["minmax"],
        norm_counts=norm_cfg["norm_counts"],
        neighb_degree=model_cfg.get("neighb_degree", 0),
        scellst=data_cfg.get("scellst", False),
        load_cell_types=False,
        shuffle=False,
    )


def _load_initial_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != INIT_SCHEMA:
        raise ValueError("Unsupported or missing G0-055 initial-state schema")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Frozen G0-055 initial state is empty")
    return state


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze the graph and one shared initialization without training."""

    for output in (Path(args.graph), Path(args.initial_state)):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite frozen prepare artifact: {output}")
    _add_upstream(args.upstream_root)
    from deepspot2cell.utils.utils import fix_seed
    from scripts.train import build_model

    cfg = _load_config(args.config)
    seed = int(cfg["experiment"]["random_seed"])
    if seed not in FROZEN_SEEDS:
        raise ValueError(f"G0-055 formal seed must be one of {FROZEN_SEEDS}, found {seed}")
    fix_seed(seed)
    dataset = _build_g0052_dataset(cfg)
    train_idx, val_idx, cutoff = spatial_holdout_indices(
        dataset.patch_barcodes, val_fraction=args.val_fraction
    )
    graph = build_frozen_train_adjacency(dataset.patch_barcodes, train_idx)
    write_frozen_adjacency(graph, args.graph)

    model = build_model(dataset, cfg)
    initial_state = capture_initial_state(model)
    init_path = Path(args.initial_state)
    init_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": INIT_SCHEMA, "state_dict": initial_state}, init_path)
    receipt = {
        "status": "PREPARED",
        "seed": seed,
        "train_spots": len(train_idx),
        "validation_spots": len(val_idx),
        "column_cutoff": cutoff,
        "training_edges": len(graph.edges),
        "rho_arms": [0.0, 0.5],
        "scientific_results_opened": False,
    }
    _write_receipt(args.receipt, receipt)
    return receipt


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train exactly one arm from the shared prepared state."""

    _add_upstream(args.upstream_root)
    from deepspot2cell.utils.utils import fix_seed
    from scripts.train import build_model, build_trainer, save_dataset_statistics

    if args.rho is None or float(args.rho) not in (0.0, 0.5):
        raise ValueError("train mode requires --rho 0 or --rho 0.5")
    rho = float(args.rho)
    cfg = _load_config(args.config)
    seed = int(cfg["experiment"]["random_seed"])
    if seed not in FROZEN_SEEDS:
        raise ValueError(f"G0-055 formal seed must be one of {FROZEN_SEEDS}, found {seed}")
    fix_seed(seed)
    dataset = _build_g0052_dataset(cfg)
    train_idx, val_idx, cutoff = spatial_holdout_indices(
        dataset.patch_barcodes, val_fraction=args.val_fraction
    )
    graph = load_frozen_adjacency(args.graph, dataset.patch_barcodes)
    if graph.train_indices != tuple(sorted(train_idx)):
        raise RuntimeError("Frozen graph does not match the G0-052 training partition")

    dl_cfg = cfg.get("dataloader", {})
    batch_size = int(dl_cfg.get("batch_size", 64))
    num_workers = int(dl_cfg.get("num_workers", 4))
    loaders = make_matched_training_loaders(
        dataset,
        train_idx,
        graph,
        batch_size=batch_size,
        ordinary_seed=seed,
        edge_seed=seed + 100_003,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = loaders.baseline if rho == 0.0 else loaders.intervention
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2 if num_workers > 2 else 0),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(dataset, cfg)
    restore_initial_state(model, _load_initial_state(args.initial_state))
    install_local_difference_overlay(model, rho=rho)

    default_output = (
        Path(cfg["experiment"]["output_dir"]).resolve()
        / f"{cfg['experiment']['name']}_g0055_rho{rho:g}"
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_statistics(dataset, output_dir)
    trainer = build_trainer(cfg, output_dir)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    receipt = {
        "status": "TRAINED",
        "rho": rho,
        "seed": seed,
        "train_spots": len(train_idx),
        "validation_spots": len(val_idx),
        "column_cutoff": cutoff,
        "training_edges": len(graph.edges),
        "ordinary_stream_seed": seed,
        "edge_stream_seed": seed + 100_003,
        "validation_loss": "upstream_unchanged",
        "scientific_results_opened": False,
    }
    _write_receipt(args.receipt, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "train"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    result = prepare(args) if args.mode == "prepare" else train(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
