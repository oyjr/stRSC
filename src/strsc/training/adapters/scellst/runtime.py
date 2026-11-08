"""Shared configuration, initialization and input validation for sCellST runners."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    EdgePairDataset,
    PairedSpotEdgeLoader,
    PreparedSpatialData,
    build_training_adjacency,
    freeze_scellst_tenx39_split,
    scellst_edge_pair_collate,
    scellst_spatial_collate,
)


SOURCE_REVISION = "503e37ac9cf994f0cb14e02a608c45a204ebdfd1"
INITIAL_STATE_SCHEMA = "g0055_scellst_paired_initial_state_v1"
SEEDS = (17, 23, 41)
RHOS = (0.0, 0.5)
SPLIT_SEED = 0
EDGE_SEED_OFFSET = 100_003
TEST_SPECIMENS = (
    "TENX98",
    "TENX99",
    "TENX95",
    "TENX97",
    "TENX94",
    "TENX96",
    "NCBI784",
    "NCBI785",
    "NCBI783",
)


class SCellSTRunnerError(RuntimeError):
    """A pinned source, data, split, training, or output contract failed."""


def validate_seed(seed: int) -> int:
    value = int(seed)
    if value not in SEEDS:
        raise SCellSTRunnerError(f"SEED_FROZEN:{value}:allowed={SEEDS}")
    return value


def validate_rho(rho: float) -> float:
    value = float(rho)
    if value not in RHOS:
        raise SCellSTRunnerError(f"RHO_FROZEN:{value}:allowed={RHOS}")
    return value


def bootstrap_upstream(upstream_root: str | Path) -> tuple[Path, str]:
    """Pin imports to the audited upstream checkout and verify its revision."""

    root = Path(upstream_root).resolve()
    required = root / "scellst" / "lightning_model" / "gene_lightning_model.py"
    if not required.is_file():
        raise SCellSTRunnerError(f"UPSTREAM_SOURCE_MISSING:{required}")
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip() if result.returncode == 0 else ""
    if revision != SOURCE_REVISION:
        raise SCellSTRunnerError(
            f"UPSTREAM_REVISION:{revision or 'UNKNOWN'}:{SOURCE_REVISION}"
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    loaded = sys.modules.get("scellst")
    if loaded is not None and getattr(loaded, "__file__", None):
        loaded_path = Path(loaded.__file__).resolve()
        if root not in loaded_path.parents:
            raise SCellSTRunnerError(f"UPSTREAM_IMPORT_COLLISION:{loaded_path}:{root}")
    return root, revision


def _config_value(config: Any, dotted: str) -> Any:
    value = config
    for component in dotted.split("."):
        value = getattr(value, component)
    return value


def validate_frozen_config(config: Any) -> None:
    """Validate the required training and data configuration."""

    exact = {
        "data.list_training_ids": ["TENX39"],
        "data.embedding_tag": "imagenet-rn50_train",
        "data.genes": "xenium_Breast",
        "data.dataset_handler": "mil",
        "data.shape_name": "cellvit",
        "data.frac_train": 0.8,
        "data.seed": SPLIT_SEED,
        "data.normalize": True,
        "data.log1p": True,
        "data.scale": "no_scaling",
        "predictor.hidden_dim": [256, 256, 256],
        "predictor.dropout_rate": 0.1,
        "predictor.final_activation": "softplus",
        "model.task_type": "regression",
        "model.criterion": "mse",
        "model.lr": 1e-4,
        "trainer.patience": 20,
        "trainer.max_epoch": 400,
    }
    for dotted, expected in exact.items():
        observed = _config_value(config, dotted)
        if isinstance(expected, list):
            observed = list(observed) if observed is not None else observed
        if observed != expected:
            raise SCellSTRunnerError(
                f"FROZEN_CONFIG:{dotted}:{observed!r}:{expected!r}"
            )
    data_dir = str(_config_value(config, "data.data_dir"))
    if not data_dir:
        raise SCellSTRunnerError("FROZEN_CONFIG:data.data_dir:EMPTY")


def load_frozen_config(config_path: str | Path) -> Any:
    from omegaconf import OmegaConf
    from scellst.utils.io_utils import load_config

    path = Path(config_path).resolve()
    if not path.is_file():
        raise SCellSTRunnerError(f"CONFIG_MISSING:{path}")
    config = load_config(path)
    try:
        OmegaConf.resolve(config)
    except Exception as exc:
        raise SCellSTRunnerError(f"CONFIG_UNRESOLVED:{type(exc).__name__}:{exc}") from exc
    validate_frozen_config(config)
    return config


def set_run_seed(seed: int) -> None:
    value = validate_seed(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.random.get_rng_state().clone(),
    }


def restore_rng_state(state: Mapping[str, Any], *, seed: int) -> None:
    random.setstate(copy.deepcopy(state["python"]))
    np.random.set_state(copy.deepcopy(state["numpy"]))
    torch.random.set_rng_state(state["torch"].clone())
    if torch.cuda.is_available():
        # Model construction is CPU-side; paired CUDA streams start from the
        # same frozen run seed in each separate arm process.
        torch.cuda.manual_seed_all(validate_seed(seed))


def prepare_training_data(
    config: Any,
    *,
    split_manifest: str | Path,
) -> tuple[Any, PreparedSpatialData]:
    """Load TENX39, replace upstream random_split, and freeze gene order."""

    from scellst.dataset.data_module import prepare_data_module

    data_module = prepare_data_module(
        config.data,
        stage="fit",
        task_type=config.model.task_type,
    )
    prepared = freeze_scellst_tenx39_split(
        data_module,
        manifest_path=split_manifest,
        split_seed=SPLIT_SEED,
        frac_train=0.8,
    )
    genes = [str(value) for value in data_module.get_gene_names()]
    if not genes or len(set(genes)) != len(genes):
        raise SCellSTRunnerError(f"GENE_AXIS:{len(genes)}:{len(set(genes))}")
    config.model.gene_names = genes
    config.predictor.output_dim = len(genes)
    return data_module, prepared


def build_model(config: Any, *, rho: float) -> torch.nn.Module:
    from .lightning_overlay import prepare_local_difference_model

    return prepare_local_difference_model(config, rho=validate_rho(rho))


@dataclass(frozen=True)
class LoaderBundle:
    train: Iterable[dict[str, Any]]
    validation: DataLoader
    ordinary: DataLoader
    edge: DataLoader | None
    graph_metadata: dict[str, Any] | None


def make_loaders(
    prepared: PreparedSpatialData,
    config: Any,
    *,
    seed: int,
    rho: float,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> LoaderBundle:
    """Build matched ordinary streams; baseline never constructs an edge stream."""

    run_seed = validate_seed(seed)
    arm = validate_rho(rho)
    size = int(config.data.batch_size if batch_size is None else batch_size)
    workers = int(config.data.num_workers if num_workers is None else num_workers)
    if size <= 0 or workers < 0:
        raise SCellSTRunnerError(f"LOADER_CONFIG:{size}:{workers}")
    ordinary = DataLoader(
        prepared.train_dataset,
        batch_size=size,
        shuffle=True,
        num_workers=workers,
        collate_fn=scellst_spatial_collate,
        generator=torch.Generator().manual_seed(run_seed),
    )
    validation = DataLoader(
        prepared.val_dataset,
        batch_size=size,
        shuffle=False,
        num_workers=workers,
        collate_fn=scellst_spatial_collate,
    )
    if arm == 0.0:
        return LoaderBundle(ordinary, validation, ordinary, None, None)

    graph = build_training_adjacency(prepared.train_dataset)
    edge_dataset = EdgePairDataset(prepared.train_dataset, graph.edge_index)
    edge = DataLoader(
        edge_dataset,
        batch_size=size,
        shuffle=True,
        num_workers=workers,
        collate_fn=scellst_edge_pair_collate,
        generator=torch.Generator().manual_seed(run_seed + EDGE_SEED_OFFSET),
    )
    paired = PairedSpotEdgeLoader(ordinary, edge)
    return LoaderBundle(paired, validation, ordinary, edge, graph.to_metadata())


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def named_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def frozen_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def require_finite_backward(loss: torch.Tensor, model: torch.nn.Module) -> None:
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise SCellSTRunnerError("LOSS_NONFINITE")
    loss.backward()
    invalid = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    if invalid:
        raise SCellSTRunnerError(f"GRADIENT_NONFINITE:{invalid[:5]}")


def config_contract(config: Any) -> str:
    from omegaconf import OmegaConf

    payload = OmegaConf.to_container(config, resolve=True)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def initial_state_payload(
    model: torch.nn.Module,
    config: Any,
    prepared: PreparedSpatialData,
    *,
    seed: int,
    source_revision: str,
) -> dict[str, Any]:
    return {
        "schema": INITIAL_STATE_SCHEMA,
        "seed": validate_seed(seed),
        "source_revision": str(source_revision),
        "split_manifest": str(prepared.split.manifest_path.resolve()),
        "split_seed": int(prepared.split.seed),
        "frac_train": float(prepared.split.frac_train),
        "n_train": len(prepared.split.train_indices),
        "n_validation": len(prepared.split.val_indices),
        "genes": [str(value) for value in config.model.gene_names],
        "config_contract": config_contract(config),
        "model_state": frozen_model_state(model),
        "rng_state": capture_rng_state(),
    }


def validate_initial_state(
    payload: Mapping[str, Any],
    config: Any,
    prepared: PreparedSpatialData,
    *,
    seed: int,
    source_revision: str,
) -> None:
    expected = {
        "schema": INITIAL_STATE_SCHEMA,
        "seed": validate_seed(seed),
        "source_revision": str(source_revision),
        "split_manifest": str(prepared.split.manifest_path.resolve()),
        "split_seed": SPLIT_SEED,
        "frac_train": 0.8,
        "n_train": len(prepared.split.train_indices),
        "n_validation": len(prepared.split.val_indices),
        "genes": [str(value) for value in config.model.gene_names],
        "config_contract": config_contract(config),
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise SCellSTRunnerError(
                f"INITIAL_STATE_MISMATCH:{key}:{payload.get(key)!r}:{expected_value!r}"
            )
    if not isinstance(payload.get("model_state"), Mapping) or not isinstance(
        payload.get("rng_state"), Mapping
    ):
        raise SCellSTRunnerError("INITIAL_STATE_CONTENT")


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, output)
    return output


def load_initial_state(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise SCellSTRunnerError(f"INITIAL_STATE_MISSING:{source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != INITIAL_STATE_SCHEMA:
        raise SCellSTRunnerError("INITIAL_STATE_SCHEMA")
    return payload


def prediction_filename(specimen: str, *, seed: int, rho: float) -> str:
    if specimen not in TEST_SPECIMENS:
        raise SCellSTRunnerError(f"PREDICTION_SPECIMEN:{specimen}")
    rho_tag = str(validate_rho(rho)).replace(".", "p")
    return (
        f"STAudit_sCellST_G0055_TENX39_seed={validate_seed(seed)};rho={rho_tag};"
        f"test_slide={specimen};shape=xenium_nucleus;infer_mode=inference.h5ad"
    )
