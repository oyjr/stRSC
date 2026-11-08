"""Reversible overlay for the released HEST spot-level DeepSpot model."""

from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any, Mapping

import torch

from .contract import HestDeepSpotHiSTError, validate_rho
from .data import DEEPSPOT_CONTEXT
from .objective import ExpressionLoss, mixed_separate_expression_loss


_ORIGINAL_TRAINING_STEP = "_g0055_deepspot_original_training_step"
_RHO = "_g0055_deepspot_rho"
_LAST_PARTS = "_g0055_deepspot_last_loss_parts"


def add_deepspot_to_path(deps_root: str | Path) -> Path:
    root = Path(deps_root).resolve()
    if not (root / "deepspot" / "spot" / "model.py").is_file():
        raise HestDeepSpotHiSTError(f"DEEPSPOT_SOURCE_MISSING:{root}")
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


def build_upstream_deepspot(
    *,
    deps_root: str | Path,
    input_size: int,
    output_size: int = 50,
    seed: int,
    scaler: Any,
):
    """Instantiate the released spot model, never the repository's cell model."""

    add_deepspot_to_path(deps_root)
    from deepspot.spot.model import DeepSpot

    if int(output_size) != 50:
        raise HestDeepSpotHiSTError(f"DEEPSPOT_OUTPUT_GENES:{output_size}")
    return DeepSpot(
        input_size=int(input_size),
        output_size=50,
        loss_func="mse",
        lr=1e-4,
        p=0.3,
        n_ensemble=10,
        phi2rho_size=512,
        weight_decay=1e-6,
        random_seed=int(seed),
        scaler=scaler,
        spot_context=DEEPSPOT_CONTEXT,
    )


def assert_deepspot_spot_contract(model: torch.nn.Module, *, strict_source: bool = True) -> None:
    if strict_source and not model.__class__.__module__.startswith("deepspot.spot"):
        raise HestDeepSpotHiSTError(
            f"DEEPSPOT_NOT_SPOT_MODEL:{model.__class__.__module__}.{model.__class__.__name__}"
        )
    loss = getattr(model, "loss_func", None)
    if not callable(loss) or getattr(loss, "__name__", "") != "loss_mse_function":
        raise HestDeepSpotHiSTError("DEEPSPOT_DIRECT_MSE_REQUIRED")
    hparams = getattr(model, "hparams", None)
    if strict_source and hparams is not None:
        context = getattr(hparams, "spot_context", None)
        if context != DEEPSPOT_CONTEXT:
            raise HestDeepSpotHiSTError(f"DEEPSPOT_CONTEXT:{context}:{DEEPSPOT_CONTEXT}")


def _ordinary_batch(batch: Mapping[str, Any]):
    if not isinstance(batch, Mapping) or "ordinary" not in batch:
        raise HestDeepSpotHiSTError("DEEPSPOT_ORDINARY_STREAM_REQUIRED")
    upstream = batch["ordinary"]
    if not isinstance(upstream, (tuple, list)) or len(upstream) != 2:
        raise HestDeepSpotHiSTError("DEEPSPOT_UPSTREAM_BATCH")
    return upstream


def _edge_batch(batch: Mapping[str, Any]) -> tuple[Any, torch.Tensor]:
    edge = batch.get("edge")
    if not isinstance(edge, Mapping) or "upstream_batch" not in edge:
        raise HestDeepSpotHiSTError("DEEPSPOT_EDGE_STREAM_REQUIRED")
    upstream = edge["upstream_batch"]
    edge_index = edge.get("edge_index")
    if not isinstance(upstream, (tuple, list)) or len(upstream) != 2:
        raise HestDeepSpotHiSTError("DEEPSPOT_EDGE_UPSTREAM_BATCH")
    if not isinstance(edge_index, torch.Tensor):
        raise HestDeepSpotHiSTError("DEEPSPOT_EDGE_TENSOR")
    return upstream, edge_index


def _overlay_training_step(self, batch: Mapping[str, Any], batch_idx: int):
    ordinary_batch = _ordinary_batch(batch)
    rho = float(getattr(self, _RHO))
    original = getattr(self, _ORIGINAL_TRAINING_STEP)
    if rho == 0.0:
        # Load-bearing no-op: do not inspect or evaluate the edge stream.
        return original(ordinary_batch, batch_idx)

    edge_batch, edge_index = _edge_batch(batch)
    ordinary_x, ordinary_target = ordinary_batch
    edge_x, edge_target = edge_batch
    ordinary_x = [value.float() for value in ordinary_x] if isinstance(ordinary_x, (list, tuple)) else ordinary_x.float()
    edge_x = [value.float() for value in edge_x] if isinstance(edge_x, (list, tuple)) else edge_x.float()
    ordinary_target = ordinary_target.float()
    edge_target = edge_target.float()
    ordinary_prediction = self(ordinary_x)
    edge_prediction = self(edge_x)
    parts = mixed_separate_expression_loss(
        ordinary_prediction,
        ordinary_target,
        rho=rho,
        edge_prediction=edge_prediction,
        edge_target=edge_target,
        edge_index=edge_index,
    )
    setattr(self, _LAST_PARTS, parts)
    self.log("train", parts.total, on_epoch=True, prog_bar=True)
    if hasattr(self, "training_loss"):
        self.training_loss.append(parts.total.detach().cpu().item())
    return parts.total


def install_deepspot_overlay(
    model: torch.nn.Module,
    *,
    rho: float,
    strict_source: bool = True,
) -> torch.nn.Module:
    assert_deepspot_spot_contract(model, strict_source=strict_source)
    value = validate_rho(rho)
    if hasattr(model, _ORIGINAL_TRAINING_STEP):
        raise HestDeepSpotHiSTError("DEEPSPOT_OVERLAY_ALREADY_INSTALLED")
    original = getattr(model, "training_step", None)
    if not callable(original):
        raise HestDeepSpotHiSTError("DEEPSPOT_TRAINING_STEP_MISSING")
    setattr(model, _ORIGINAL_TRAINING_STEP, original)
    setattr(model, _RHO, value)
    model.training_step = types.MethodType(_overlay_training_step, model)
    return model


def remove_deepspot_overlay(model: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(model, _ORIGINAL_TRAINING_STEP):
        return model
    model.training_step = getattr(model, _ORIGINAL_TRAINING_STEP)
    delattr(model, _ORIGINAL_TRAINING_STEP)
    delattr(model, _RHO)
    if hasattr(model, _LAST_PARTS):
        delattr(model, _LAST_PARTS)
    return model


def last_deepspot_loss_parts(model: torch.nn.Module) -> ExpressionLoss | None:
    return getattr(model, _LAST_PARTS, None)
