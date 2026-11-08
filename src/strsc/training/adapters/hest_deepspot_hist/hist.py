"""Reversible training-only loss overlay for the released HiST model."""

from __future__ import annotations

from pathlib import Path
import types
from typing import Any

import torch
import torch.nn.functional as F

from .contract import HestDeepSpotHiSTError, validate_rho
from .data import HIST_BACKBONE, add_hist_to_path
from .objective import ExpressionLoss


_ORIGINAL_FORWARD = "_g0055_hist_original_forward"
_RHO = "_g0055_hist_rho"
_LAST_PARTS = "_g0055_hist_last_loss_parts"


def compose_hist_config(deps_root: str | Path, *, overrides: list[str] | None = None):
    root = add_hist_to_path(deps_root)
    try:
        from hydra import compose, initialize_config_dir
    except Exception as exc:  # pragma: no cover - exercised by the pinned real gate.
        raise HestDeepSpotHiSTError(f"HIST_PINNED_ENV_IMPORT:{type(exc).__name__}:{exc}") from exc
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        cfg = compose(config_name="gene_prediction", overrides=list(overrides or ()))
    if cfg.model.train_mode != "precomputed" or cfg.patch_backbone != HIST_BACKBONE:
        raise HestDeepSpotHiSTError(
            f"HIST_RELEASED_FEATURE_CONTRACT:{cfg.model.train_mode}:{cfg.patch_backbone}"
        )
    return cfg


def build_upstream_hist(
    *, deps_root: str | Path, cfg: Any | None = None, num_genes: int = 50
):
    """Build the released sparse encoder-decoder and log-expression head."""

    root = add_hist_to_path(deps_root)
    if int(num_genes) != 50:
        raise HestDeepSpotHiSTError(f"HIST_OUTPUT_GENES:{num_genes}")
    config = compose_hist_config(root) if cfg is None else cfg
    try:
        from tasks.gene_prediction.main import build_model_components
        from tasks.gene_prediction.model import PredictionModel

        _, _, _, padder, slide_encoder, decoder, segmentor = build_model_components(
            config, 50
        )
        return PredictionModel(
            padder=padder,
            slide_encoder=slide_encoder,
            decoder=decoder,
            segmentor=segmentor,
            image_suffix=config.patch_backbone,
            cfg=config,
        )
    except Exception as exc:  # DGL/CUDA extension failures are classified by the gate.
        raise HestDeepSpotHiSTError(
            f"HIST_PINNED_MODEL_BUILD:{type(exc).__name__}:{exc}"
        ) from exc


def assert_hist_contract(model: torch.nn.Module, *, strict_source: bool = True) -> None:
    if strict_source and model.__class__.__module__ != "tasks.gene_prediction.model":
        raise HestDeepSpotHiSTError(
            f"HIST_NOT_RELEASED_MODEL:{model.__class__.__module__}.{model.__class__.__name__}"
        )
    if not callable(getattr(model, "forward", None)) or not isinstance(
        getattr(model, "segmentor", None), torch.nn.Module
    ):
        raise HestDeepSpotHiSTError("HIST_FORWARD_CONTRACT")


def _capture_segmentor_output(storage: list[torch.Tensor]):
    def hook(module, inputs, output):
        del module, inputs
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise HestDeepSpotHiSTError("HIST_SEGMENTOR_OUTPUT")
        storage.append(value)

    return hook


def _overlay_forward(self, batch, return_slide_token: bool = False):
    original = getattr(self, _ORIGINAL_FORWARD)
    rho = float(getattr(self, _RHO))
    if rho == 0.0 or not self.training:
        # Validation/checkpoint selection and rho=0 are the exact upstream path.
        return original(batch, return_slide_token=return_slide_token)
    edge_index = getattr(batch, "g0055_edge_index", None)
    if not isinstance(edge_index, torch.Tensor) or edge_index.shape[0] != 2:
        raise HestDeepSpotHiSTError("HIST_TRAIN_EDGE_INDEX_REQUIRED")

    captured: list[torch.Tensor] = []
    handle = self.segmentor.register_forward_hook(_capture_segmentor_output(captured))
    try:
        result = original(batch, return_slide_token=return_slide_token)
    finally:
        handle.remove()
    if not isinstance(result, tuple) or len(result) not in (2, 3) or len(captured) != 1:
        raise HestDeepSpotHiSTError("HIST_FORWARD_RESULT")
    latent = captured[0][1:-1]
    target = torch.log1p(batch.mol_feats)
    ordinary = result[1]
    reference = F.mse_loss(latent, target)
    if not torch.equal(ordinary, reference):
        raise HestDeepSpotHiSTError("HIST_UPSTREAM_MSE_COMPONENT_DRIFT")

    from strsc.training.shared.losses import local_difference_mse

    local = local_difference_mse(latent, target, edge_index)
    total = (1.0 - rho) * ordinary + rho * local
    setattr(self, _LAST_PARTS, ExpressionLoss(total, ordinary, local))
    if len(result) == 2:
        return result[0], total
    return result[0], total, result[2]


def install_hist_overlay(
    model: torch.nn.Module,
    *,
    rho: float,
    strict_source: bool = True,
) -> torch.nn.Module:
    assert_hist_contract(model, strict_source=strict_source)
    value = validate_rho(rho)
    if hasattr(model, _ORIGINAL_FORWARD):
        raise HestDeepSpotHiSTError("HIST_OVERLAY_ALREADY_INSTALLED")
    setattr(model, _ORIGINAL_FORWARD, model.forward)
    setattr(model, _RHO, value)
    model.forward = types.MethodType(_overlay_forward, model)
    return model


def remove_hist_overlay(model: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(model, _ORIGINAL_FORWARD):
        return model
    model.forward = getattr(model, _ORIGINAL_FORWARD)
    delattr(model, _ORIGINAL_FORWARD)
    delattr(model, _RHO)
    if hasattr(model, _LAST_PARTS):
        delattr(model, _LAST_PARTS)
    return model


def last_hist_loss_parts(model: torch.nn.Module) -> ExpressionLoss | None:
    return getattr(model, _LAST_PARTS, None)


def build_hist_optimizer(model: torch.nn.Module, cfg: Any):
    """The released non-LoRA optimizer; formal config remains authoritative."""

    if cfg.model.train_mode != "precomputed":
        raise HestDeepSpotHiSTError(f"HIST_TRAIN_MODE:{cfg.model.train_mode}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
