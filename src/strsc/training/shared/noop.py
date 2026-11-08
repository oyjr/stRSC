"""Numerical equivalence gate for model-specific ``rho=0`` overlays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch


class NoOpCheckError(RuntimeError):
    """Raised when a rho-zero overlay changes the original computation."""


@dataclass(frozen=True)
class NoOpReceipt:
    status: str
    atol: float
    rtol: float
    loss_max_abs_difference: float
    prediction_max_abs_difference: float
    gradient_max_abs_difference: float
    tensors_compared: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _compare(left: Any, right: Any, *, name: str, atol: float, rtol: float) -> float:
    a = _array(left)
    b = _array(right)
    if a.shape != b.shape:
        raise NoOpCheckError(f"NOOP_SHAPE:{name}:{a.shape}:{b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise NoOpCheckError(f"NOOP_NONFINITE:{name}")
    maximum = float(np.max(np.abs(a - b), initial=0.0))
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise NoOpCheckError(
            f"NOOP_VALUE:{name}:max_abs={maximum}:atol={atol}:rtol={rtol}"
        )
    return maximum


def check_rho_zero_noop(
    *,
    original_loss: Any,
    rho_zero_loss: Any,
    original_predictions: Any,
    rho_zero_predictions: Any,
    original_gradients: Mapping[str, Any],
    rho_zero_gradients: Mapping[str, Any],
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> NoOpReceipt:
    """Assert equality of loss, predictions, and every named parameter gradient."""

    if atol < 0 or rtol < 0:
        raise NoOpCheckError(f"NOOP_TOLERANCE:{atol}:{rtol}")
    original_names = set(original_gradients)
    overlay_names = set(rho_zero_gradients)
    if original_names != overlay_names:
        raise NoOpCheckError(
            f"NOOP_GRADIENT_KEYS:{sorted(original_names)}:{sorted(overlay_names)}"
        )
    loss_difference = _compare(
        original_loss, rho_zero_loss, name="loss", atol=atol, rtol=rtol
    )
    prediction_difference = _compare(
        original_predictions,
        rho_zero_predictions,
        name="predictions",
        atol=atol,
        rtol=rtol,
    )
    gradient_difference = 0.0
    for name in sorted(original_names):
        left = original_gradients[name]
        right = rho_zero_gradients[name]
        if (left is None) != (right is None):
            raise NoOpCheckError(f"NOOP_GRADIENT_NONE:{name}")
        if left is None:
            continue
        gradient_difference = max(
            gradient_difference,
            _compare(left, right, name=f"gradient:{name}", atol=atol, rtol=rtol),
        )
    return NoOpReceipt(
        status="PASS",
        atol=float(atol),
        rtol=float(rtol),
        loss_max_abs_difference=loss_difference,
        prediction_max_abs_difference=prediction_difference,
        gradient_max_abs_difference=gradient_difference,
        tensors_compared=2 + len(original_names),
    )
