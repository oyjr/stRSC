"""Paired-arm initialization helpers without digest-based proxies."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch


def capture_initial_state(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Clone the initialized parameters/buffers before either arm trains."""

    return OrderedDict(
        (name, value.detach().cpu().clone()) for name, value in model.state_dict().items()
    )


def restore_initial_state(
    model: torch.nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    """Load and tensor-compare the shared initial state for one paired arm."""

    model.load_state_dict(state, strict=True)
    restored = model.state_dict()
    if tuple(restored) != tuple(state):
        raise RuntimeError("Restored state keys differ from the frozen initialization")
    mismatched = [
        name
        for name, expected in state.items()
        if not torch.equal(restored[name].detach().cpu(), expected.detach().cpu())
    ]
    if mismatched:
        raise RuntimeError(
            "Restored initialization differs for: " + ", ".join(mismatched[:10])
        )
