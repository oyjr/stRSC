"""Paired-arm initialization and RNG reset helpers."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import random

import numpy as np
import torch


@dataclass(frozen=True)
class PairedArmState:
    model_state: dict[str, torch.Tensor]
    python_state: object
    numpy_state: tuple
    torch_state: torch.Tensor
    cuda_state: list[torch.Tensor] | None
    seed: int

    @classmethod
    def capture(cls, model: torch.nn.Module, *, seed: int) -> "PairedArmState":
        return cls(
            model_state={
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            python_state=copy.deepcopy(random.getstate()),
            numpy_state=copy.deepcopy(np.random.get_state()),
            torch_state=torch.random.get_rng_state().clone(),
            cuda_state=[value.clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None,
            seed=int(seed),
        )

    def restore(self, model: torch.nn.Module) -> torch.Generator:
        model.load_state_dict(self.model_state, strict=True)
        model.zero_grad(set_to_none=True)
        random.setstate(copy.deepcopy(self.python_state))
        np.random.set_state(copy.deepcopy(self.numpy_state))
        torch.random.set_rng_state(self.torch_state.clone())
        if self.cuda_state is not None:
            torch.cuda.set_rng_state_all([value.clone() for value in self.cuda_state])
        return torch.Generator().manual_seed(self.seed)


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
