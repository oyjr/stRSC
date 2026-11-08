"""Deterministic paired-arm initialization, order and edge-batch schedules."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass(frozen=True)
class PairedArmState:
    seed: int
    initial_state: dict[str, torch.Tensor]

    def load_into(self, model: torch.nn.Module) -> None:
        model.load_state_dict(copy.deepcopy(self.initial_state), strict=True)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def freeze_initialization(
    factory: Callable[[], torch.nn.Module], *, seed: int
) -> tuple[torch.nn.Module, PairedArmState]:
    seed_everything(seed)
    model = factory()
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    return model, PairedArmState(int(seed), state)


def epoch_order(length: int, *, seed: int, epoch: int) -> np.ndarray:
    if length < 1 or epoch < 0:
        raise ValueError(f"PAIR_ORDER_ARGUMENT:{length}:{epoch}")
    sequence = np.random.SeedSequence([int(seed), int(epoch), 0xE611])
    return np.random.default_rng(sequence).permutation(length)


def edge_generator(*, seed: int, epoch: int, step: int) -> np.random.Generator:
    if epoch < 0 or step < 0:
        raise ValueError(f"PAIR_EDGE_ARGUMENT:{epoch}:{step}")
    sequence = np.random.SeedSequence([int(seed), int(epoch), int(step), 0xED63])
    return np.random.default_rng(sequence)


def step_rng_seed(*, seed: int, epoch: int, step: int, stream: str) -> int:
    """Stable per-step RNG so edge forwards cannot shift ordinary augmentation."""

    labels = {"ordinary": 0x0D17, "edge_left": 0xED61, "edge_right": 0xED62}
    if stream not in labels or epoch < 0 or step < 0:
        raise ValueError(f"PAIR_STEP_RNG_ARGUMENT:{stream}:{epoch}:{step}")
    sequence = np.random.SeedSequence(
        [int(seed), int(epoch), int(step), labels[stream]]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])
