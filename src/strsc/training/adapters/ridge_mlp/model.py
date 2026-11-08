"""Two-layer MLP prediction head for HEST embeddings."""

from __future__ import annotations

import torch
from torch import nn


class TwoLayerMLP(nn.Module):
    """Linear -> GELU -> Linear, with no stochastic layers."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) < 1:
            raise ValueError(f"MLP_DIMENSIONS:{input_dim}:{hidden_dim}:{output_dim}")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
