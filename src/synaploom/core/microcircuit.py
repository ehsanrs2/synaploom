from __future__ import annotations

from torch import Tensor, nn


class MicroCircuit(nn.Module):
    """Small residual expert used as one graph node in M1."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))
