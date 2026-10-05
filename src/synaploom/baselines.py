from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from synaploom.config import ModelConfig
from synaploom.model import SynapLoomModel


@dataclass
class DenseOutput:
    logits: Tensor
    representation: Tensor


class DenseControlModel(nn.Module):
    """Simple dense control for harness validation.

    M1.2 does not claim parameter matching. Fair matched baselines are an M1.3 task.
    """

    def __init__(self, *, input_dim: int, d_model: int, d_hidden: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Identity() if input_dim == d_model else nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.body = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> DenseOutput:
        z = self.encoder(x)
        representation = z + self.body(self.norm(z))
        return DenseOutput(logits=self.head(representation), representation=representation)


def make_no_growth_routed_control(config: ModelConfig) -> SynapLoomModel:
    """SynapLoom execution/routing with no structural controller attached."""
    return SynapLoomModel(config)
