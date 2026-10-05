from __future__ import annotations

from torch import Tensor, nn


class SharedRelationProjector(nn.Module):
    """Shared dense transform per relation type; edge-specific state stays scalar/lightweight."""

    def __init__(self, relation_types: int, d_model: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Linear(d_model, d_model, bias=False) for _ in range(relation_types)
        )

    def forward(self, x: Tensor, relation: int) -> Tensor:
        return self.projections[relation](x)
