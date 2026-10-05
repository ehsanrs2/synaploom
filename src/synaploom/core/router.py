from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from synaploom.config import GraphConfig
from synaploom.core.graph_state import GraphState
from synaploom.core.outputs import RouterOutput


class SparseRouter(nn.Module):
    """Batch-level top-k router over recruited nodes only."""

    def __init__(self, config: GraphConfig) -> None:
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.d_model, config.router_dim, bias=False)

    def forward(self, x: Tensor, graph_state: GraphState) -> RouterOutput:
        if x.ndim != 2 or x.shape[-1] != self.config.d_model:
            raise ValueError("router expects [batch, d_model]")

        recruited_ids = graph_state.recruited_ids()
        if recruited_ids.numel() == 0:
            raise RuntimeError("router has no recruited nodes")

        batch_query = self.query(x).mean(dim=0)
        keys = graph_state.routing_keys.index_select(0, recruited_ids)
        scores = torch.mv(keys, batch_query) / math.sqrt(self.config.router_dim)
        k = min(self.config.top_k, int(recruited_ids.numel()))
        selected_scores, local_ids = torch.topk(scores, k=k, dim=0)
        selected_node_ids = recruited_ids.index_select(0, local_ids)
        weights = torch.softmax(selected_scores, dim=0)

        return RouterOutput(
            selected_node_ids=selected_node_ids,
            selected_scores=selected_scores,
            weights=weights,
            batch_query=batch_query,
        )
