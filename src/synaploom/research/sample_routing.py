"""Isolated H1a reference; shares parameters, never updates statistics.

Rows are deliberately executed separately. This is a correctness reference,
not an optimized executor or a TrainingEngine-compatible replacement.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class SampleOutput:
    logits: torch.Tensor
    selected_node_ids: torch.Tensor
    weights: torch.Tensor
    queries: torch.Tensor


class SampleRoutingReference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._validate()

    def _validate(self):
        g = self.model.config.graph
        if g.q_max != 0 or g.message_rounds != 0:
            raise ValueError("reference requires q_max=0 and message_rounds=0")
        self.model.graph_state.validate()

    def forward(self, x):
        self._validate()
        if x.ndim != 2 or len(x) == 0 or x.shape[1] != self.model.config.input_dim:
            raise ValueError("expected nonempty [batch,input_dim]")
        model = self.model
        recruited = model.graph_state.recruited_ids()  # ascending IDs
        if not len(recruited):
            raise ValueError("no recruited experts")
        logits, ids, weights, queries = [], [], [], []
        for row in x.split(1):
            z = model.encoder(row)
            query = model.router.query(z)[0]
            scores = torch.mv(model.graph_state.routing_keys[recruited], query)
            scores = scores / math.sqrt(model.config.graph.router_dim)
            if not torch.isfinite(scores).all():
                raise ValueError("nonfinite routing scores")
            # Stable score sort makes exact ties choose the lowest node ID.
            local = torch.argsort(scores, descending=True, stable=True)[:model.config.graph.top_k]
            selected = recruited[local]
            w = scores[local].softmax(0)
            representation, _ = model.executor(z, selected, w, model.graph_state)
            logits.append(model.head(representation))
            ids.append(selected)
            weights.append(w)
            queries.append(query)
        return SampleOutput(torch.cat(logits), torch.stack(ids), torch.stack(weights), torch.stack(queries))
