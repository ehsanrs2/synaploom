from __future__ import annotations

import torch
from torch import nn

from synaploom.core.graph_state import GraphState


@torch.no_grad()
def update_node_importance(
    executor: nn.Module,
    graph_state: GraphState,
    *,
    decay: float = 0.99,
) -> None:
    """EMA of mean |parameter * gradient| per microcircuit."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")
    microcircuits = getattr(executor, "microcircuits", None)
    if microcircuits is None:
        raise TypeError("executor must expose a microcircuits ModuleList")

    graph_state.node_importance_ema.mul_(decay)
    for node_id, module in enumerate(microcircuits):
        score = None
        count = 0
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            contribution = (parameter.detach() * parameter.grad.detach()).abs().sum()
            score = contribution if score is None else score + contribution
            count += parameter.numel()
        if score is not None and count > 0:
            graph_state.node_importance_ema[node_id] += (1.0 - decay) * (score / count)
