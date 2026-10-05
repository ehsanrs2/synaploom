from __future__ import annotations

from torch import nn

from synaploom.core.graph_state import GraphState


def scale_microcircuit_gradients(executor: nn.Module, graph_state: GraphState) -> None:
    """Scale per-node gradients after backward and before optimizer.step()."""
    microcircuits = getattr(executor, "microcircuits", None)
    if microcircuits is None:
        raise TypeError("executor must expose a microcircuits ModuleList")
    for node_id, module in enumerate(microcircuits):
        scale = float(graph_state.node_plasticity[node_id].item())
        for parameter in module.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
