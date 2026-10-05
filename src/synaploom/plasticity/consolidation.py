from __future__ import annotations

from torch import nn

from synaploom.core.graph_state import GraphState


def scale_node_owned_gradients(executor: nn.Module, graph_state: GraphState) -> None:
    """Scale gradients owned by each node according to its plasticity coefficient.

    Node-owned parameters in M1 are:
    - the microcircuit parameters,
    - the node's routing-key row,
    - the node's outgoing edge-gate row.

    Shared relation projections remain shared/plastic and are not scaled here.
    """
    microcircuits = getattr(executor, "microcircuits", None)
    if microcircuits is None:
        raise TypeError("executor must expose a microcircuits ModuleList")

    for node_id, module in enumerate(microcircuits):
        scale = float(graph_state.node_plasticity[node_id].item())
        for parameter in module.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)

        if graph_state.routing_keys.grad is not None:
            graph_state.routing_keys.grad[node_id].mul_(scale)
        if graph_state.edge_gate.grad is not None:
            graph_state.edge_gate.grad[node_id].mul_(scale)


def scale_microcircuit_gradients(executor: nn.Module, graph_state: GraphState) -> None:
    """Backward-compatible alias for the M1.0 API."""
    scale_node_owned_gradients(executor, graph_state)
