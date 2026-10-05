from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from synaploom.core.graph_state import GraphState


@dataclass(frozen=True)
class StructuralCounts:
    total_nodes: int
    recruited_nodes: int
    executed_nodes: int
    active_edges: int


@dataclass(frozen=True)
class ParameterAccounting:
    total_parameters: int
    shared_parameters: int
    total_microcircuit_parameters: int
    recruited_microcircuit_parameters: int
    executed_microcircuit_parameters: int


def structural_counts(graph_state: GraphState, selected_node_ids: Tensor) -> StructuralCounts:
    return StructuralCounts(
        total_nodes=graph_state.config.num_nodes,
        recruited_nodes=graph_state.recruited_count(),
        executed_nodes=int(selected_node_ids.numel()),
        active_edges=graph_state.active_edge_count(),
    )


def parameter_accounting(model: nn.Module, selected_node_ids: Tensor) -> ParameterAccounting:
    executor = getattr(model, "executor", None)
    graph_state = getattr(model, "graph_state", None)
    if executor is None or graph_state is None or not hasattr(executor, "microcircuits"):
        raise TypeError("model must expose executor.microcircuits and graph_state")

    per_node = [sum(p.numel() for p in module.parameters()) for module in executor.microcircuits]
    total_micro = sum(per_node)
    total = sum(p.numel() for p in model.parameters())
    recruited = sum(per_node[i] for i in graph_state.recruited_ids().tolist())
    executed = sum(per_node[i] for i in selected_node_ids.detach().cpu().tolist())
    return ParameterAccounting(
        total_parameters=total,
        shared_parameters=total - total_micro,
        total_microcircuit_parameters=total_micro,
        recruited_microcircuit_parameters=recruited,
        executed_microcircuit_parameters=executed,
    )
