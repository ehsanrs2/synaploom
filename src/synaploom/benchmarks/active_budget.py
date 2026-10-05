from __future__ import annotations

from dataclasses import dataclass

from synaploom.model import SynapLoomModel


@dataclass(frozen=True)
class SynapLoomCapacity:
    total_parameters: int
    shared_parameters: int
    per_microcircuit_parameters: int
    total_microcircuit_parameters: int
    estimated_executed_parameters: int
    estimated_recruited_parameters: int


def synaploom_capacity(
    model: SynapLoomModel,
    *,
    executed_nodes: int,
    recruited_nodes: int,
) -> SynapLoomCapacity:
    if executed_nodes <= 0 or recruited_nodes <= 0:
        raise ValueError("node counts must be positive")
    if executed_nodes > recruited_nodes:
        raise ValueError("executed_nodes cannot exceed recruited_nodes")
    if recruited_nodes > model.config.graph.num_nodes:
        raise ValueError("recruited_nodes exceeds model capacity")

    micro_counts = [
        sum(parameter.numel() for parameter in module.parameters())
        for module in model.executor.microcircuits
    ]
    if len(set(micro_counts)) != 1:
        raise RuntimeError("M1.3 assumes equal-size microcircuits")
    per_micro = micro_counts[0]
    total = sum(parameter.numel() for parameter in model.parameters())
    total_micro = per_micro * len(micro_counts)

    # Shared compute includes encoder, router query, head, and relation operators.
    # Routing keys and edge gates are node-owned, so count only relevant rows.
    routing_row = model.graph_state.routing_keys.shape[1]
    edge_row = model.graph_state.edge_gate.shape[1]
    node_owned_non_micro = routing_row + edge_row
    all_node_owned_non_micro = node_owned_non_micro * model.config.graph.num_nodes

    shared = total - total_micro - all_node_owned_non_micro
    executed = shared + executed_nodes * (per_micro + node_owned_non_micro)
    recruited = shared + recruited_nodes * (per_micro + node_owned_non_micro)
    return SynapLoomCapacity(
        total_parameters=total,
        shared_parameters=shared,
        per_microcircuit_parameters=per_micro,
        total_microcircuit_parameters=total_micro,
        estimated_executed_parameters=executed,
        estimated_recruited_parameters=recruited,
    )
