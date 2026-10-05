from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor, nn

from synaploom.config import GraphConfig
from synaploom.core.edge_ops import SharedRelationProjector
from synaploom.core.graph_state import GraphState
from synaploom.core.microcircuit import MicroCircuit


class GraphExecutor(nn.Module):
    """Executes only the batch-selected microcircuits and their selected-endpoint edges."""

    def __init__(self, config: GraphConfig) -> None:
        super().__init__()
        self.config = config
        self.microcircuits = nn.ModuleList(
            MicroCircuit(config.d_model, config.d_hidden) for _ in range(config.num_nodes)
        )
        self.relations = SharedRelationProjector(config.relation_types, config.d_model)

    def forward(
        self,
        x: Tensor,
        selected_node_ids: Tensor,
        router_weights: Tensor,
        graph_state: GraphState,
    ) -> tuple[Tensor, int]:
        node_ids = [int(v) for v in selected_node_ids.detach().cpu().tolist()]
        if not node_ids:
            raise RuntimeError("executor received an empty selection")

        states: dict[int, Tensor] = {node_id: self.microcircuits[node_id](x) for node_id in node_ids}
        selected_set = set(node_ids)
        executed_edge_count = 0

        for _ in range(self.config.message_rounds):
            aggregates: dict[int, Tensor] = {}
            counts: dict[int, int] = defaultdict(int)
            projected_cache: dict[tuple[int, int], Tensor] = {}
            round_edge_count = 0

            for src in node_ids:
                for slot in range(self.config.q_max):
                    if not bool(graph_state.edge_enabled[src, slot].item()):
                        continue
                    dst = int(graph_state.edge_dst[src, slot].item())
                    if dst not in selected_set:
                        continue
                    relation = int(graph_state.edge_relation[src, slot].item())
                    cache_key = (src, relation)
                    if cache_key not in projected_cache:
                        projected_cache[cache_key] = self.relations(states[src], relation)
                    gate = torch.sigmoid(graph_state.edge_gate[src, slot])
                    message = gate * projected_cache[cache_key]
                    aggregates[dst] = message if dst not in aggregates else aggregates[dst] + message
                    counts[dst] += 1
                    round_edge_count += 1

            if round_edge_count == 0:
                break

            new_states: dict[int, Tensor] = {}
            for node_id in node_ids:
                mixed = states[node_id]
                if node_id in aggregates:
                    mixed = mixed + aggregates[node_id] / counts[node_id]
                new_states[node_id] = self.microcircuits[node_id](mixed)
            states = new_states
            executed_edge_count += round_edge_count

        stacked = torch.stack([states[node_id] for node_id in node_ids], dim=0)
        representation = (router_weights[:, None, None] * stacked).sum(dim=0)
        return representation, executed_edge_count
