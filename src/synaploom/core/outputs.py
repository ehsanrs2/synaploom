from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class RouterOutput:
    selected_node_ids: Tensor
    selected_scores: Tensor
    weights: Tensor
    batch_query: Tensor


@dataclass
class ForwardTrace:
    recruited_node_ids: Tensor
    selected_node_ids: Tensor
    router_weights: Tensor
    routing_context: Tensor
    executed_edge_count: int
    structural_version: int

    def detached(self) -> "ForwardTrace":
        return ForwardTrace(
            recruited_node_ids=self.recruited_node_ids.detach().cpu(),
            selected_node_ids=self.selected_node_ids.detach().cpu(),
            router_weights=self.router_weights.detach().cpu(),
            routing_context=self.routing_context.detach().cpu(),
            executed_edge_count=self.executed_edge_count,
            structural_version=self.structural_version,
        )


@dataclass
class ModelOutput:
    logits: Tensor
    representation: Tensor
    aux_loss: Tensor
    trace: ForwardTrace
