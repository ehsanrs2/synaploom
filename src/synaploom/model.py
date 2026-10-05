from __future__ import annotations

from torch import Tensor, nn

from synaploom.config import ModelConfig
from synaploom.core.executor import GraphExecutor
from synaploom.core.graph_state import GraphState
from synaploom.core.outputs import ForwardTrace, ModelOutput
from synaploom.core.router import SparseRouter


class SynapLoomModel(nn.Module):
    """M1 single-modality structural core."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        g = config.graph
        self.encoder = (
            nn.Identity() if config.input_dim == g.d_model else nn.Linear(config.input_dim, g.d_model)
        )
        self.graph_state = GraphState(g)
        self.router = SparseRouter(g)
        self.executor = GraphExecutor(g)
        self.head = nn.Linear(g.d_model, config.num_classes)

    def forward(self, x: Tensor, *, update_statistics: bool = False) -> ModelOutput:
        z = self.encoder(x)
        routed = self.router(z, self.graph_state)
        representation, executed_edges = self.executor(
            z,
            routed.selected_node_ids,
            routed.weights,
            self.graph_state,
        )
        logits = self.head(representation)
        if update_statistics:
            self.graph_state.update_utilization(routed.selected_node_ids, routed.weights)

        aux_loss = logits.new_zeros(())
        trace = ForwardTrace(
            recruited_node_ids=self.graph_state.recruited_ids(),
            selected_node_ids=routed.selected_node_ids,
            router_weights=routed.weights,
            routing_context=routed.batch_query,
            executed_edge_count=executed_edges,
            structural_version=int(self.graph_state.structural_version.item()),
        )
        return ModelOutput(
            logits=logits,
            representation=representation,
            aux_loss=aux_loss,
            trace=trace,
        )
