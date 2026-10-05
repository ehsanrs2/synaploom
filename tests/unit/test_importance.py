import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.plasticity.statistics import update_node_importance


def test_importance_updates_only_microcircuits_with_gradients() -> None:
    torch.manual_seed(4)
    graph = GraphConfig(
        num_nodes=5,
        initial_recruited=3,
        top_k=1,
        d_model=8,
        d_hidden=16,
        router_dim=8,
    )
    model = SynapLoomModel(ModelConfig(graph=graph, input_dim=8, num_classes=2))
    out = model(torch.randn(4, 8))
    selected = int(out.trace.selected_node_ids.item())
    out.logits.sum().backward()

    update_node_importance(model.executor, model.graph_state, decay=0.9)

    for node_id, value in enumerate(model.graph_state.node_importance_ema.tolist()):
        if node_id == selected:
            assert value > 0
        else:
            assert value == 0
