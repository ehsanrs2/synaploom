import torch
import torch.nn.functional as F

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel


def test_unselected_and_dormant_microcircuits_do_not_receive_gradients() -> None:
    torch.manual_seed(7)
    graph = GraphConfig(
        num_nodes=6,
        initial_recruited=3,
        top_k=1,
        d_model=8,
        d_hidden=16,
        router_dim=8,
        message_rounds=0,
    )
    model = SynapLoomModel(ModelConfig(graph=graph, input_dim=8, num_classes=3))
    x = torch.randn(4, 8)
    y = torch.tensor([0, 1, 2, 1])

    out = model(x)
    selected = int(out.trace.selected_node_ids.item())
    loss = F.cross_entropy(out.logits, y)
    loss.backward()

    for node_id, module in enumerate(model.executor.microcircuits):
        grads = [p.grad for p in module.parameters()]
        if node_id == selected:
            assert any(g is not None and torch.count_nonzero(g).item() > 0 for g in grads)
        else:
            assert all(g is None for g in grads)


def test_only_selected_endpoint_edges_execute() -> None:
    torch.manual_seed(3)
    graph = GraphConfig(
        num_nodes=4,
        initial_recruited=2,
        top_k=2,
        d_model=8,
        d_hidden=16,
        router_dim=8,
        q_max=2,
        message_rounds=1,
    )
    model = SynapLoomModel(ModelConfig(graph=graph, input_dim=8, num_classes=2))
    model.graph_state.add_edge(0, 1)
    model.graph_state.add_edge(0, 2)
    out = model(torch.randn(3, 8))
    assert out.trace.executed_edge_count == 1
