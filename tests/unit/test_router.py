import torch

from synaploom.config import GraphConfig
from synaploom.core.graph_state import GraphState
from synaploom.core.router import SparseRouter


def test_router_never_selects_dormant_nodes() -> None:
    torch.manual_seed(0)
    cfg = GraphConfig(num_nodes=8, initial_recruited=3, top_k=2, d_model=8, d_hidden=16, router_dim=8)
    state = GraphState(cfg)
    router = SparseRouter(cfg)
    x = torch.randn(5, cfg.d_model)

    out = router(x, state)
    assert out.selected_node_ids.numel() == 2
    assert set(out.selected_node_ids.tolist()).issubset({0, 1, 2})
    assert torch.allclose(out.weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_only_selected_routing_keys_receive_nonzero_gradient() -> None:
    torch.manual_seed(2)
    cfg = GraphConfig(num_nodes=6, initial_recruited=4, top_k=2, d_model=8, d_hidden=16, router_dim=8)
    state = GraphState(cfg)
    router = SparseRouter(cfg)
    x = torch.randn(5, cfg.d_model, requires_grad=True)

    out = router(x, state)
    loss = (out.weights * out.selected_scores).sum()
    loss.backward()

    grad = state.routing_keys.grad
    assert grad is not None
    selected = set(out.selected_node_ids.tolist())
    for node_id in range(cfg.num_nodes):
        nz = torch.count_nonzero(grad[node_id]).item()
        if node_id in selected:
            assert nz > 0
        else:
            assert nz == 0
