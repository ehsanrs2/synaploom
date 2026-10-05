from pathlib import Path

import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.training.checkpoint import load_checkpoint, save_checkpoint


def test_structural_state_and_routing_restore_exactly(tmp_path: Path) -> None:
    torch.manual_seed(11)
    graph = GraphConfig(num_nodes=5, initial_recruited=2, top_k=1, d_model=8, d_hidden=16, router_dim=8, q_max=2)
    cfg = ModelConfig(graph=graph, input_dim=8, num_classes=3)
    model = SynapLoomModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.graph_state.recruit(2, torch.arange(8, dtype=torch.float32))
    model.graph_state.add_edge(0, 1, gate=0.7)
    x = torch.randn(4, 8)
    before = model(x).trace.detached()

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer=optimizer, global_step=17, extra={"tag": "test"})

    model.graph_state.recruit(3, torch.zeros(8))
    with torch.no_grad():
        model.graph_state.routing_keys.add_(10.0)

    meta = load_checkpoint(path, model, optimizer=optimizer, restore_rng=True)
    after = model(x).trace.detached()

    assert meta == {"global_step": 17, "extra": {"tag": "test"}}
    assert torch.equal(before.recruited_node_ids, after.recruited_node_ids)
    assert torch.equal(before.selected_node_ids, after.selected_node_ids)
    assert before.structural_version == after.structural_version
