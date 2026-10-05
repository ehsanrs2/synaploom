import pytest
import torch

from synaploom.config import GraphConfig
from synaploom.core.graph_state import GraphState
from synaploom.core.states import NodeState


def test_recruitment_and_bounded_edge_slots() -> None:
    cfg = GraphConfig(num_nodes=5, initial_recruited=2, top_k=1, q_max=2, router_dim=8, d_model=8, d_hidden=16)
    state = GraphState(cfg)

    assert state.recruited_count() == 2
    assert state.node_state[2].item() == int(NodeState.DORMANT)

    key = torch.ones(cfg.router_dim)
    state.recruit(2, key)
    assert state.recruited_count() == 3
    assert state.node_state[2].item() == int(NodeState.PLASTIC)
    assert torch.equal(state.routing_keys[2].detach(), key)

    state.add_edge(0, 1)
    state.add_edge(0, 2)
    assert state.active_edge_count() == 2
    with pytest.raises(RuntimeError):
        state.add_edge(0, 3)

    state.validate()


def test_invalid_transition_is_rejected() -> None:
    cfg = GraphConfig(num_nodes=4, initial_recruited=2, top_k=1, d_model=8, d_hidden=16, router_dim=8)
    state = GraphState(cfg)
    with pytest.raises(ValueError):
        state.recruit(0)
    with pytest.raises(ValueError):
        state.consolidate([3])
