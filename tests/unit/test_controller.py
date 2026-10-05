import torch

from synaploom.config import GraphConfig
from synaploom.core.graph_state import GraphState
from synaploom.core.states import NodeState
from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import (
    ConnectToSelectedTopR,
    FixedScheduleRecruitment,
    NoRecruitment,
    StructuralObservation,
    TaskBoundaryTopImportance,
)


def _observation(cfg: GraphConfig, *, step: int, boundary: bool = False) -> StructuralObservation:
    return StructuralObservation(
        step=step,
        task_boundary=boundary,
        selected_node_ids=(0, 1),
        router_weights=(0.8, 0.2),
        routing_context=torch.arange(1, cfg.router_dim + 1, dtype=torch.float32),
    )


def test_fixed_schedule_recruits_lowest_dormant_and_imprints_context() -> None:
    torch.manual_seed(0)
    cfg = GraphConfig(
        num_nodes=5,
        initial_recruited=2,
        top_k=2,
        q_max=2,
        d_model=8,
        d_hidden=16,
        router_dim=8,
    )
    state = GraphState(cfg)
    expected_norm = state.routing_keys[state.recruited_mask].detach().norm(dim=1).median()

    controller = StructuralController(
        recruitment_policy=FixedScheduleRecruitment([3]),
        edge_init_policy=ConnectToSelectedTopR(1),
    )
    obs = _observation(cfg, step=3)
    controller.observe(obs)

    assert controller.should_update(state)
    plan = controller.propose(state)
    assert plan.recruit_node_id == 2
    assert plan.edge_targets == (0,)

    events = controller.apply(state, plan)
    assert [event.kind for event in events] == ["recruit", "edge_add"]
    assert state.node_state[2].item() == int(NodeState.PLASTIC)
    assert state.edge_dst[2, 0].item() == 0

    key = state.routing_keys[2].detach()
    cosine = torch.nn.functional.cosine_similarity(key, obs.routing_context, dim=0)
    assert cosine.item() > 0.999
    assert torch.allclose(key.norm(), expected_norm, rtol=1e-5, atol=1e-7)

    assert controller.apply(state, plan) == []
    assert state.recruited_count() == 3


def test_task_boundary_consolidates_highest_importance_plastic_node() -> None:
    cfg = GraphConfig(
        num_nodes=4,
        initial_recruited=3,
        top_k=2,
        d_model=8,
        d_hidden=16,
        router_dim=8,
    )
    state = GraphState(cfg)
    state.node_importance_ema[:] = torch.tensor([0.1, 0.9, 0.4, 0.0])

    controller = StructuralController(
        recruitment_policy=NoRecruitment(),
        consolidation_policy=TaskBoundaryTopImportance(top_n=1),
    )
    controller.observe(_observation(cfg, step=5, boundary=True))
    plan = controller.propose(state)
    assert plan.consolidate_node_ids == (1,)

    events = controller.apply(state, plan)
    assert [event.kind for event in events] == ["consolidate"]
    assert state.node_state[1].item() == int(NodeState.CONSOLIDATED)
