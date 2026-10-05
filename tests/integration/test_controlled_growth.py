import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import ConnectToSelectedTopR, FixedScheduleRecruitment
from synaploom.training.engine import TrainingEngine


def test_scheduled_recruitment_runs_end_to_end() -> None:
    torch.manual_seed(12)
    graph = GraphConfig(
        num_nodes=6,
        initial_recruited=2,
        top_k=2,
        q_max=2,
        d_model=8,
        d_hidden=16,
        router_dim=8,
        message_rounds=1,
    )
    model = SynapLoomModel(ModelConfig(graph=graph, input_dim=8, num_classes=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    controller = StructuralController(
        recruitment_policy=FixedScheduleRecruitment([2, 4]),
        edge_init_policy=ConnectToSelectedTopR(1),
    )
    engine = TrainingEngine(model, optimizer, controller=controller)

    x = torch.randn(8, 8)
    y = (x[:, 0] > 0).long()
    event_steps = []

    for step in range(1, 5):
        result = engine.train_step(x, y, step=step)
        if result.events:
            event_steps.append(step)

    assert event_steps == [2, 4]
    assert model.graph_state.recruited_count() == 4
    assert model.graph_state.active_edge_count() == 2
    assert model.graph_state.structural_version.item() == 4
