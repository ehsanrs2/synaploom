from pathlib import Path

import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import FixedScheduleRecruitment, StructuralObservation
from synaploom.training.checkpoint import load_checkpoint, save_checkpoint


def test_controller_last_applied_step_is_checkpointed(tmp_path: Path) -> None:
    graph = GraphConfig(
        num_nodes=4,
        initial_recruited=2,
        top_k=1,
        d_model=8,
        d_hidden=16,
        router_dim=8,
    )
    model = SynapLoomModel(ModelConfig(graph=graph, input_dim=8, num_classes=2))
    controller = StructuralController(recruitment_policy=FixedScheduleRecruitment([2]))
    obs = StructuralObservation(
        step=2,
        task_boundary=False,
        selected_node_ids=(0,),
        router_weights=(1.0,),
        routing_context=torch.ones(8),
    )
    controller.observe(obs)
    controller.apply(model.graph_state, controller.propose(model.graph_state))

    path = tmp_path / "controller.pt"
    save_checkpoint(path, model, controller=controller)

    restored_model = SynapLoomModel(ModelConfig(graph=graph, input_dim=8, num_classes=2))
    restored_controller = StructuralController(recruitment_policy=FixedScheduleRecruitment([2]))
    load_checkpoint(path, restored_model, controller=restored_controller)

    restored_controller.observe(obs)
    assert not restored_controller.should_update(restored_model.graph_state)
    assert restored_controller.apply(
        restored_model.graph_state,
        restored_controller.propose(restored_model.graph_state),
    ) == []
    assert restored_model.graph_state.recruited_count() == 3
