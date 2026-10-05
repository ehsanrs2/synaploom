import torch

from synaploom.baselines import DenseControlModel, make_no_growth_routed_control
from synaploom.config import GraphConfig, ModelConfig
from synaploom.data.synthetic import make_gaussian_cluster_stream
from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import NoConsolidation, NoEdges, TaskBoundaryRecruitment
from synaploom.training.continual import ContinualConfig, run_continual_experiment
from synaploom.training.engine import TrainingEngine


def _stream():
    return make_gaussian_cluster_stream(
        num_tasks=2,
        classes_per_task=2,
        input_dim=6,
        train_per_class=10,
        eval_per_class=8,
        separation=5.0,
        noise_std=0.6,
        seed=5,
    )


def test_dense_control_runs_same_continual_matrix_pipeline() -> None:
    torch.manual_seed(1)
    model = DenseControlModel(input_dim=6, d_model=8, d_hidden=12, num_classes=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    result = run_continual_experiment(
        model,
        _stream(),
        optimizer,
        config=ContinualConfig(epochs_per_task=1, batch_size=8, eval_batch_size=16, seed=1),
    )

    assert result.accuracy_matrix.shape == (2, 2)
    assert torch.isnan(result.accuracy_matrix[0, 1])
    assert not torch.isnan(result.accuracy_matrix[1]).any()
    assert result.structural_history[0].recruited_nodes is None


def test_no_growth_and_controlled_growth_are_distinguishable() -> None:
    graph = GraphConfig(
        num_nodes=5,
        initial_recruited=2,
        top_k=1,
        d_model=8,
        d_hidden=12,
        router_dim=8,
        q_max=1,
        message_rounds=0,
    )
    config = ModelConfig(graph=graph, input_dim=6, num_classes=4)
    run_config = ContinualConfig(epochs_per_task=1, batch_size=8, eval_batch_size=16, seed=4)

    torch.manual_seed(4)
    fixed_model = make_no_growth_routed_control(config)
    fixed_optimizer = torch.optim.SGD(fixed_model.parameters(), lr=0.05)
    fixed_result = run_continual_experiment(
        fixed_model, _stream(), fixed_optimizer, config=run_config
    )
    assert [s.recruited_nodes for s in fixed_result.structural_history] == [2, 2]
    assert any(
        value > 0
        for value in fixed_result.structural_history[-1].node_utilization_ema.values()
    )

    torch.manual_seed(4)
    growth_model = make_no_growth_routed_control(config)
    growth_optimizer = torch.optim.SGD(growth_model.parameters(), lr=0.05)
    controller = StructuralController(
        recruitment_policy=TaskBoundaryRecruitment(),
        edge_init_policy=NoEdges(),
        consolidation_policy=NoConsolidation(),
    )
    engine = TrainingEngine(growth_model, growth_optimizer, controller=controller)
    growth_result = run_continual_experiment(
        growth_model,
        _stream(),
        growth_optimizer,
        config=run_config,
        engine=engine,
    )

    assert [s.recruited_nodes for s in growth_result.structural_history] == [3, 3]
    assert growth_result.structural_history[0].mean_executed_nodes == 1.0
    assert growth_result.global_steps > 0
