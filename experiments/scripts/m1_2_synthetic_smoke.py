from __future__ import annotations

import json

import torch

from synaploom.baselines import DenseControlModel, make_no_growth_routed_control
from synaploom.config import GraphConfig, ModelConfig
from synaploom.data.synthetic import make_gaussian_cluster_stream
from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import NoConsolidation, NoEdges, TaskBoundaryRecruitment
from synaploom.training.continual import ContinualConfig, run_continual_experiment
from synaploom.training.engine import TrainingEngine


def run() -> dict[str, object]:
    stream = make_gaussian_cluster_stream(
        num_tasks=3,
        classes_per_task=2,
        input_dim=8,
        train_per_class=16,
        eval_per_class=16,
        separation=5.0,
        noise_std=0.7,
        seed=21,
    )
    run_cfg = ContinualConfig(
        epochs_per_task=1,
        batch_size=16,
        eval_batch_size=64,
        seed=21,
    )

    graph = GraphConfig(
        num_nodes=7,
        initial_recruited=2,
        top_k=1,
        d_model=12,
        d_hidden=20,
        router_dim=12,
        q_max=1,
        message_rounds=0,
    )
    model_cfg = ModelConfig(graph=graph, input_dim=8, num_classes=6)

    torch.manual_seed(21)
    growth = make_no_growth_routed_control(model_cfg)
    growth_opt = torch.optim.SGD(growth.parameters(), lr=0.06)
    controller = StructuralController(
        recruitment_policy=TaskBoundaryRecruitment(),
        edge_init_policy=NoEdges(),
        consolidation_policy=NoConsolidation(),
    )
    growth_engine = TrainingEngine(growth, growth_opt, controller=controller)
    growth_result = run_continual_experiment(
        growth, stream, growth_opt, config=run_cfg, engine=growth_engine
    )

    torch.manual_seed(21)
    fixed = make_no_growth_routed_control(model_cfg)
    fixed_opt = torch.optim.SGD(fixed.parameters(), lr=0.06)
    fixed_result = run_continual_experiment(fixed, stream, fixed_opt, config=run_cfg)

    torch.manual_seed(21)
    dense = DenseControlModel(input_dim=8, d_model=12, d_hidden=20, num_classes=6)
    dense_opt = torch.optim.SGD(dense.parameters(), lr=0.06)
    dense_result = run_continual_experiment(dense, stream, dense_opt, config=run_cfg)

    growth_history = [s.recruited_nodes for s in growth_result.structural_history]
    fixed_history = [s.recruited_nodes for s in fixed_result.structural_history]
    assert growth_history == [3, 4, 4]
    assert fixed_history == [2, 2, 2]
    assert growth_result.accuracy_matrix.shape == (3, 3)
    assert fixed_result.accuracy_matrix.shape == (3, 3)
    assert dense_result.accuracy_matrix.shape == (3, 3)

    return {
        "kind": "engineering_smoke_only",
        "growth_recruited": growth_history,
        "fixed_recruited": fixed_history,
        "growth_summary": growth_result.summary(),
        "fixed_summary": fixed_result.summary(),
        "dense_summary": dense_result.summary(),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
