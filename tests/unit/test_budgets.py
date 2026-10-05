import torch

from synaploom.baselines import DenseControlModel
from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.research.budgets import (
    count_trainable_parameters,
    dense_hidden_width_for_target,
    synaploom_budget,
)


def test_budget_distinguishes_total_and_executed_experts() -> None:
    model = SynapLoomModel(
        ModelConfig(
            graph=GraphConfig(
                num_nodes=6,
                initial_recruited=3,
                top_k=2,
                d_model=8,
                d_hidden=12,
                router_dim=8,
                q_max=0,
                message_rounds=0,
            ),
            input_dim=10,
            num_classes=5,
        )
    )
    budget = synaploom_budget(model)
    assert budget.total_nodes == 6
    assert budget.recruited_nodes == 3
    assert budget.executed_microcircuit_parameters == 2 * budget.per_microcircuit_parameters
    assert budget.total_microcircuit_parameters == 6 * budget.per_microcircuit_parameters
    assert budget.relation_parameters_executed == 0


def test_dense_width_search_matches_target_nearly_exactly() -> None:
    target = 50_000
    width, reported = dense_hidden_width_for_target(
        target,
        input_dim=32,
        d_model=16,
        num_classes=10,
    )
    dense = DenseControlModel(input_dim=32, d_model=16, d_hidden=width, num_classes=10)
    actual = count_trainable_parameters(dense)
    assert actual == reported
    assert abs(actual - target) <= 2 * 16 + 1
