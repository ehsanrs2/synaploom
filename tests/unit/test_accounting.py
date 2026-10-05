import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.training.metrics import parameter_accounting, structural_counts


def test_structural_and_parameter_accounting_separates_capacity_levels() -> None:
    graph = GraphConfig(num_nodes=6, initial_recruited=3, top_k=2, d_model=8, d_hidden=16, router_dim=8)
    model = SynapLoomModel(ModelConfig(graph=graph, input_dim=8, num_classes=3))
    selected = torch.tensor([0, 2])

    counts = structural_counts(model.graph_state, selected)
    params = parameter_accounting(model, selected)

    assert counts.total_nodes == 6
    assert counts.recruited_nodes == 3
    assert counts.executed_nodes == 2
    assert params.total_parameters > params.recruited_microcircuit_parameters > params.executed_microcircuit_parameters > 0
    assert params.total_microcircuit_parameters > params.recruited_microcircuit_parameters
