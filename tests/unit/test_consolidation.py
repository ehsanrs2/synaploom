import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.plasticity.consolidation import scale_microcircuit_gradients


def test_consolidated_node_gradient_scaling() -> None:
    graph = GraphConfig(
        num_nodes=3,
        initial_recruited=2,
        top_k=1,
        d_model=4,
        d_hidden=8,
        router_dim=4,
        consolidated_plasticity=0.2,
    )
    model = SynapLoomModel(ModelConfig(graph=graph, input_dim=4, num_classes=2))
    model.graph_state.consolidate([0])

    for module in model.executor.microcircuits:
        for parameter in module.parameters():
            parameter.grad = torch.ones_like(parameter)

    scale_microcircuit_gradients(model.executor, model.graph_state)

    for parameter in model.executor.microcircuits[0].parameters():
        assert torch.allclose(parameter.grad, torch.full_like(parameter.grad, 0.2))
    for parameter in model.executor.microcircuits[1].parameters():
        assert torch.allclose(parameter.grad, torch.ones_like(parameter.grad))
    for parameter in model.executor.microcircuits[2].parameters():
        assert torch.count_nonzero(parameter.grad).item() == 0
