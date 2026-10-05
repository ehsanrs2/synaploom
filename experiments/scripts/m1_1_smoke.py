from __future__ import annotations

import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import ConnectToSelectedTopR, FixedScheduleRecruitment
from synaploom.training.engine import TrainingEngine


def main() -> None:
    torch.manual_seed(2026)
    graph = GraphConfig(
        num_nodes=8,
        initial_recruited=2,
        top_k=2,
        q_max=2,
        d_model=16,
        d_hidden=32,
        router_dim=16,
        message_rounds=1,
    )
    model = SynapLoomModel(ModelConfig(graph=graph, input_dim=16, num_classes=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    controller = StructuralController(
        recruitment_policy=FixedScheduleRecruitment([3, 6, 9]),
        edge_init_policy=ConnectToSelectedTopR(1),
    )
    engine = TrainingEngine(model, optimizer, controller=controller)

    x = torch.randn(32, 16)
    y = (x[:, 0] + 0.25 * x[:, 1] > 0).long()

    for step in range(1, 10):
        result = engine.train_step(x, y, step=step)
        if result.events:
            summary = ", ".join(
                f"{event.kind}:{event.node_id if event.node_id is not None else event.dst}"
                for event in result.events
            )
            print(
                f"step={step} loss={result.loss:.4f} "
                f"recruited={model.graph_state.recruited_count()} events={summary}"
            )

    assert model.graph_state.recruited_count() == 5
    assert model.graph_state.active_edge_count() == 3
    print(
        "M1.1 smoke passed:",
        f"recruited={model.graph_state.recruited_count()}",
        f"edges={model.graph_state.active_edge_count()}",
        f"version={int(model.graph_state.structural_version.item())}",
    )


if __name__ == "__main__":
    main()
