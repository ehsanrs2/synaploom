from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.data.vision import make_split_mnist_stream
from synaploom.model import SynapLoomModel
from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import NoConsolidation, NoEdges, TaskBoundaryRecruitment
from synaploom.training.continual import ContinualConfig, run_continual_experiment, save_continual_result
from synaploom.training.engine import TrainingEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Functional SplitMNIST smoke; not a research benchmark.")
    parser.add_argument("--root", type=Path, default=Path("data/mnist"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit-per-class", type=int, default=128)
    parser.add_argument("--artifact", type=Path, default=None)
    args = parser.parse_args()

    stream = make_split_mnist_stream(
        args.root,
        download=args.download,
        train_limit_per_class=args.limit_per_class,
        eval_limit_per_class=min(args.limit_per_class, 128),
        seed=7,
    )
    graph = GraphConfig(
        num_nodes=8,
        initial_recruited=2,
        top_k=2,
        d_model=64,
        d_hidden=96,
        router_dim=64,
        q_max=2,
        message_rounds=0,
    )
    model = SynapLoomModel(ModelConfig(graph=graph, input_dim=784, num_classes=10))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    controller = StructuralController(
        recruitment_policy=TaskBoundaryRecruitment(),
        edge_init_policy=NoEdges(),
        consolidation_policy=NoConsolidation(),
    )
    engine = TrainingEngine(model, optimizer, controller=controller)
    result = run_continual_experiment(
        model,
        stream,
        optimizer,
        config=ContinualConfig(
            epochs_per_task=1,
            batch_size=64,
            eval_batch_size=256,
            seed=7,
        ),
        engine=engine,
    )
    if args.artifact is not None:
        save_continual_result(result, args.artifact)
    print(json.dumps({
        "kind": "functional_smoke_only",
        "recruited_history": [s.recruited_nodes for s in result.structural_history],
        "summary": result.summary(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
