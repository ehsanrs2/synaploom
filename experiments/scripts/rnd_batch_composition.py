"""Manual diagnostic entrypoint. Never trains or automatically chooses CUDA."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from synaploom.baselines import DenseControlModel
from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.research.batch_composition import diagnose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if any((args.checkpoint, args.config, args.dataset)) and not all((args.checkpoint, args.config, args.dataset)):
        parser.error("trained diagnostics require checkpoint, config and dataset together")
    # Exclusive directory creation prevents overwriting any evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    results = {}
    if args.checkpoint:
        config = json.loads(args.config.read_text())
        kind = config.pop("kind")
        if kind == "dense":
            model = DenseControlModel(**config)
        elif kind in {"progressive_reserve", "all_recruited"}:
            model = SynapLoomModel(ModelConfig(graph=GraphConfig(**config.pop("graph")), **config))
        else:
            raise ValueError("unknown model kind")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
        dataset = torch.load(args.dataset, map_location="cpu", weights_only=True)
        x, y = dataset["x"], dataset["y"]
        results[kind] = diagnose(model.to(args.device), x.to(args.device), y,
            sample_ids=dataset["sample_ids"], seed=args.seed)
        provenance = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                      (("checkpoint", args.checkpoint), ("config", args.config), ("dataset", args.dataset))}
    else:
        generator = torch.Generator().manual_seed(args.seed)
        x, y = torch.randn(24, 8, generator=generator), torch.arange(24) % 3
        provenance = {"states": "NEWLY_CONSTRUCTED_UNTRAINED_SYNTHETIC"}
        for kind in ("progressive_reserve_early", "progressive_reserve_later", "all_recruited", "dense"):
            torch.manual_seed(args.seed)
            if kind == "dense":
                model = DenseControlModel(input_dim=8, d_model=8, d_hidden=16, num_classes=3)
            else:
                graph = GraphConfig(num_nodes=6, initial_recruited=6 if kind == "all_recruited" else 2,
                                    d_model=8, d_hidden=16, router_dim=8, top_k=2)
                model = SynapLoomModel(ModelConfig(graph=graph, input_dim=8, num_classes=3))
                if kind == "progressive_reserve_later":
                    for node in range(2, 6):
                        model.graph_state.recruit(node)
            results[kind] = diagnose(model.to(args.device), x.to(args.device), y, seed=args.seed)
    (args.output / "results.json").write_text(json.dumps({"provenance": provenance, "results": results}, indent=2))


if __name__ == "__main__":
    main()
