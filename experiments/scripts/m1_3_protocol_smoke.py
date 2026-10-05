from __future__ import annotations

import json

import torch

from synaploom.data.synthetic import make_gaussian_cluster_stream
from synaploom.protocols.h1a import H1AProtocol, build_h1a_variant
from synaploom.training.continual import run_continual_experiment


def run() -> dict[str, object]:
    protocol = H1AProtocol(
        feature_dim=6,
        num_classes=6,
        classes_per_task=2,
        d_model=8,
        d_hidden=12,
        num_nodes=5,
        initial_recruited=2,
        top_k=2,
        epochs_per_task=1,
        batch_size=4,
        eval_batch_size=8,
        seeds=(3,),
        current_task_eval_every_epochs=1,
    )
    stream = make_gaussian_cluster_stream(
        num_tasks=3,
        classes_per_task=2,
        input_dim=6,
        train_per_class=8,
        eval_per_class=6,
        separation=5.0,
        noise_std=0.5,
        seed=10,
    )

    outputs = {}
    for variant in ("progressive_reserve", "all_recruited", "fixed_small", "dense_active_proxy", "dense_total"):
        model, optimizer, engine, metadata = build_h1a_variant(
            variant, stream=stream, seed=3, protocol=protocol
        )
        result = run_continual_experiment(
            model,
            stream,
            optimizer,
            config=protocol.continual_config(seed=3, device="cpu"),
            engine=engine,
        )
        outputs[variant] = {
            "summary": result.summary(),
            "metadata": metadata,
            "recruited": [s.recruited_nodes for s in result.structural_history],
        }

    assert outputs["progressive_reserve"]["recruited"] == [2, 3, 4]
    assert outputs["all_recruited"]["recruited"] == [5, 5, 5]
    assert outputs["fixed_small"]["recruited"] == [2, 2, 2]
    assert len(outputs["progressive_reserve"]["summary"]["within_task_mean_accuracy_by_task"]) == 3
    return {"kind": "protocol_smoke_only", "variants": outputs}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
