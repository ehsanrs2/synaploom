from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from synaploom.data.features import make_split_feature_stream
from synaploom.protocols.h1a import H1A_CLASS_ORDER, H1A_VARIANTS, H1AProtocol, build_h1a_variant
from synaploom.research.budgets import (
    count_trainable_parameters,
    estimate_dense_linear_macs_per_batch,
    estimate_synaploom_linear_macs_per_batch,
    synaploom_budget,
)
from synaploom.training.continual import run_continual_experiment, save_continual_result


def _aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and row[key] is not None]
        if not values:
            continue
        out[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": float(len(values)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="M1.3 H1a mechanism-isolation benchmark.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/m1_3_h1a"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()

    protocol = H1AProtocol()
    seeds = tuple(args.seeds) if args.seeds else protocol.seeds
    variants = tuple(args.variants) if args.variants else H1A_VARIANTS
    unknown = set(variants).difference(H1A_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")

    stream = make_split_feature_stream(
        args.features,
        class_order=H1A_CLASS_ORDER,
        classes_per_task=protocol.classes_per_task,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    aggregate_rows: dict[str, list[dict[str, float]]] = {variant: [] for variant in variants}

    for seed in seeds:
        for variant in variants:
            model, optimizer, engine, build_meta = build_h1a_variant(
                variant,
                stream=stream,
                seed=seed,
                protocol=protocol,
            )
            run_config = protocol.continual_config(seed=seed, device=args.device)

            if hasattr(model, "graph_state"):
                initial_budget = synaploom_budget(model)
                initial_macs = estimate_synaploom_linear_macs_per_batch(
                    model, batch_size=protocol.batch_size
                )
                dense_width = None
            else:
                initial_budget = None
                dense_width = int(build_meta["dense_hidden_width"])
                initial_macs = estimate_dense_linear_macs_per_batch(
                    input_dim=protocol.feature_dim,
                    d_model=protocol.d_model,
                    d_hidden=dense_width,
                    num_classes=protocol.num_classes,
                    batch_size=protocol.batch_size,
                )

            result = run_continual_experiment(
                model,
                stream,
                optimizer,
                config=run_config,
                engine=engine,
                metadata={
                    "milestone": "M1.3",
                    "hypothesis": "H1a-controlled-reserve-recruitment",
                    "variant": variant,
                    "seed": seed,
                    "protocol": protocol.as_dict(),
                    "class_order": list(H1A_CLASS_ORDER),
                    "feature_bundle": str(args.features),
                    "build": build_meta,
                    "initial_total_parameters": count_trainable_parameters(model),
                    "initial_linear_macs_per_batch_estimate": initial_macs,
                },
            )

            if hasattr(model, "graph_state"):
                result.metadata["final_budget"] = synaploom_budget(model).__dict__
                result.metadata["final_linear_macs_per_batch_estimate"] = (
                    estimate_synaploom_linear_macs_per_batch(
                        model, batch_size=protocol.batch_size
                    )
                )

            path = args.output / f"{variant}__seed-{seed}.json"
            save_continual_result(result, path)

            summary = result.summary()
            scalar_summary = {
                key: float(value)
                for key, value in summary.items()
                if isinstance(value, (int, float)) and value is not None
            }
            aggregate_rows[variant].append(scalar_summary)
            print(json.dumps({
                "variant": variant,
                "seed": seed,
                "result": str(path),
                "summary": scalar_summary,
            }, sort_keys=True))

    aggregate = {
        "milestone": "M1.3",
        "hypothesis": "H1a-controlled-reserve-recruitment",
        "protocol": protocol.as_dict(),
        "class_order": list(H1A_CLASS_ORDER),
        "variants": {
            variant: _aggregate(rows)
            for variant, rows in aggregate_rows.items()
        },
    }
    aggregate_path = args.output / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"aggregate": str(aggregate_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
