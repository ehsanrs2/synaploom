"""Manual M1.3b CUDA train-only feasibility. Never runs on CI by default.

Acknowledge a deliberately small training-only pilot. It is not the
confirmatory CIFAR-100 experiment and has no --science or --seeds override.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import traceback

import torch

from synaploom.protocols.h1a2 import H1A2_FEATURE_SHA256
from synaploom.research.m13b_trainonly_cuda_pilot import (
    ENGINEERING_SEED, HOLDOUT_PER_CLASS, KIND, MASK_SEED,
    TASKS, TRAIN_PER_CLASS, load_trusted_train_only_features,
    run_m13b_trainonly_cuda_feasibility, train_only_two_task_stream,
)


def _write_new(path: Path, obj: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MANUAL CUDA 3-arm train-only tiny pilot; no full CIFAR science execution."
    )
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--describe-only", action="store_true",
                      help="Show fixed GPU pilot design without opening data or CUDA.")
    kind.add_argument("--manual-gpu-trainonly-pilot", action="store_true",
                      help="Run exactly two tiny training-only tasks on CUDA:0.")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trusted-feature-source", action="store_true")
    parser.add_argument("--acknowledge-no-scientific-claim", action="store_true")
    args = parser.parse_args(argv)

    plan = {
        "kind": KIND, "device": "cuda:0", "training_part_only": True,
        "scientific_claim": "NONE",
        "expected_frozen_feature_sha256": H1A2_FEATURE_SHA256,
        "arms": ["progressive_reserve", "all_recruited", "fixed_sparse_mlp"],
        "tasks": TASKS, "epochs_per_task": 1, "train_per_class": TRAIN_PER_CLASS,
        "holdout_per_class": HOLDOUT_PER_CLASS,
        "engineering_seed": ENGINEERING_SEED, "mask_seed": MASK_SEED,
        "protected_confirmation_seeds_used": False,
        "full_30_or_90_scientific_runs_allowed": False,
    }
    if args.describe_only:
        if (args.features is not None or args.output_dir is not None
                or args.trusted_feature_source or args.acknowledge_no_scientific_claim):
            parser.error("--describe-only forbids execution arguments")
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if (args.features is None or args.output_dir is None
            or not args.trusted_feature_source or not args.acknowledge_no_scientific_claim):
        parser.error("manual pilot requires --features, --output-dir, --trusted-feature-source and --acknowledge-no-scientific-claim")
    if args.output_dir.exists():
        parser.error("output directory already exists; refusing to overwrite pilot evidence")
    if not torch.cuda.is_available():
        parser.error("no CUDA device; manual GPU pilot not started")
    if not args.features.is_file():
        parser.error("trusted frozen feature path does not exist")

    # Reserve an evidence directory before any PyTorch deserialization.
    args.output_dir.mkdir(parents=False, exist_ok=False)
    _write_new(args.output_dir / "intent.json", plan)
    stage = "trusted_data"
    try:
        train_x, train_y, digest = load_trusted_train_only_features(
            args.features, trusted_source=True
        )
        stage = "train_only_split"
        stream, split_digest = train_only_two_task_stream(train_x, train_y)
        stage = "manual_gpu_pilot"

        def record_arm(name: str, record: dict) -> None:
            _write_new(args.output_dir / ("arm-" + name + ".json"), record)

        result = run_m13b_trainonly_cuda_feasibility(
            stream, frozen_data_sha256=digest,
            train_only_split_sha256=split_digest,
            on_arm_complete=record_arm,
        )
        stage = "finalize"
        _write_new(args.output_dir / "pilot.json", result)
        canonical = json.dumps(
            result, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        with (args.output_dir / "pilot.canonical-sha256").open("x", encoding="ascii") as handle:
            handle.write(sha256(canonical).hexdigest() + "\n")
        print(json.dumps({"status": "GPU_TRAIN_ONLY_ENGINEERING_COMPLETE",
                          "output_dir": str(args.output_dir),
                          "kind": KIND}, sort_keys=True))
        return 0
    except Exception as exc:
        _write_new(args.output_dir / "failure.json", {
            "status": "ENGINEERING_PILOT_FAILED", "failed_stage": stage,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:2000],
            "scientific_claim": "NONE",
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
