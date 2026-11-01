"""Opt-in, single-arm, train-partition-only M1.3b GPU cost *engineering*.

The default operation is describe-only. An actual calibration needs an
explicit CLI mode, trusted pinned local input and a fresh evidence directory.
No confirmatory seeds, no full-science epoch count or scientific result path.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import torch

from synaploom.research.m13b_trainonly_cuda_pilot import load_trusted_train_only_features
from synaploom.research.m13b_cost_calibration import (
    KIND, TASK_OPTIONS, describe_cost_calibration,
    make_training_partition_calibration_stream,
    run_one_arm_trainonly_cost_calibration,
)
from synaploom.research.m13b_synthetic_pilot import ARMS


def _write_exclusive(path: Path, record: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Bounded manual 1-arm, 2-or-10-task train-only GPU cost calibration."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe-only", action="store_true",
                      help="Show fixed engineering plan; no data, optimizer or CUDA access.")
    mode.add_argument("--manual-trainonly-cost-calibration", action="store_true",
                      help="Run 78 or 390 train-only optimizer steps for ONE arm.")
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--tasks", type=int, choices=TASK_OPTIONS)
    p.add_argument("--features", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--trusted-feature-source", action="store_true")
    p.add_argument("--acknowledge-engineering-no-science", action="store_true")
    args = p.parse_args(argv)

    descriptor = describe_cost_calibration()
    if args.describe_only:
        if (args.arm is not None or args.tasks is not None
                or args.features is not None or args.output_dir is not None
                or args.trusted_feature_source or args.acknowledge_engineering_no_science):
            p.error("--describe-only forbids all execution arguments")
        print(json.dumps(descriptor, indent=2, sort_keys=True))
        return 0

    if (args.arm is None or args.tasks is None or args.features is None
            or args.output_dir is None or not args.trusted_feature_source
            or not args.acknowledge_engineering_no_science):
        p.error("manual calibration requires --arm, --tasks, --features, --output-dir, "
                "--trusted-feature-source, and --acknowledge-engineering-no-science")
    if args.output_dir.exists():
        p.error("existing evidence directory; refusing to overwrite")
    if not args.features.is_file():
        p.error("trusted frozen feature file does not exist")
    if not torch.cuda.is_available():
        p.error("no CUDA device available; calibration not started")

    # Exclusive evidence root reserved before trusted-feature deserialization.
    args.output_dir.mkdir(parents=False, exist_ok=False)
    _write_exclusive(args.output_dir / "intent.json", {
        **descriptor,
        "selected_arm": args.arm,
        "selected_tasks": args.tasks,
        "observed_git_head_not_attested": _git_head(),
        "execution_was_explicitly_invoked": True,
    })
    stage = "feature_trust_gate"
    try:
        x, y, digest = load_trusted_train_only_features(
            args.features, trusted_source=True
        )
        stage = "bounded_train_only_split"
        stream, split_sha = make_training_partition_calibration_stream(
            x, y, tasks=args.tasks
        )
        stage = "single_arm_train_only_cuda_calibration"
        observation = run_one_arm_trainonly_cost_calibration(
            stream, arm=args.arm, tasks=args.tasks,
            frozen_sha256=digest, split_sha256=split_sha,
        )
        stage = "success_evidence"
        _write_exclusive(args.output_dir / "cost-observation.json", observation)
        canonical = json.dumps(
            observation, sort_keys=True, ensure_ascii=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("ascii")
        with (args.output_dir / "cost-observation.canonical-sha256").open("x", encoding="ascii") as f:
            f.write(sha256(canonical).hexdigest() + "\n")
        print(json.dumps({
            "status": "ENGINEERING_COST_OBSERVATION_COMPLETE",
            "scientific_claim": "NONE",
            "arm": args.arm, "tasks": args.tasks,
            "output_dir": str(args.output_dir),
        }, sort_keys=True))
        return 0
    except Exception as exc:
        _write_exclusive(args.output_dir / "failure.json", {
            "status": "ENGINEERING_COST_CALIBRATION_FAILED",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "scientific_claim": "NONE",
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
