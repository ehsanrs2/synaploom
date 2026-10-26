"""Explicit tiny CPU synthetic M1.3b ENGINEERING smoke only.

This is deliberately a different CLI from m13b_external_preflight.py.
It cannot load CIFAR features, choose science seeds, use CUDA or execute the
proposed 30/90 trajectories.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from synaploom.research.m13b_synthetic_pilot import (
    pilot_canonical_json,
    run_m13b_synthetic_engineering_pilot,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Only fixed three-arm CPU Gaussian engineering pilot; no CIFAR science."
    )
    parser.add_argument("--synthetic-engineering-pilot", action="store_true",
                        help="Execute only the registered 3x2 Gaussian toy smoke.")
    parser.add_argument("--acknowledge-engineering-only", action="store_true",
                        help="Acknowledge synthetic metrics are not research results.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optionally create a NEW directory to preserve pilot JSON.")
    args = parser.parse_args(argv)
    if not args.synthetic_engineering_pilot or not args.acknowledge_engineering_only:
        parser.error("both --synthetic-engineering-pilot and --acknowledge-engineering-only are required")

    if args.output_dir is not None and args.output_dir.exists():
        parser.error("output directory already exists; cannot overwrite evidence")

    result = run_m13b_synthetic_engineering_pilot()
    payload = pilot_canonical_json(result)
    digest = sha256(payload).hexdigest()
    if args.output_dir is not None:
        # Exclusive output path: never mix engineering smoke evidence with an
        # existing experiment/confirmatory result directory.
        args.output_dir.mkdir(parents=False, exist_ok=False)
        with (args.output_dir / "engineering-pilot.json").open("x", encoding="utf-8") as out:
            out.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        with (args.output_dir / "engineering-pilot.sha256").open("x", encoding="ascii") as out:
            out.write(digest + "  engineering-pilot.canonical-json\n")
    print(json.dumps({"sha256_canonical_json": digest, **result},
                     indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
