"""Explicit *synthetic only* M1.3b inferential contract demonstration.

No real-results file, feature path, GPU, owner approval, science run or
preregistration freeze is accepted by this CLI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from synaploom.research.m13b_statistical_dryrun import (
    SCENARIOS, analyze_synthetic_fixture, build_synthetic_fixture,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulate proposed M1.3b inference on invented metrics ONLY; not science."
    )
    parser.add_argument("--synthetic-self-test", action="store_true",
                        help="Validate all 90 toy slots and demonstrate synthetic-only paired CIs.")
    parser.add_argument("--scenario", choices=SCENARIOS, default="positive",
                        help="Only invented toy cases: positive, negative, null, opposed.")
    parser.add_argument("--output", type=Path,
                        help="Optionally create a NEW dry-run report; never overwrite existing results.")
    args = parser.parse_args(argv)
    if not args.synthetic_self_test:
        parser.error("--synthetic-self-test is required; no real result inference exists")
    report = analyze_synthetic_fixture(build_synthetic_fixture(args.scenario))
    blob = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(blob)
    print(blob, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
