"""M1.3b draft dry-run / trusted local feature preflight — NEVER TRAINS.

Usage:
  python experiments/scripts/m13b_external_preflight.py --dry-run
  python experiments/scripts/m13b_external_preflight.py --dry-run --plan-output /tmp/m13b-plan.json
  python experiments/scripts/m13b_external_preflight.py --verify-features \
    --features path/to/trusted.pt --trusted-source
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from synaploom.research.m13b_manifest import (
    build_feature_verified_manifest,
    build_m13b_draft_manifest,
    verify_manifest_integrity,
    verify_trusted_feature_file,
)


def _commit() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return value if len(value) == 40 and all(
            c in "0123456789abcdef" for c in value
        ) else None
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Draft M1.3b dry-run and trusted local data audit; no training allowed."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true",
                        help="Print only proposed matrices and budget ledger; do not read feature file.")
    action.add_argument("--verify-features", action="store_true",
                        help="Verify SHA first, then inspect explicitly trusted frozen local feature bundle.")
    parser.add_argument("--features", type=Path,
                        help="Required only for --verify-features; never opened by --dry-run.")
    parser.add_argument("--trusted-source", action="store_true",
                        help="Explicitly affirm local artifact origin is trusted for torch.load pickle.")
    parser.add_argument("--plan-output", type=Path,
                        help="Write an optional NEW draft JSON plan; refuse overwrite or missing parent.")
    args = parser.parse_args(argv)
    if args.dry_run and (args.features is not None or args.trusted_source):
        parser.error("--dry-run forbids feature-file options")
    if args.verify_features and (args.features is None or not args.trusted_source):
        parser.error("--verify-features requires --features and --trusted-source")

    source = _commit()
    if args.dry_run:
        plan = build_m13b_draft_manifest(source_commit=source)
    else:
        audit = verify_trusted_feature_file(args.features, trusted_source=True)
        plan = build_feature_verified_manifest(source_commit=source, feature_audit=audit)
    verify_manifest_integrity(plan)

    serialized = json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.plan_output is not None:
        # A plan is NOT an output directory, results directory or approval token.
        # Exclusive mode avoids silently changing an existing evidence artifact.
        with args.plan_output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
