"""Generate/verify an UNAPPROVED M1.3b registration *candidate* (offline only).

No training mode, feature-file path, CUDA/device choice or write-back to old
experiments. --review-plan permits exclusive new JSON file creation only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from synaploom.research.m13b_prereg_candidate import (
    build_m13b_prereg_candidate, verify_m13b_prereg_candidate,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Offline proposed 90-slot M1.3b registration, never an approved GPU job."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review-plan", action="store_true",
                      help="Generate the complete non-executable candidate plan.")
    mode.add_argument("--verify-plan", type=Path,
                      help="Check an existing candidate JSON against the exact proposal template.")
    p.add_argument("--output", type=Path,
                   help="Write the reviewed plan to a NEW JSON file; refuse overwrite.")
    args = p.parse_args(argv)
    if args.verify_plan is not None:
        if args.output is not None:
            p.error("--verify-plan forbids --output")
        candidate = json.loads(args.verify_plan.read_text(encoding="utf-8"))
        verify_m13b_prereg_candidate(candidate)
        print(json.dumps({
            "verification": "MATCHES_UNAPPROVED_CANDIDATE_TEMPLATE",
            "owner_approval_present": False,
            "science_executable": False,
            "plan_sha256": candidate["record_sha256"],
        }, sort_keys=True))
        return 0

    candidate = build_m13b_prereg_candidate()
    verify_m13b_prereg_candidate(candidate)
    blob = json.dumps(candidate, sort_keys=True, indent=2,
                      ensure_ascii=True, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(blob)
    print(blob, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
