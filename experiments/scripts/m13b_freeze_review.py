"""Offline M1.3b conditional owner design packet and synthetic method diagnostic.

No final freeze, training, science-run authorization, feature paths, or GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from synaploom.research.m13b_freeze_readiness import (
    build_freeze_readiness_packet, verify_freeze_readiness_packet,
)
from synaploom.research.m13b_smalln_method_sensitivity import (
    simulate_n10_method_sensitivity,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only M1.3b owner direction / artificial bootstrap diagnostic."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--readiness", action="store_true",
                      help="Output blocked freeze-readiness candidate, never a science GO.")
    mode.add_argument("--verify-readiness", type=Path,
                      help="Verify a previously generated readiness JSON against exact source.")
    mode.add_argument("--synthetic-method-review", action="store_true",
                      help="Run only bounded Gaussian toy bootstrap sensitivity.")
    parser.add_argument("--output", type=Path,
                        help="Create a NEW JSON file, never overwrite existing evidence.")
    args = parser.parse_args(argv)
    if args.verify_readiness is not None:
        if args.output is not None:
            parser.error("--verify-readiness forbids --output")
        data = json.loads(args.verify_readiness.read_text(encoding="utf-8"))
        verify_freeze_readiness_packet(data)
        print(json.dumps({
            "status": "MATCHES_NONEXECUTABLE_OWNER_REVIEW_PACKET",
            "record_sha256": data["record_sha256"],
            "ready_for_scientific_execution": False,
        }, sort_keys=True))
        return 0

    if args.readiness:
        data = build_freeze_readiness_packet()
        verify_freeze_readiness_packet(data)
    else:
        data = simulate_n10_method_sensitivity()
    body = json.dumps(data, sort_keys=True, indent=2,
                      ensure_ascii=True, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(body)
    print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
