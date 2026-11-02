"""Read-only review of D3 paired-mean estimand and hypothetical n=10 power."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from synaploom.research.m13b_d3_mean_power_planning import (
    build_mean_power_planning_packet,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(
        "Review owner-selected paired MEAN benefit and synthetic normal/known-SD "
        "planning proxies. Does not use research outcomes or authorize science."
    ))
    p.add_argument("--review-mean-power", action="store_true", required=True,
                   help="Emit the fixed non-executable mean power review packet.")
    p.add_argument("--output", type=Path,
                   help="Optionally create a NEW exclusive JSON file; refuse overwrite.")
    args = p.parse_args(argv)
    report = build_mean_power_planning_packet()
    raw = json.dumps(
        report, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(raw)
    print(raw, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
