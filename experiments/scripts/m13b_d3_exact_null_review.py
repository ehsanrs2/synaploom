"""Offline D3 finite-sample mathematical counterexample; no science/GPU access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from synaploom.research.m13b_d3_exact_null_review import build_exact_null_review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Read-only, exact synthetic n=10 zero-mean percentile-bootstrap "
        "counterexample. Never analyzes SynapLoom outcomes or runs training."
    ))
    parser.add_argument(
        "--exact-synthetic-null-review", action="store_true", required=True,
        help="Emit the fixed synthetic zero-mean mathematical counterexample."
    )
    parser.add_argument(
        "--output", type=Path,
        help="Create a NEW output JSON file only (exclusive, never overwrite)."
    )
    args = parser.parse_args(argv)
    report = build_exact_null_review()
    serial = json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(serial)
    print(serial, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
