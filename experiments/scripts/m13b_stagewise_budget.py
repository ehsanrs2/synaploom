"""Read-only analytical M1.3b stagewise resource audit: never trains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from synaploom.research.m13b_stagewise_budget import build_m13b_stagewise_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="No-data, CPU analytical H1a2/M1.3b MAC budget audit only."
    )
    parser.add_argument("--audit", action="store_true", help="Calculate stagewise mathematical MAC estimates.")
    parser.add_argument("--output", type=Path, help="Optional NEW JSON file; never overwrites existing evidence.")
    args = parser.parse_args(argv)
    if not args.audit:
        parser.error("--audit must be provided")
    result = build_m13b_stagewise_audit()
    printable = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(printable)
    print(printable, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
