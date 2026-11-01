"""Offline D3 estimand options only; no scientific data or trained models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from synaploom.research.m13b_d3_estimand_options import (
    build_estimand_options_review,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Review exact n=10 paired-seed sign and bounded mean inference options."
    )
    p.add_argument(
        "--review-estimands", action="store_true", required=True,
        help="Show four prospective inference families, select none, run no science."
    )
    p.add_argument(
        "--output", type=Path,
        help="Optionally create an exclusive NEW JSON review file, never overwrite."
    )
    args = p.parse_args(argv)
    payload = build_estimand_options_review()
    result = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False,
    ) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(result)
    print(result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
