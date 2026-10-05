from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from synaploom.benchmarks.h1 import default_h1_protocol
from synaploom.benchmarks.models import H1ModelShape
from synaploom.benchmarks.runner import aggregate_runs, build_dense_matches, run_h1_suite, save_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="M1.3 H1 benchmark: recruitment-only Split CIFAR-100.")
    parser.add_argument("--root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/m1_3_h1.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = default_h1_protocol()
    shape = H1ModelShape()
    if args.dry_run:
        print(json.dumps({
            "protocol": protocol.to_dict(),
            "shape": asdict(shape),
            "dense_matches": build_dense_matches(shape),
        }, indent=2, sort_keys=True))
        return

    runs = run_h1_suite(
        args.root,
        protocol=protocol,
        shape=shape,
        download=args.download,
    )
    save_suite(runs, args.artifact)
    print(json.dumps(aggregate_runs(runs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
