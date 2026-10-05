from __future__ import annotations

import argparse
import json
from pathlib import Path

from synaploom.protocols.h1a import H1A_CLASS_ORDER, H1AProtocol
from synaploom.research.preflight import audit_h1a_feature_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the frozen feature artifact before EXP-H1A-001."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/cifar100_resnet18_imagenet1k_v1.pt"),
    )
    args = parser.parse_args()

    audit = audit_h1a_feature_bundle(args.features)
    protocol = H1AProtocol()
    if len(H1A_CLASS_ORDER) != protocol.num_classes:
        raise RuntimeError("frozen H1a class order length does not match protocol")
    if len(set(H1A_CLASS_ORDER)) != protocol.num_classes:
        raise RuntimeError("frozen H1a class order is not a permutation")
    if sorted(H1A_CLASS_ORDER) != list(range(protocol.num_classes)):
        raise RuntimeError("frozen H1a class order does not cover CIFAR-100 exactly")

    print(
        json.dumps(
            {
                "status": "ready",
                "feature_bundle": audit.to_dict(),
                "protocol": protocol.as_dict(),
                "class_order_sha": __import__("hashlib").sha256(
                    bytes(H1A_CLASS_ORDER)
                ).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
