from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class FeatureBundleAudit:
    path: str
    sha256: str
    train_examples: int
    eval_examples: int
    feature_dim: int
    num_classes: int
    train_examples_per_class: int
    eval_examples_per_class: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def audit_feature_bundle(
    path: str | Path,
    *,
    expected_train_examples: int,
    expected_eval_examples: int,
    expected_feature_dim: int,
    expected_num_classes: int,
    expected_train_per_class: int,
    expected_eval_per_class: int,
) -> FeatureBundleAudit:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "train_features",
        "train_targets",
        "eval_features",
        "eval_targets",
        "metadata",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"feature bundle missing keys: {sorted(missing)}")

    train_x = bundle["train_features"]
    train_y = bundle["train_targets"]
    eval_x = bundle["eval_features"]
    eval_y = bundle["eval_targets"]
    metadata = dict(bundle["metadata"])

    if train_x.ndim != 2 or eval_x.ndim != 2:
        raise ValueError("features must be rank-2 [examples, feature_dim]")
    if train_y.ndim != 1 or eval_y.ndim != 1:
        raise ValueError("targets must be rank-1")
    if train_x.shape[0] != train_y.shape[0] or eval_x.shape[0] != eval_y.shape[0]:
        raise ValueError("feature/target example counts do not match")
    if train_x.shape[1] != eval_x.shape[1]:
        raise ValueError("train/eval feature dimensions differ")
    if not train_x.is_floating_point() or not eval_x.is_floating_point():
        raise ValueError("features must use a floating dtype")
    if train_y.is_floating_point() or eval_y.is_floating_point():
        raise ValueError("targets must use an integer dtype")
    if not bool(torch.isfinite(train_x).all()) or not bool(torch.isfinite(eval_x).all()):
        raise ValueError("feature bundle contains NaN or Inf")

    if int(train_x.shape[0]) != expected_train_examples:
        raise ValueError(
            f"unexpected train example count: {train_x.shape[0]} != {expected_train_examples}"
        )
    if int(eval_x.shape[0]) != expected_eval_examples:
        raise ValueError(
            f"unexpected eval example count: {eval_x.shape[0]} != {expected_eval_examples}"
        )
    if int(train_x.shape[1]) != expected_feature_dim:
        raise ValueError(
            f"unexpected feature dim: {train_x.shape[1]} != {expected_feature_dim}"
        )

    expected_classes = torch.arange(expected_num_classes, dtype=torch.long)
    train_classes, train_counts = torch.unique(train_y.long(), sorted=True, return_counts=True)
    eval_classes, eval_counts = torch.unique(eval_y.long(), sorted=True, return_counts=True)
    if not torch.equal(train_classes, expected_classes):
        raise ValueError("train targets do not contain exactly the expected class ids")
    if not torch.equal(eval_classes, expected_classes):
        raise ValueError("eval targets do not contain exactly the expected class ids")
    if not bool((train_counts == expected_train_per_class).all()):
        raise ValueError("unexpected per-class train counts")
    if not bool((eval_counts == expected_eval_per_class).all()):
        raise ValueError("unexpected per-class eval counts")

    metadata_expectations = {
        "dataset": "CIFAR100",
        "feature_extractor": "torchvision.resnet18",
        "weights": "IMAGENET1K_V1",
        "feature_dim": expected_feature_dim,
        "train_examples": expected_train_examples,
        "eval_examples": expected_eval_examples,
    }
    for key, expected in metadata_expectations.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"metadata mismatch for {key}: {metadata.get(key)!r} != {expected!r}"
            )

    return FeatureBundleAudit(
        path=str(path),
        sha256=sha256_file(path),
        train_examples=int(train_x.shape[0]),
        eval_examples=int(eval_x.shape[0]),
        feature_dim=int(train_x.shape[1]),
        num_classes=expected_num_classes,
        train_examples_per_class=expected_train_per_class,
        eval_examples_per_class=expected_eval_per_class,
        metadata=metadata,
    )


def audit_h1a_feature_bundle(path: str | Path) -> FeatureBundleAudit:
    return audit_feature_bundle(
        path,
        expected_train_examples=50_000,
        expected_eval_examples=10_000,
        expected_feature_dim=512,
        expected_num_classes=100,
        expected_train_per_class=500,
        expected_eval_per_class=100,
    )
