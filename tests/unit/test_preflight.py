from pathlib import Path

import pytest
import torch

from synaploom.research.preflight import audit_feature_bundle, sha256_file


def _write_bundle(path: Path, *, bad_count: bool = False) -> None:
    classes = 3
    train_per = 2
    eval_per = 1
    train_y = torch.arange(classes).repeat_interleave(train_per)
    eval_y = torch.arange(classes).repeat_interleave(eval_per)
    if bad_count:
        train_y = train_y[:-1]
    train_x = torch.arange(train_y.numel() * 4, dtype=torch.float32).reshape(-1, 4)
    eval_x = torch.arange(eval_y.numel() * 4, dtype=torch.float32).reshape(-1, 4)
    torch.save(
        {
            "train_features": train_x,
            "train_targets": train_y,
            "eval_features": eval_x,
            "eval_targets": eval_y,
            "metadata": {
                "dataset": "CIFAR100",
                "feature_extractor": "torchvision.resnet18",
                "weights": "IMAGENET1K_V1",
                "feature_dim": 4,
                "train_examples": 6,
                "eval_examples": 3,
            },
        },
        path,
    )


def test_feature_bundle_audit_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "features.pt"
    _write_bundle(path)
    audit = audit_feature_bundle(
        path,
        expected_train_examples=6,
        expected_eval_examples=3,
        expected_feature_dim=4,
        expected_num_classes=3,
        expected_train_per_class=2,
        expected_eval_per_class=1,
    )
    assert audit.sha256 == sha256_file(path)
    assert audit.num_classes == 3
    assert len(audit.sha256) == 64


def test_feature_bundle_audit_rejects_class_count_corruption(tmp_path: Path) -> None:
    path = tmp_path / "features.pt"
    _write_bundle(path, bad_count=True)
    with pytest.raises(ValueError):
        audit_feature_bundle(
            path,
            expected_train_examples=5,
            expected_eval_examples=3,
            expected_feature_dim=4,
            expected_num_classes=3,
            expected_train_per_class=2,
            expected_eval_per_class=1,
        )
