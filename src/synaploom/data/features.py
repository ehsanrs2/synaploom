from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from synaploom.data.streams import ContinualStream, TensorTask


def load_feature_bundle(path: str | Path) -> dict[str, Any]:
    bundle = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"train_features", "train_targets", "eval_features", "eval_targets"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"feature bundle missing keys: {sorted(missing)}")
    return bundle


def make_split_feature_stream(
    path: str | Path,
    *,
    class_order: tuple[int, ...],
    classes_per_task: int = 10,
) -> ContinualStream:
    """Create a class-incremental stream from a frozen feature bundle."""
    if classes_per_task <= 0:
        raise ValueError("classes_per_task must be positive")
    if len(class_order) % classes_per_task != 0:
        raise ValueError("class_order length must be divisible by classes_per_task")
    if len(set(class_order)) != len(class_order):
        raise ValueError("class_order must contain unique classes")

    bundle = load_feature_bundle(path)
    train_x = bundle["train_features"].float()
    train_y = bundle["train_targets"].long()
    eval_x = bundle["eval_features"].float()
    eval_y = bundle["eval_targets"].long()

    available = set(int(v) for v in torch.cat([train_y, eval_y]).unique().tolist())
    requested = set(int(v) for v in class_order)
    if not requested.issubset(available):
        raise ValueError("class_order requests classes missing from the feature bundle")

    tasks: list[TensorTask] = []
    for task_id, start in enumerate(range(0, len(class_order), classes_per_task)):
        classes = tuple(int(v) for v in class_order[start : start + classes_per_task])
        train_mask = torch.zeros_like(train_y, dtype=torch.bool)
        eval_mask = torch.zeros_like(eval_y, dtype=torch.bool)
        for class_id in classes:
            train_mask |= train_y == class_id
            eval_mask |= eval_y == class_id
        tasks.append(
            TensorTask(
                task_id=task_id,
                classes=classes,
                train_x=train_x[train_mask],
                train_y=train_y[train_mask],
                eval_x=eval_x[eval_mask],
                eval_y=eval_y[eval_mask],
            )
        )
    return ContinualStream(tasks)
