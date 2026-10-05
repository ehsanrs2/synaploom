from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TensorTask:
    """One class/domain-incremental task backed by in-memory tensors."""

    task_id: int
    classes: tuple[int, ...]
    train_x: Tensor
    train_y: Tensor
    eval_x: Tensor
    eval_y: Tensor

    def __post_init__(self) -> None:
        if self.train_x.ndim < 2 or self.eval_x.ndim < 2:
            raise ValueError("features must have batch and feature dimensions")
        if self.train_x.shape[0] != self.train_y.shape[0]:
            raise ValueError("train feature/label length mismatch")
        if self.eval_x.shape[0] != self.eval_y.shape[0]:
            raise ValueError("eval feature/label length mismatch")
        if self.train_x.shape[1:] != self.eval_x.shape[1:]:
            raise ValueError("train/eval feature shapes must match")
        if self.train_y.ndim != 1 or self.eval_y.ndim != 1:
            raise ValueError("labels must be rank-1")
        if not self.classes:
            raise ValueError("task must declare at least one class")


class ContinualStream(Sequence[TensorTask]):
    """Ordered immutable task stream used by the continual harness."""

    def __init__(self, tasks: Sequence[TensorTask]) -> None:
        if not tasks:
            raise ValueError("stream must contain at least one task")
        ids = [task.task_id for task in tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("task ids must be unique")
        feature_shape = tasks[0].train_x.shape[1:]
        if any(task.train_x.shape[1:] != feature_shape for task in tasks):
            raise ValueError("all tasks must share the same feature shape")
        self._tasks = tuple(tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, index: int) -> TensorTask:
        return self._tasks[index]

    @property
    def feature_shape(self) -> torch.Size:
        return self._tasks[0].train_x.shape[1:]

    @property
    def num_classes(self) -> int:
        return len({label for task in self._tasks for label in task.classes})


def iter_tensor_batches(
    x: Tensor,
    y: Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature/label length mismatch")
    n = int(x.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        ids = order[start : start + batch_size]
        yield x.index_select(0, ids), y.index_select(0, ids)
