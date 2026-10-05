from __future__ import annotations

from pathlib import Path

import torch

from synaploom.data.streams import ContinualStream, TensorTask
from synaploom.vision import normalize_cifar100


def cifar100_class_order(seed: int) -> tuple[int, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return tuple(int(v) for v in torch.randperm(100, generator=generator).tolist())


def make_split_cifar100_stream(
    root: str | Path,
    *,
    num_tasks: int = 10,
    class_order_seed: int = 1701,
    download: bool = False,
    normalize: bool = True,
) -> ContinualStream:
    """10-step class-incremental CIFAR-100 stream for M1.3.

    No stochastic augmentation is applied in protocol v1. All models see exactly
    the same normalized tensors and global class labels.
    """
    if num_tasks <= 0 or 100 % num_tasks != 0:
        raise ValueError("num_tasks must divide 100")
    try:
        from torchvision.datasets import CIFAR100
    except ImportError as exc:
        raise RuntimeError(
            'CIFAR-100 benchmark requires: pip install -e ".[vision]"'
        ) from exc

    train = CIFAR100(root=str(Path(root)), train=True, download=download)
    test = CIFAR100(root=str(Path(root)), train=False, download=download)

    train_x = torch.from_numpy(train.data).permute(0, 3, 1, 2).float().div_(255.0)
    train_y = torch.tensor(train.targets, dtype=torch.long)
    test_x = torch.from_numpy(test.data).permute(0, 3, 1, 2).float().div_(255.0)
    test_y = torch.tensor(test.targets, dtype=torch.long)
    if normalize:
        train_x = normalize_cifar100(train_x)
        test_x = normalize_cifar100(test_x)

    order = cifar100_class_order(class_order_seed)
    classes_per_task = 100 // num_tasks
    tasks: list[TensorTask] = []
    for task_id in range(num_tasks):
        classes = order[
            task_id * classes_per_task : (task_id + 1) * classes_per_task
        ]
        class_tensor = torch.tensor(classes, dtype=torch.long)
        train_mask = torch.isin(train_y, class_tensor)
        test_mask = torch.isin(test_y, class_tensor)
        tasks.append(
            TensorTask(
                task_id=task_id,
                classes=tuple(classes),
                train_x=train_x[train_mask],
                train_y=train_y[train_mask],
                eval_x=test_x[test_mask],
                eval_y=test_y[test_mask],
            )
        )
    return ContinualStream(tasks)
