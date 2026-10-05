from __future__ import annotations

from pathlib import Path

import torch

from synaploom.data.streams import ContinualStream, TensorTask


def make_split_mnist_stream(
    root: str | Path,
    *,
    class_pairs: tuple[tuple[int, int], ...] = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
    download: bool = False,
    train_limit_per_class: int | None = None,
    eval_limit_per_class: int | None = None,
    seed: int = 0,
) -> ContinualStream:
    """Optional real SplitMNIST adapter.

    torchvision is imported lazily so the core package and CI do not depend on it.
    Labels remain global MNIST class ids, making this a class-incremental stream.
    """
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise RuntimeError(
            'SplitMNIST requires the optional vision dependency: pip install -e ".[vision]"'
        ) from exc

    root = Path(root)
    train = MNIST(root=str(root), train=True, download=download)
    test = MNIST(root=str(root), train=False, download=download)

    def select(
        data: torch.Tensor,
        targets: torch.Tensor,
        classes: tuple[int, ...],
        per_class: int | None,
        local_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for offset, class_id in enumerate(classes):
            ids = torch.nonzero(targets == class_id, as_tuple=False).flatten()
            if per_class is not None:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(local_seed + offset * 1009)
                ids = ids[torch.randperm(ids.numel(), generator=generator)[:per_class]]
            x = data.index_select(0, ids).float().div_(255.0).flatten(1)
            y = targets.index_select(0, ids).long()
            xs.append(x)
            ys.append(y)
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    tasks = []
    for task_id, classes in enumerate(class_pairs):
        train_x, train_y = select(
            train.data, train.targets, classes, train_limit_per_class, seed + task_id * 10000
        )
        eval_x, eval_y = select(
            test.data, test.targets, classes, eval_limit_per_class, seed + 500000 + task_id * 10000
        )
        tasks.append(
            TensorTask(
                task_id=task_id,
                classes=tuple(classes),
                train_x=train_x,
                train_y=train_y,
                eval_x=eval_x,
                eval_y=eval_y,
            )
        )
    return ContinualStream(tasks)
