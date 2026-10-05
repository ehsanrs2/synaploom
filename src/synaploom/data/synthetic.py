from __future__ import annotations

import torch

from synaploom.data.streams import ContinualStream, TensorTask


def make_gaussian_cluster_stream(
    *,
    num_tasks: int = 3,
    classes_per_task: int = 2,
    input_dim: int = 8,
    train_per_class: int = 48,
    eval_per_class: int = 48,
    separation: float = 5.0,
    noise_std: float = 0.8,
    seed: int = 0,
) -> ContinualStream:
    """Deterministic class-incremental Gaussian stream for engineering validation."""
    if num_tasks <= 0 or classes_per_task <= 0:
        raise ValueError("num_tasks and classes_per_task must be positive")
    if input_dim <= 0 or train_per_class <= 0 or eval_per_class <= 0:
        raise ValueError("dimensions and sample counts must be positive")
    if separation <= 0 or noise_std <= 0:
        raise ValueError("separation and noise_std must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    total_classes = num_tasks * classes_per_task
    means = torch.randn(total_classes, input_dim, generator=generator)
    means = means / means.norm(dim=1, keepdim=True).clamp_min(1e-8) * separation

    tasks: list[TensorTask] = []
    for task_idx in range(num_tasks):
        classes = tuple(
            range(task_idx * classes_per_task, (task_idx + 1) * classes_per_task)
        )
        train_x, train_y, eval_x, eval_y = [], [], [], []
        for class_id in classes:
            mean = means[class_id]
            train_x.append(
                mean
                + noise_std
                * torch.randn(train_per_class, input_dim, generator=generator)
            )
            train_y.append(torch.full((train_per_class,), class_id, dtype=torch.long))
            eval_x.append(
                mean
                + noise_std
                * torch.randn(eval_per_class, input_dim, generator=generator)
            )
            eval_y.append(torch.full((eval_per_class,), class_id, dtype=torch.long))

        task_train_x = torch.cat(train_x, dim=0)
        task_train_y = torch.cat(train_y, dim=0)
        task_eval_x = torch.cat(eval_x, dim=0)
        task_eval_y = torch.cat(eval_y, dim=0)

        train_perm = torch.randperm(task_train_x.shape[0], generator=generator)
        eval_perm = torch.randperm(task_eval_x.shape[0], generator=generator)
        tasks.append(
            TensorTask(
                task_id=task_idx,
                classes=classes,
                train_x=task_train_x.index_select(0, train_perm),
                train_y=task_train_y.index_select(0, train_perm),
                eval_x=task_eval_x.index_select(0, eval_perm),
                eval_y=task_eval_y.index_select(0, eval_perm),
            )
        )
    return ContinualStream(tasks)
