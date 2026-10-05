from __future__ import annotations

import math

import torch
from torch import Tensor


def _as_float_matrix(matrix: Tensor) -> Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("accuracy matrix must be square [num_stages, num_tasks]")
    return matrix.float()


def average_accuracy_by_stage(matrix: Tensor) -> Tensor:
    """Mean accuracy over tasks observed up to each training stage."""
    matrix = _as_float_matrix(matrix)
    values = []
    for stage in range(matrix.shape[0]):
        row = matrix[stage, : stage + 1]
        values.append(torch.nanmean(row))
    return torch.stack(values)


def final_average_accuracy(matrix: Tensor) -> float:
    matrix = _as_float_matrix(matrix)
    return float(torch.nanmean(matrix[-1]).item())


def forgetting_per_task(matrix: Tensor) -> Tensor:
    """Final forgetting for each old task; last/current task is NaN."""
    matrix = _as_float_matrix(matrix)
    num_tasks = matrix.shape[0]
    result = torch.full((num_tasks,), float("nan"))
    final = matrix[-1]
    for task in range(num_tasks - 1):
        history = matrix[task:, task]
        best = torch.nanmax(history[:-1]) if history.numel() > 1 else history[0]
        result[task] = best - final[task]
    return result


def average_forgetting(matrix: Tensor) -> float:
    values = forgetting_per_task(matrix)
    valid = values[~torch.isnan(values)]
    return 0.0 if valid.numel() == 0 else float(valid.mean().item())


def diagonal_learning_accuracy(matrix: Tensor) -> Tensor:
    """Accuracy on each task immediately after learning that task."""
    matrix = _as_float_matrix(matrix)
    return torch.diagonal(matrix)


def late_task_learning_accuracy(matrix: Tensor, *, last_n: int = 2) -> float:
    if last_n <= 0:
        raise ValueError("last_n must be positive")
    diagonal = diagonal_learning_accuracy(matrix)
    n = min(last_n, diagonal.numel())
    return float(torch.nanmean(diagonal[-n:]).item())


def backward_transfer(matrix: Tensor) -> float:
    """Mean final minus post-learning accuracy on old tasks."""
    matrix = _as_float_matrix(matrix)
    if matrix.shape[0] <= 1:
        return 0.0
    diagonal = torch.diagonal(matrix)
    values = matrix[-1, :-1] - diagonal[:-1]
    return float(torch.nanmean(values).item())


def matrix_to_jsonable(matrix: Tensor) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    for row in matrix.detach().cpu().tolist():
        rows.append([
            None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
            for value in row
        ])
    return rows
