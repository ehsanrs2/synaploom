from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class BudgetReport:
    total_parameters: int
    trainable_parameters: int
    model_bytes_fp32: int


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def budget_report(model: nn.Module) -> BudgetReport:
    total = model_parameter_count(model)
    trainable = trainable_parameter_count(model)
    return BudgetReport(
        total_parameters=total,
        trainable_parameters=trainable,
        model_bytes_fp32=total * 4,
    )


def nearest_dense_hidden_width(
    factory,
    *,
    target_parameters: int,
    minimum: int = 8,
    maximum: int = 4096,
) -> tuple[int, int]:
    """Find a hidden width whose model parameter count is nearest to target.

    factory(width) must return a fresh nn.Module.
    """
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid search bounds")

    lo, hi = minimum, maximum
    best_width = lo
    best_count = model_parameter_count(factory(lo))
    best_error = abs(best_count - target_parameters)

    while lo <= hi:
        mid = (lo + hi) // 2
        count = model_parameter_count(factory(mid))
        error = abs(count - target_parameters)
        if error < best_error:
            best_width, best_count, best_error = mid, count, error
        if count < target_parameters:
            lo = mid + 1
        elif count > target_parameters:
            hi = mid - 1
        else:
            return mid, count

    for width in {max(minimum, best_width - 1), best_width, min(maximum, best_width + 1)}:
        count = model_parameter_count(factory(width))
        error = abs(count - target_parameters)
        if error < best_error:
            best_width, best_count, best_error = width, count, error
    return best_width, best_count
