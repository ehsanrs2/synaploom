from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class NumericalContext:
    stage: int | None = None
    task_id: int | None = None
    epoch: int | None = None
    batch_index: int | None = None
    global_step: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


class NumericalInstabilityError(RuntimeError):
    """Raised at the first detected NaN/Inf during training or evaluation."""

    def __init__(
        self,
        *,
        phase: str,
        tensor_name: str,
        context: NumericalContext | None = None,
        parameter_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.phase = phase
        self.tensor_name = tensor_name
        self.context = context or NumericalContext()
        self.parameter_name = parameter_name
        self.details = dict(details or {})

        location = ", ".join(
            f"{key}={value}"
            for key, value in self.context.to_dict().items()
            if value is not None
        )
        message = f"non-finite numerical value detected: phase={phase}, tensor={tensor_name}"
        if self.parameter_name is not None:
            message += f", parameter={self.parameter_name}"
        if location:
            message += f", {location}"
        if self.details:
            message += f", details={self.details}"
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "phase": self.phase,
            "tensor_name": self.tensor_name,
            "parameter_name": self.parameter_name,
            "context": self.context.to_dict(),
            "details": self.details,
            "message": str(self),
        }


def _tensor_details(tensor: Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    nonfinite_count = int((~finite).sum().item())
    finite_values = detached[finite]
    max_abs_finite = (
        float(finite_values.abs().max().item())
        if finite_values.numel() > 0
        else None
    )
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "nonfinite_count": nonfinite_count,
        "max_abs_finite": max_abs_finite,
    }


def ensure_finite_tensor(
    tensor: Tensor,
    *,
    phase: str,
    tensor_name: str,
    context: NumericalContext | None = None,
    parameter_name: str | None = None,
) -> None:
    if bool(torch.isfinite(tensor.detach()).all()):
        return
    raise NumericalInstabilityError(
        phase=phase,
        tensor_name=tensor_name,
        context=context,
        parameter_name=parameter_name,
        details=_tensor_details(tensor),
    )


@dataclass(frozen=True)
class GradientDiagnostics:
    total_l2_norm: float
    max_abs: float
    max_abs_parameter: str | None


@dataclass(frozen=True)
class ParameterDiagnostics:
    max_abs: float
    max_abs_parameter: str | None


def check_model_gradients(
    model: nn.Module,
    *,
    phase: str,
    context: NumericalContext | None = None,
) -> GradientDiagnostics:
    sum_sq = 0.0
    max_abs = 0.0
    max_name: str | None = None

    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        ensure_finite_tensor(
            grad,
            phase=phase,
            tensor_name="gradient",
            context=context,
            parameter_name=name,
        )
        norm = float(torch.linalg.vector_norm(grad.detach()).item())
        local_max = float(grad.detach().abs().max().item()) if grad.numel() else 0.0
        sum_sq += norm * norm
        if local_max > max_abs:
            max_abs = local_max
            max_name = name

    total = math.sqrt(sum_sq)
    if not math.isfinite(total):
        raise NumericalInstabilityError(
            phase=phase,
            tensor_name="gradient_total_l2_norm",
            context=context,
            parameter_name=max_name,
            details={"total_l2_norm": total, "max_abs": max_abs},
        )
    return GradientDiagnostics(
        total_l2_norm=total,
        max_abs=max_abs,
        max_abs_parameter=max_name,
    )


def check_model_parameters(
    model: nn.Module,
    *,
    phase: str,
    context: NumericalContext | None = None,
) -> ParameterDiagnostics:
    max_abs = 0.0
    max_name: str | None = None
    for name, parameter in model.named_parameters():
        ensure_finite_tensor(
            parameter,
            phase=phase,
            tensor_name="parameter",
            context=context,
            parameter_name=name,
        )
        local_max = (
            float(parameter.detach().abs().max().item())
            if parameter.numel()
            else 0.0
        )
        if local_max > max_abs:
            max_abs = local_max
            max_name = name
    return ParameterDiagnostics(max_abs=max_abs, max_abs_parameter=max_name)
