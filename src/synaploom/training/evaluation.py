from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from synaploom.data.streams import TensorTask, iter_tensor_batches
from synaploom.training.class_incremental import mask_logits_to_classes


@dataclass(frozen=True)
class TaskEvaluation:
    loss: float
    accuracy: float
    examples: int


def logits_from_output(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, Tensor):
        return logits
    raise TypeError("model output must be a Tensor or expose a Tensor .logits")


@torch.no_grad()
def evaluate_task(
    model: nn.Module,
    task: TensorTask,
    *,
    batch_size: int,
    device: torch.device | str = "cpu",
    allowed_classes: tuple[int, ...] | list[int] | None = None,
) -> TaskEvaluation:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for x, y in iter_tensor_batches(
        task.eval_x, task.eval_y, batch_size=batch_size, shuffle=False, seed=0
    ):
        x, y = x.to(device), y.to(device)
        logits = logits_from_output(model(x))
        if allowed_classes is not None:
            logits = mask_logits_to_classes(logits, allowed_classes)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_examples += int(y.numel())
    if total_examples == 0:
        raise RuntimeError("evaluation task has no examples")
    return TaskEvaluation(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        examples=total_examples,
    )
