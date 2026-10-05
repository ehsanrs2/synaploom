from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def mask_logits_to_classes(logits: Tensor, allowed_classes: tuple[int, ...] | list[int]) -> Tensor:
    """Mask logits outside the globally-known class set without task labels."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    allowed = torch.as_tensor(allowed_classes, device=logits.device, dtype=torch.long)
    if allowed.numel() == 0:
        raise ValueError("allowed_classes cannot be empty")
    if int(allowed.min().item()) < 0 or int(allowed.max().item()) >= logits.shape[-1]:
        raise ValueError("allowed class id outside logits range")
    keep = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.bool)
    keep[allowed] = True
    floor = torch.finfo(logits.dtype).min
    return logits.masked_fill(~keep.unsqueeze(0), floor)


def class_incremental_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    allowed_classes: tuple[int, ...] | list[int] | None = None,
) -> Tensor:
    if allowed_classes is not None:
        allowed_set = set(int(v) for v in allowed_classes)
        labels = set(int(v) for v in targets.detach().cpu().unique().tolist())
        if not labels.issubset(allowed_set):
            raise ValueError("targets contain a class outside allowed_classes")
        logits = mask_logits_to_classes(logits, allowed_classes)
    return F.cross_entropy(logits, targets)
