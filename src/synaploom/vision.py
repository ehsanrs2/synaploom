from __future__ import annotations

import torch
from torch import Tensor, nn


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class CifarConvEncoder(nn.Module):
    """Small shared vision stem for controlled CIFAR experiments."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(96, d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("CifarConvEncoder expects [batch, 3, H, W]")
        return self.projection(self.features(x).flatten(1))


def normalize_cifar100(x: Tensor) -> Tensor:
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("expected [batch, 3, H, W]")
    mean = x.new_tensor(CIFAR100_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(CIFAR100_STD).view(1, 3, 1, 1)
    return (x - mean) / std
