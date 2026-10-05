from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    import numpy as np
except ImportError:
    np = None


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if np is not None:
        state["numpy"] = np.random.get_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if np is not None and "numpy" in state:
        np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    global_step: int = 0,
    extra: dict[str, Any] | None = None,
    controller: Any | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "global_step": int(global_step),
        "rng_state": capture_rng_state(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if controller is not None:
        payload["controller"] = controller.state_dict()
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
    controller: Any | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if controller is not None and "controller" in payload:
        controller.load_state_dict(payload["controller"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return {
        "global_step": int(payload["global_step"]),
        "extra": payload.get("extra", {}),
    }
