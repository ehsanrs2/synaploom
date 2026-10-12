from __future__ import annotations

import hashlib
import json

import torch
from torch import nn


def model_state_sha256(model: nn.Module) -> str:
    """Hash every state_dict tensor using the frozen canonical encoding.

    Entries are sorted by name. Each entry contributes its UTF-8 name, ASCII
    dtype, compact-JSON shape, and raw contiguous CPU tensor bytes. NUL bytes
    delimit the metadata fields so the digest is unambiguous. Sparse or
    unsupported state entries fail closed rather than being silently omitted.
    """
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"unsupported state_dict entry {name!r}: expected Tensor, "
                f"got {type(value).__name__}"
            )
        if value.layout != torch.strided:
            raise TypeError(
                f"unsupported state_dict tensor layout for {name!r}: {value.layout}"
            )
        tensor = value.detach().to(device="cpu").contiguous()
        try:
            raw_bytes = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        except Exception as exc:
            raise TypeError(
                f"cannot encode state_dict tensor {name!r} for SHA-256"
            ) from exc

        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(raw_bytes)
    return digest.hexdigest()
