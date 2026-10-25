"""Research-only, pinned-byte model checkpoint loader for routing evaluation.

This loader deliberately does NOT resume training, optimizer/controller state,
or checkpoint RNG. It accepts only explicitly trusted, independently SHA-pinned
local checkpoint bytes. torch.load(weights_only=False) executes pickle: never
use with untrusted files, even if someone supplies a matching digest.
Production checkpoint APIs are untouched.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from typing import Any, Mapping

import torch
from torch import Tensor

from synaploom.model import SynapLoomModel
from synaploom.research.routing_provenance import (
    RoutingPreflightV1,
    preflight_routing_extra,
)
from synaploom.research.routing_session import RoutingSpec
from synaploom.training.state_digest import model_state_sha256


DEFAULT_MAX_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ModelCheckpointLoadV1:
    checkpoint_sha256: str
    checkpoint_bytes: int
    global_step: int
    routing: RoutingPreflightV1
    loaded_model_sha256: str
    optimizer_restored: bool = False
    controller_restored: bool = False
    rng_restored: bool = False


def _validate_envelope(payload: Any, model: SynapLoomModel) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must be a mapping")
    required = {"model", "global_step", "rng_state", "extra"}
    if not required.issubset(payload):
        raise ValueError("checkpoint envelope is missing required fields")
    if set(payload) - (required | {"optimizer", "controller"}):
        raise ValueError("unknown checkpoint envelope fields")
    if type(payload["global_step"]) is not int or payload["global_step"] < 0:
        raise ValueError("invalid checkpoint global_step")
    if not isinstance(payload["extra"], Mapping) or not isinstance(payload["rng_state"], Mapping):
        raise ValueError("invalid checkpoint extra or RNG envelope")
    for field in ("optimizer", "controller"):
        if field in payload and not isinstance(payload[field], Mapping):
            raise ValueError("invalid checkpoint " + field + " envelope")

    incoming = payload["model"]
    reference = model.state_dict()
    if not isinstance(incoming, Mapping) or set(incoming) != set(reference):
        raise ValueError("checkpoint state_dict keys mismatch")
    for name, current in reference.items():
        supplied = incoming[name]
        if type(supplied) is not Tensor or type(current) is not Tensor:
            raise ValueError("checkpoint state_dict tensor type mismatch: " + name)
        if (supplied.shape != current.shape or supplied.dtype != current.dtype
                or supplied.layout != current.layout or supplied.layout != torch.strided):
            raise ValueError("checkpoint state_dict shape/dtype/layout mismatch: " + name)
    return payload


def load_trusted_routing_model_checkpoint(
    path: str | Path,
    model: SynapLoomModel,
    *,
    requested: RoutingSpec,
    expected_sha256: str,
    trusted_source: bool = False,
    intent: str = "replay",
    allow_counterfactual: bool = False,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ModelCheckpointLoadV1:
    """Load ONLY model state after immutable-byte, semantic and shadow-model gates.

    This is not a general production loader or a true atomic/concurrent transaction.
    For failures before the commit, live model, optimizer, controller and RNG are
    untouched. During commit, a best-effort model-state rollback protects against
    ordinary load/validation errors; hardware/process failure cannot be undone.

    Caller MUST independently trust the file origin AND the supplied digest.
    SHA-256 alone does not authenticate an untrusted source or safe pickle.
    """
    if not isinstance(model, SynapLoomModel):
        raise TypeError("model must be SynapLoomModel")
    if not isinstance(requested, RoutingSpec):
        raise TypeError("requested must be RoutingSpec")
    if not trusted_source:
        raise ValueError("explicit trusted_source=True required; unsafe pickle format")
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected_sha256 must be independently pinned lowercase SHA-256")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")

    with Path(path).open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if not raw or len(raw) > max_bytes:
        raise ValueError("checkpoint file is empty or exceeds byte limit")
    digest = sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("checkpoint byte SHA-256 mismatch")

    # CAUTION: this unpickles code. Do NOT use with untrusted or web-downloaded
    # checkpoints. Load from the same pinned immutable bytes, not the path.
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=False)
    envelope = _validate_envelope(payload, model)

    routing = preflight_routing_extra(
        envelope["extra"],
        model=model,
        requested=requested,
        intent=intent,
        allow_counterfactual=allow_counterfactual,
    )
    if "routing" in envelope["extra"]:
        saved_version = int(envelope["model"]["graph_state.structural_version"].item())
        if saved_version != envelope["extra"]["routing"]["structural_version"]:
            raise ValueError("checkpoint routing structural version differs from stored graph")

    # A detached shadow verifies strict load and graph invariants before any
    # modification of the live model. Gradients/parameters remain separate.
    shadow = copy.deepcopy(model)
    shadow.load_state_dict(envelope["model"], strict=True)
    shadow.graph_state.validate()
    expected_model_digest = model_state_sha256(shadow)

    # Only state_dict tensors can be changed. Never touch optimizer/controller,
    # update stats, restore RNG, change modes, or run an optimizer step.
    snapshot = {name: value.detach().clone()
                for name, value in model.state_dict().items()}
    try:
        model.load_state_dict(envelope["model"], strict=True)
        model.graph_state.validate()
        if model_state_sha256(model) != expected_model_digest:
            raise RuntimeError("loaded model differs from validated shadow")
    except Exception:
        try:
            model.load_state_dict(snapshot, strict=True)
        except Exception as rollback_error:
            raise RuntimeError("checkpoint commit failed AND model rollback failed") from rollback_error
        raise

    return ModelCheckpointLoadV1(
        checkpoint_sha256=digest,
        checkpoint_bytes=len(raw),
        global_step=envelope["global_step"],
        routing=routing,
        loaded_model_sha256=expected_model_digest,
    )
