"""Research-only full-envelope shadow audit; NEVER restores live training state.

Trusted pickle warning: SHA-256 checks identity, not origin or code safety.
Caller MUST independently trust the file and expected digest.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import random
import re
from typing import Any, Mapping

import torch

from synaploom.model import SynapLoomModel
from synaploom.plasticity.controller import StructuralController
from synaploom.research.routing_provenance import RoutingPreflightV1, preflight_routing_extra
from synaploom.research.routing_session import RoutingSpec
from synaploom.research.trusted_routing_checkpoint import DEFAULT_MAX_BYTES, _validate_envelope
from synaploom.training.state_digest import model_state_sha256


@dataclass(frozen=True)
class EnvelopeAuditV1:
    checkpoint_sha256: str
    checkpoint_bytes: int
    global_step: int
    routing: RoutingPreflightV1
    shadow_model_sha256: str
    optimizer_groups: int
    optimizer_state_entries: int
    controller_last_applied_step: int
    cpu_rng_checked: bool
    numpy_rng_checked: bool
    cuda_rng_entries: int
    live_state_restored: bool = False
    resume_authorized: bool = False


def _validate_rng(state: Mapping[str, Any]) -> tuple[bool, int]:
    allowed = {"python", "torch_cpu", "numpy", "torch_cuda"}
    if not {"python", "torch_cpu"}.issubset(state) or set(state) - allowed:
        raise ValueError("invalid RNG state keys")
    # Validate in isolated PRNG instances; do not set global RNG state.
    try:
        random.Random().setstate(state["python"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Python RNG state") from exc
    cpu = state["torch_cpu"]
    if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1 or cpu.device.type != "cpu":
        raise ValueError("invalid CPU torch RNG state")
    try:
        torch.Generator(device="cpu").set_state(cpu)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("invalid CPU torch RNG state") from exc
    if "numpy" in state:
        try:
            import numpy as np
            np.random.RandomState().set_state(state["numpy"])
        except (ImportError, TypeError, ValueError) as exc:
            raise ValueError("invalid NumPy RNG state") from exc

    cuda = state.get("torch_cuda", [])
    if not isinstance(cuda, (list, tuple)) or any(
        not isinstance(v, torch.Tensor) or v.dtype != torch.uint8
        or v.ndim != 1 or v.device.type != "cpu" for v in cuda
    ):
        raise ValueError("invalid CUDA RNG state structure")
    # A CPU CI runner cannot verify CUDA RNG generator state restoration.
    # Report count only. Live CUDA-device roundtrip remains a separate gate.
    return "numpy" in state, len(cuda)


def _check_optimizer_links(model: SynapLoomModel, optimizer: torch.optim.Optimizer) -> None:
    model_ids = [id(p) for p in model.parameters()]
    group_ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    if len(group_ids) != len(model_ids) or set(group_ids) != set(model_ids):
        raise ValueError("optimizer must reference exactly the model's parameters")
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("optimizer parameter is repeated in groups")



def _assert_optimizer_tree_equal(saved: Any, restored: Any, path: str = "optimizer") -> None:
    """Require exact CPU-normalized optimizer state *values*, not just counts."""
    if isinstance(saved, torch.Tensor):
        if not isinstance(restored, torch.Tensor) or (
            saved.shape != restored.shape or saved.dtype != restored.dtype
            or not torch.equal(saved.detach().cpu(), restored.detach().cpu())
        ):
            raise ValueError("optimizer state value mismatch: " + path)
        return
    if isinstance(saved, Mapping):
        if not isinstance(restored, Mapping) or set(saved) != set(restored):
            raise ValueError("optimizer state keys mismatch: " + path)
        for key in saved:
            _assert_optimizer_tree_equal(saved[key], restored[key], path + "." + str(key))
        return
    if isinstance(saved, (list, tuple)):
        if not isinstance(restored, type(saved)) or len(saved) != len(restored):
            raise ValueError("optimizer state sequence mismatch: " + path)
        for i, (a, b) in enumerate(zip(saved, restored)):
            _assert_optimizer_tree_equal(a, b, path + "[" + str(i) + "]")
        return
    if type(saved) is not type(restored) or saved != restored:
        raise ValueError("optimizer state scalar mismatch: " + path)


def audit_trusted_routing_envelope(
    path: str | Path,
    model: SynapLoomModel,
    optimizer: torch.optim.Optimizer,
    controller: StructuralController,
    *,
    requested: RoutingSpec,
    expected_sha256: str,
    trusted_source: bool = False,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> EnvelopeAuditV1:
    """Full envelope *shadow* roundtrip for legacy batch semantics only.

    Does not mutate live model, optimizer, controller or any global RNG.
    Does not authorize training, resume or sample-trained checkpoints.
    Controller policies/observation are NOT saved by production state_dict:
    therefore full science/trajectory equivalence is impossible to claim here.
    """
    if not isinstance(model, SynapLoomModel) or not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("expected SynapLoomModel and torch optimizer")
    if not isinstance(controller, StructuralController):
        raise TypeError("expected StructuralController")
    if not isinstance(requested, RoutingSpec):
        raise TypeError("requested must be RoutingSpec")
    if requested != RoutingSpec():
        raise ValueError("full-envelope audit is restricted to legacy batch routing")
    if controller.event_logger is not None:
        raise ValueError("controller with external event logger cannot be shadow-audited")
    if not trusted_source:
        raise ValueError("trusted_source=True required for pickle checkpoint")
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected_sha256 must be pinned lowercase SHA-256")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("invalid checkpoint byte limit")
    _check_optimizer_links(model, optimizer)

    with Path(path).open("rb") as source:
        raw = source.read(max_bytes + 1)
    if not raw or len(raw) > max_bytes:
        raise ValueError("checkpoint byte limit exceeded or file empty")
    digest = sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    # WARNING: this executes trusted pickle. No authentication or sandbox.
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=False)
    envelope = _validate_envelope(payload, model)
    if "optimizer" not in envelope or "controller" not in envelope:
        raise ValueError("full envelope requires optimizer and controller")
    if not isinstance(envelope["optimizer"], Mapping) or not isinstance(envelope["controller"], Mapping):
        raise ValueError("invalid optimizer/controller envelope")
    # Narrow V1 qualification: insist that the saved optimizer's declared
    # param groups and hyperparameters match the caller's expected optimizer
    # configuration before attempting ANY restore, even on the shadow.
    saved_groups = envelope["optimizer"].get("param_groups")
    actual_groups = optimizer.state_dict()["param_groups"]
    if not isinstance(saved_groups, list) or len(saved_groups) != len(actual_groups):
        raise ValueError("optimizer parameter group count mismatch")
    for saved, expected in zip(saved_groups, actual_groups):
        if not isinstance(saved, Mapping) or set(saved) != set(expected):
            raise ValueError("optimizer parameter group schema mismatch")
        if not isinstance(saved["params"], list) or len(saved["params"]) != len(expected["params"]):
            raise ValueError("optimizer parameter group size mismatch")
        for key in expected:
            if key != "params" and saved[key] != expected[key]:
                raise ValueError("optimizer hyperparameters mismatch: " + key)
    if set(envelope["controller"]) != {"last_applied_step"} or type(envelope["controller"]["last_applied_step"]) is not int:
        raise ValueError("invalid controller state schema")
    step = envelope["controller"]["last_applied_step"]
    if step < -1 or step > envelope["global_step"]:
        raise ValueError("controller step outside checkpoint global_step")
    with torch.no_grad():
        version = int(envelope["model"]["graph_state.structural_version"].item())
    if "routing" in envelope["extra"] and (
        envelope["extra"]["routing"]["structural_version"] != version
    ):
        raise ValueError("routing metadata structural version mismatch")
    routing = preflight_routing_extra(
        envelope["extra"], model=model, requested=requested, intent="replay"
    )
    numpy_checked, cuda_count = _validate_rng(envelope["rng_state"])

    # One memo preserves the model-optimizer parameter relationships.
    # No global randomness, training step, event logging or live mutation.
    shadow_model, shadow_optimizer, shadow_controller = copy.deepcopy(
        (model, optimizer, controller)
    )
    _check_optimizer_links(shadow_model, shadow_optimizer)
    shadow_model.load_state_dict(envelope["model"], strict=True)
    shadow_model.graph_state.validate()
    shadow_optimizer.load_state_dict(envelope["optimizer"])
    shadow_controller.load_state_dict(envelope["controller"])
    if shadow_controller.state_dict() != dict(envelope["controller"]):
        raise ValueError("controller shadow state does not roundtrip")
    if len(shadow_optimizer.param_groups) != len(optimizer.param_groups):
        raise ValueError("optimizer param-group count changed")
    _assert_optimizer_tree_equal(envelope["optimizer"], shadow_optimizer.state_dict())
    _check_optimizer_links(shadow_model, shadow_optimizer)

    return EnvelopeAuditV1(
        checkpoint_sha256=digest,
        checkpoint_bytes=len(raw),
        global_step=envelope["global_step"],
        routing=routing,
        shadow_model_sha256=model_state_sha256(shadow_model),
        optimizer_groups=len(shadow_optimizer.param_groups),
        optimizer_state_entries=len(shadow_optimizer.state_dict()["state"]),
        controller_last_applied_step=step,
        cpu_rng_checked=True,
        numpy_rng_checked=numpy_checked,
        cuda_rng_entries=cuda_count,
    )
