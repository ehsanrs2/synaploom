"""Frozen, allowlisted controller-policy identity for checkpoint research.

Pure data only: never reconstruct/execute an untrusted policy, resume training,
or modify a controller. A digest identifies bytes; it does NOT attest origin.
This contract is deliberately separate from legacy StructuralController.state_dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Mapping

from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import (
    ConnectToSelectedTopR,
    FixedScheduleRecruitment,
    NoConsolidation,
    NoEdges,
    NoRecruitment,
    SeededRandomR,
    TaskBoundaryRecruitment,
    TaskBoundaryTopImportance,
)


@dataclass(frozen=True)
class ControllerPolicyPreflightV1:
    schema: int
    policy_sha256: str
    policy_matches: bool
    resume_authorized: bool = False


def _recruitment(policy: Any) -> dict[str, Any]:
    if type(policy) is NoRecruitment:
        return {"type": "NoRecruitment"}
    if type(policy) is FixedScheduleRecruitment:
        steps = policy.steps
        if not isinstance(steps, frozenset) or any(type(s) is not int for s in steps):
            raise ValueError("invalid FixedScheduleRecruitment steps")
        return {"type": "FixedScheduleRecruitment", "steps": sorted(steps)}
    if type(policy) is TaskBoundaryRecruitment:
        return {"type": "TaskBoundaryRecruitment"}
    raise ValueError("unsupported recruitment policy type")


def _edges(policy: Any) -> dict[str, Any]:
    if type(policy) is NoEdges:
        return {"type": "NoEdges"}
    if type(policy) is ConnectToSelectedTopR:
        if type(policy.top_r) is not int or policy.top_r < 0:
            raise ValueError("invalid edge top_r")
        return {"type": "ConnectToSelectedTopR", "top_r": policy.top_r}
    if type(policy) is SeededRandomR:
        if (type(policy.top_r) is not int or policy.top_r < 0
                or type(policy.seed) is not int):
            raise ValueError("invalid random edge policy")
        return {"type": "SeededRandomR", "top_r": policy.top_r, "seed": policy.seed}
    raise ValueError("unsupported edge policy type")


def _consolidation(policy: Any) -> dict[str, Any]:
    if type(policy) is NoConsolidation:
        return {"type": "NoConsolidation"}
    if type(policy) is TaskBoundaryTopImportance:
        if type(policy.top_n) is not int or policy.top_n < 0:
            raise ValueError("invalid consolidation top_n")
        return {"type": "TaskBoundaryTopImportance", "top_n": policy.top_n}
    raise ValueError("unsupported consolidation policy type")


def snapshot_controller_policy(controller: StructuralController) -> dict[str, Any]:
    """Capture M1 policy configuration, NOT last step, observation, or logger state.

    The unsupported event logger is rejected because its output/cursor cannot
    be certified by this self-contained policy-identity record.
    """
    if type(controller) is not StructuralController:
        raise TypeError("requires exact StructuralController, not custom subclass")
    if controller.event_logger is not None:
        raise ValueError("external event logger requires separate provenance")
    noise = controller.imprint_noise_std
    gate = controller.edge_gate_logit
    if (type(noise) is not float or not math.isfinite(noise) or noise < 0
            or type(gate) is not float or not math.isfinite(gate)):
        raise ValueError("controller scalar policy settings must be finite")
    return {
        "schema": 1,
        "kind": "synaploom.controller_policy.v1",
        "recruitment": _recruitment(controller.recruitment_policy),
        "edge_init": _edges(controller.edge_init_policy),
        "consolidation": _consolidation(controller.consolidation_policy),
        "imprint_noise_std": noise,
        "edge_gate_logit": gate,
    }


def _canonical_bytes(record: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("noncanonical policy record") from exc


def controller_policy_sha256(record: Mapping[str, Any]) -> str:
    if not isinstance(record, Mapping):
        raise TypeError("policy record must be a mapping")
    return sha256(_canonical_bytes(record)).hexdigest()


def verify_controller_policy(
    record: Mapping[str, Any],
    controller: StructuralController,
    *,
    expected_sha256: str,
) -> ControllerPolicyPreflightV1:
    """Fail closed on record changes or target-policy mismatch; read-only.

    The caller must pin expected_sha256 from an independent trusted manifest.
    Providing a digest calculated from the same untrusted record is NOT trust.
    """
    if not isinstance(record, Mapping):
        raise TypeError("policy record must be a mapping")
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected_sha256 must be an independently pinned lowercase SHA-256")
    if set(record) != {
        "schema", "kind", "recruitment", "edge_init", "consolidation",
        "imprint_noise_std", "edge_gate_logit",
    } or type(record.get("schema")) is not int or record["schema"] != 1:
        raise ValueError("unsupported policy identity schema")
    if record["kind"] != "synaploom.controller_policy.v1":
        raise ValueError("unsupported policy identity kind")

    digest = controller_policy_sha256(record)
    if digest != expected_sha256:
        raise ValueError("policy identity SHA-256 mismatch")

    # Canonical generated structure is the entire V1 allowlist. This also
    # rejects unknown policy types/fields, incorrect numeric types and changes
    # even when a caller supplies a newly calculated hash of altered metadata.
    target = snapshot_controller_policy(controller)
    if record != target:
        raise ValueError("checkpoint controller policy differs from target")
    return ControllerPolicyPreflightV1(
        schema=1, policy_sha256=digest, policy_matches=True,
    )
