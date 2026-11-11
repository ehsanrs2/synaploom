"""D2: detached-pin synthetic evidence-chain contract; never scientific approval.

Only small, entirely synthetic, fixed registries are accepted. No files,
features, torch, network, CUDA, processes, or actual results are read.

A SHA-256 digest supplied independently of the record detects changes only
when the supplier/verification channel is trusted; there is NO signature,
external custody, owner approval, or immutable storage here.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

KIND = "M13B_D2_DETACHED_PIN_SYNTHETIC_ENVELOPE_V1"
SEEDS = (13, 17)
ARMS = ("progressive_reserve", "fixed_sparse_mlp")
BATCH_SIZES = (256, 256, 256, 232)
STEPS_TINY = 12
ZERO_SHA = "0" * 64
SLOTS = tuple(f"O0__seed-{seed}__{arm}" for seed in SEEDS for arm in ARMS)
SUCCEEDED = "TINY_SYNTHETIC_CPU_SUCCEEDED"
FAILED = "TINY_SYNTHETIC_CPU_FAILED"
STATES = (SUCCEEDED, FAILED)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


def digest_json(value: Any) -> str:
    return sha256(canonical_json(value)).hexdigest()


def _hex(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase 64-hex SHA256")
    return value


def _event(seq: int, slot: str, state: str, steps: int, prev: str) -> dict[str, Any]:
    body = {
        "sequence": seq, "slot_key": slot,
        "terminal_state": state, "cpu_optimizer_post_steps": steps,
        "previous_sha256": prev,
    }
    return {**body, "event_sha256": digest_json(body)}


def build_synthetic_envelope(
    *,
    independently_expected_source_sha256: str,
    independently_expected_sample_layout_sha256: str,
    independently_expected_fixed_mask_sha256: str,
    attempts: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    """Construct synthetic toy evidence; 'independent' denotes caller *role*, not proof.

    Caller must separately preserve the returned root. Inputs are digests of
    invented fixtures, NOT real protected features or a verified Git checkout.
    """
    source = _hex(independently_expected_source_sha256, "source pin")
    samples = _hex(independently_expected_sample_layout_sha256, "sample pin")
    mask = _hex(independently_expected_fixed_mask_sha256, "mask pin")
    if not isinstance(attempts, (list, tuple)) or len(attempts) != len(SLOTS):
        raise ValueError("exact synthetic four-slot registry required")
    events: list[dict[str, Any]] = []
    previous = ZERO_SHA
    for i, (slot, item) in enumerate(zip(SLOTS, attempts)):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("invalid synthetic attempt row")
        state, steps = item
        if state not in STATES or type(steps) is not int or not (0 <= steps <= STEPS_TINY):
            raise ValueError("invalid synthetic terminal state or step count")
        if state == SUCCEEDED and steps != STEPS_TINY:
            raise ValueError("a successful tiny CPU attempt requires 12 observed steps")
        event = _event(i, slot, state, steps, previous)
        events.append(event)
        previous = event["event_sha256"]
    return {
        "schema": 1, "kind": KIND,
        "status": "RESEARCH_ONLY_NO_SCIENTIFIC_PROTOCOL",
        "expected_seed_registry": list(SEEDS),
        "expected_arm_registry": list(ARMS),
        "order_id": "O0",
        "expected_trajectory_slots": list(SLOTS),
        "declared_source_sha256": source,
        "declared_synthetic_sample_layout_sha256": samples,
        "declared_synthetic_fixed_mask_sha256": mask,
        "declared_eval_batch_sizes": list(BATCH_SIZES),
        "events": events,
        "chain_head_sha256": previous,
        "data_protected": False,
        "sciencerun_optimizer_steps": 0,
        "owner_signed": False,
        "external_witness_signed": False,
        "science_executable": False,
        "scientific_claim": "NONE",
    }


def verify_synthetic_envelope(
    envelope: Mapping[str, Any], *,
    detached_expected_root_sha256: str,
    expected_source_sha256: str,
    expected_sample_layout_sha256: str,
    expected_fixed_mask_sha256: str,
) -> dict[str, Any]:
    """Fail closed if even a rehashed event chain deviates from detached pin.

    Does NOT verify signer identity, file custody, observed model execution,
    independence of caller-supplied pins or any full scientific run.
    """
    root = _hex(detached_expected_root_sha256, "detached root pin")
    pins = (
        _hex(expected_source_sha256, "expected source"),
        _hex(expected_sample_layout_sha256, "expected sample layout"),
        _hex(expected_fixed_mask_sha256, "expected mask"),
    )
    if not isinstance(envelope, Mapping):
        raise ValueError("envelope must be a mapping")
    expected_keys = {
        "schema", "kind", "status", "expected_seed_registry",
        "expected_arm_registry", "order_id", "expected_trajectory_slots",
        "declared_source_sha256", "declared_synthetic_sample_layout_sha256",
        "declared_synthetic_fixed_mask_sha256", "declared_eval_batch_sizes",
        "events", "chain_head_sha256", "data_protected",
        "sciencerun_optimizer_steps", "owner_signed",
        "external_witness_signed", "science_executable", "scientific_claim",
    }
    if set(envelope) != expected_keys:
        raise ValueError("unexpected or missing envelope contract fields")
    if (type(envelope["schema"]) is not int or envelope["schema"] != 1
            or envelope["kind"] != KIND
            or envelope["status"] != "RESEARCH_ONLY_NO_SCIENTIFIC_PROTOCOL"
            or envelope["expected_seed_registry"] != list(SEEDS)
            or envelope["expected_arm_registry"] != list(ARMS)
            or envelope["order_id"] != "O0"
            or envelope["expected_trajectory_slots"] != list(SLOTS)
            or envelope["declared_eval_batch_sizes"] != list(BATCH_SIZES)
            or envelope["data_protected"] is not False
            or type(envelope["sciencerun_optimizer_steps"]) is not int
            or envelope["sciencerun_optimizer_steps"] != 0
            or envelope["owner_signed"] is not False
            or envelope["external_witness_signed"] is not False
            or envelope["science_executable"] is not False
            or envelope["scientific_claim"] != "NONE"):
        raise ValueError("synthetic study-policy or authorization drift")
    declared = (
        envelope["declared_source_sha256"],
        envelope["declared_synthetic_sample_layout_sha256"],
        envelope["declared_synthetic_fixed_mask_sha256"],
    )
    if declared != pins:
        raise ValueError("declared source/sample/mask digests differ from separately pinned expectations")
    events = envelope["events"]
    if not isinstance(events, list) or len(events) != len(SLOTS):
        raise ValueError("missing or duplicated synthetic trajectory event")
    prev = ZERO_SHA
    failures = 0
    for idx, (slot, item) in enumerate(zip(SLOTS, events)):
        if not isinstance(item, Mapping) or set(item) != {
            "sequence", "slot_key", "terminal_state", "cpu_optimizer_post_steps",
            "previous_sha256", "event_sha256",
        }:
            raise ValueError("malformed event")
        state = item["terminal_state"]
        steps = item["cpu_optimizer_post_steps"]
        if (type(item["sequence"]) is not int or item["sequence"] != idx
                or item["slot_key"] != slot or state not in STATES
                or type(steps) is not int or not 0 <= steps <= STEPS_TINY
                or (state == SUCCEEDED and steps != STEPS_TINY)
                or item["previous_sha256"] != prev):
            raise ValueError("altered slot, omitted failure or inaccurate step witness")
        body = {key: val for key, val in item.items() if key != "event_sha256"}
        if item["event_sha256"] != digest_json(body):
            raise ValueError("event hash chain corrupted")
        prev = item["event_sha256"]
        failures += state == FAILED
    if envelope["chain_head_sha256"] != prev:
        raise ValueError("chain head does not match all attempts")
    if digest_json(envelope) != root:
        raise ValueError("detached external root pin mismatch, possibly rehashed replacement")
    return {
        "kind": KIND,
        "status": "DETACHED_PIN_MATCH_FOR_SYNTHETIC_FIXTURE_ONLY",
        "toy_slot_count": len(SLOTS),
        "toy_failure_count": failures,
        "all_toy_slots_present": True,
        "detached_root_sha256": root,
        "source_sample_mask_declared_pins_match": True,
        "caller_pin_origin_cryptographically_authenticated": False,
        "owner_signed": False,
        "witness_signed": False,
        "real_runner_integrated": False,
        "real_protected_samples_examined": False,
        "science_executable": False,
        "scientific_claim": "NONE",
    }


def d2_trust_requirements() -> dict[str, Any]:
    """Prospective unapproved external-evidence contract, not a GO checklist."""
    requirements = {
        "source": "trusted commit/tree, all imports and lockfile hashes; independent checkout witness",
        "data": "trusted full bundle byte digest, nonexecuting format review, true row/label IDs",
        "evaluation": "ordered per-task 1000-row membership digests and 256/256/256/232 forwards",
        "steps": "8000 actually observed optimizer steps, 40 per epoch, 200 evaluated epochs",
        "mask_optimizer": "seed-bound mask construction digest, every-step projection, momentum state",
        "scope": "owner-ratified D1 arms and class orders, registered seed-generating process",
        "inference": "owner-ratified D3 paired mean analysis, joint power assumptions and n",
        "budget": "owner-ratified D4 bounded GPU budget and manual abort controls",
        "failure": "all attempt IDs, terminal failures, predeclared retry/exclusion and timeouts",
        "custody": "owner-attested pre-outcome manifest and independent postrun witness storage",
    }
    return {
        "schema": 1, "kind": KIND,
        "status": "PROSPECTIVE_REQUIREMENTS_NOT_IMPLEMENTED",
        "requirements": requirements,
        "implementation_proof_for_science": False,
        "owner_signed_protocol": False,
        "independent_witness": False,
        "science_executable": False,
    }
