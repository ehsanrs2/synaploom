"""M1.3b D2 offline Runner result-boundary qualification (no execution).

Consumes ContinualRunResult.to_dict() shaped observations, not a training
command. The strict checks below are necessary but NOT sufficient for science:
source/device/data/mask provenance cannot be attested from self-reported JSON.
This module performs no model execution, file access, GPU operation or inference.
"""
from __future__ import annotations

from hashlib import sha256
import math
from typing import Any, Mapping, Sequence

from synaploom.protocols.h1a2 import H1A2_FEATURE_SHA256
from synaploom.protocols.h1a_order_sensitivity import (
    CLASS_ORDERS, EXPECTED_ORDER_FINGERPRINTS,
)
from synaploom.research.m13b_d2_eval_guard import (
    validate_co_primary_observation,
)
from synaploom.research.m13b_manifest import canonical_json

KIND = "M13B_D2_CPU_RUNNER_OUTPUT_CONTRACT_V1"
ARMS = ("progressive_reserve", "fixed_sparse_mlp")
SYNTHETIC_ENGINEERING_SEEDS = (13, 17)
TASKS = 10
EPOCHS = 20
TRAIN_EXAMPLES_PER_TASK = 5000
EVAL_EXAMPLES_PER_TASK = 1000
TRAIN_BATCH = 128
EVAL_BATCH = 256
STEPS_PER_EPOCH = 40
STEPS_PER_TRAJECTORY = TASKS * EPOCHS * STEPS_PER_EPOCH
REFERENCE_SOURCE_MAIN = "9b0c30657d237a23d2e85896dce195ac5997261c"


def reference_batch_ranges() -> tuple[tuple[int, int], ...]:
    """Ordered *positions*, not actual feature IDs or verified batch membership."""
    return tuple(
        (start, min(start + EVAL_BATCH, EVAL_EXAMPLES_PER_TASK))
        for start in range(0, EVAL_EXAMPLES_PER_TASK, EVAL_BATCH)
    )


def reference_position_digest() -> str:
    """Hash of [0:256,256:512,512:768,768:1000] (shape policy ONLY)."""
    return sha256(canonical_json([list(pair) for pair in reference_batch_ranges()])).hexdigest()


def _finite_nonnegative(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite nonnegative numeric")
    return float(value)


def _hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _require_sequence(value: Any, label: str, n: int) -> Sequence[Any]:
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"{label} requires exactly {n} entries")
    return value


def validate_shape_and_policy(
    payload: Mapping[str, Any], *, seed: int, arm: str,
    expected_mask_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail-closed *post-run-shaped* contract; NEVER evidence of completed science.

    Source/feature/order/mask digests included here are merely DECLARED JSON
    equality checks, not independent file/optimizer/device attestations.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("runner output must be a mapping")
    if type(seed) is not int or seed < 0 or arm not in ARMS:
        raise ValueError("seed and arm must be declared independently")
    if seed in range(501, 511):
        raise ValueError("reserved proposed scientific seeds forbidden in CPU qualification")
    if payload.get("global_steps") != STEPS_PER_TRAJECTORY or type(payload.get("global_steps")) is not int:
        raise ValueError("global_steps must be the exact 8000-step full shape")
    config = payload.get("run_config")
    if not isinstance(config, Mapping) or type(config.get("seed")) is not int or config.get("seed") != seed:
        raise ValueError("missing or mismatched seed")
    if config.get("device") != "cpu":
        raise ValueError("CPU qualification refuses CUDA or other devices")
    if config.get("apply_final_task_boundary") not in (False, None):
        raise ValueError("unexpected final task boundary policy")

    # Mandatory pre-existing guard checks lower-triangle accuracy and all curves.
    endpoints = validate_co_primary_observation(payload)

    # The old standalone validator required increasing steps; exactness is
    # stronger: 5000 train examples / B128 = 40 steps per epoch.
    curves = payload["current_task_learning_curves"]
    for stage, curve in enumerate(curves):
        for epoch_index, point in enumerate(curve):
            exact = (stage * EPOCHS + epoch_index + 1) * STEPS_PER_EPOCH
            if type(point.get("global_step")) is not int or point["global_step"] != exact:
                raise ValueError("missing, duplicated or shifted optimizer-step checkpoint")
    if curves[-1][-1]["global_step"] != STEPS_PER_TRAJECTORY:
        raise ValueError("final step not witnessed by final epoch")

    manifest = _require_sequence(payload.get("stream_manifest"), "stream_manifest", TASKS)
    reference_order = CLASS_ORDERS["O0"]
    for stage, row in enumerate(manifest):
        if not isinstance(row, Mapping):
            raise ValueError("invalid stream task record")
        if (
            type(row.get("task_id")) is not int or row["task_id"] != stage
            or type(row.get("train_examples")) is not int
            or row["train_examples"] != TRAIN_EXAMPLES_PER_TASK
            or type(row.get("eval_examples")) is not int
            or row["eval_examples"] != EVAL_EXAMPLES_PER_TASK
            or row.get("classes") != list(reference_order[10*stage:10*(stage+1)])
        ):
            raise ValueError("stream task ID, class order or dataset geometry mismatch")

    eval_losses = _require_sequence(payload.get("eval_loss_matrix"), "eval_loss_matrix", TASKS)
    for stage, row in enumerate(eval_losses):
        row = _require_sequence(row, f"eval_loss_matrix[{stage}]", TASKS)
        for task, val in enumerate(row):
            if task <= stage:
                _finite_nonnegative(val, f"eval_loss_matrix[{stage}][{task}]")
            elif val is not None:
                raise ValueError("future-task eval losses must be explicit null")
    for name in ("train_loss_by_task", "train_seconds_by_task",
                 "eval_seconds_by_stage", "max_gradient_norm_by_task",
                 "max_parameter_abs_by_task"):
        for index, val in enumerate(_require_sequence(payload.get(name), name, TASKS)):
            _finite_nonnegative(val, f"{name}[{index}]")
    snapshots = _require_sequence(payload.get("structural_history"), "structural_history", TASKS)
    for i, snap in enumerate(snapshots):
        if not isinstance(snap, Mapping) or type(snap.get("stage")) is not int or snap["stage"] != i:
            raise ValueError("structural history stage mismatch")
        if type(snap.get("task_id")) is not int or snap["task_id"] != i:
            raise ValueError("structural history task ID mismatch")
    if payload.get("evaluation_batch_size_results") not in (None, {}):
        raise ValueError("no extra evaluation batch-size sweep admitted to this O0 reference contract")
    if payload.get("peak_cuda_memory_bytes") is not None:
        raise ValueError("CPU fixture cannot claim peak CUDA allocator memory")

    md = payload.get("metadata")
    if not isinstance(md, Mapping) or md.get("kind") != "SYNTHETIC_CPU_D2_RUNNER_FIXTURE_V1":
        raise ValueError("only synthetic CPU boundary fixtures accepted")
    if (md.get("synthetic") is not True or md.get("scientific_claim") != "NONE"
        or md.get("protected_eval_accessed") is not False
        or md.get("actual_optimizer_steps") != 0
        or type(md.get("attempt_index")) is not int or md["attempt_index"] != 0
        or md.get("terminal_state") != "SYNTHETIC_COMPLETE_NOT_TRAINED"
        or md.get("arm") != arm or type(md.get("seed")) is not int or md["seed"] != seed
        or md.get("order_id") != "O0"):
        raise ValueError("synthetic provenance/attempt fields fail closed")
    if md.get("source_commit_declared") != REFERENCE_SOURCE_MAIN:
        raise ValueError("declared source candidate baseline mismatch")
    if md.get("feature_sha256_declared") != H1A2_FEATURE_SHA256:
        raise ValueError("declared feature digest mismatch")
    if md.get("order_sha256_declared") != EXPECTED_ORDER_FINGERPRINTS["O0"]:
        raise ValueError("declared class-order digest mismatch")
    if md.get("position_only_batch_digest_declared") != reference_position_digest():
        raise ValueError("reference 256/232 position layout mismatch")
    if arm == "fixed_sparse_mlp":
        expected = _hex64(expected_mask_sha256, "independently supplied mask digest")
        declared = _hex64(md.get("mask_sha256_declared"), "declared mask digest")
        if declared != expected:
            raise ValueError("mask digest mismatch")
    elif md.get("mask_sha256_declared") is not None or expected_mask_sha256 is not None:
        raise ValueError("progressive arm must not impersonate fixed mask digest")

    return {
        "kind": KIND,
        "status": "SYNTHETIC_CPU_SHAPE_GUARD_PASS__NO_PROVENANCE_ATTESTATION",
        "slot_key": f"O0__seed-{seed}__{arm}",
        "co_primary_metric_values": endpoints["co_primary_metric_values"],
        "position_only_batch_layout": [list(x) for x in reference_batch_ranges()],
        "position_only_batch_digest": reference_position_digest(),
        "self_reported_payload_sha256": sha256(canonical_json(payload)).hexdigest(),
        "independent_source_feature_mask_attestation": False,
        "runner_integrated_with_science": False,
        "science_executable": False,
        "scientific_claim": "NONE",
    }


def validate_synthetic_registry(
    records: Sequence[Mapping[str, Any]], *,
    seeds: tuple[int, ...] = SYNTHETIC_ENGINEERING_SEEDS,
    expected_mask_sha256_by_seed: Mapping[int, str],
) -> dict[str, Any]:
    """Require complete balanced two-arm synthetic slots; no omissions/retries."""
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    if (
        not isinstance(seeds, tuple) or not seeds
        or any(type(s) is not int or s < 0 or s in range(501, 511) for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("invalid synthetic seed registry")
    keys = {f"O0__seed-{s}__{arm}" for s in seeds for arm in ARMS}
    if len(records) != len(keys):
        raise ValueError("missing or excess paired-arm records")
    accepted = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"seed", "arm", "observation"}:
            raise ValueError("invalid synthetic slot record")
        seed, arm = record["seed"], record["arm"]
        if type(seed) is not int or seed not in seeds or arm not in ARMS:
            raise ValueError("unregistered seed or arm")
        key = f"O0__seed-{seed}__{arm}"
        if key not in keys or key in accepted:
            raise ValueError("duplicate or unexpected arm/seed slot")
        if arm == "fixed_sparse_mlp":
            if seed not in expected_mask_sha256_by_seed:
                raise ValueError("fixed sparse mask expectation missing")
            mask_digest = expected_mask_sha256_by_seed[seed]
        else:
            mask_digest = None
        accepted[key] = validate_shape_and_policy(
            record["observation"], seed=seed, arm=arm, expected_mask_sha256=mask_digest
        )
    if set(accepted) != keys:
        raise ValueError("incomplete paired-seed slot registry")
    return {
        "kind": KIND,
        "status": "COMPLETE_SYNTHETIC_CPU_PAIR_MATRIX__NOT_SCIENTIFIC",
        "paired_seed_count": len(seeds),
        "record_count": len(accepted),
        "registered_slot_keys": sorted(accepted),
        "full_shape_steps_declared_per_synthetic_slot": STEPS_PER_TRAJECTORY,
        "actual_optimizer_steps": 0,
        "protected_eval_accessed": False,
        "source_feature_mask_crypto_attested": False,
        "science_executable": False,
        "scientific_claim": "NONE",
    }
