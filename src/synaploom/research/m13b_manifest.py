"""M1.3b proposed pre-registration manifest and *read-only* preflight.

Only immutable, auditable plans and trusted-feature byte checks are supported.
NO GPU/training execution, optimizer steps, experiment registration or freeze.
The present scope is a candidate, not an owner-approved scientific protocol.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from synaploom.protocols.h1a2 import H1A2_FEATURE_SHA256, h1a2_protocol
from synaploom.protocols.h1a_order_sensitivity import (
    CLASS_ORDERS, EXPECTED_ORDER_FINGERPRINTS, validate_class_orders,
)
from synaploom.research.fixed_sparse_baseline import build_h1a_fixed_sparse_feasibility
from synaploom.research.preflight import audit_h1a_feature_bundle, sha256_file


STUDY_ID = "EXP-M13B-EXTERNAL-001"
PROPOSED_SEEDS = tuple(range(501, 511))
PROPOSED_ARMS = ("progressive_reserve", "all_recruited", "fixed_sparse_mlp")
MASK_SEED = 20261026
MATCH_TOTAL_LIMIT = 0.02
MATCH_IDEAL_MAC_LIMIT = 0.10
AUTHORIZATION = "NO_GO_DRAFT_NOT_FROZEN"
KNOWN_ORDERS = ("O0", "O1", "O2")
NO_LIVE_RESTORE = True


def canonical_json(obj: Any) -> bytes:
    """Stable UTF-8 serialization, no floating NaN/Inf."""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")


def _proposed_matrix(orders: tuple[str, ...]) -> dict[str, Any]:
    return {
        "orders": list(orders),
        "seed_ids": list(PROPOSED_SEEDS),
        "arms": list(PROPOSED_ARMS),
        "expected_trajectories": len(orders) * len(PROPOSED_SEEDS) * len(PROPOSED_ARMS),
        "paired_blocks": len(orders) * len(PROPOSED_SEEDS),
        "trajectory_keys": [
            f"{order}__seed-{seed}__{arm}"
            for order in orders for seed in PROPOSED_SEEDS for arm in PROPOSED_ARMS
        ],
    }


def build_m13b_draft_manifest(*, source_commit: str | None = None) -> dict[str, Any]:
    """Propose BOTH O0-only and O0/O1/O2 designs; never select one silently.

    This function does not access feature files and does not train a model.
    Builds a deterministic CPU resource candidate once, independently of
    proposed confirmatory outcomes and with restored global CPU RNG state.
    """
    hashes = validate_class_orders()
    if tuple(hashes) != KNOWN_ORDERS or hashes != EXPECTED_ORDER_FINGERPRINTS:
        raise ValueError("frozen class-order identifiers or digests changed")
    for name, order in CLASS_ORDERS.items():
        if len(order) != 100 or set(order) != set(range(100)):
            raise ValueError("invalid frozen class order " + name)
    protocol = h1a2_protocol()
    candidate, report = build_h1a_fixed_sparse_feasibility(
        protocol, seed=PROPOSED_SEEDS[0], mask_seed=MASK_SEED
    )
    candidate.assert_masks()
    if not report.matched_total_2pct or not report.matched_ideal_macs_10pct:
        raise ValueError("candidate fixed-sparse arithmetic budget gate not met")
    if source_commit is not None:
        if (not isinstance(source_commit, str) or len(source_commit) != 40
                or any(c not in "0123456789abcdef" for c in source_commit)):
            raise ValueError("source_commit must be an exact lowercase 40-hex commit")
    record: dict[str, Any] = {
        "schema": 1,
        "study_id": STUDY_ID,
        "status": "DRAFT_DESIGN_PREVIEW",
        "authorization": AUTHORIZATION,
        "science_executable": False,
        "source_commit": source_commit,
        "source_revision_verified": False,
        "origin": {
            "protocol": "docs/research/m13b_external_baselines/PROTOCOL_DRAFT.md",
            "anchor": "EXP-H1A-002",
            "historical_result_unchanged": True,
            "routing_semantics": "legacy_batch",
            "training_sample_routing_allowed": False,
        },
        "dataset": {
            "id": "CIFAR100_RESNET18_IMAGENET1K_V1_FROZEN",
            "feature_sha256_expected": H1A2_FEATURE_SHA256,
            "feature_sha256_verified": None,
            "feature_validation": "NOT_CHECKED",
            "feature_dim": protocol.feature_dim,
            "num_classes": protocol.num_classes,
            "train_examples": 50_000,
            "eval_examples": 10_000,
        },
        "fixed_conditions": {
            "tasks": 10,
            "classes_per_task": protocol.classes_per_task,
            "task_id_at_inference": False,
            "seen_class_mask": True,
            "epochs_per_task": protocol.epochs_per_task,
            "train_batch_size": protocol.batch_size,
            "eval_batch_size": protocol.eval_batch_size,
            "learning_rate": protocol.learning_rate,
            "momentum": protocol.momentum,
            "weight_decay": protocol.weight_decay,
            "scheduler": None,
            "replay": False,
            "consolidation": False,
            "q_max": 0,
            "message_rounds": 0,
        },
        "class_order_fingerprints": hashes,
        "unresolved_owner_decisions": [
            "O0_ONLY_OR_O0_O1_O2",
            "EXTERNAL_BASELINE_CONFIRMATORY_FAMILY_AND_MULTIPLICITY",
            "PROPOSED_SEEDS_501_TO_510",
            "FIXED_SPARSE_MASK_AND_MATCHING_TIER",
            "NUMERICAL_PILOT_AND_METHOD_FIDELITY_APPROVAL",
            "TRUSTED_DATA_AND_SOURCE_ORIGIN",
            "EXPERIMENT_RUN_DIR_AND_IMMUTABLE_EVIDENCE_POLICY",
        ],
        "candidate_matrices": {
            "O0_ONLY": _proposed_matrix(("O0",)),
            "O0_O1_O2": _proposed_matrix(KNOWN_ORDERS),
        },
        "fixed_sparse_candidate": {
            **report.as_dict(),
            "mask_fingerprint_is_independently_attested": False,
            "mask_seed_is_proposed_not_frozen": True,
        },
        "budget_thresholds_proposed": {
            "physical_trainable_relative_error_max": MATCH_TOTAL_LIMIT,
            "ideal_linear_mac_relative_error_max": MATCH_IDEAL_MAC_LIMIT,
            "dense_masked_execution_is_real_sparse_compute": False,
        },
        "metrics_proposed": [
            "final_average_accuracy",
            "late_task_within_task_mean_accuracy",
            "average_forgetting",
            "backward_transfer",
            "train_wall_time",
            "peak_gpu_memory",
        ],
        "statistical_rule_proposed": {
            "paired_bootstrap_resamples": 10_000,
            "bootstrap_seed": 1337,
            "both_co_primary_confidence_intervals_above_zero_required": True,
            "historical_h1a_reinterpretation_allowed": False,
        },
        "run_status": "NO_RUNS_EXECUTED",
    }
    return _with_digest(record)


def _with_digest(record: Mapping[str, Any]) -> dict[str, Any]:
    if "record_sha256" in record:
        raise ValueError("record_sha256 must not be present in canonical payload")
    result = dict(record)
    result["record_sha256"] = sha256(canonical_json(record)).hexdigest()
    return result


def verify_manifest_integrity(manifest: Mapping[str, Any]) -> None:
    """Detect byte-level changes to the preview; not owner authentication."""
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be mapping")
    digest = manifest.get("record_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("missing plan digest")
    bare = {k: v for k, v in manifest.items() if k != "record_sha256"}
    if digest != sha256(canonical_json(bare)).hexdigest():
        raise ValueError("draft plan SHA-256 mismatch")
    if bare.get("status") != "DRAFT_DESIGN_PREVIEW" or bare.get("science_executable") is not False:
        raise ValueError("only draft, nonexecutable plans are supported")
    if bare.get("authorization") != AUTHORIZATION:
        raise ValueError("scientific execution is not authorized")


def verify_trusted_feature_file(
    path: str | Path,
    *,
    trusted_source: bool,
) -> dict[str, Any]:
    """Validate only local trusted artifact, checking SHA before unsafe unpickle.

    audit_h1a_feature_bundle uses torch.load(weights_only=False); never invoke
    on unknown or web-downloaded pickle files. Hash matches independently
    recorded feature identity but is not publisher trust authentication.
    """
    if trusted_source is not True:
        raise ValueError("explicit trusted feature source approval required")
    actual = sha256_file(path)
    if actual != H1A2_FEATURE_SHA256:
        raise ValueError("frozen feature SHA-256 mismatch")
    audit = audit_h1a_feature_bundle(path)
    if audit.sha256 != H1A2_FEATURE_SHA256:
        raise ValueError("feature bytes changed during audit")
    return audit.to_dict()


def build_feature_verified_manifest(
    *, source_commit: str | None, feature_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Add an audit result to a draft preview without authorizing training."""
    if feature_audit.get("sha256") != H1A2_FEATURE_SHA256:
        raise ValueError("feature audit digest mismatch")
    base = build_m13b_draft_manifest(source_commit=source_commit)
    bare = {key: value for key, value in base.items() if key != "record_sha256"}
    bare["dataset"] = {
        **bare["dataset"],
        "feature_sha256_verified": feature_audit["sha256"],
        "feature_validation": "TRUSTED_LOCAL_CHECK_PASSED",
        "audit": dict(feature_audit),
    }
    return _with_digest(bare)
