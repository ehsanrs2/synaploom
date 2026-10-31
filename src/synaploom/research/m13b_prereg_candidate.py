"""M1.3b complete *candidate* registration contract, NOT approval or a runner.

Deterministically enumerates the proposed 90 trajectories for owner review;
verifies published frozen order fingerprints and pre-existing per-task resource
audit. No feature files, training, CUDA, subprocesses, checkpoints, or network.
A candidate digest is NOT an owner signature or permission to train.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping

from synaploom.protocols.h1a_order_sensitivity import (
    EXPECTED_ORDER_FINGERPRINTS,
    validate_class_orders,
)
from synaploom.protocols.h1a2 import H1A2_FEATURE_SHA256
from synaploom.research.m13b_manifest import (
    MASK_SEED, PROPOSED_ARMS, PROPOSED_SEEDS, build_m13b_draft_manifest,
    canonical_json,
)
from synaploom.research.m13b_stagewise_budget import build_m13b_stagewise_audit


KIND = "M13B_PREREGISTRATION_CANDIDATE_V1"
BASE_MAIN_COMMIT = "ce0fe5270d406b5f09cadb7d050eaaff737977e8"
ORDERS = ("O0", "O1", "O2")
CONFIRMATORY_ORDER = "O0"
ROBUSTNESS_ORDERS = ("O1", "O2")
ARMS = ("progressive_reserve", "all_recruited", "fixed_sparse_mlp")
PROPOSED_SEED_IDS = tuple(range(501, 511))
PILOT_SEED_IDS = frozenset((13, 17, 41))
METRICS = ("final_average_accuracy", "late_task_within_task_mean_accuracy")
SCIENCE_RUN_AUTHORIZED = False


def _approval_queue() -> dict[str, dict[str, Any]]:
    """Concrete recommended options, all explicitly PENDING."""
    return {
        "D1_ORDER_SCOPE": {
            "status": "PENDING",
            "proposal": "O0 primary plus mandatory predeclared O1/O2 descriptive robustness",
            "alternate": "O0 only; report inability to claim order robustness",
        },
        "D2_RESOURCE_FAIRNESS": {
            "status": "PENDING",
            "proposal": "allocated capacity plus ideal MAC tiers; disclose actual GPU work separately",
            "requires_review": "dense masked compute, optimizer states and full-task measured resource fairness",
        },
        "D3_STATISTICS": {
            "status": "PENDING",
            "proposal": "one O0 external paired contrast, both co-primary CIs and descriptive secondary arms",
            "requires_review": "coverage, small n=10, multiplicity and robustness sensitivity policy",
        },
        "D4_EXECUTION_SCOPE": {
            "status": "PENDING",
            "proposal": "10 paired seeds 501..510, 3 arms, 3 orders, 90 trajectories after explicit owner GO",
            "alternate": "bounded O0-only study; no post-hoc robustness claims",
        },
    }


def _slots() -> list[dict[str, Any]]:
    """Balance which arm runs first within each deterministic order/seed block.

    The schedule balances sequences for engineering fairness but does NOT by
    itself guarantee equal GPU state, temperature, or execution cost.
    """
    slots = []
    n = 0
    for order_index, order_id in enumerate(ORDERS):
        for seed_index, seed in enumerate(PROPOSED_SEED_IDS):
            shift = (seed_index + order_index) % len(ARMS)
            arm_sequence = ARMS[shift:] + ARMS[:shift]
            for arm_position, arm in enumerate(arm_sequence):
                slots.append({
                    "slot_index": n,
                    "slot_key": f"{order_id}__seed-{seed}__{arm}",
                    "order_id": order_id,
                    "order_sha256": EXPECTED_ORDER_FINGERPRINTS[order_id],
                    "seed": seed,
                    "arm": arm,
                    "comparison_role": (
                        "PRIMARY_O0" if order_id == CONFIRMATORY_ORDER
                        else "DESCRIPTIVE_ORDER_ROBUSTNESS"
                    ),
                    "arm_execution_position_in_block": arm_position,
                    "optimizer_steps_expected": 8_000,
                    "train_examples_presented_expected": 1_000_000,
                    "science_executable": False,
                })
                n += 1
    return slots


def build_m13b_prereg_candidate() -> dict[str, Any]:
    """Build all 90 proposed slots with fail-closed, scientifically honest gates.

    The verified SHAs in this candidate cover analytical *content*, not source
    authorship. A later separately approved version must verify executable
    source revision, trusted feature bytes, mask provenance and user GO.
    """
    registered = validate_class_orders()
    if registered != EXPECTED_ORDER_FINGERPRINTS or tuple(registered) != ORDERS:
        raise ValueError("registered O0/O1/O2 class-order source has changed")
    if PROPOSED_SEEDS != PROPOSED_SEED_IDS or PROPOSED_ARMS != ARMS:
        raise ValueError("new proposed seeds or arms diverge from existing M1.3b manifest")
    if PILOT_SEED_IDS.intersection(PROPOSED_SEED_IDS):
        raise ValueError("engineering and proposed confirmation seed sets overlap")
    initial = build_m13b_draft_manifest()
    stage = build_m13b_stagewise_audit()
    if stage["owner_decision_pending"] is not True or stage["science_executable"] is not False:
        raise ValueError("stagewise analytical audit was incorrectly promoted")
    if not (initial["science_executable"] is False
            and initial["authorization"] == "NO_GO_DRAFT_NOT_FROZEN"):
        raise ValueError("base M1.3b draft authorization changed")
    if (initial["class_order_fingerprints"] != registered
            or initial["dataset"]["feature_sha256_expected"] != H1A2_FEATURE_SHA256):
        raise ValueError("draft class-order or feature pin drift")
    if (initial["fixed_sparse_candidate"]["mask_sha256"]
            != stage["fixed_sparse_mask_sha256"]):
        raise ValueError("proposed fixed mask differs across resource audits")
    if (stage["total_optimizer_steps_per_arm"] != 8000
            or stage["total_optimizer_steps_for_3_arms_per_order"] != 24000):
        raise ValueError("stagewise optimization geometry drifted")

    slots = _slots()
    if len(slots) != 90 or len({s["slot_key"] for s in slots}) != len(slots):
        raise ValueError("slot multiplicity or uniqueness failure")
    if tuple(s["slot_index"] for s in slots) != tuple(range(90)):
        raise ValueError("unstable slot ordering")
    if sum(s["optimizer_steps_expected"] for s in slots) != 720_000:
        raise ValueError("unexpected proposed total optimizer steps")

    data: dict[str, Any] = {
        "schema": 1,
        "kind": KIND,
        "study_id": "EXP-M13B-EXTERNAL-001",
        "status": "DESIGN_REVIEW_ONLY_UNFROZEN",
        "authorization": "NO_GO_CONFIRMATORY_SCIENCE",
        "science_executable": SCIENCE_RUN_AUTHORIZED,
        "confirmatory_outcomes_observed": False,
        "historical_h1a_results_unchanged": True,
        "source_base_commit": BASE_MAIN_COMMIT,
        "source_commit_verified_on_execution_host": False,
        "owner_approval_present": False,
        "independent_method_review_complete": False,
        "final_protocol_frozen": False,
        "science_runner_present": False,
        "local_feature_bytes_verified_now": False,
        "feature_bundle_sha256_expected": H1A2_FEATURE_SHA256,
        "order_fingerprints": registered,
        "order_scope_proposal": {
            "primary_order": CONFIRMATORY_ORDER,
            "robustness_orders": list(ROBUSTNESS_ORDERS),
            "scope_has_owner_approval": False,
        },
        "seeds_proposed": list(PROPOSED_SEED_IDS),
        "arms_proposed": list(ARMS),
        "pilot_seed_ids_not_used": sorted(PILOT_SEED_IDS),
        "mask_seed_proposed": MASK_SEED,
        "mask_sha256_proposed": stage["fixed_sparse_mask_sha256"],
        "training_policy_proposed": {
            "task_count": 10,
            "classes_per_task": 10,
            "epochs_per_task": 20,
            "train_batch_size": 128,
            "eval_batch_size": 256,
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "replay": False,
            "sample_routing_training_allowed": False,
            "task_id_at_inference": False,
        },
        "planned_trajectory_count": len(slots),
        "primary_trajectory_count": 30,
        "robustness_trajectory_count": 60,
        "optimizer_steps_per_trajectory": 8000,
        "optimizer_steps_total_proposed": 720_000,
        "expected_train_example_presentations_per_trajectory": 1_000_000,
        "statistical_plan_proposed": {
            "primary_contrast": ["progressive_reserve", "fixed_sparse_mlp"],
            "order": CONFIRMATORY_ORDER,
            "paired_seed_ids": list(PROPOSED_SEED_IDS),
            "co_primary_metrics": list(METRICS),
            "percentile_bootstrap_draws": 10_000,
            "bootstrap_seed": 1337,
            "favorable_if_both_lower_two_sided_95ci_gt_zero": True,
            "unfavorable_if_both_upper_two_sided_95ci_lt_zero": True,
            "other_results_are_descriptive_not_confirmatory": True,
            "missing_or_nonfinite_trajectory_is_exclusion_allowed": False,
            "methodological_review_pending": True,
        },
        "resource_ledger": {
            "source_stagewise_digest": stage["record_sha256"],
            "reference_total_trainable": stage["reference_total_trainable"],
            "fixed_sparse_total_trainable": stage["fixed_sparse_total_trainable"],
            "allocated_parameter_error_relative": stage["physical_trainable_relative_error"],
            "per_task_ideal_linear_mac_error_within_10pct": (
                stage["per_task_ideal_relative_error_at_most_10pct"]
            ),
            "per_order_ideal_linear_macs": stage["all_task_linear_mac_estimates"],
            "actual_gpu_work_measured_for_full_runs": False,
            "masked_dense_matmul_is_sparse_gpu_kernel": False,
            "equivalent_real_gpu_compute_established": False,
        },
        "owner_decisions": _approval_queue(),
        "proposed_trajectory_slots": slots,
        "historical_technique_status": {
            "EXP-H1A-002": "INCONCLUSIVE",
            "EXP-H1A-ORDER-001": "ORDER_SENSITIVE",
            "EXP-H1A-EVALBATCH-001": "DIRECTION_FLIP_OBSERVED",
            "RigL_NISPA_admitted": False,
        },
    }
    data["record_sha256"] = sha256(canonical_json(data)).hexdigest()
    return data


def verify_m13b_prereg_candidate(data: Mapping[str, Any]) -> None:
    """Reject even rehashed edits to the fixed *candidate* plan.

    This deliberately does NOT accept any owner-approved/GO variant; an actual
    signed frozen protocol requires a separate new workflow and implementation.
    """
    if not isinstance(data, Mapping):
        raise TypeError("candidate must be a mapping")
    observed_sha = data.get("record_sha256")
    if not isinstance(observed_sha, str) or len(observed_sha) != 64:
        raise ValueError("missing candidate digest")
    bare = {k: v for k, v in data.items() if k != "record_sha256"}
    if sha256(canonical_json(bare)).hexdigest() != observed_sha:
        raise ValueError("candidate digest mismatch")
    expected = build_m13b_prereg_candidate()
    if canonical_json(data) != canonical_json(expected):
        raise ValueError("candidate diverges from exact approved-for-review template")
    if (data.get("science_executable") is not False
            or data.get("final_protocol_frozen") is not False
            or data.get("owner_approval_present") is not False):
        raise ValueError("this module refuses any science authorization")
