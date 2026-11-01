"""M1.3b conditional owner design decision and preregistration READINESS only.

No training, GPU, data loading, final protocol signature, owner execution
authorization, or implicit approval from merging review pull requests.
The owner accepted the *design direction* in 2026-11-01 conversation; the
separate scientific review and final GPU authorization remain unresolved.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping

from synaploom.research.m13b_manifest import canonical_json
from synaploom.research.m13b_prereg_candidate import (
    build_m13b_prereg_candidate, verify_m13b_prereg_candidate,
)

KIND = "M13B_CONDITIONAL_OWNER_DESIGN_READINESS_V1"
OWNER_MESSAGE_DATE = "2026-11-01"
OWNER_EVIDENCE = "https://github.com/ehsanrs2/synaploom/issues/29#issuecomment-6553724926"
EXPECTED_BASE_MAIN_COMMIT = "7957a06e9858ce7a4e54c41313e1a079422fa477"
NO_GO = "NO_GO_CONFIRMATORY_TRAINING"


def _decision_record() -> dict[str, dict[str, Any]]:
    return {
        "D1_SCOPE": {
            "owner_direction": "ACCEPTED_FOR_CONTRACT_PREPARATION",
            "selection": "O0 primary, O1/O2 predeclared descriptive robustness",
            "planned_trajectories": 90,
            "approved_to_execute": False,
        },
        "D2_FAIRNESS": {
            "owner_direction": "ACCEPTED_FOR_CONTRACT_PREPARATION",
            "selection": "Separate physical allocated capacity, ideal MAC proxy, and measured actual GPU work",
            "method_fidelity_review": "PENDING",
            "approved_to_claim_equal_gpu_compute": False,
        },
        "D3_STATISTICS": {
            "owner_direction": "ACCEPTED_SUBJECT_TO_INDEPENDENT_METHODOLOGY_REVIEW",
            "selection": "Single O0 progressive vs fixed-sparse external contrast; both co-primary endpoints",
            "bootstrap_small_n_coverage_review": "PENDING",
            "multiple_testing_review": "PENDING",
            "approved_to_claim_statistical_validity": False,
        },
        "D4_RUN_MATRIX": {
            "owner_direction": "ACCEPTED_FOR_CONTRACT_PREPARATION",
            "selection": "3 arms, 10 paired proposed seeds 501..510, O0/O1/O2",
            "projected_optimizer_steps": 720_000,
            "maximum_gpu_hours_approved": None,
            "maximum_wall_hours_approved": None,
            "execution_command_approved": False,
        },
    }


def build_freeze_readiness_packet() -> dict[str, Any]:
    """Produce an exact, self-contained non-executable review packet.

    This packet never turns a user message into authority for an experiment
    runner. Checks are always conditional/fail-closed. SHA is a checksum only.
    """
    proposal = build_m13b_prereg_candidate()
    verify_m13b_prereg_candidate(proposal)
    if (proposal["source_base_commit"] !=
            "ce0fe5270d406b5f09cadb7d050eaaff737977e8"):
        raise ValueError("unexpected upstream candidate base")
    if proposal["planned_trajectory_count"] != 90:
        raise ValueError("proposed scientific trajectory count changed")
    if proposal["optimizer_steps_total_proposed"] != 720_000:
        raise ValueError("proposed optimizer workload changed")
    if proposal["science_executable"] is not False:
        raise ValueError("candidate incorrectly became executable")
    if proposal["historical_h1a_results_unchanged"] is not True:
        raise ValueError("historical evidence policy changed")
    decisions = _decision_record()
    outstanding = [
        "PR38_SYNTHETIC_STATISTICS_REVIEW_MERGE_STATUS_MUST_BE_RECHECKED",
        "INDEPENDENT_N10_STATISTICAL_COVERAGE_AND_MULTIPLICITY_REVIEW",
        "EXTERNAL_FIXED_SPARSE_METHOD_FIDELITY_AND_GPU_RESOURCE_FAIRNESS_REVIEW",
        "OWNER_APPROVED_GPU_BUDGET_AND_FAILURE_RETRY_POLICY",
        "FINAL_EXECUTION_SOURCE_COMMIT_AND_CLEAN_HOST_IDENTITY",
        "TRUSTED_FEATURE_AND_MASK_DIGEST_RECHECK_ON_EXECUTION_HOST",
        "FINAL_IMMUTABLE_PROTOCOL_HASH_WITH_INDEPENDENT_REVIEW",
        "MANUAL_EXPLICIT_SCIENCE_EXECUTION_AUTHORIZATION",
        "SCIENTIFIC_RUNNER_TESTS_AND_BOUNDED_HOST_QUALIFICATION",
    ]
    packet: dict[str, Any] = {
        "schema": 1,
        "kind": KIND,
        "study": proposal["study_id"],
        "status": "OWNER_DIRECTION_RECORDED__SCIENTIFIC_FREEZE_BLOCKED",
        "authorization": NO_GO,
        "science_executable": False,
        "owner_design_direction_accepted": True,
        "owner_execution_authorization_present": False,
        "independent_methodology_signoff_present": False,
        "final_protocol_frozen": False,
        "scientific_runner_authorized": False,
        "real_scientific_outcomes_observed": False,
        "evidence_source": {
            "owner_message_date": OWNER_MESSAGE_DATE,
            "owner_message": "ادغام انجام شد و با پیشنهادت برای مرحله بعد موافقم",
            "public_issue_audit_url": OWNER_EVIDENCE,
            "is_cryptographic_owner_attestation": False,
            "scope": "DESIGN_DIRECTION_ONLY_NOT_GPU_EXECUTION",
        },
        "git_provenance": {
            "main_commit_observed_when_preparing": EXPECTED_BASE_MAIN_COMMIT,
            "scientific_runner_source_sha": None,
            "git_head_verified_on_execution_host": False,
            "github_pr38_status_runtime_verified": False,
        },
        "candidate_preregistration_sha256": proposal["record_sha256"],
        "expected_feature_sha256": proposal["feature_bundle_sha256_expected"],
        "expected_order_sha256": proposal["order_fingerprints"],
        "candidate_mask_sha256": proposal["mask_sha256_proposed"],
        "decision_record": decisions,
        "proposed_scope": {
            "orders": ["O0", "O1", "O2"],
            "primary_order": "O0",
            "arms": list(proposal["arms_proposed"]),
            "seeds": list(proposal["seeds_proposed"]),
            "primary_trajectories": 30,
            "descriptive_robustness_trajectories": 60,
            "total_trajectories": 90,
            "optimizer_steps_per_trajectory": 8000,
            "optimizer_steps_total": 720_000,
        },
        "statistical_scope": {
            "primary_contrast": proposal["statistical_plan_proposed"]["primary_contrast"],
            "co_primary_metrics": proposal["statistical_plan_proposed"]["co_primary_metrics"],
            "candidate_inference_method": "paired bootstrap percentile, 10k draws",
            "independent_methodological_validation": "PENDING",
            "order_robustness": "descriptive, cluster ten shared seeds, never 30 independent seeds",
            "failed_seed_exclusion_permitted": False,
        },
        "resource_truth": {
            "parameter_match_only_is_not_compute_match": True,
            "ideal_macs_are_not_measured_gpu_flops": True,
            "masked_dense_kernel_is_not_sparse_compute": True,
            "measured_full_run_gpu_cost": None,
            "owner_approved_gpu_hours": None,
        },
        "blocking_gates": outstanding,
        "all_blocking_gates_cleared": False,
        "ready_for_scientific_execution": False,
        "next_state": "REQUEST_INDEPENDENT_REVIEW_AND_GPU_COST_LIMIT",
    }
    packet["record_sha256"] = sha256(canonical_json(packet)).hexdigest()
    return packet


def verify_freeze_readiness_packet(data: Mapping[str, Any]) -> None:
    """Verify exact approved-for-review template and refuse any claimed GO."""
    if not isinstance(data, Mapping):
        raise TypeError("expected mapping")
    supplied = data.get("record_sha256")
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise ValueError("missing freeze-readiness checksum")
    base = {k: v for k, v in data.items() if k != "record_sha256"}
    if sha256(canonical_json(base)).hexdigest() != supplied:
        raise ValueError("readiness checksum mismatch")
    reference = build_freeze_readiness_packet()
    if canonical_json(data) != canonical_json(reference):
        raise ValueError("readiness packet diverges from offline review contract")
    if (data["ready_for_scientific_execution"] is not False
            or data["owner_execution_authorization_present"] is not False
            or data["all_blocking_gates_cleared"] is not False):
        raise ValueError("cannot authorize scientific execution")
