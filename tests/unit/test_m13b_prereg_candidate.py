"""Exact, strictly non-executable M1.3b preregistration candidate tests."""
from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from synaploom.protocols.h1a2 import H1A2_FEATURE_SHA256
from synaploom.protocols.h1a_order_sensitivity import EXPECTED_ORDER_FINGERPRINTS
from synaploom.research.m13b_manifest import canonical_json
from synaploom.research import m13b_prereg_candidate as proposal


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "experiments" / "scripts" / "m13b_prereg_candidate.py"


@pytest.fixture(scope="module")
def candidate():
    return proposal.build_m13b_prereg_candidate()


def test_full_matrix_permutation_and_balanced_arm_first(candidate):
    proposal.verify_m13b_prereg_candidate(candidate)
    assert candidate["planned_trajectory_count"] == 90
    assert candidate["primary_trajectory_count"] == 30
    assert candidate["robustness_trajectory_count"] == 60
    assert candidate["optimizer_steps_total_proposed"] == 720_000
    assert candidate["optimizer_steps_per_trajectory"] == 8_000
    assert candidate["expected_train_example_presentations_per_trajectory"] == 1_000_000
    assert candidate["seeds_proposed"] == list(range(501, 511))
    assert candidate["arms_proposed"] == list(proposal.ARMS)
    assert candidate["order_fingerprints"] == EXPECTED_ORDER_FINGERPRINTS
    assert set(candidate["pilot_seed_ids_not_used"]).isdisjoint(candidate["seeds_proposed"])
    keys = [s["slot_key"] for s in candidate["proposed_trajectory_slots"]]
    assert len(keys) == len(set(keys)) == 90

    for order_id in ("O0", "O1", "O2"):
        subset = [s for s in candidate["proposed_trajectory_slots"]
                  if s["order_id"] == order_id]
        assert len(subset) == 30
        for seed in proposal.PROPOSED_SEED_IDS:
            triple = [s for s in subset if s["seed"] == seed]
            assert len(triple) == 3
            assert set(s["arm"] for s in triple) == set(proposal.ARMS)
            assert sorted(s["arm_execution_position_in_block"] for s in triple) == [0, 1, 2]
            assert all(s["optimizer_steps_expected"] == 8000 for s in triple)
            assert all(s["order_sha256"] == EXPECTED_ORDER_FINGERPRINTS[order_id]
                       for s in triple)
            assert all(s["comparison_role"] == (
                "PRIMARY_O0" if order_id == "O0"
                else "DESCRIPTIVE_ORDER_ROBUSTNESS"
            ) for s in triple)
        counts = {arm: sum(s["arm"] == arm and
                           s["arm_execution_position_in_block"] == 0 for s in subset)
                  for arm in proposal.ARMS}
        assert max(counts.values()) - min(counts.values()) <= 1


def test_proposal_is_explicitly_not_owner_approved(candidate):
    assert candidate["kind"] == proposal.KIND
    assert candidate["status"] == "DESIGN_REVIEW_ONLY_UNFROZEN"
    assert candidate["authorization"] == "NO_GO_CONFIRMATORY_SCIENCE"
    assert candidate["science_executable"] is False
    assert candidate["confirmatory_outcomes_observed"] is False
    assert candidate["owner_approval_present"] is False
    assert candidate["independent_method_review_complete"] is False
    assert candidate["final_protocol_frozen"] is False
    assert candidate["science_runner_present"] is False
    assert candidate["local_feature_bytes_verified_now"] is False
    assert candidate["feature_bundle_sha256_expected"] == H1A2_FEATURE_SHA256
    assert candidate["historical_h1a_results_unchanged"] is True
    assert all(decision["status"] == "PENDING"
               for decision in candidate["owner_decisions"].values())
    assert set(candidate["owner_decisions"]) == {
        "D1_ORDER_SCOPE", "D2_RESOURCE_FAIRNESS",
        "D3_STATISTICS", "D4_EXECUTION_SCOPE",
    }


def test_statistical_scope_resource_accounting_and_no_promotion(candidate):
    stats = candidate["statistical_plan_proposed"]
    assert stats["order"] == "O0"
    assert stats["primary_contrast"] == ["progressive_reserve", "fixed_sparse_mlp"]
    assert stats["co_primary_metrics"] == [
        "final_average_accuracy", "late_task_within_task_mean_accuracy"
    ]
    assert stats["percentile_bootstrap_draws"] == 10_000
    assert stats["bootstrap_seed"] == 1337
    assert stats["missing_or_nonfinite_trajectory_is_exclusion_allowed"] is False
    assert stats["other_results_are_descriptive_not_confirmatory"] is True
    assert stats["methodological_review_pending"] is True
    resources = candidate["resource_ledger"]
    assert resources["reference_total_trainable"] == 1_172_196
    assert resources["fixed_sparse_total_trainable"] == 1_172_226
    assert resources["allocated_parameter_error_relative"] < 0.02
    assert resources["per_task_ideal_linear_mac_error_within_10pct"] is True
    assert resources["equivalent_real_gpu_compute_established"] is False
    assert resources["actual_gpu_work_measured_for_full_runs"] is False
    assert resources["masked_dense_matmul_is_sparse_gpu_kernel"] is False
    assert len(resources["source_stagewise_digest"]) == 64
    assert candidate["historical_technique_status"]["RigL_NISPA_admitted"] is False


def test_canonical_digest_and_identical_repeated_offline_candidate(candidate):
    state = torch.get_rng_state().clone()
    another = proposal.build_m13b_prereg_candidate()
    assert torch.equal(torch.get_rng_state(), state)
    assert another == candidate
    without = {k: v for k, v in candidate.items() if k != "record_sha256"}
    assert sha256(canonical_json(without)).hexdigest() == candidate["record_sha256"]


@pytest.mark.parametrize("tamper", [
    lambda c: c.update(science_executable=True),
    lambda c: c.update(owner_approval_present=True),
    lambda c: c["owner_decisions"]["D1_ORDER_SCOPE"].update(status="APPROVED"),
    lambda c: c["proposed_trajectory_slots"].pop(),
    lambda c: c["proposed_trajectory_slots"][0].update(seed=502),
    lambda c: c["resource_ledger"].update(equivalent_real_gpu_compute_established=True),
    lambda c: c["statistical_plan_proposed"].update(
        primary_contrast=["all_recruited", "fixed_sparse_mlp"]
    ),
])
def test_rehashed_tampering_still_rejected(candidate, tamper):
    forged = copy.deepcopy(candidate)
    tamper(forged)
    without = {k: v for k, v in forged.items() if k != "record_sha256"}
    forged["record_sha256"] = sha256(canonical_json(without)).hexdigest()
    with pytest.raises(ValueError, match="exact approved-for-review template"):
        proposal.verify_m13b_prereg_candidate(forged)


def test_corrupted_digest_rejected(candidate):
    forged = copy.deepcopy(candidate)
    forged["proposed_trajectory_slots"][0]["seed"] = 502
    with pytest.raises(ValueError, match="digest mismatch"):
        proposal.verify_m13b_prereg_candidate(forged)


def _cli(*flags):
    return subprocess.run([sys.executable, str(CLI), *flags],
                          cwd=ROOT, capture_output=True, text=True, check=False)


def test_cli_full_review_verify_and_exclusive_new_file(candidate, tmp_path):
    target = tmp_path / "candidate.json"
    run = _cli("--review-plan", "--output", str(target))
    assert run.returncode == 0, run.stderr
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved == candidate
    assert json.loads(run.stdout) == candidate
    verified = _cli("--verify-plan", str(target))
    assert verified.returncode == 0, verified.stderr
    verify_result = json.loads(verified.stdout)
    assert verify_result["verification"] == "MATCHES_UNAPPROVED_CANDIDATE_TEMPLATE"
    assert verify_result["science_executable"] is False
    assert verify_result["owner_approval_present"] is False
    assert verify_result["plan_sha256"] == candidate["record_sha256"]
    again = _cli("--review-plan", "--output", str(target))
    assert again.returncode != 0
    assert json.loads(target.read_text()) == candidate


@pytest.mark.parametrize("flags", [
    (), ("--execute",), ("--science",), ("--review-plan", "--cuda"),
    ("--review-plan", "--features", "data/feature.pt"),
    ("--review-plan", "--seed", "501"),
    ("--review-plan", "--arm", "progressive_reserve"),
    ("--review-plan", "--approve"), ("--review-plan", "--freeze"),
    ("--review-plan", "--verify-plan", "unused"),
])
def test_cli_refuses_unsupported_training_and_approval_flags(flags):
    failed = _cli(*flags)
    assert failed.returncode != 0
    assert "error:" in failed.stderr
