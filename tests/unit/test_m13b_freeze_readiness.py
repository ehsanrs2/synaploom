"""Offline owner decision packet checks. Never approve scientific execution."""
from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from synaploom.research.m13b_manifest import canonical_json
from synaploom.research.m13b_freeze_readiness import (
    KIND, NO_GO, OWNER_EVIDENCE, build_freeze_readiness_packet,
    verify_freeze_readiness_packet,
)
from synaploom.research.m13b_smalln_method_sensitivity import (
    KIND as TOY_KIND, SCENARIOS, simulate_n10_method_sensitivity,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "experiments" / "scripts" / "m13b_freeze_review.py"


@pytest.fixture(scope="module")
def packet():
    return build_freeze_readiness_packet()


def test_owner_design_accepted_but_not_scientific_go(packet):
    verify_freeze_readiness_packet(packet)
    assert packet["kind"] == KIND
    assert packet["authorization"] == NO_GO
    assert packet["owner_design_direction_accepted"] is True
    assert packet["science_executable"] is False
    assert packet["owner_execution_authorization_present"] is False
    assert packet["independent_methodology_signoff_present"] is False
    assert packet["final_protocol_frozen"] is False
    assert packet["scientific_runner_authorized"] is False
    assert packet["real_scientific_outcomes_observed"] is False
    assert packet["ready_for_scientific_execution"] is False
    assert packet["all_blocking_gates_cleared"] is False
    assert packet["evidence_source"]["public_issue_audit_url"] == OWNER_EVIDENCE
    assert packet["evidence_source"]["is_cryptographic_owner_attestation"] is False
    assert packet["git_provenance"]["scientific_runner_source_sha"] is None
    assert packet["git_provenance"]["github_pr38_status_runtime_verified"] is False
    assert len(packet["blocking_gates"]) >= 7


def test_owner_direction_is_specific_and_budget_unknown(packet):
    d = packet["decision_record"]
    assert set(d) == {"D1_SCOPE", "D2_FAIRNESS", "D3_STATISTICS", "D4_RUN_MATRIX"}
    assert d["D1_SCOPE"]["owner_direction"] == "ACCEPTED_FOR_CONTRACT_PREPARATION"
    assert d["D1_SCOPE"]["planned_trajectories"] == 90
    assert d["D1_SCOPE"]["approved_to_execute"] is False
    assert d["D2_FAIRNESS"]["approved_to_claim_equal_gpu_compute"] is False
    assert d["D2_FAIRNESS"]["method_fidelity_review"] == "PENDING"
    assert d["D3_STATISTICS"]["bootstrap_small_n_coverage_review"] == "PENDING"
    assert d["D3_STATISTICS"]["multiple_testing_review"] == "PENDING"
    assert d["D3_STATISTICS"]["approved_to_claim_statistical_validity"] is False
    assert d["D4_RUN_MATRIX"]["projected_optimizer_steps"] == 720_000
    assert d["D4_RUN_MATRIX"]["execution_command_approved"] is False
    assert d["D4_RUN_MATRIX"]["maximum_gpu_hours_approved"] is None
    assert d["D4_RUN_MATRIX"]["maximum_wall_hours_approved"] is None
    assert packet["proposed_scope"]["total_trajectories"] == 90
    assert packet["proposed_scope"]["optimizer_steps_total"] == 720_000
    assert packet["statistical_scope"]["failed_seed_exclusion_permitted"] is False
    assert packet["resource_truth"]["masked_dense_kernel_is_not_sparse_compute"] is True
    assert packet["resource_truth"]["measured_full_run_gpu_cost"] is None


def test_original_candidate_hash_stays_stable_and_packet_rng_isolation(packet):
    old = torch.get_rng_state().clone()
    next_packet = build_freeze_readiness_packet()
    assert torch.equal(old, torch.get_rng_state())
    assert packet == next_packet
    pure = {k: v for k, v in packet.items() if k != "record_sha256"}
    assert sha256(canonical_json(pure)).hexdigest() == packet["record_sha256"]
    assert len(packet["candidate_preregistration_sha256"]) == 64


@pytest.mark.parametrize("change", [
    lambda o: o.update(science_executable=True),
    lambda o: o.update(ready_for_scientific_execution=True),
    lambda o: o.update(all_blocking_gates_cleared=True),
    lambda o: o.update(owner_execution_authorization_present=True),
    lambda o: o["decision_record"]["D4_RUN_MATRIX"].update(maximum_gpu_hours_approved=10),
    lambda o: o["decision_record"]["D3_STATISTICS"].update(multiple_testing_review="PASS"),
    lambda o: o["blocking_gates"].clear(),
    lambda o: o["proposed_scope"].update(total_trajectories=30),
    lambda o: o["evidence_source"].update(is_cryptographic_owner_attestation=True),
])
def test_even_rehashed_approval_forgery_is_rejected(packet, change):
    forged = copy.deepcopy(packet)
    change(forged)
    payload = {k: v for k, v in forged.items() if k != "record_sha256"}
    forged["record_sha256"] = sha256(canonical_json(payload)).hexdigest()
    with pytest.raises(ValueError, match="diverges"):
        verify_freeze_readiness_packet(forged)


def test_corrupt_digest_rejected(packet):
    forged = copy.deepcopy(packet)
    forged["blocking_gates"].pop()
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_freeze_readiness_packet(forged)


def test_synthetic_small_n_scenarios_are_bounded_and_do_not_claim_validity():
    one = simulate_n10_method_sensitivity(trials=30, bootstrap_draws=120, seed=19)
    again = simulate_n10_method_sensitivity(trials=30, bootstrap_draws=120, seed=19)
    assert one == again
    assert one["kind"] == TOY_KIND
    assert one["synthetic_only"] is True
    assert one["actual_scientific_results_loaded"] is False
    assert one["science_execution_authorized"] is False
    assert one["scientific_claim"] == "NONE"
    assert one["n_paired_seeds"] == 10
    assert one["monte_carlo_trials_per_scenario"] == 30
    assert one["bootstrap_draws_per_trial"] == 120
    assert set(one["toy_results"]) == {x[0] for x in SCENARIOS}
    for scenario, data in one["toy_results"].items():
        counts = data["toy_classification_counts"]
        assert set(counts) == {
            "SYNTHETIC_FAVORABLE", "SYNTHETIC_UNFAVORABLE",
            "SYNTHETIC_INCONCLUSIVE",
        }
        assert sum(counts.values()) == 30
        assert 0 <= data["toy_favorable_rate"] <= 1
        if scenario in ("both_null", "partial_null_other_strong"):
            assert data["toy_favorable_when_at_least_one_null_count"] == counts["SYNTHETIC_FAVORABLE"]
        else:
            assert data["toy_favorable_when_at_least_one_null_count"] is None
    pure = {k: v for k, v in one.items() if k != "record_sha256"}
    assert sha256(canonical_json(pure)).hexdigest() == one["record_sha256"]


@pytest.mark.parametrize("config", [
    {"trials": 0}, {"trials": True}, {"trials": 501},
    {"bootstrap_draws": 0}, {"bootstrap_draws": 1001},
    {"seed": -1}, {"seed": 1.5},
])
def test_toy_calibration_bounds(config):
    with pytest.raises(ValueError, match="invalid bounded"):
        simulate_n10_method_sensitivity(**config)


def _cli(*flags):
    return subprocess.run([sys.executable, str(CLI), *flags],
                          cwd=ROOT, capture_output=True, text=True, check=False)


def test_cli_readiness_and_exclusive_output(tmp_path, packet):
    path = tmp_path / "m13b-review.json"
    result = _cli("--readiness", "--output", str(path))
    assert result.returncode == 0, result.stderr
    from_stdout = json.loads(result.stdout)
    assert from_stdout == packet == json.loads(path.read_text())
    verified = _cli("--verify-readiness", str(path))
    assert verified.returncode == 0, verified.stderr
    data = json.loads(verified.stdout)
    assert data["ready_for_scientific_execution"] is False
    assert data["status"] == "MATCHES_NONEXECUTABLE_OWNER_REVIEW_PACKET"
    repeat = _cli("--readiness", "--output", str(path))
    assert repeat.returncode != 0
    assert json.loads(path.read_text()) == packet


def test_cli_synthetic_review_only():
    outcome = _cli("--synthetic-method-review")
    assert outcome.returncode == 0, outcome.stderr
    parsed = json.loads(outcome.stdout)
    assert parsed["kind"] == TOY_KIND
    assert parsed["science_execution_authorized"] is False
    assert parsed["actual_scientific_results_loaded"] is False


@pytest.mark.parametrize("flags", [
    (), ("--execute",), ("--readiness", "--approve"),
    ("--readiness", "--freeze"), ("--readiness", "--gpu"),
    ("--readiness", "--budget-hours", "10"),
    ("--readiness", "--scientific-seed", "501"),
    ("--synthetic-method-review", "--features", "frozen.pt"),
    ("--readiness", "--synthetic-method-review"),
    ("--verify-readiness", "any.json", "--output", "new.json"),
])
def test_cli_forbids_training_approval_and_real_data(flags):
    x = _cli(*flags)
    assert x.returncode != 0
    assert "error:" in x.stderr
