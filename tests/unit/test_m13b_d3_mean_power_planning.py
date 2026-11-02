"""D3 selected mean estimand: review-only Gaussian power planning tests.

No actual model outputs, no CIFAR, no GPU, no protocol finalization.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import ceil, isclose, log, sqrt
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys

import pytest

from synaploom.research.m13b_manifest import canonical_json
from synaploom.research.m13b_d3_mean_power_planning import (
    ASSUMED_MEAN_EFFECTS, ASSUMED_PAIRED_SEED_SDS, CANDIDATE_N, KIND,
    OWNER_DECISION_URL, PRIMARY_ENDPOINTS,
    build_mean_power_planning_packet,
    joint_success_probability_bounds,
    normal_known_sd_one_sided_power,
    normal_known_sd_required_n,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "experiments" / "scripts" / "m13b_d3_mean_power_planning.py"


def test_owner_mean_benefit_decision_but_no_inference_approval():
    packet = build_mean_power_planning_packet()
    assert packet["kind"] == KIND
    assert packet["status"] == "OWNER_MEAN_ESTIMAND_RECORDED__METHOD_AND_SAMPLE_SIZE_PENDING"
    assert packet["science_executable"] is False
    assert packet["real_outcomes_read"] is False
    assert packet["optimizer_steps_executed"] == 0
    assert packet["scientific_claim"] == "NONE"
    assert packet["method_chosen"] is None
    assert packet["independent_statistical_review_complete"] is False
    assert packet["d2_baseline_fidelity_review_complete"] is False
    assert packet["owner_gpu_hours_budget"] is None
    assert packet["final_protocol_sha256"] is None
    assert packet["science_runner_present"] is False
    assert packet["ready_for_science"] is False
    assert packet["owner_decision"]["public_audit_url"] == OWNER_DECISION_URL
    assert packet["owner_decision"]["selected_estimand"] == "MEAN_OF_PAIRED_SEED_ACCURACY_DIFFERENCES"
    assert packet["owner_decision"]["contrast"] == ["progressive_reserve", "fixed_sparse_mlp"]
    assert packet["owner_decision"]["primary_order"] == "O0"
    assert packet["owner_decision"]["co_primary_accuracy_endpoints"] == list(PRIMARY_ENDPOINTS)
    assert packet["owner_decision"]["majority_positive_seed_claim_is_not_primary"] is True
    assert packet["owner_decision"]["owner_approved_final_inferential_method"] is False
    assert packet["owner_decision"]["owner_approved_confirmatory_seed_count"] is False
    assert packet["owner_decision"]["owner_approved_gpu_execution"] is False
    assert packet["current_proposed_seed_count"] == CANDIDATE_N == 10
    assert packet["current_proposed_seed_ids"] == list(range(501, 511))
    assert packet["initial_90_trajectory_study_remains_proposed_only"] is True
    assert packet["formal_negative_claim_policy"] == "UNDECIDED"
    assert packet["true_paired_seed_sd_known"] is False
    assert packet["effect_size_of_scientific_interest_owner_approved"] is False
    assert packet["record_sha256"] == sha256(canonical_json({
        k: v for k, v in packet.items() if k != "record_sha256"
    })).hexdigest()
    assert packet == build_mean_power_planning_packet()


def test_n10_known_sigma_normal_planning_proxy_is_not_joint_power():
    packet = build_mean_power_planning_packet()
    grid = packet["illustrative_paired_effect_sd_scenarios_only"]
    assert len(grid) == len(ASSUMED_MEAN_EFFECTS) * len(ASSUMED_PAIRED_SEED_SDS) == 9
    representative = next(x for x in grid if (
        x["invented_mean_gain_fraction"] == .02 and
        x["invented_paired_seed_sd_fraction"] == .05
    ))
    assert representative["illustrative_single_endpoint_power_at_n10"] == pytest.approx(
        0.3519940258168499, rel=1e-10
    )
    assert representative["marginal_80pct_power_necessary_n_for_joint_80pct"] == 39
    assert representative["marginal_90pct_power_sufficient_n_for_joint_80pct"] == 54
    assert representative["joint_both_needed_power_lower_bound_at_n10"] == 0
    assert representative["joint_both_needed_power_upper_bound_at_n10"] == pytest.approx(
        representative["illustrative_single_endpoint_power_at_n10"]
    )
    for row in grid:
        n1 = row["marginal_80pct_power_necessary_n_for_joint_80pct"]
        n2 = row["marginal_90pct_power_sufficient_n_for_joint_80pct"]
        assert n2 >= n1 >= 2
        delta = row["invented_mean_gain_fraction"]
        sd = row["invented_paired_seed_sd_fraction"]
        assert normal_known_sd_one_sided_power(
            n=n1, mean_effect=delta, paired_sd=sd
        ) >= .8 - 1e-12
        assert normal_known_sd_one_sided_power(
            n=n2, mean_effect=delta, paired_sd=sd
        ) >= .9 - 1e-12
        if n1 > 2:
            assert normal_known_sd_one_sided_power(
                n=n1-1, mean_effect=delta, paired_sd=sd
            ) < .8 + 1e-10


def test_analytic_normal_formula_is_explicit_and_monotonic():
    normal = NormalDist()
    critical = normal.inv_cdf(.95)
    actual = normal_known_sd_one_sided_power(n=10, mean_effect=.02, paired_sd=.05)
    assert actual == pytest.approx(
        normal.cdf(.02 * sqrt(10) / .05 - critical)
    )
    assert normal_known_sd_one_sided_power(n=30, mean_effect=.02, paired_sd=.05) > actual
    assert normal_known_sd_one_sided_power(n=10, mean_effect=.05, paired_sd=.05) > actual
    assert normal_known_sd_one_sided_power(n=10, mean_effect=.02, paired_sd=.10) < actual
    assert normal_known_sd_required_n(
        mean_effect=.01, paired_sd=.10
    ) == 619


def test_sharp_joint_power_bounds_do_not_assume_endpoint_independence():
    assert joint_success_probability_bounds(.35, .35) == (0, .35)
    low, high = joint_success_probability_bounds(.9, .9)
    assert low == pytest.approx(.8)
    assert high == pytest.approx(.9)
    assert joint_success_probability_bounds(1, 0) == (0, 0)
    assert joint_success_probability_bounds(.8, .9) == pytest.approx((.7, .8))


@pytest.mark.parametrize("args", [
    {"n": True, "mean_effect": .02, "paired_sd": .05},
    {"n": 1, "mean_effect": .02, "paired_sd": .05},
    {"n": 10, "mean_effect": 0., "paired_sd": .05},
    {"n": 10, "mean_effect": float("nan"), "paired_sd": .05},
    {"n": 10, "mean_effect": .02, "paired_sd": 0.},
    {"n": 10, "mean_effect": .02, "paired_sd": float("inf")},
    {"n": 10, "mean_effect": .02, "paired_sd": .05, "alpha": 1.},
    {"n": 10, "mean_effect": .02, "paired_sd": .05, "alpha": True},
])
def test_mean_power_inputs_fail_closed(args):
    with pytest.raises(ValueError):
        normal_known_sd_one_sided_power(**args)


@pytest.mark.parametrize("args", [
    {"mean_effect": -.01, "paired_sd": .05},
    {"mean_effect": .02, "paired_sd": .05, "alpha": 0},
    {"mean_effect": .02, "paired_sd": .05, "power": 0},
    {"mean_effect": .02, "paired_sd": "0.05"},
    {"mean_effect": .00000001, "paired_sd": 1.},
])
def test_required_n_invalid_or_unbounded(args):
    with pytest.raises(ValueError):
        normal_known_sd_required_n(**args)


@pytest.mark.parametrize("args", [
    (float("nan"), .2), (-.2, .4), (1.1, .3),
    (.5, float("inf")), (True, .5),
])
def test_joint_bounds_bad_inputs(args):
    with pytest.raises(ValueError):
        joint_success_probability_bounds(*args)


def _run_cli(*args):
    return subprocess.run([sys.executable, str(CLI), *args],
                          cwd=ROOT, text=True, capture_output=True, check=False)


def test_cli_review_and_exclusive_no_overwrite(tmp_path):
    out = tmp_path / "d3_mean_planning.json"
    first = _run_cli("--review-mean-power", "--output", str(out))
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout) == build_mean_power_planning_packet()
    assert json.loads(out.read_text()) == build_mean_power_planning_packet()
    second = _run_cli("--review-mean-power", "--output", str(out))
    assert second.returncode != 0
    assert json.loads(out.read_text()) == build_mean_power_planning_packet()


@pytest.mark.parametrize("args", [
    (), ("--execute",), ("--train",), ("--features", "data.pt"),
    ("--results", "real.json"), ("--method", "bootstrap"),
    ("--seed", "501"), ("--approve",), ("--freeze",),
    ("--review-mean-power", "--delta", ".02"),
    ("--review-mean-power", "--sigma", ".05"),
    ("--review-mean-power", "--owner-approve", "true"),
])
def test_cli_rejects_training_data_and_unapproved_choices(args):
    result = _run_cli(*args)
    assert result.returncode != 0
    assert "error:" in result.stderr
