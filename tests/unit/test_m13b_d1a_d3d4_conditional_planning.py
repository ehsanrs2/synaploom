"""No-GPU exact combinatorial tests for D1-A / D3-D4 design-only planning."""
from fractions import Fraction
import json

import pytest

from synaploom.research.m13b_d1a_d3d4_conditional_planning import (
    ALPHA, N_GRID, SCENARIOS, build_conditional_plan,
    exact_binomial_upper, rejection_positive_count,
    scenario_capacity, scenario_cost, two_endpoint_scenario,
)


def test_owner_D1_A_is_recorded_only_for_design_and_denies_science():
    report = build_conditional_plan()
    selected = report["D1_owner_selected_design_scope"]
    assert selected["option"] == "A_O0_TWO_ARM"
    assert selected["order"] == "O0"
    assert selected["arms"] == ["progressive_reserve", "fixed_sparse_mlp"]
    assert selected["historical_90_slot_candidate_requires_prospective_amendment"]
    assert selected["owner_approved_final_scientific_protocol"] is False
    assert selected["order_robustness_inferred"] is False
    assert selected["RigL_or_NISPA_qualified"] is False
    assert report["final_protocol_frozen"] is False
    assert report["science_executable"] is False
    assert report["gpu_used"] is False
    assert report["D3"]["method_selected"] is None
    assert report["D3"]["paired_seed_n_selected"] is None
    assert report["D4"]["approved_total_gpu_hours"] is None
    assert report["D4"]["gpu_execution_authorized"] is False
    assert report["scientific_claim"] == "NONE"


def test_exact_small_binomial_tails():
    assert exact_binomial_upper(3, 2, Fraction(1, 2)) == Fraction(1, 2)
    assert exact_binomial_upper(3, 4, Fraction(1, 2)) == 0
    assert exact_binomial_upper(3, 0, Fraction(1, 2)) == 1
    assert exact_binomial_upper(4, 4, Fraction(3, 4)) == (Fraction(3, 4) ** 4)


def test_exact_threshold_n210_sd5pp_and_frechet_joint_bound():
    p = Fraction(1, 2)
    hi, lo = Fraction(7, 100), Fraction(-3, 100)
    threshold = rejection_positive_count(210, high=hi, low=lo, p=p)
    assert threshold == 96
    record = two_endpoint_scenario(210, high=hi, low=lo, p=p)
    assert record["positive_draws_required"] == 96
    assert abs(record["one_endpoint_power_exact"] - 0.905155) < 2e-6
    assert abs(record["joint_power_frechet_lower"] - 0.81031) < 2e-6
    assert record["joint_power_frechet_lower"] <= record["joint_if_opposite_signs_exact"]
    assert record["joint_if_opposite_signs_exact"] <= record["joint_if_independent_signs_exact"]
    assert record["joint_if_independent_signs_exact"] <= record["joint_if_identical_signs_exact"]


def test_n10_has_no_fixed_bet_rejection_even_with_all_highs_in_sd5pp():
    high, low, p = Fraction(7, 100), Fraction(-3, 100), Fraction(1, 2)
    record = two_endpoint_scenario(10, high=high, low=low, p=p)
    assert record["positive_draws_required"] == 11
    assert record["one_endpoint_power_exact"] == 0
    assert record["joint_power_frechet_upper"] == 0


def test_rare_collapse_alternative_avoids_false_power_promise():
    high, low, p = Fraction(3, 50), Fraction(-47, 50), Fraction(24, 25)
    assert p*high+(1-p)*low == Fraction(1, 50)
    record = two_endpoint_scenario(210, high=high, low=low, p=p)
    assert record["positive_draws_required"] == 207
    assert abs(record["one_endpoint_power_exact"] - 0.02987967) < 1e-8
    assert record["joint_power_frechet_upper"] < 0.03


def test_all_constructed_scenarios_exact_two_percent_mean_and_support():
    for _, high, low, p in SCENARIOS:
        assert -1 <= low < high <= 1
        assert p*high+(1-p)*low == Fraction(1, 50)
        for n in N_GRID:
            record = two_endpoint_scenario(n, high=high, low=low, p=p)
            assert record["n_scenario_only"] == n
            assert 0 <= record["joint_power_frechet_lower"] <= record["joint_power_frechet_upper"] <= 1


def test_cost_calculation_is_point_rate_only_not_budget_approval():
    assert scenario_cost(210)["O0_two_arm_trajectories"] == 420
    assert abs(scenario_cost(210)[
        "arithmetic_gpu_hours_at_single_engineering_point_rate"
    ] - 9.674155) < 0.00002
    assert abs(scenario_cost(152)[
        "arithmetic_gpu_hours_at_single_engineering_point_rate"
    ] - 7.00224) < 0.00002
    assert scenario_capacity(15)[
        "arithmetic_max_n_under_assumed_1_5x_not_guaranteed"
    ] == 217
    assert not scenario_capacity(15)["real_gpu_budget_approved"]


@pytest.mark.parametrize("bad_n", [0, 401, True, 1.0, "210", -1])
def test_fail_closed_invalid_n(bad_n):
    with pytest.raises(ValueError, match="1..400"):
        scenario_cost(bad_n)


@pytest.mark.parametrize("bad", [0, 73, True, "12"])
def test_fail_closed_invalid_budget_scenario(bad):
    with pytest.raises(ValueError, match="GPU-hours"):
        scenario_capacity(bad)


def test_report_deterministic_json_and_content_hash_is_not_signature():
    a = build_conditional_plan()
    b = build_conditional_plan()
    assert a == b
    assert a["content_sha256"] == b["content_sha256"]
    assert len(a["content_sha256"]) == 64
    json.dumps(a, allow_nan=False)
    assert a["D3"]["real_paired_seed_SD_or_dependence_measured"] is False
    assert a["D4"]["costs_are_not_science_walltime_bounds"] is True
    assert ALPHA == Fraction(1, 20)
