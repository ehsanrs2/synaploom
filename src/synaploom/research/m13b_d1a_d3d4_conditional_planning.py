"""M1.3b D1 option A: nonexecuting D3/D4 conditional planning research.

All distributions are invented exact rational two-point examples, not estimated
from protected results. The owner selected O0 / two arms for STUDY DESIGN only.
Nothing in this module can run a model, fetch data, select science n, or approve
GPU execution. Exact binomial probabilities are only conditional scenarios.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from math import comb
import json
from typing import Any

KIND = "M13B_D1A_D3D4_CONDITIONAL_FEASIBILITY_V1"
ALPHA = Fraction(1, 20)
THRESHOLD = 20
MEAN_ALTERNATIVE = Fraction(1, 50)
EVAL_BS = (256, 256, 256, 232)
N_GRID = (10, 54, 152, 210, 358)
PROGRESSIVE_SECONDS = 130.52830013199127
FIXED_SECONDS = 35.31436062100693
PAIR_SECONDS = PROGRESSIVE_SECONDS + FIXED_SECONDS
# Finite support; these are deliberately FABRICATED, not science variance.
SCENARIOS = (
    ("symmetric_sd_2pp", Fraction(1, 25), Fraction(0), Fraction(1, 2)),
    ("symmetric_sd_5pp", Fraction(7, 100), Fraction(-3, 100), Fraction(1, 2)),
    ("symmetric_sd_10pp", Fraction(3, 25), Fraction(-2, 25), Fraction(1, 2)),
    ("rare_large_loss", Fraction(3, 50), Fraction(-47, 50), Fraction(24, 25)),
)


def _validate_n(n: int) -> None:
    if type(n) is not int or not 1 <= n <= 400:
        raise ValueError("only 1..400 hypothetical paired seeds are permitted")


def _validate_two_point(high: Fraction, low: Fraction, p: Fraction) -> None:
    if (type(high) is not Fraction or type(low) is not Fraction
            or type(p) is not Fraction or not -1 <= low < high <= 1
            or not 0 < p < 1 or p * high + (1 - p) * low != MEAN_ALTERNATIVE):
        raise ValueError("invented supported two-point alternative must have exactly +2pp mean")


def rejection_positive_count(n: int, *, high: Fraction, low: Fraction,
                             p: Fraction) -> int:
    """Exact min #high draws for fixed-stake E=prod(1+D)>=20; n+1 if impossible."""
    _validate_n(n)
    _validate_two_point(high, low, p)
    def wins(k: int) -> bool:
        return (1 + high) ** k * (1 + low) ** (n - k) >= THRESHOLD
    if not wins(n):
        return n + 1
    left, right = 0, n
    while left < right:
        mid = (left + right) // 2
        if wins(mid):
            right = mid
        else:
            left = mid + 1
    return left


def exact_binomial_upper(n: int, k: int, p: Fraction) -> Fraction:
    """Exact P[Binomial(n,p)>=k] with built-in integers, no simulation."""
    _validate_n(n)
    if type(k) is not int or type(p) is not Fraction or not 0 <= p <= 1:
        raise ValueError("invalid exact binomial input")
    if k <= 0:
        return Fraction(1)
    if k > n:
        return Fraction(0)
    q = 1 - p
    return sum((Fraction(comb(n, j)) * p**j * q**(n-j)
                for j in range(k, n + 1)), Fraction(0))


def two_endpoint_scenario(n: int, *, high: Fraction,
                          low: Fraction, p: Fraction) -> dict[str, Any]:
    """Exact marginal and Fréchet bounds; coupling examples when p=1/2.

    Identical/independent/opposite *per-seed signs* are mathematical alternative
    couplings, not measured endpoint correlation or generic bounds.
    """
    k = rejection_positive_count(n, high=high, low=low, p=p)
    marginal = exact_binomial_upper(n, k, p)
    lower = max(Fraction(0), 2 * marginal - 1)
    result: dict[str, Any] = {
        "n_scenario_only": n,
        "positive_draws_required": k,
        "one_endpoint_power_exact": round(float(marginal), 8),
        "joint_power_frechet_lower": round(float(lower), 8),
        "joint_power_frechet_upper": round(float(marginal), 8),
        "two_endpoint_coupling_unobserved": True,
    }
    if p == Fraction(1, 2):
        if k > n // 2:
            opposite = Fraction(0)
        else:
            opposite = sum(
                (Fraction(comb(n, j), 2**n)
                 for j in range(k, n-k+1)), Fraction(0)
            )
        result.update({
            "joint_if_identical_signs_exact": round(float(marginal), 8),
            "joint_if_independent_signs_exact": round(float(marginal**2), 8),
            "joint_if_opposite_signs_exact": round(float(opposite), 8),
        })
    return result


def scenario_cost(n: int) -> dict[str, Any]:
    _validate_n(n)
    return {
        "n_scenario_only": n,
        "O0_two_arm_trajectories": 2 * n,
        "observed_engineering_pair_seconds_not_science": PAIR_SECONDS,
        "arithmetic_gpu_hours_at_single_engineering_point_rate":
            round(n * PAIR_SECONDS / 3600, 6),
        "illustrative_1_5x_hours_NOT_a_bound":
            round(1.5 * n * PAIR_SECONDS / 3600, 6),
        "actual_science_runtime_measured": False,
        "authorized_to_run": False,
    }


def scenario_capacity(hours: int) -> dict[str, Any]:
    if type(hours) is not int or not 1 <= hours <= 72:
        raise ValueError("scenario GPU-hours must be int in 1..72")
    max_n = int(hours * 3600 // (1.5 * PAIR_SECONDS))
    return {
        "budget_hours_scenario_not_owner_approved": hours,
        "arithmetic_max_n_under_assumed_1_5x_not_guaranteed": max_n,
        "real_gpu_budget_approved": False,
    }


def build_conditional_plan() -> dict[str, Any]:
    """No scientific results, data access, choices, or execution side effects."""
    report: dict[str, Any] = {
        "schema": 1,
        "kind": KIND,
        "status": "D1_OPTION_A_OWNER_SELECTED_FOR_DESIGN_ONLY_D3_D4_OPEN",
        "D1_owner_selected_design_scope": {
            "option": "A_O0_TWO_ARM",
            "order": "O0",
            "arms": ["progressive_reserve", "fixed_sparse_mlp"],
            "comparison": "population_mean_of_paired_seed_differences",
            "eval_grouping_positions_only": list(EVAL_BS),
            "order_robustness_inferred": False,
            "recruitment_timing_causality_inferred": False,
            "RigL_or_NISPA_qualified": False,
            "historical_90_slot_candidate_requires_prospective_amendment": True,
            "owner_approved_final_scientific_protocol": False,
        },
        "D3": {
            "two_coprimary_metrics": [
                "final_average_accuracy",
                "late_task_within_task_mean_accuracy",
            ],
            "meaningful_alternative_each_endpoint_fraction": float(MEAN_ALTERNATIVE),
            "positive_null_each_endpoint": "population_paired_mean <= 0",
            "global_claim": "both_valid_one_sided_component_tests_reject_IUT",
            "candidate_fixed_stake_e_test_only_NOT_SELECTED": True,
            "null_conditions_for_bet": "independent_bounded_pairs_with_common_mu_or_conditional_null_mean<=0",
            "per_endpoint_alpha": float(ALPHA),
            "joint_power_target_0_80_is_illustrative_not_owner_approved": True,
            "real_paired_seed_SD_or_dependence_measured": False,
            "actual_seed_generating_population_approved": False,
            "method_selected": None,
            "paired_seed_n_selected": None,
            "method_power_scenarios": [
                {
                    "invented_scenario": name,
                    "high_delta": float(high),
                    "low_delta": float(low),
                    "high_probability": float(p),
                    "true_mean_fraction_exact": float(MEAN_ALTERNATIVE),
                    "outcomes_protected_or_observed": False,
                    "grid": [
                        two_endpoint_scenario(n, high=high, low=low, p=p)
                        for n in N_GRID
                    ],
                }
                for name, high, low, p in SCENARIOS
            ],
        },
        "D4": {
            "owner_local_engineering_pair_seconds": PAIR_SECONDS,
            "costs_are_not_science_walltime_bounds": True,
            "approved_total_gpu_hours": None,
            "retry_timeout_policy_approved": False,
            "n_grid_cost": [scenario_cost(n) for n in N_GRID],
            "hour_capacity_scenarios": [
                scenario_capacity(h) for h in (3, 6, 12, 15, 24)
            ],
            "gpu_execution_authorized": False,
        },
        "final_protocol_frozen": False,
        "science_executable": False,
        "protected_scientific_outcomes_examined": False,
        "gpu_used": False,
        "scientific_claim": "NONE",
    }
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"),
                     allow_nan=False, ensure_ascii=True).encode("ascii")
    report["content_sha256"] = sha256(raw).hexdigest()
    return report


def main() -> None:
    print(json.dumps(build_conditional_plan(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
