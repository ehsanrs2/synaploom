"""M1.3b owner-selected mean estimand: prospective Gaussian *planning* proxy.

No inference from real experiments. Normal location/known-SD power is only a
transparent design approximation; real accuracy differences are bounded and
the true paired-seed SD and distribution are unknown. Never let this module
select a method, infer power from observed confirmatory scores or approve GPU.
"""
from __future__ import annotations

from hashlib import sha256
from math import ceil, isfinite, sqrt
from statistics import NormalDist
from typing import Any

from synaploom.research.m13b_manifest import canonical_json


KIND = "M13B_D3_OWNER_MEAN_ESTIMAND_POWER_PLANNING_V1"
OWNER_DECISION_URL = "https://github.com/ehsanrs2/synaploom/issues/29#issuecomment-6557812225"
PRIMARY_ENDPOINTS = (
    "final_average_accuracy",
    "late_task_within_task_mean_accuracy",
)
ASSUMED_MEAN_EFFECTS = (0.01, 0.02, 0.05)
ASSUMED_PAIRED_SEED_SDS = (0.02, 0.05, 0.10)
CANDIDATE_N = 10
MARGINAL_ONE_SIDED_ALPHA = 0.05
PLANNING_TARGET_JOINT_POWER = 0.80
STANDARD_NORMAL = NormalDist()


def _valid_int(value: int, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be integer >= {minimum}")


def _valid_probability(value: float, name: str) -> None:
    if type(value) not in (int, float) or not isfinite(float(value)) or not 0 < value < 1:
        raise ValueError(f"{name} must be a finite probability strictly between 0 and 1")


def _valid_mean_sd(mean_effect: float, paired_sd: float) -> None:
    if (type(mean_effect) not in (int, float)
            or not isfinite(float(mean_effect)) or not 0 < mean_effect <= 1):
        raise ValueError("mean_effect must be finite and in (0,1]")
    if (type(paired_sd) not in (int, float)
            or not isfinite(float(paired_sd)) or not 0 < paired_sd <= 1):
        raise ValueError("paired_sd must be finite and in (0,1]")


def normal_known_sd_one_sided_power(
    *, n: int, mean_effect: float, paired_sd: float,
    alpha: float = MARGINAL_ONE_SIDED_ALPHA,
) -> float:
    """One-ENDPOINT power under an approximate Gaussian with KNOWN seed SD.

    P(Z > z_(1-alpha) | true location mean_effect)
      = Phi(mean_effect*sqrt(n)/paired_sd - z_(1-alpha)).
    It is NOT paired-t power (unknown SD), not distribution-free, not real
    measured SynapLoom power and not joint two-endpoint power.
    """
    _valid_int(n, "n", minimum=2)
    _valid_mean_sd(mean_effect, paired_sd)
    _valid_probability(alpha, "alpha")
    critical = STANDARD_NORMAL.inv_cdf(1 - alpha)
    return STANDARD_NORMAL.cdf(mean_effect * sqrt(n) / paired_sd - critical)


def normal_known_sd_required_n(
    *, mean_effect: float, paired_sd: float,
    alpha: float = MARGINAL_ONE_SIDED_ALPHA, power: float = 0.80,
) -> int:
    """Smallest integer n>=2 meeting the Gaussian known-SD one-sided proxy.

    This is a planning scenario only, and n is not owner approved. For
    two co-primary endpoints, setting marginal power = (1+target_joint)/2
    gives a distribution-independent-between-ENDPOINTS union-bound
    sufficiency condition, but still assumes each marginal Gaussian proxy.
    """
    _valid_mean_sd(mean_effect, paired_sd)
    _valid_probability(alpha, "alpha")
    _valid_probability(power, "power")
    z_alpha = STANDARD_NORMAL.inv_cdf(1 - alpha)
    z_beta = STANDARD_NORMAL.inv_cdf(power)
    raw = ((z_alpha + z_beta) * paired_sd / mean_effect) ** 2
    if not isfinite(raw) or raw > 10_000_000:
        raise ValueError("unbounded planning sample size")
    n = max(2, ceil(raw))
    # Floating-point guard for borderline integer threshold.
    while normal_known_sd_one_sided_power(
        n=n, mean_effect=mean_effect, paired_sd=paired_sd, alpha=alpha
    ) + 1e-13 < power:
        n += 1
    return n


def joint_success_probability_bounds(
    first_endpoint_power: float, second_endpoint_power: float,
) -> tuple[float, float]:
    """Sharp Frechet bounds without assuming endpoint independence."""
    for value in (first_endpoint_power, second_endpoint_power):
        if type(value) not in (int, float) or not isfinite(float(value)) or not 0 <= value <= 1:
            raise ValueError("endpoint rejection probabilities must be in [0,1]")
    return (max(0., first_endpoint_power + second_endpoint_power - 1.),
            min(first_endpoint_power, second_endpoint_power))


def build_mean_power_planning_packet() -> dict[str, Any]:
    """Hash-bound source-derived, non-executable prospective D3 review packet."""
    scenarios: list[dict[str, Any]] = []
    for delta in ASSUMED_MEAN_EFFECTS:
        for sigma in ASSUMED_PAIRED_SEED_SDS:
            p10 = normal_known_sd_one_sided_power(
                n=CANDIDATE_N, mean_effect=delta, paired_sd=sigma
            )
            min_joint, max_joint = joint_success_probability_bounds(p10, p10)
            n_marginal_80 = normal_known_sd_required_n(
                mean_effect=delta, paired_sd=sigma,
                power=PLANNING_TARGET_JOINT_POWER,
            )
            # By P(A∩B)>=P(A)+P(B)-1: if both per-endpoint
            # powers >=0.9, global both-needed power >=0.8.
            n_joint_sufficient = normal_known_sd_required_n(
                mean_effect=delta, paired_sd=sigma,
                power=(1 + PLANNING_TARGET_JOINT_POWER) / 2,
            )
            if n_joint_sufficient < n_marginal_80:
                raise AssertionError("joint sufficient planning n below marginal necessary n")
            scenarios.append({
                "invented_mean_gain_fraction": delta,
                "invented_paired_seed_sd_fraction": sigma,
                "illustrative_single_endpoint_power_at_n10": p10,
                "joint_both_needed_power_lower_bound_at_n10": min_joint,
                "joint_both_needed_power_upper_bound_at_n10": max_joint,
                "marginal_80pct_power_necessary_n_for_joint_80pct": n_marginal_80,
                "marginal_90pct_power_sufficient_n_for_joint_80pct": n_joint_sufficient,
                "guarantee_scope": "GAUSSIAN_KNOWN_SD_PROXY_ONLY",
            })
    packet: dict[str, Any] = {
        "schema": 1,
        "kind": KIND,
        "status": "OWNER_MEAN_ESTIMAND_RECORDED__METHOD_AND_SAMPLE_SIZE_PENDING",
        "science_executable": False,
        "scientific_claim": "NONE",
        "real_outcomes_read": False,
        "optimizer_steps_executed": 0,
        "owner_decision": {
            "public_audit_url": OWNER_DECISION_URL,
            "selected_estimand": "MEAN_OF_PAIRED_SEED_ACCURACY_DIFFERENCES",
            "contrast": ["progressive_reserve", "fixed_sparse_mlp"],
            "primary_order": "O0",
            "co_primary_accuracy_endpoints": list(PRIMARY_ENDPOINTS),
            "global_positive_claim_rule": "BOTH_CO_PRIMARY_MEAN_EFFECTS_POSITIVE_WITH_VALID_INFERENCE",
            "majority_positive_seed_claim_is_not_primary": True,
            "owner_approved_final_inferential_method": False,
            "owner_approved_confirmatory_seed_count": False,
            "owner_approved_gpu_execution": False,
        },
        "current_proposed_seed_count": CANDIDATE_N,
        "current_proposed_seed_ids": list(range(501, 511)),
        "initial_90_trajectory_study_remains_proposed_only": True,
        "formal_negative_claim_policy": "UNDECIDED",
        "alpha_per_one_sided_positive_endpoint_for_review": MARGINAL_ONE_SIDED_ALPHA,
        "joint_positive_rule": "INTERSECTION_UNION_IF_COMPONENT_TESTS_VALID",
        "planning_target_joint_power": PLANNING_TARGET_JOINT_POWER,
        "proxy_model": "NORMAL_KNOWN_PAIRED_SEED_SD_FOR_PLANNING_NOT_INFERENCE",
        "true_paired_seed_sd_known": False,
        "effect_size_of_scientific_interest_owner_approved": False,
        "illustrative_paired_effect_sd_scenarios_only": scenarios,
        "reviewer_questions": [
            "Are the fixed seed IDs a defensible independent sample from an explicit stochastic training-seed population?",
            "What minimum mean accuracy gain is scientifically meaningful for both endpoints, before actual outcomes?",
            "What predeclared plausible paired-seed variability is externally justified without protected M1.3b outcome leakage?",
            "Can an inferential method for means control small-n size under plausible skew/outliers, unlike the original percentile bootstrap counterexample?",
            "Can n=10 attain target joint power under a defensible model, or must the seed/90-slot proposal change before freezing?",
            "Will negative verdict be descriptive or a separately protected inferential direction?",
            "Are all missing/failed/nonfinite seeds preserved and handled fail-closed with precommitted bounded retries?",
            "Does an independent D2 reviewer accept fixed-mask MLP as a parameter-capacity comparator without compute-matched claims?",
        ],
        "method_chosen": None,
        "independent_statistical_review_complete": False,
        "d2_baseline_fidelity_review_complete": False,
        "owner_gpu_hours_budget": None,
        "final_protocol_sha256": None,
        "science_runner_present": False,
        "ready_for_science": False,
        "limitations": [
            "Normal known-SD proxy assumes a Gaussian approximation and a known paired-seed SD; neither is established for n=10 bounded accuracies.",
            "These SDs and mean gains are invented planning assumptions, NOT estimates from real SynapLoom paired outcomes.",
            "Per-endpoint power and Frechet bounds do not assert independence of the co-primary outcomes.",
            "A necessary marginal n and a sufficient worst-correlation n are planning quantities, not an approved power calculation for the real experiment.",
            "If formal claims are made in both directions, the alpha allocation requires separate pre-outcome independent review.",
            "No inference method, new seeds, budget, real data, evaluation or GPU execution has been authorized.",
        ],
    }
    packet["record_sha256"] = sha256(canonical_json(packet)).hexdigest()
    return packet
