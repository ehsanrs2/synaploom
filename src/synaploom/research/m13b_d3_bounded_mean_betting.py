"""M1.3b D3: CPU-only candidate bounded paired-MEAN e-test feasibility.

For IID paired accuracy differences D_i in [-1, 1], under H0: E[D]<=0,
the fixed (non-adaptive) betting factor 1+D_i is nonnegative and has
expected value <=1. Thus E_n = product_i(1+D_i) has null expectation <=1
for independently sampled seeds. Markov gives P_H0(E_n>=20)<=0.05
without requiring symmetric or normal seed effects.

With two co-primary endpoints, an IUT of BOTH individually valid one-sided
component e-tests controls the global false-positive rate at <=0.05 under
the union null, regardless of inter-endpoint dependence. This is a
mathematical candidate and often underpowered: DO NOT approve science or n.

This module accepts no external files or actual research outcomes, touches
no GPU or model, and executes only fixed synthetic scenarios. Its generic
numeric helpers are tested on artificial arrays only.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
from math import ceil, comb, fsum, isfinite, log, log1p
from typing import Any, Sequence

from synaploom.research.m13b_manifest import canonical_json


KIND = "M13B_D3_BOUNDED_MEAN_EVALUE_FEASIBILITY_V1"
ALPHA = Fraction(1, 20)
THRESHOLD = Fraction(20)
MAX_ENUMERATION_N = 320
NULL_P_GRID = (
    Fraction(1, 2), Fraction(4, 5), Fraction(9, 10), Fraction(19, 20),
)
NULL_N_GRID = (10, 55, 100, 320)
ALT_N_GRID = (10, 55, 100, 160)
CONSTANT_GAIN_GRID = (
    Fraction(1, 100), Fraction(2, 100),
    Fraction(5, 100), Fraction(10, 100),
)
ALT_FAMILIES = (
    # (P(positive), positive delta, negative delta) -- invented distributions
    (Fraction(9, 10), Fraction(6, 100), Fraction(-10, 100)),
    (Fraction(9, 10), Fraction(3, 100), Fraction(-3, 100)),
)


def fixed_betting_log_evalue(deltas: Sequence[float]) -> float:
    """Compute log(product(1+D_i)) without numeric overflow.

    D_i are paired *accuracy fraction* differences in [-1,1]. The
    statistical interpretation requires independent IID training seeds;
    accepting numeric inputs does NOT verify that assumption or authorize
    their scientific use. We do not attempt data-dependent bet selection.
    """
    if not isinstance(deltas, (list, tuple)) or not deltas:
        raise ValueError("requires a nonempty sequence of paired differences")
    logs: list[float] = []
    for item in deltas:
        if type(item) not in (int, float):
            raise ValueError("paired differences must be numeric and not boolean")
        d = float(item)
        if not isfinite(d) or not -1 <= d <= 1:
            raise ValueError("paired differences must be finite fractions in [-1,1]")
        if d == -1:
            return float("-inf")
        logs.append(log1p(d))
    return fsum(logs)


def one_sided_betting_rejects(log_evalue: float) -> bool:
    """Fixed one-sided e-value threshold 1/alpha=20; no adaptive tuning."""
    if type(log_evalue) not in (float, int) or (
        not isfinite(float(log_evalue)) and float(log_evalue) != float("-inf")
    ):
        raise ValueError("invalid log e-value")
    return float(log_evalue) >= log(float(THRESHOLD))


def joint_both_needed_rejects(first: Sequence[float], second: Sequence[float]) -> bool:
    """IUT conjunctive positive claim, without assuming endpoint independence."""
    if len(first) != len(second):
        raise ValueError("co-primary endpoints require the same paired seed count")
    return (
        one_sided_betting_rejects(fixed_betting_log_evalue(first))
        and one_sided_betting_rejects(fixed_betting_log_evalue(second))
    )


def _two_point_distribution(
    p_positive: Fraction, positive: Fraction, negative: Fraction
) -> Fraction:
    if (type(p_positive) is not Fraction or not 0 < p_positive < 1
            or type(positive) is not Fraction or not 0 < positive <= 1
            or type(negative) is not Fraction or not -1 <= negative < 0):
        raise ValueError("must be bounded, rational two-point effect population")
    return p_positive * positive + (1-p_positive) * negative


def _exact_betting_threshold(
    n: int, positive: Fraction, negative: Fraction,
) -> int:
    """Smallest k with (1+positive)^k (1+negative)^(n-k)>=20, exact.

    Returns n+1 if even all n positive effects cannot reject.
    Monotonicity in k permits a bounded binary search using Fraction only.
    """
    if type(n) is not int or not 2 <= n <= MAX_ENUMERATION_N:
        raise ValueError("sample count must be a supported bounded integer")
    if not (type(positive) is Fraction and type(negative) is Fraction
            and -1 <= negative < 0 < positive <= 1):
        raise ValueError("invalid rational bounded two-point differences")

    upper = 1 + positive
    lower = 1 + negative

    def qualifies(k: int) -> bool:
        return upper**k * lower**(n-k) >= THRESHOLD

    if not qualifies(n):
        return n+1
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if qualifies(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _exact_binomial_upper_tail(n: int, k_min: int, p: Fraction) -> Fraction:
    """Exact success frequency for a fixed toy two-point population."""
    if (type(n) is not int or not 2 <= n <= MAX_ENUMERATION_N
            or type(k_min) is not int or not 0 <= k_min <= n+1
            or type(p) is not Fraction or not 0 <= p <= 1):
        raise ValueError("invalid bounded exact binomial inputs")
    return sum((
        Fraction(comb(n, k)) * p**k * (1-p)**(n-k)
        for k in range(k_min, n+1)
    ), Fraction(0))


def _smallest_constant_gain_n(delta: Fraction) -> int:
    """For all D_i=delta, smallest n where (1+delta)^n >=20, exactly."""
    if type(delta) is not Fraction or not 0 < delta <= 1:
        raise ValueError("constant gain must be a positive bounded rational")
    guess = max(1, ceil(log(float(THRESHOLD)) / log1p(float(delta))))
    factor = 1 + delta
    while factor**guess < THRESHOLD:
        guess += 1
    while guess > 1 and factor**(guess-1) >= THRESHOLD:
        guess -= 1
    return guess


def build_bounded_mean_evalue_review() -> dict[str, Any]:
    """Prospective CPU-only review: exact type-I in toys and power tradeoff."""
    null_cells: list[dict[str, Any]] = []
    for p in NULL_P_GRID:
        positive = Fraction(1, 100)
        negative = -positive * p / (1-p)
        assert _two_point_distribution(p, positive, negative) == 0
        for n in NULL_N_GRID:
            minimum_k = _exact_betting_threshold(n, positive, negative)
            exact = _exact_binomial_upper_tail(n, minimum_k, p)
            if exact > ALPHA:
                raise AssertionError("fixed betting e-test violated alpha in constructed null")
            null_cells.append({
                "hypothetical_p_positive": float(p),
                "hypothetical_positive_difference": float(positive),
                "hypothetical_negative_difference": float(negative),
                "population_mean_exactly_zero": True,
                "n_paired_seeds": n,
                "minimum_positive_count_for_rejection": (
                    None if minimum_k > n else minimum_k
                ),
                "exact_false_positive_probability": float(exact),
                "exact_false_positive_fraction": (
                    f"{exact.numerator}/{exact.denominator}"
                ),
                "nominal_one_sided_alpha": float(ALPHA),
            })

    constant_rows: list[dict[str, Any]] = []
    for delta in CONSTANT_GAIN_GRID:
        needed = _smallest_constant_gain_n(delta)
        assert (1 + delta)**(needed-1) < THRESHOLD <= (1+delta)**needed
        constant_rows.append({
            "invented_identical_gain_per_seed": float(delta),
            "first_n_able_to_reject_if_every_seed_has_exactly_this_gain": needed,
            "interpretation": "DEGENERATE_POSITIVE_ALTERNATIVE_NOT_REAL_POWER",
        })

    power_cells: list[dict[str, Any]] = []
    for p, positive, negative in ALT_FAMILIES:
        actual_mean = _two_point_distribution(p, positive, negative)
        if actual_mean <= 0:
            raise AssertionError("fabricated alternative must have positive mean")
        for n in ALT_N_GRID:
            minimum_k = _exact_betting_threshold(n, positive, negative)
            exact = _exact_binomial_upper_tail(n, minimum_k, p)
            power_cells.append({
                "invented_positive_seed_rate": float(p),
                "invented_positive_difference": float(positive),
                "invented_negative_difference": float(negative),
                "invented_population_mean_difference": float(actual_mean),
                "n_paired_seeds": n,
                "minimum_positive_count_for_rejection": (
                    None if minimum_k > n else minimum_k
                ),
                "one_endpoint_exact_power_under_this_toy": float(exact),
                "joint_both_needed_power_without_endpoint_dependence": (
                    "UNKNOWN__FRECHET_BOUNDS_ONLY"
                ),
            })

    report: dict[str, Any] = {
        "schema": 1,
        "kind": KIND,
        "status": "D3_CALIBRATED_BOUNDED_MEAN_TEST_CANDIDATE__NOT_SELECTED",
        "science_executable": False,
        "real_outcomes_loaded": False,
        "actual_optimizer_steps": 0,
        "training_or_gpu_accessed": False,
        "scientific_claim": "NONE",
        "owner_mean_estimand_preserved": True,
        "primary_order": "O0",
        "primary_contrast": ["progressive_reserve", "fixed_sparse_mlp"],
        "co_primary_metrics": [
            "final_average_accuracy", "late_task_within_task_mean_accuracy"
        ],
        "test_family": "FIXED_LAMBDA_ONE_BOUNDED_MEAN_EVALUE",
        "alpha_one_sided_positive_per_endpoint": float(ALPHA),
        "evalue_threshold": int(THRESHOLD),
        "evalue_equation": "PRODUCT_OVER_PAIRED_SEEDS(1 + DELTA_i)",
        "iid_training_seed_population_assumed_not_verified": True,
        "distribution_free_within_assumptions": True,
        "required_difference_support": [-1, 1],
        "null": "COMMON_IID_POPULATION_PAIRED_MEAN_DIFFERENCE_LE_ZERO",
        "one_endpoint_size_proof": (
            "For IID D in [-1,1] with E[D]<=0, each nonnegative (1+D) "
            "has expectation <=1; independence gives E[product(1+D)]<=1. "
            "Markov: P(product>=20)<=1/20."
        ),
        "global_positive_rule": "BOTH_ONE_SIDED_COMPONENT_EVALUES_AT_LEAST_20",
        "global_null": "AT_LEAST_ONE_COPRIMARY_POPULATION_MEAN_LE_ZERO",
        "joint_error_control": (
            "Intersection-union: if either component null holds, "
            "the joint event is a subset of rejecting that valid component. "
            "No assumption of independence between co-primary endpoints."
        ),
        "formal_negative_claim": "NOT_DEFINED__DESCRIPTIVE_ONLY",
        "method_chosen_for_actual_science": None,
        "final_seed_count": None,
        "minimum_mean_effect_owner_approved": False,
        "joint_power_established": False,
        "ready_for_science": False,
        "exact_zero_mean_null_stress": null_cells,
        "all_identical_positive_gain_thresholds": constant_rows,
        "exact_positive_mean_toy_power": power_cells,
        "source": {
            "paper": "Waudby-Smith and Ramdas (2024), Estimating means of bounded random variables by betting",
            "doi": "10.1093/jrsssb/qkad009",
            "use": "Elementary fixed-bet nonnegative-supermartingale special case, not a reproduction of full adaptive procedure",
        },
        "limitations": [
            "IID paired training-seed differences are a substantive assumption; consecutively named seeds do not prove independent sampling.",
            "This fixed lambda=1 construction is valid but can be severely underpowered for small accuracy improvements; a green toy-null size test is not evidence of adequate real power.",
            "All invented two-point null/positive-mean scenarios are mathematical only, not observed SynapLoom effects or variance estimates.",
            "The minimum all-positive n values are deterministic extreme best-case scenarios, NOT realistic power-based sample-size recommendations.",
            "The fixed bet and threshold were selected before outcomes; changing lambda/threshold after real results invalidates the advertised guarantee.",
            "A later model-specific optimizer/evaluation/failure policy and complete seed registry are still missing.",
            "Distinct primary negative claims would require their own preregistered error and multiplicity rules.",
            "Neither a CPU toy calculation, merged PR, nor acceptance of AI advice is GPU execution authorization.",
        ],
    }
    report["record_sha256"] = sha256(canonical_json(report)).hexdigest()
    return report


def main() -> int:
    import sys
    if len(sys.argv) != 1:
        raise SystemExit("No file/results/seed/method flags; CPU fixed toy study only")
    print(json.dumps(build_bounded_mean_evalue_review(), indent=2, sort_keys=True,
                     ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
