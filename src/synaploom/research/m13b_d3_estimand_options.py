"""D3 prospective estimand choices, exact n=10 sign counts, bounded mean limits.

No scientific results, no GPU, no automatic test selection. The exact
binomial statements apply to positive-sign prevalence under iid seeds,
whereas Hoeffding applies to a mean of independent bounded differences.
They answer different scientific questions. Neither is selected or frozen.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from math import ceil, comb, isfinite, log, sqrt
from typing import Any

from synaploom.research.m13b_manifest import canonical_json

KIND = "M13B_D3_PROSPECTIVE_ESTIMAND_OPTIONS_V1"
N = 10
ALPHA = Fraction(1, 20)
SINGLE_SEED_DIFFERENCE_LOWER = -1
SINGLE_SEED_DIFFERENCE_UPPER = +1
POWER_POSITIVE_PROBABILITIES = (
    Fraction(3, 5), Fraction(7, 10), Fraction(4, 5), Fraction(9, 10)
)


def sign_upper_tail(k: int, *, n: int = N) -> Fraction:
    """Exact one-sided worst-null P(Bin(n,0.5)>=k) for sign prevalence.

    The null tested is P(delta>0)<=0.5. Observed zero deltas are NOT
    deleted: they count as nonpositive in n; independence of seed trials is
    an assumption. This is NOT a test of E(delta)>0.
    """
    if type(k) is not int or type(n) is not int or not (0 <= k <= n == N):
        raise ValueError("requires exactly ten fixed paired seed records")
    return sum((Fraction(comb(n, j), 2**n) for j in range(k, n + 1)), Fraction(0))


def sign_power(k_min: int, p_positive: Fraction, *, n: int = N) -> Fraction:
    """Binomial rejection probability for an invented iid positive-sign rate."""
    if (type(k_min) is not int or type(n) is not int
            or not 0 <= k_min <= n == N
            or type(p_positive) is not Fraction or not 0 <= p_positive <= 1):
        raise ValueError("invalid exact sign power parameters")
    return sum(
        (Fraction(comb(n, k)) * p_positive**k
         * (1-p_positive)**(n-k) for k in range(k_min, n+1)),
        Fraction(0),
    )


def hoeffding_mean_margin(
    *, n: int = N, alpha: Fraction = ALPHA,
    lower: int = SINGLE_SEED_DIFFERENCE_LOWER,
    upper: int = SINGLE_SEED_DIFFERENCE_UPPER,
) -> float:
    """One-sided bound on a mean of independent bounded per-seed differences.

    For Xi in [a,b], P(xbar - E[xbar] >= t) <= exp(-2 n t²/(b-a)²).
    If Xi are iid, E[xbar] is a common seed-population mean. With
    independently generated but nonidentically distributed seeds, the
    target is the average of those means.

    This is a distribution-free bounded-mean bound *conditional on the
    independence and bound assumptions*, not a CI for a median.
    """
    if (type(n) is not int or n < 1 or type(alpha) is not Fraction
            or not 0 < alpha < 1 or type(lower) is not int
            or type(upper) is not int or not lower < upper):
        raise ValueError("invalid Hoeffding bound parameters")
    radius = (upper-lower) * sqrt(log(1/float(alpha)) / (2*n))
    if not isfinite(radius):
        raise ValueError("nonfinite confidence radius")
    return radius


def build_estimand_options_review() -> dict[str, Any]:
    """Offline review packet with explicit different hypotheses and no GO."""
    threshold = next((k for k in range(N+1) if sign_upper_tail(k) <= ALPHA), None)
    if threshold is None:
        raise RuntimeError("no exact one-sided sign rejection threshold")
    p_at_threshold = sign_upper_tail(threshold)
    p_below = sign_upper_tail(threshold-1)
    if not p_at_threshold <= ALPHA < p_below:
        raise AssertionError("sign threshold not minimal")
    mean_one_sided = hoeffding_mean_margin()
    mean_two_direction = hoeffding_mean_margin(alpha=ALPHA/2)
    options: dict[str, Any] = {
        "MEAN_OF_PAIRED_SEED_DIFFERENCES": {
            "target": "E[accuracy(progressive)-accuracy(fixed_sparse)] over independently drawn random training seeds",
            "endpoint_unit": "accuracy fraction, difference bounded [-1,+1]",
            "null_for_positive": "mean difference <= 0 for at least one co-primary endpoint",
            "joint_positive": "both co-primary endpoint tests reject their nonpositive mean null",
            "conditional_guarantee": (
                "One-sided Hoeffding per endpoint controls size at 0.05 "
                "if paired seed differences are independent and in [-1,+1]. "
                "Joint all-positive via intersection-union has size <=0.05 "
                "if each component is valid; no independence between endpoints required."
            ),
            "one_sided_alpha": "1/20",
            "one_sided_hoeffding_lower_bound_radius": mean_one_sided,
            "minimum_observed_mean_to_certify_positive": mean_one_sided,
            "if_both_positive_and_negative_are_formal_directions": (
                "Split error between the two possible directions, "
                "e.g. two-sided 95% bounds with alpha/2 per tail. "
                "Conjunction still requires both co-primary endpoints."
            ),
            "two_direction_95pct_hoeffding_radius": mean_two_direction,
            "two_direction_alpha_per_tail": "1/40",
            "practicality": (
                "With n=10 and [-1,1] bounds the guaranteed mean-effect threshold "
                "is very large; this conservative check does not establish usable power."
            ),
            "decision": "NOT_SELECTED__INDEPENDENT_STATISTICAL_REVIEW_REQUIRED",
        },
        "POSITIVE_SIGN_PREVALENCE": {
            "target": "P(paired seed delta > 0) over independent identically distributed seed draws",
            "null_for_positive": "probability of a positive delta <= 0.5 for at least one co-primary endpoint",
            "zero_delta_policy": "Count zero as nonpositive, retain all ten seeds; no dropping ties",
            "joint_positive": "at least the exact one-sided sign-rejection threshold for BOTH endpoints",
            "one_sided_alpha": "1/20",
            "minimal_positive_seed_count_out_of_10": threshold,
            "at_threshold_exact_one_sided_p": (
                f"{p_at_threshold.numerator}/{p_at_threshold.denominator}"
            ),
            "at_threshold_one_sided_p_value": float(p_at_threshold),
            "one_fewer_exact_one_sided_p": (
                f"{p_below.numerator}/{p_below.denominator}"
            ),
            "one_fewer_one_sided_p_value": float(p_below),
            "toy_exact_rejection_probability_if_true_positive_sign_rate": {
                str(float(p)): float(sign_power(threshold, p))
                for p in POWER_POSITIVE_PROBABILITIES
            },
            "critical_limitation": (
                "A sign test targets positive-sign prevalence, NOT positive mean "
                "accuracy difference under skewed seed effects. "
                "One-sided exact binomial calibration requires independent "
                "signs and a fixed number of seed trials."
            ),
            "if_negative_direction_is_formally_tested": (
                "Specify a two-direction alpha allocation and exact negative-sign "
                "testing before outcomes; zeros count as failures for either direction."
            ),
            "decision": "NOT_SELECTED__DIFFERENT_SCIENTIFIC_ESTIMAND",
        },
        "PARAMETRIC_PAIRED_MEAN": {
            "target": "mean of paired differences, with explicit distributional assumptions",
            "candidate": "one-sided paired t tests or confidence intervals for both co-primary means",
            "requirement": (
                "A real reviewer must justify assumptions, small n behavior, "
                "outlier sensitivity, null/alternative, multiplicity and power. "
                "Do not label parametric t intervals distribution-free."
            ),
            "decision": "NOT_SELECTED__EXTERNAL_CALIBRATION_REQUIRED",
        },
        "SIGN_FLIP_SYMMETRY_TEST": {
            "target": "symmetric paired-seed location shift or a justified exchangeable-label null",
            "candidate": "exact paired sign-flip of mean differences, 2**10 sign patterns",
            "requirement": (
                "An exact sign-flip test requires suitable symmetry or sharp "
                "exchangeability assumptions; mean-zero alone is insufficient. "
                "Do not silently substitute this method for the skew-sensitive bootstrap."
            ),
            "decision": "NOT_SELECTED__ASSUMPTIONS_UNPROVEN",
        },
    }
    report: dict[str, Any] = {
        "schema": 1,
        "kind": KIND,
        "status": "D3_ESTIMAND_SELECTION_PENDING_INDEPENDENT_REVIEW",
        "science_executable": False,
        "real_data_loaded": False,
        "optimizer_steps": 0,
        "scientific_claim": "NONE",
        "candidate_co_primary_metrics": [
            "final_average_accuracy", "late_task_within_task_mean_accuracy"
        ],
        "candidate_primary_order": "O0",
        "candidate_paired_seed_count": N,
        "candidate_primary_contrast": ["progressive_reserve", "fixed_sparse_mlp"],
        "accepted_method": None,
        "accepted_estimand": None,
        "approved_type_i_error_plan": None,
        "reviewer_signoff": False,
        "owner_science_run_authorization": False,
        "alternative_estimands_and_tests": options,
        "review_requirements": [
            "Choose the scientific estimand before exposing any real confirmatory outcomes.",
            "Define iid random-seed population and independence; fixed ten predetermined seeds may support descriptive inference only.",
            "Predeclare 2-endpoint joint positive rule and whether negative verdict is formal or descriptive.",
            "No post-hoc seed exclusion or changes to O0/O1/O2 family; exact 90-slot registry is not GO.",
            "Provide independent distributional assumptions, alpha calibration and target power under n=10.",
            "Reassess the 10-seed budget if useful mean-effect inference cannot be supported.",
            "Approve a revised D3 separately from D2 baseline fidelity and GPU resource budget.",
        ],
    }
    report["record_sha256"] = sha256(canonical_json(report)).hexdigest()
    return report
