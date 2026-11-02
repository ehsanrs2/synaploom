"""M1.3b D3: exact finite-n Student-t null stress test, NO real science.

A one-sided paired Student t test is a one-sample t test on the paired-seed
differences. The ideal normal-differences t critical is NOT distribution-free.
For a constructed iid bounded TWO-POINT zero-mean population, enumerate the
full binomial sampling distribution exactly to calculate false rejection rates.

No Monte Carlo, no CIFAR, no file inputs, no model, no GPU, no test selection,
no inference for SynapLoom outcomes and no approved confirmatory seed count.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
from math import comb, isfinite, sqrt
from typing import Any

from synaploom.research.m13b_manifest import canonical_json


KIND = "M13B_D3_EXACT_STUDENT_T_SKEW_NULL_STRESS_V1"
ALPHA = Fraction(1, 20)
POSITIVE_EFFECT = Fraction(1, 100)
PROPOSED_N = 10
N_GRID = (10, 20, 40, 55, 100)
P_GRID = (
    Fraction(1, 2), Fraction(4, 5), Fraction(9, 10), Fraction(19, 20)
)
# One-sided alpha=.05 Student t quantiles (df=n-1), computed offline with
# SciPy 1.17.0 scipy.stats.t.ppf(.95, n-1). Frozen numerical references
# are used because scipy is NOT a runtime/dev dependency of SynapLoom.
T_CRITICAL_95 = {
    10: 1.833112932656237,
    20: 1.7291328115213682,
    40: 1.6848751217112248,
    55: 1.6735649063521605,
    100: 1.6603911560169906,
}


def two_point_zero_mean(p_positive: Fraction) -> tuple[Fraction, Fraction]:
    """Return (positive a=.01, negative b), with E[D] exactly zero."""
    if type(p_positive) is not Fraction or not 0 < p_positive < 1:
        raise ValueError("p_positive must be a strict rational probability")
    a = POSITIVE_EFFECT
    b = -p_positive * a / (1 - p_positive)
    if not -1 <= b < 0 < a <= 1 or p_positive * a + (1-p_positive) * b != 0:
        raise ValueError("not a bounded, exactly centered paired-difference null")
    return a, b


def student_t_from_binary_count(
    n: int, k: int, positive: Fraction, negative: Fraction,
) -> float:
    """Usual paired Student-t statistic from k positives and n-k negatives.

    Two-point sample variance, ddof=1:
      t = [k*a+(n-k)*b]*sqrt(n-1) / [(a-b)*sqrt(k*(n-k))].
    For a constant positive sample t=+infinity (as in standard t-test
    arithmetic) and for a constant negative sample t=-infinity. Both
    boundary conventions are explicit parts of the synthetic stress test.
    """
    if type(n) is not int or n < 2 or type(k) is not int or not 0 <= k <= n:
        raise ValueError("invalid n or positive count")
    if (type(positive) is not Fraction or type(negative) is not Fraction
            or not -1 <= negative < 0 < positive <= 1):
        raise ValueError("requires rational bounded opposite-sign effects")
    if k == n:
        return float("inf")
    if k == 0:
        return float("-inf")
    result = (
        float(k * positive + (n-k) * negative) * sqrt(n-1)
        / (float(positive-negative) * sqrt(k*(n-k)))
    )
    if not isfinite(result):
        raise ArithmeticError("nonfinite interior t statistic")
    return result


def binomial_upper_tail(n: int, k_min: int, p: Fraction) -> Fraction:
    """Exact Pr(K>=k_min) for K~Binomial(n,p), rational arithmetic."""
    if (type(n) is not int or n < 2 or type(k_min) is not int
            or not 0 <= k_min <= n+1
            or type(p) is not Fraction or not 0 <= p <= 1):
        raise ValueError("invalid exact binomial arguments")
    return sum((
        Fraction(comb(n, k)) * p**k * (1-p)**(n-k)
        for k in range(k_min, n+1)
    ), Fraction(0))


def exact_one_sided_t_null_rejection(n: int, p: Fraction) -> tuple[Fraction, int]:
    """Exact null false-rejection chance for the frozen test and toy family."""
    if n not in T_CRITICAL_95 or type(n) is not int:
        raise ValueError("n not in prospectively selected comparison grid")
    positive, negative = two_point_zero_mean(p)
    critical = T_CRITICAL_95[n]
    rejected_counts = [
        k for k in range(n+1)
        if student_t_from_binary_count(n, k, positive, negative) > critical
    ]
    if not rejected_counts or rejected_counts[-1] != n:
        raise AssertionError("all-positive zero-variance rejection convention changed")
    if rejected_counts != list(range(rejected_counts[0], n+1)):
        raise AssertionError("two-point rejection set is not a binomial upper tail")
    k_min = rejected_counts[0]
    return binomial_upper_tail(n, k_min, p), k_min


def build_exact_t_stress_report() -> dict[str, Any]:
    """Review-only exact null-risk table; no data-driven test selection."""
    cells: list[dict[str, Any]] = []
    for p in P_GRID:
        pos, neg = two_point_zero_mean(p)
        for n in N_GRID:
            actual, k_min = exact_one_sided_t_null_rejection(n, p)
            at_least = p**n
            if actual < at_least:
                raise AssertionError("null false rejection must include all-positive event")
            cells.append({
                "seed_count": n,
                "one_seed_positive_probability": float(p),
                "positive_difference_fraction": float(pos),
                "negative_difference_fraction": float(neg),
                "true_mean_difference_fraction": float(p*pos+(1-p)*neg),
                "paired_t_df": n-1,
                "one_sided_t_critical_alpha_005": T_CRITICAL_95[n],
                "minimum_positive_count_for_rejection": k_min,
                "null_rejection_probability_exact": (
                    f"{actual.numerator}/{actual.denominator}"
                ),
                "null_rejection_probability": float(actual),
                "all_positive_event_probability_lower_bound": float(at_least),
                "nominal_one_sided_alpha": float(ALPHA),
                "exceeds_nominal_005": actual > ALPHA,
                "both_endpoint_joint_false_favorable_if_identical_deltas": float(actual),
                "joint_two_endpoint_scope": "IDENTICAL_COPRIMARY_SEED_DELTAS_ONLY",
            })
    report: dict[str, Any] = {
        "schema": 1,
        "kind": KIND,
        "status": "CPU_EXACT_NULL_STRESS_ONLY__D3_METHOD_NOT_SELECTED",
        "real_outcomes_loaded": False,
        "model_or_features_loaded": False,
        "optimizer_steps_executed": 0,
        "science_executable": False,
        "scientific_claim": "NONE",
        "owner_mean_estimand_preserved": True,
        "test_under_stress": "PAIRED_STUDENT_T_ONE_SIDED_POSITIVE_MEAN",
        "test_alpha": float(ALPHA),
        "sample_sizes": list(N_GRID),
        "toy_positive_seed_probabilities": [float(p) for p in P_GRID],
        "pairing": "EACH_SYNTHETIC_SEED_DELTA_IS_PAIRED_ARM_DIFFERENCE",
        "sampling": "IID_BOUNDED_TWO_POINT_NULL_EXACT_BINOMIAL_ENUMERATION",
        "t_critical_source": "SciPy 1.17.0 scipy.stats.t.ppf(0.95,n-1) offline frozen table",
        "variance_zero_policy": "CONSTANT_POSITIVE_SAMPLE_T_PLUS_INFINITY_AND_REJECT",
        "co_primary_positive_conjunction": "IDENTICAL_DELTAS_MEANS_JOINT_EQUALS_SINGLE_ENDPOINT",
        "original_percentile_bootstrap": "ALREADY_INVALIDATED_UNDER_SKEW_IN_MERGED_PR42",
        "comparison_to_bounded_hoeffding": (
            "One-sided Hoeffding with full [-1,1] range is size-controlled "
            "if paired seeds are independent, but its n<=100 bound radius "
            "exceeds all .01 observed positive-toy sample means."
        ),
        "selection_outcome": "NONE__TECHNICAL_ADVISORY_NOT_AN_ADOPTED_INFERENCE_METHOD",
        "approved_paired_seed_count": None,
        "actual_type_i_for_synaploom": None,
        "limitations": [
            "Exact probabilities refer to a manufactured iid two-point distribution, not measured SynapLoom accuracy differences.",
            "No finite stress grid can prove or disprove adequate type-I behavior for all real possible seed distributions.",
            "The Student-t size property is exact under iid normal differences, a condition not provided by this bounded toy.",
            "The two endpoints are set identically to show a valid joint-null counterexample, not to estimate their actual correlation.",
            "The no-variance positive event uses the standard infinite-positive-t rejection convention; alternative policies require preregistration.",
            "Power for positive nonzero mean effects and real two-endpoint dependence remain unknown.",
            "Changing n or replacing the test based on real confirmatory outcomes is prohibited.",
            "Owner acceptance of AI advisory and success of CI do not authorize scientific GPU experiments.",
        ],
    }
    report["scenarios"] = cells
    report["record_sha256"] = sha256(canonical_json(report)).hexdigest()
    return report


def main() -> int:
    import sys
    if len(sys.argv) != 1:
        raise SystemExit("No external inputs/flags accepted; synthetic exact review only")
    print(json.dumps(build_exact_t_stress_report(), sort_keys=True,
                     indent=2, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
