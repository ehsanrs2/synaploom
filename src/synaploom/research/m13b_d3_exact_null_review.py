"""M1.3b D3: an exact small-n null counterexample, NOT science inference.

No Monte Carlo error: enumerate the ideal percentile bootstrap event for iid
two-point distributions with exactly ten paired seeds. Both co-primary
endpoints share identical paired effects (correlation 1), so an all-positive
sample causes the proposed BOTH-needed 95% percentile CI rule to claim
"SYNTHETIC_FAVORABLE" under a *joint zero-mean null*. This demonstrably
precludes assuming distribution-free 5% size control.

This module NEVER consumes experiment outcomes, validates science records,
trains a model, selects a real-world inference method, or grants execution.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from math import comb
from typing import Any

from synaploom.research.m13b_manifest import canonical_json

KIND = "M13B_D3_EXACT_SYNTHETIC_ZERO_MEAN_COUNTEREXAMPLE_V1"
N = 10
LOWER_TAIL = Fraction(1, 40)  # lower tail of each two-sided 95% percentile interval
CONFIGURATIONS = {
    "symmetric_zero_mean": {
        "p_positive": Fraction(1, 2),
        "positive_effect": Fraction(1, 100),
        "negative_effect": Fraction(-1, 100),
    },
    "skewed_zero_mean": {
        "p_positive": Fraction(9, 10),
        "positive_effect": Fraction(1, 100),
        "negative_effect": Fraction(-9, 100),
    },
}


def _iid_positive_count_probability(k: int, p_positive: Fraction) -> Fraction:
    if type(k) is not int or not 0 <= k <= N:
        raise ValueError("invalid positive sample count")
    if not isinstance(p_positive, Fraction) or not 0 <= p_positive <= 1:
        raise ValueError("invalid probability")
    return Fraction(comb(N, k)) * p_positive**k * (1 - p_positive)**(N - k)


def _resampled_nonpositive_probability(
    observed_positive_count: int, positive_effect: Fraction,
    negative_effect: Fraction,
) -> Fraction:
    """Infinite-resample distribution: J~Binomial(n,observed_k/n).

    Computes the exact conditional mass of non-positive bootstrap means,
    including ties at zero. No RNG and no science data.
    """
    if type(observed_positive_count) is not int or not 0 <= observed_positive_count <= N:
        raise ValueError("invalid observed positive count")
    if not (isinstance(positive_effect, Fraction)
            and isinstance(negative_effect, Fraction)
            and 0 < positive_effect <= 1
            and -1 <= negative_effect < 0):
        raise ValueError("requires opposite-sign bounded exact toy effects")
    q = Fraction(observed_positive_count, N)
    return sum((
        _iid_positive_count_probability(j, q)
        for j in range(N + 1)
        if j * positive_effect + (N - j) * negative_effect <= 0
    ), start=Fraction(0))


def _exact_ideal_bootstrap_favorable_probability(
    positive_probability: Fraction,
    positive_effect: Fraction,
    negative_effect: Fraction,
) -> tuple[Fraction, list[int]]:
    """Ideal percentile-bootstrap favorable event for perfectly correlated endpoints.

    If P*(mean*<=0) < .025, the exact lower 2.5% percentile is >0;
    hence both identical co-primary endpoints are favorable. We avoid the
    quantile-boundary equality case in the frozen toy configurations.
    """
    favorable_counts = [
        k for k in range(N + 1)
        if _resampled_nonpositive_probability(
            k, positive_effect, negative_effect
        ) < LOWER_TAIL
    ]
    probability = sum((
        _iid_positive_count_probability(k, positive_probability)
        for k in favorable_counts
    ), start=Fraction(0))
    return probability, favorable_counts


def build_exact_null_review() -> dict[str, Any]:
    """Read-only finite-probability assessment of a candidate inference rule."""
    scenarios: dict[str, Any] = {}
    for name, spec in CONFIGURATIONS.items():
        p, pos, neg = (
            spec["p_positive"], spec["positive_effect"],
            spec["negative_effect"],
        )
        mean = p * pos + (1 - p) * neg
        if mean != 0:
            raise ValueError("toy null distribution is not exactly centered")
        if not (0 < p < 1 and 0 < pos <= 1 and -1 <= neg < 0):
            raise ValueError("invalid bounded synthetic null")
        exact_rate, favorable_k = _exact_ideal_bootstrap_favorable_probability(
            p, pos, neg
        )
        all_positive_probability = p**N
        if exact_rate < all_positive_probability or N not in favorable_k:
            raise AssertionError("infinite percentile calculation omitted all-positive sample")
        scenarios[name] = {
            "one_seed_positive_probability": float(p),
            "positive_delta_fraction": float(pos),
            "negative_delta_fraction": float(neg),
            "true_expected_delta_fraction": float(mean),
            "n_paired_seeds": N,
            "co_primary_endpoints": "IDENTICAL_PAIRED_EFFECTS_PER_SEED_SYNTHETIC_ONLY",
            "correlation_both_endpoints": 1.0,
            "all_positive_sample_probability_exact": (
                f"{all_positive_probability.numerator}/"
                f"{all_positive_probability.denominator}"
            ),
            "all_positive_sample_probability": float(all_positive_probability),
            "all_positive_sample_bootstrap_ci": [float(pos), float(pos)],
            "ideal_infinite_bootstrap_favorable_positive_counts": favorable_k,
            "ideal_infinite_bootstrap_favorable_probability_exact": (
                f"{exact_rate.numerator}/{exact_rate.denominator}"
            ),
            "ideal_infinite_bootstrap_favorable_probability": float(exact_rate),
        }

    skew = scenarios["skewed_zero_mean"]
    if skew["all_positive_sample_probability"] <= 0.05:
        raise AssertionError("counterexample no longer exceeds illustrative 5%")
    report: dict[str, Any] = {
        "schema": 1,
        "kind": KIND,
        "status": "MATHEMATICAL_D3_COUNTEREXAMPLE__INDEPENDENT_REVIEW_PENDING",
        "science_executable": False,
        "real_scientific_data_loaded": False,
        "actual_optimizer_steps": 0,
        "scientific_claim": "NONE",
        "paired_seed_count": N,
        "bootstrap_rule_reviewed": (
            "Both marginal two-sided 95pct percentile CI lower bounds >0 "
            "for perfectly correlated synthetic endpoints"
        ),
        "bootstrap_resampling": "EXACT_INFINITE_RESAMPLE_LIMIT",
        "finite_10000_draw_result_claimed": False,
        "synthetic_null_scenarios": scenarios,
        "conclusion": (
            "At n=10 and under skewed but bounded iid zero-mean seed effects, "
            "the original proposed percentile-bootstrap both-needed favorable "
            "rule can have at least 34.86784401% false favorable frequency. "
            "It is therefore not distribution-free size-controlled."
        ),
        "limitations": [
            "This is a purely mathematical constructed counterexample, not a distribution observed in SynapLoom.",
            "The two invented endpoints are perfectly correlated; this is allowed by the unqualified proposed test.",
            "Exact ideal bootstrap probability uses the infinite-resample limit; the all-positive lower bound holds for ANY finite bootstrap count and RNG seed.",
            "A seed-paired sign-flip test also requires symmetry/exchangeability: zero expected difference alone is insufficient.",
            "A conjunction of calibrated component tests can control a composite null, but the proposed n=10 percentile tests are not automatically calibrated.",
            "No method choice, learned effect, study freeze, owner execution authorization or actual GPU work follows from this diagnostic.",
        ],
    }
    report["record_sha256"] = sha256(canonical_json(report)).hexdigest()
    return report
