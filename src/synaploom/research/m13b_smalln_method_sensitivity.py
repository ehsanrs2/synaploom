"""Small-n proposed M1.3b paired-bootstrap sensitivity simulation, NOT science.

Uses wholly invented Gaussian ten-seed, two-endpoint deltas. Illustrates
finite-sample behavior under synthetic partial/global nulls; no guarantee
of test calibration, no empirical H1a/M1.3b observations, no GPU/data access.
"""
from __future__ import annotations

from hashlib import sha256
import math
import random
import statistics
from typing import Any

from synaploom.research.m13b_manifest import canonical_json


KIND = "M13B_N10_BOOTSTRAP_METHOD_SENSITIVITY_TOY_V1"
SCENARIOS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("both_null", (0.0, 0.0)),
    ("partial_null_other_strong", (0.0, 0.20)),
    ("both_positive", (0.16, 0.16)),
    ("opposed_effects", (0.16, -0.16)),
)
N_SEEDS = 10
SYNTHETIC_SIGMA = 0.12
SYNTHETIC_RHO = 0.7


def _ci_two_endpoints(
    rows: list[tuple[float, float]],
    rng: random.Random,
    *,
    draws: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Shared seed-resampling indices for two correlated paired endpoints."""
    if len(rows) != N_SEEDS:
        raise ValueError("expected exactly 10 paired deltas")
    if draws < 100:
        raise ValueError("simulation needs >=100 bootstrap draws")
    a: list[float] = []
    b: list[float] = []
    for _ in range(draws):
        indices = [rng.randrange(N_SEEDS) for _ in range(N_SEEDS)]
        a.append(statistics.fmean(rows[i][0] for i in indices))
        b.append(statistics.fmean(rows[i][1] for i in indices))
    a.sort()
    b.sort()
    lo, hi = int(0.025 * (draws - 1)), int(0.975 * (draws - 1))
    return (a[lo], a[hi]), (b[lo], b[hi])


def simulate_n10_method_sensitivity(
    *, trials: int = 120, bootstrap_draws: int = 400, seed: int = 8927,
) -> dict[str, Any]:
    """Return a bounded toy Monte Carlo illustration; NOT a calibrated test.

    All distributional assumptions (Gaussian variance, correlation and effect
    sizes) are illustrative inventions, not fit to any research outcomes.
    """
    if (type(trials) is not int or not 20 <= trials <= 500
            or type(bootstrap_draws) is not int or not 100 <= bootstrap_draws <= 1000
            or type(seed) is not int or seed < 0):
        raise ValueError("invalid bounded toy simulation settings")
    rng = random.Random(seed)
    results: dict[str, Any] = {}
    for name, (mu1, mu2) in SCENARIOS:
        classified = {"SYNTHETIC_FAVORABLE": 0, "SYNTHETIC_UNFAVORABLE": 0,
                      "SYNTHETIC_INCONCLUSIVE": 0}
        margin_false_positive = 0
        for _ in range(trials):
            observations: list[tuple[float, float]] = []
            for _ in range(N_SEEDS):
                shared = rng.gauss(0, 1)
                independent = rng.gauss(0, 1)
                first = mu1 + SYNTHETIC_SIGMA * shared
                second = mu2 + SYNTHETIC_SIGMA * (
                    SYNTHETIC_RHO * shared
                    + math.sqrt(1 - SYNTHETIC_RHO ** 2) * independent
                )
                observations.append((first, second))
            ci1, ci2 = _ci_two_endpoints(observations, rng, draws=bootstrap_draws)
            if ci1[0] > 0 and ci2[0] > 0:
                category = "SYNTHETIC_FAVORABLE"
            elif ci1[1] < 0 and ci2[1] < 0:
                category = "SYNTHETIC_UNFAVORABLE"
            else:
                category = "SYNTHETIC_INCONCLUSIVE"
            classified[category] += 1
            if name in ("both_null", "partial_null_other_strong") and category == "SYNTHETIC_FAVORABLE":
                margin_false_positive += 1
        results[name] = {
            "generating_mean_deltas": [mu1, mu2],
            "toy_classification_counts": classified,
            "toy_favorable_rate": classified["SYNTHETIC_FAVORABLE"] / trials,
            "toy_favorable_when_at_least_one_null_count": (
                margin_false_positive if name in (
                    "both_null", "partial_null_other_strong"
                ) else None
            ),
        }

    data: dict[str, Any] = {
        "schema": 1,
        "kind": KIND,
        "scientific_claim": "NONE",
        "actual_scientific_results_loaded": False,
        "science_execution_authorized": False,
        "synthetic_only": True,
        "n_paired_seeds": N_SEEDS,
        "synthetic_sigma": SYNTHETIC_SIGMA,
        "synthetic_endpoint_correlation": SYNTHETIC_RHO,
        "monte_carlo_trials_per_scenario": trials,
        "bootstrap_draws_per_trial": bootstrap_draws,
        "independent_simulation_seed": seed,
        "toy_results": results,
        "limitations": [
            "Toy Gaussian assumptions/effect scales are invented, not estimated from SynapLoom.",
            "Short bootstrap (not the proposed 10000) and modest Monte Carlo sample cause large simulation uncertainty.",
            "Marginal percentile-bootstrap tests with n=10 require an independent statistical size/power/coverage review.",
            "A synthetic false-favorable frequency is not an accepted type-I error bound for real experiments.",
            "This result cannot freeze the protocol or authorize GPU training.",
        ],
    }
    data["record_sha256"] = sha256(canonical_json(data)).hexdigest()
    return data
