from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import replace
from typing import Mapping, Sequence

from synaploom.protocols.h1a import H1A_CLASS_ORDER
from synaploom.protocols.h1a2 import h1a2_protocol


H1A_ORDER_EXPERIMENT_ID = "EXP-H1A-ORDER-001"
H1A_ORDER_SEEDS: tuple[int, ...] = tuple(range(301, 311))
H1A_ORDER_VARIANTS: tuple[str, ...] = (
    "progressive_reserve",
    "all_recruited",
)
H1A_ORDER_IDS: tuple[str, ...] = ("O0", "O1", "O2")
H1A_ORDER_RUNS = 60
BOOTSTRAP_RESAMPLES = 10_000
ORDER_BOOTSTRAP_SEEDS: dict[str, int] = {
    "O0": 1337,
    "O1": 1338,
    "O2": 1339,
}
POOLED_BOOTSTRAP_SEED = 7331
CO_PRIMARY_METRICS: tuple[str, ...] = (
    "final_average_accuracy",
    "late_task_within_task_mean_accuracy",
)
PERCENT_POINT_METRICS = frozenset(
    {
        "final_average_accuracy",
        "late_task_within_task_mean_accuracy",
        "late_task_learning_accuracy",
        "average_forgetting",
        "backward_transfer",
    }
)

# These literal permutations and their fingerprints are frozen in
# docs/H1A_CLASS_ORDER_SENSITIVITY_PROTOCOL.md. Do not regenerate at runtime.
_ORDER_O1: tuple[int, ...] = (
    63, 10, 86, 36, 96, 65, 25, 71, 29, 67,
    80, 90, 69, 4, 13, 1, 30, 14, 98, 32,
    59, 73, 54, 47, 27, 0, 81, 20, 99, 17,
    75, 16, 21, 58, 7, 3, 12, 45, 87, 97,
    11, 70, 94, 24, 31, 82, 78, 22, 77, 2,
    89, 55, 74, 40, 43, 15, 41, 85, 6, 57,
    50, 26, 48, 53, 39, 61, 52, 46, 88, 18,
    84, 9, 28, 44, 5, 83, 38, 60, 66, 91,
    33, 64, 51, 72, 56, 49, 95, 23, 37, 19,
    35, 8, 68, 76, 62, 92, 93, 79, 42, 34,
)
_ORDER_O2: tuple[int, ...] = (
    29, 45, 9, 37, 77, 79, 90, 72, 12, 35,
    16, 64, 71, 13, 86, 60, 0, 80, 24, 15,
    69, 21, 22, 14, 40, 25, 46, 61, 30, 83,
    43, 33, 6, 27, 7, 67, 11, 2, 1, 62,
    32, 92, 99, 65, 75, 54, 18, 3, 20, 55,
    52, 49, 47, 57, 91, 59, 81, 44, 74, 8,
    76, 34, 93, 94, 50, 5, 42, 23, 84, 87,
    39, 96, 63, 73, 19, 58, 85, 66, 95, 56,
    26, 82, 88, 97, 98, 17, 36, 28, 41, 70,
    10, 31, 38, 4, 89, 48, 51, 53, 78, 68,
)
CLASS_ORDERS: dict[str, tuple[int, ...]] = {
    "O0": tuple(H1A_CLASS_ORDER),
    "O1": _ORDER_O1,
    "O2": _ORDER_O2,
}
EXPECTED_ORDER_FINGERPRINTS: dict[str, str] = {
    "O0": "81d7f0637b8e61477c1ef80b7eef33270aae8d81dfcaffad6e58252fd004b633",
    "O1": "c0dc7d8b234ec06670526cbc8d917f0e7e14ed9bbaee640272eab95573f1f8ed",
    "O2": "b610995c70c2269fb28e7492f1c1eb2870703fa134f967e185ca02dbd44e2e63",
}


def class_order_fingerprint(order: Sequence[int]) -> str:
    """Hash a class order as compact JSON, with no trailing newline."""
    payload = json.dumps([int(value) for value in order], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_class_orders() -> dict[str, str]:
    """Fail closed if a preregistered class order or fingerprint changes."""
    if tuple(CLASS_ORDERS) != H1A_ORDER_IDS:
        raise ValueError(
            f"class-order IDs must be exactly {H1A_ORDER_IDS}, got {tuple(CLASS_ORDERS)}"
        )

    fingerprints: dict[str, str] = {}
    observed_orders: set[tuple[int, ...]] = set()
    expected_classes = set(range(100))
    for order_id in H1A_ORDER_IDS:
        order = tuple(CLASS_ORDERS[order_id])
        if len(order) != 100 or set(order) != expected_classes:
            raise ValueError(
                f"{order_id} must be a permutation of all CIFAR-100 class IDs 0..99"
            )
        if order in observed_orders:
            raise ValueError("all preregistered class orders must be distinct")
        observed_orders.add(order)
        actual = class_order_fingerprint(order)
        expected = EXPECTED_ORDER_FINGERPRINTS[order_id]
        if actual != expected:
            raise ValueError(
                f"class-order fingerprint mismatch for {order_id}: "
                f"expected {expected}, got {actual}"
            )
        fingerprints[order_id] = actual

    return fingerprints


def h1a_order_sensitivity_protocol():
    """Return EXP-H1A-002's frozen protocol with only its registered seed block changed."""
    return replace(h1a2_protocol(), seeds=H1A_ORDER_SEEDS)


def paired_bootstrap_ci(
    deltas_by_seed: Mapping[int, float],
    *,
    samples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 1337,
) -> list[float]:
    """Percentile CI for the paired mean, resampling paired seed deltas."""
    if samples < 1:
        raise ValueError("samples must be positive")
    values = [float(deltas_by_seed[s]) for s in sorted(deltas_by_seed)]
    if not values:
        raise ValueError("paired bootstrap requires at least one delta")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("paired bootstrap deltas must be finite")

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(draw))
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return [lo, hi]


def seed_cluster_bootstrap_ci(
    deltas_by_order: Mapping[str, Mapping[int, float]],
    *,
    seeds: Sequence[int] = H1A_ORDER_SEEDS,
    samples: int = BOOTSTRAP_RESAMPLES,
    seed: int = POOLED_BOOTSTRAP_SEED,
) -> list[float]:
    """Percentile CI for an equal-order-weighted mean, resampling seed clusters.

    Each draw samples seed IDs with replacement; each selected seed carries its
    delta from all three fixed class orders into the replicate. This avoids
    falsely treating the 30 order-by-seed observations as independent.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    seed_ids = tuple(int(value) for value in seeds)
    if not seed_ids:
        raise ValueError("cluster bootstrap requires at least one seed")
    if tuple(deltas_by_order) != H1A_ORDER_IDS:
        raise ValueError(
            f"cluster bootstrap orders must be exactly {H1A_ORDER_IDS}"
        )
    for order_id in H1A_ORDER_IDS:
        missing = set(seed_ids).difference(deltas_by_order[order_id])
        if missing:
            raise ValueError(
                f"missing seed deltas for {order_id}: {sorted(missing)}"
            )
    cluster_means = {
        seed_id: statistics.fmean(
            float(deltas_by_order[order_id][seed_id])
            for order_id in H1A_ORDER_IDS
        )
        for seed_id in seed_ids
    }
    if not all(math.isfinite(value) for value in cluster_means.values()):
        raise ValueError("cluster bootstrap deltas must be finite")

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        draw = [seed_ids[rng.randrange(len(seed_ids))] for _ in seed_ids]
        means.append(statistics.fmean(cluster_means[seed_id] for seed_id in draw))
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return [lo, hi]


def classify_order_sensitivity(
    pooled_cis: Mapping[str, Sequence[float]],
    order_mean_deltas: Mapping[str, Mapping[str, float]],
) -> str:
    """Apply the exact preregistered descriptive class-order classification."""
    if any(metric not in pooled_cis for metric in CO_PRIMARY_METRICS):
        return "INCONCLUSIVE"
    if any(metric not in order_mean_deltas for metric in CO_PRIMARY_METRICS):
        return "INCONCLUSIVE"

    cis: dict[str, tuple[float, float]] = {}
    means: dict[str, list[float]] = {}
    for metric in CO_PRIMARY_METRICS:
        ci = pooled_cis[metric]
        if len(ci) != 2:
            return "INCONCLUSIVE"
        lo, hi = float(ci[0]), float(ci[1])
        metric_means = [
            float(order_mean_deltas[metric].get(order_id, float("nan")))
            for order_id in H1A_ORDER_IDS
        ]
        if (
            not math.isfinite(lo)
            or not math.isfinite(hi)
            or lo > hi
            or not all(math.isfinite(value) for value in metric_means)
        ):
            return "INCONCLUSIVE"
        cis[metric] = (lo, hi)
        means[metric] = metric_means

    favorable = all(cis[metric][0] > 0.0 for metric in CO_PRIMARY_METRICS)
    favorable = favorable and all(
        all(value > 0.0 for value in means[metric])
        for metric in CO_PRIMARY_METRICS
    )
    if favorable:
        return "ORDER_ROBUST_FAVORABLE"

    unfavorable = all(cis[metric][1] < 0.0 for metric in CO_PRIMARY_METRICS)
    unfavorable = unfavorable and all(
        all(value < 0.0 for value in means[metric])
        for metric in CO_PRIMARY_METRICS
    )
    if unfavorable:
        return "ORDER_ROBUST_UNFAVORABLE"

    if any(min(values) < 0.0 < max(values) for values in means.values()):
        return "ORDER_SENSITIVE"
    return "INCONCLUSIVE"
