from __future__ import annotations

import random
import statistics

from synaploom.protocols.h1a import H1A_CLASS_ORDER
from synaploom.protocols.h1a2 import h1a2_protocol
from synaploom.protocols.h1a_order_sensitivity import (
    CLASS_ORDERS,
    EXPECTED_ORDER_FINGERPRINTS,
    H1A_ORDER_EXPERIMENT_ID,
    H1A_ORDER_IDS,
    H1A_ORDER_RUNS,
    H1A_ORDER_SEEDS,
    H1A_ORDER_VARIANTS,
    class_order_fingerprint,
    classify_order_sensitivity,
    h1a_order_sensitivity_protocol,
    paired_bootstrap_ci,
    seed_cluster_bootstrap_ci,
    validate_class_orders,
)


def test_registered_orders_match_frozen_fingerprints_and_cover_all_classes() -> None:
    assert tuple(CLASS_ORDERS) == ("O0", "O1", "O2")
    assert CLASS_ORDERS["O0"] == H1A_CLASS_ORDER
    fingerprints = validate_class_orders()
    assert fingerprints == EXPECTED_ORDER_FINGERPRINTS
    for order_id, order in CLASS_ORDERS.items():
        assert len(order) == 100
        assert set(order) == set(range(100))
        assert class_order_fingerprint(order) == EXPECTED_ORDER_FINGERPRINTS[order_id]
    assert len({CLASS_ORDERS[key] for key in H1A_ORDER_IDS}) == 3


def test_registered_run_matrix_and_protocol_are_frozen() -> None:
    assert H1A_ORDER_EXPERIMENT_ID == "EXP-H1A-ORDER-001"
    assert H1A_ORDER_SEEDS == tuple(range(301, 311))
    assert H1A_ORDER_VARIANTS == ("progressive_reserve", "all_recruited")
    assert len(H1A_ORDER_IDS) * len(H1A_ORDER_SEEDS) * len(H1A_ORDER_VARIANTS) == H1A_ORDER_RUNS == 60

    original = h1a2_protocol().as_dict()
    sensitivity = h1a_order_sensitivity_protocol().as_dict()
    changed = {key for key in original if original[key] != sensitivity[key]}
    assert changed == {"seeds"}
    assert sensitivity["seeds"] == H1A_ORDER_SEEDS
    assert sensitivity["learning_rate"] == 0.01
    assert sensitivity["batch_size"] == 128
    assert sensitivity["eval_batch_size"] == 256
    assert sensitivity["epochs_per_task"] == 20
    assert sensitivity["momentum"] == 0.9
    assert sensitivity["weight_decay"] == 0.0005


def test_paired_bootstrap_is_deterministic_and_reports_constant_effect_exactly() -> None:
    values = {seed: 1.25 for seed in (301, 302, 303, 304)}
    first = paired_bootstrap_ci(values, samples=500, seed=1337)
    second = paired_bootstrap_ci(values, samples=500, seed=1337)
    assert first == second == [1.25, 1.25]


def test_cluster_bootstrap_resamples_seed_clusters_not_order_seed_rows() -> None:
    seeds = (301, 302, 303, 304)
    by_order = {
        "O0": {301: 0.0, 302: 1.0, 303: 10.0, 304: 11.0},
        "O1": {301: 100.0, 302: 101.0, 303: 110.0, 304: 111.0},
        "O2": {301: -100.0, 302: -99.0, 303: -90.0, 304: -89.0},
    }
    first = seed_cluster_bootstrap_ci(
        by_order, seeds=seeds, samples=500, seed=7331
    )
    second = seed_cluster_bootstrap_ci(
        by_order, seeds=seeds, samples=500, seed=7331
    )
    assert first == second

    # Reproduce the registered cluster-resampling unit: each sampled seed keeps
    # all three of its order-specific deltas together for that bootstrap draw.
    cluster_means = [
        statistics.fmean(by_order[order_id][seed] for order_id in ("O0", "O1", "O2"))
        for seed in seeds
    ]
    rng = random.Random(7331)
    means = sorted(
        statistics.fmean(
            cluster_means[rng.randrange(len(cluster_means))]
            for _ in cluster_means
        )
        for _ in range(500)
    )
    expected = [
        means[int(0.025 * (500 - 1))],
        means[int(0.975 * (500 - 1))],
    ]
    assert first == expected
    assert 0.0 <= first[0] <= first[1] <= 11.0

def test_order_sensitivity_classification_obeys_registered_rules() -> None:
    metrics = (
        "final_average_accuracy",
        "late_task_within_task_mean_accuracy",
    )

    positive_ci = {metric: [0.1, 0.4] for metric in metrics}
    positive_means = {
        metric: {"O0": 0.2, "O1": 0.1, "O2": 0.3}
        for metric in metrics
    }
    assert classify_order_sensitivity(positive_ci, positive_means) == "ORDER_ROBUST_FAVORABLE"

    negative_ci = {metric: [-0.4, -0.1] for metric in metrics}
    negative_means = {
        metric: {"O0": -0.2, "O1": -0.1, "O2": -0.3}
        for metric in metrics
    }
    assert classify_order_sensitivity(negative_ci, negative_means) == "ORDER_ROBUST_UNFAVORABLE"

    mixed_means = {
        "final_average_accuracy": {"O0": 0.2, "O1": -0.1, "O2": 0.3},
        "late_task_within_task_mean_accuracy": {"O0": 0.2, "O1": 0.1, "O2": 0.3},
    }
    assert classify_order_sensitivity(positive_ci, mixed_means) == "ORDER_SENSITIVE"

    weak_ci = {
        "final_average_accuracy": [-0.1, 0.4],
        "late_task_within_task_mean_accuracy": [-0.2, 0.3],
    }
    same_sign_means = {
        metric: {"O0": 0.2, "O1": 0.1, "O2": 0.3}
        for metric in metrics
    }
    assert classify_order_sensitivity(weak_ci, same_sign_means) == "INCONCLUSIVE"
