from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import replace
from typing import Mapping

from synaploom.protocols.h1a import H1A_CLASS_ORDER
from synaploom.protocols.h1a2 import H1A2_FEATURE_SHA256, h1a2_protocol


H1A_EVALBATCH_EXPERIMENT_ID = "EXP-H1A-EVALBATCH-001"
H1A_EVALBATCH_SEEDS: tuple[int, ...] = tuple(range(401, 411))
H1A_EVALBATCH_VARIANTS: tuple[str, ...] = (
    "progressive_reserve",
    "all_recruited",
)
H1A_EVAL_BATCH_SIZES: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)
H1A_EVALBATCH_REFERENCE_SIZE = 256
H1A_EVALBATCH_RUNS = 20
H1A_EVALBATCH_SUMMARIES = 120
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 7331
H1A_EVALBATCH_ORDER_FINGERPRINT = (
    "81d7f0637b8e61477c1ef80b7eef33270aae8d81dfcaffad6e58252fd004b633"
)
CO_PRIMARY_METRICS: tuple[str, ...] = (
    "final_average_accuracy",
    "late_task_within_task_mean_accuracy",
)
SECONDARY_METRICS: tuple[str, ...] = (
    "average_forgetting",
    "backward_transfer",
)
PERCENT_POINT_METRICS = frozenset((*CO_PRIMARY_METRICS, *SECONDARY_METRICS))


def class_order_fingerprint(order: tuple[int, ...] | list[int] = H1A_CLASS_ORDER) -> str:
    """SHA-256 of the frozen compact JSON class-order list."""
    payload = json.dumps(list(order), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def h1a_eval_batch_sensitivity_protocol():
    """Return the H1A-002 training protocol with only the fresh seed block changed."""
    return replace(h1a2_protocol(), seeds=H1A_EVALBATCH_SEEDS)


def validate_protocol() -> None:
    protocol = h1a_eval_batch_sensitivity_protocol()
    original = h1a2_protocol().as_dict()
    current = protocol.as_dict()
    changed = {key for key in original if original[key] != current[key]}
    if changed != {"seeds"}:
        raise ValueError(
            "EXP-H1A-EVALBATCH-001 may change only the registered seed block "
            f"relative to EXP-H1A-002; changed={sorted(changed)}"
        )
    if protocol.eval_batch_size != H1A_EVALBATCH_REFERENCE_SIZE:
        raise ValueError("the training protocol reference eval_batch_size must remain 256")
    if protocol.seeds != H1A_EVALBATCH_SEEDS:
        raise ValueError("protocol seeds differ from the frozen seed block")
    if H1A_EVALBATCH_RUNS != len(H1A_EVALBATCH_SEEDS) * len(H1A_EVALBATCH_VARIANTS):
        raise ValueError("registered training trajectory count is inconsistent")
    if H1A_EVALBATCH_SUMMARIES != H1A_EVALBATCH_RUNS * len(H1A_EVAL_BATCH_SIZES):
        raise ValueError("registered run-size summary count is inconsistent")
    if len(set(H1A_EVAL_BATCH_SIZES)) != len(H1A_EVAL_BATCH_SIZES):
        raise ValueError("registered evaluation batch sizes must be unique")
    if H1A_EVAL_BATCH_SIZES.count(H1A_EVALBATCH_REFERENCE_SIZE) != 1:
        raise ValueError("reference batch size 256 must occur exactly once")
    if class_order_fingerprint() != H1A_EVALBATCH_ORDER_FINGERPRINT:
        raise ValueError("frozen primary class order fingerprint changed")
    if H1A2_FEATURE_SHA256 != (
        "c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47"
    ):
        raise ValueError("frozen feature hash constant changed")


def dry_run_payload() -> dict[str, object]:
    """Build the registered plan without loading features or constructing models."""
    validate_protocol()
    return {
        "experiment_id": H1A_EVALBATCH_EXPERIMENT_ID,
        "status": "DRY_RUN_ONLY",
        "feature_sha256": H1A2_FEATURE_SHA256,
        "class_order_fingerprint_sha256": H1A_EVALBATCH_ORDER_FINGERPRINT,
        "class_order": list(H1A_CLASS_ORDER),
        "protocol": h1a_eval_batch_sensitivity_protocol().as_dict(),
        "seeds": list(H1A_EVALBATCH_SEEDS),
        "variants": list(H1A_EVALBATCH_VARIANTS),
        "evaluation_batch_sizes": list(H1A_EVAL_BATCH_SIZES),
        "reference_batch_size": H1A_EVALBATCH_REFERENCE_SIZE,
        "expected_training_runs": H1A_EVALBATCH_RUNS,
        "expected_run_size_summaries": H1A_EVALBATCH_SUMMARIES,
        "expected_state_hash_sweeps_per_run": (
            10 * 20 + 10
        ),
        "co_primary_metrics": list(CO_PRIMARY_METRICS),
        "secondary_metrics": list(SECONDARY_METRICS),
        "bootstrap": {
            "method": "paired seed-cluster bootstrap",
            "resamples": BOOTSTRAP_RESAMPLES,
            "rng_seed": BOOTSTRAP_SEED,
            "unit": "paired seed; each draw retains both variants and all batch sizes",
        },
        "registered_direction_classification": [
            "DIRECTION_FLIP_OBSERVED",
            "NO_DIRECTION_FLIP_OBSERVED",
        ],
    }


def _sign(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("direction-classification values must be finite")
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def classify_batch_size_direction(
    mean_effects_by_metric: Mapping[str, Mapping[int, float]],
) -> str:
    """Apply the preregistered descriptive treatment-effect sign comparison."""
    for metric in CO_PRIMARY_METRICS:
        if metric not in mean_effects_by_metric:
            raise ValueError(f"missing co-primary metric {metric}")
        values = mean_effects_by_metric[metric]
        if set(values) != set(H1A_EVAL_BATCH_SIZES):
            raise ValueError(f"incomplete batch-size effects for {metric}")
        reference_sign = _sign(float(values[H1A_EVALBATCH_REFERENCE_SIZE]))
        if any(
            _sign(float(values[batch_size])) != reference_sign
            for batch_size in H1A_EVAL_BATCH_SIZES
            if batch_size != H1A_EVALBATCH_REFERENCE_SIZE
        ):
            return "DIRECTION_FLIP_OBSERVED"
    return "NO_DIRECTION_FLIP_OBSERVED"


def _percentile_ci(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("bootstrap distribution cannot be empty")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("bootstrap distribution must be finite")
    ordered = sorted(float(value) for value in values)
    return [
        ordered[int(0.025 * (len(ordered) - 1))],
        ordered[int(0.975 * (len(ordered) - 1))],
    ]


def _describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot describe an empty set")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("descriptive values must be finite")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def paired_batch_size_comparisons(
    per_seed_variant_batch_metrics: Mapping[
        int, Mapping[str, Mapping[int, Mapping[str, float]]]
    ],
    *,
    samples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Summarize paired treatment effects and drift from batch size 256.

    Input metric values are percentages (0--100). Deltas and drift are in
    percentage points. Every bootstrap draw samples seed IDs once and reuses
    that paired draw for both variants, both co-primary metrics, and every
    evaluation batch size.
    """
    seeds = tuple(sorted(int(item) for item in per_seed_variant_batch_metrics))
    if seeds != H1A_EVALBATCH_SEEDS:
        raise ValueError(
            f"expected exact paired seed block {H1A_EVALBATCH_SEEDS}, got {seeds}"
        )
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    batch_deltas: dict[str, dict[int, dict[int, float]]] = {
        metric: {batch_size: {} for batch_size in H1A_EVAL_BATCH_SIZES}
        for metric in CO_PRIMARY_METRICS
    }
    metric_values_by_variant: dict[str, dict[int, dict[str, dict[int, float]]]] = {
        metric: {
            batch_size: {variant: {} for variant in H1A_EVALBATCH_VARIANTS}
            for batch_size in H1A_EVAL_BATCH_SIZES
        }
        for metric in (*CO_PRIMARY_METRICS, *SECONDARY_METRICS)
    }
    for run_seed in H1A_EVALBATCH_SEEDS:
        row = per_seed_variant_batch_metrics[run_seed]
        if set(row) != set(H1A_EVALBATCH_VARIANTS):
            raise ValueError(f"invalid variants for paired seed {run_seed}")
        for variant in H1A_EVALBATCH_VARIANTS:
            by_size = row[variant]
            if set(by_size) != set(H1A_EVAL_BATCH_SIZES):
                raise ValueError(f"incomplete batch-size summaries for {variant}/seed-{run_seed}")
            for batch_size in H1A_EVAL_BATCH_SIZES:
                values = by_size[batch_size]
                for metric in (*CO_PRIMARY_METRICS, *SECONDARY_METRICS):
                    value = float(values[metric])
                    if not math.isfinite(value):
                        raise ValueError(
                            f"non-finite {metric} at seed={run_seed}, variant={variant}, "
                            f"batch_size={batch_size}"
                        )
                    metric_values_by_variant[metric][batch_size][variant][run_seed] = value
                for metric in CO_PRIMARY_METRICS:
                    batch_deltas[metric][batch_size][run_seed] = (
                        float(values[metric])
                    )

    for metric in CO_PRIMARY_METRICS:
        for batch_size in H1A_EVAL_BATCH_SIZES:
            for run_seed in H1A_EVALBATCH_SEEDS:
                treatment = metric_values_by_variant[metric][batch_size]["progressive_reserve"][run_seed]
                control = metric_values_by_variant[metric][batch_size]["all_recruited"][run_seed]
                batch_deltas[metric][batch_size][run_seed] = treatment - control

    rng = random.Random(seed)
    bootstrap_draws: dict[str, dict[int, dict[str, list[float]]]] = {
        metric: {
            batch_size: {"effect": [], "drift": []}
            for batch_size in H1A_EVAL_BATCH_SIZES
        }
        for metric in CO_PRIMARY_METRICS
    }
    for _ in range(samples):
        sampled_seeds = [
            seeds[rng.randrange(len(seeds))]
            for _ in seeds
        ]
        for metric in CO_PRIMARY_METRICS:
            reference = batch_deltas[metric][H1A_EVALBATCH_REFERENCE_SIZE]
            reference_mean = statistics.fmean(reference[item] for item in sampled_seeds)
            for batch_size in H1A_EVAL_BATCH_SIZES:
                deltas = batch_deltas[metric][batch_size]
                effect_mean = statistics.fmean(deltas[item] for item in sampled_seeds)
                drift_mean = statistics.fmean(
                    deltas[item] - reference[item] for item in sampled_seeds
                )
                bootstrap_draws[metric][batch_size]["effect"].append(effect_mean)
                bootstrap_draws[metric][batch_size]["drift"].append(drift_mean)

    by_metric: dict[str, object] = {}
    mean_effects: dict[str, dict[int, float]] = {}
    for metric in CO_PRIMARY_METRICS:
        by_size: dict[str, object] = {}
        mean_effects[metric] = {}
        for batch_size in H1A_EVAL_BATCH_SIZES:
            treatment = metric_values_by_variant[metric][batch_size]["progressive_reserve"]
            control = metric_values_by_variant[metric][batch_size]["all_recruited"]
            deltas = batch_deltas[metric][batch_size]
            mean_delta = statistics.fmean(deltas.values())
            mean_effects[metric][batch_size] = mean_delta
            reference = batch_deltas[metric][H1A_EVALBATCH_REFERENCE_SIZE]
            per_seed: dict[str, dict[str, float]] = {}
            for run_seed in H1A_EVALBATCH_SEEDS:
                per_seed[str(run_seed)] = {
                    "progressive_reserve_percent": treatment[run_seed],
                    "all_recruited_percent": control[run_seed],
                    "treatment_effect_pp": deltas[run_seed],
                    "drift_vs_256_pp": deltas[run_seed] - reference[run_seed],
                }
            draws = bootstrap_draws[metric][batch_size]
            by_size[str(batch_size)] = {
                "progressive_reserve": _describe(list(treatment.values())),
                "all_recruited": _describe(list(control.values())),
                "mean_treatment_effect_pp": mean_delta,
                "ci95_treatment_effect_pp": _percentile_ci(draws["effect"]),
                "mean_drift_vs_256_pp": statistics.fmean(
                    deltas[item] - reference[item] for item in H1A_EVALBATCH_SEEDS
                ),
                "ci95_drift_vs_256_pp": _percentile_ci(draws["drift"]),
                "per_seed": per_seed,
            }
        by_metric[metric] = by_size

    return {
        "reference_batch_size": H1A_EVALBATCH_REFERENCE_SIZE,
        "bootstrap_resamples": samples,
        "bootstrap_seed": seed,
        "co_primary_metrics": by_metric,
        "mean_treatment_effects_pp": mean_effects,
        "registered_direction_classification": classify_batch_size_direction(mean_effects),
    }


def secondary_batch_size_descriptives(
    per_seed_variant_batch_metrics: Mapping[
        int, Mapping[str, Mapping[int, Mapping[str, float]]]
    ],
) -> dict[str, object]:
    """Descriptive-only secondary metric deltas by size; no inferential labels."""
    per_metric: dict[str, object] = {}
    for metric in SECONDARY_METRICS:
        per_size: dict[str, object] = {}
        for batch_size in H1A_EVAL_BATCH_SIZES:
            by_variant = {
                variant: [
                    float(per_seed_variant_batch_metrics[run_seed][variant][batch_size][metric])
                    for run_seed in H1A_EVALBATCH_SEEDS
                ]
                for variant in H1A_EVALBATCH_VARIANTS
            }
            per_size[str(batch_size)] = {
                "variant_descriptives": {
                    variant: _describe(values) for variant, values in by_variant.items()
                },
                "mean_treatment_minus_control_pp": statistics.fmean(
                    float(per_seed_variant_batch_metrics[run_seed]["progressive_reserve"][batch_size][metric])
                    - float(per_seed_variant_batch_metrics[run_seed]["all_recruited"][batch_size][metric])
                    for run_seed in H1A_EVALBATCH_SEEDS
                ),
            }
        per_metric[metric] = per_size
    return per_metric
