from __future__ import annotations

from synaploom.protocols.h1a2 import H1A2_FEATURE_SHA256, h1a2_protocol
from synaploom.protocols.h1a_eval_batch_sensitivity import (
    BOOTSTRAP_RESAMPLES,
    H1A_EVAL_BATCH_SIZES,
    H1A_EVALBATCH_EXPERIMENT_ID,
    H1A_EVALBATCH_ORDER_FINGERPRINT,
    H1A_EVALBATCH_REFERENCE_SIZE,
    H1A_EVALBATCH_RUNS,
    H1A_EVALBATCH_SEEDS,
    H1A_EVALBATCH_SUMMARIES,
    H1A_EVALBATCH_VARIANTS,
    classify_batch_size_direction,
    dry_run_payload,
    h1a_eval_batch_sensitivity_protocol,
    paired_batch_size_comparisons,
    validate_protocol,
)


def _synthetic_values(effect_by_size: dict[int, float]):
    data = {}
    for seed in H1A_EVALBATCH_SEEDS:
        data[seed] = {}
        seed_offset = (seed - 405) * 0.01
        for variant in H1A_EVALBATCH_VARIANTS:
            data[seed][variant] = {}
            for batch_size in H1A_EVAL_BATCH_SIZES:
                control = 50.0 + seed_offset + batch_size / 100_000.0
                effect = effect_by_size[batch_size] + seed_offset * 0.1
                treatment = control + effect
                data[seed][variant][batch_size] = {
                    "final_average_accuracy": (
                        treatment if variant == "progressive_reserve" else control
                    ),
                    "late_task_within_task_mean_accuracy": (
                        treatment if variant == "progressive_reserve" else control
                    ),
                    "average_forgetting": (
                        10.0 if variant == "all_recruited" else 10.1
                    ),
                    "backward_transfer": (
                        -2.0 if variant == "all_recruited" else -1.9
                    ),
                }
    return data


def test_protocol_is_frozen_against_h1a_002_except_seed_block() -> None:
    validate_protocol()
    original = h1a2_protocol().as_dict()
    sensitivity = h1a_eval_batch_sensitivity_protocol().as_dict()

    assert {key for key in original if original[key] != sensitivity[key]} == {"seeds"}
    assert H1A_EVALBATCH_EXPERIMENT_ID == "EXP-H1A-EVALBATCH-001"
    assert H1A_EVALBATCH_SEEDS == tuple(range(401, 411))
    assert H1A_EVALBATCH_VARIANTS == ("progressive_reserve", "all_recruited")
    assert H1A_EVAL_BATCH_SIZES == (32, 64, 128, 256, 512, 1024)
    assert H1A_EVALBATCH_REFERENCE_SIZE == 256
    assert H1A_EVALBATCH_RUNS == 20
    assert H1A_EVALBATCH_SUMMARIES == 120
    assert BOOTSTRAP_RESAMPLES == 10_000
    assert H1A_EVALBATCH_ORDER_FINGERPRINT == (
        "81d7f0637b8e61477c1ef80b7eef33270aae8d81dfcaffad6e58252fd004b633"
    )
    assert H1A2_FEATURE_SHA256 == (
        "c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47"
    )


def test_dry_run_is_complete_and_does_not_need_features() -> None:
    payload = dry_run_payload()

    assert payload["status"] == "DRY_RUN_ONLY"
    assert payload["expected_training_runs"] == 20
    assert payload["expected_run_size_summaries"] == 120
    assert payload["evaluation_batch_sizes"] == [32, 64, 128, 256, 512, 1024]
    assert payload["expected_state_hash_sweeps_per_run"] == 210


def test_registered_classifier_flags_treatment_effect_sign_flip() -> None:
    mean_effects = {
        "final_average_accuracy": {
            32: -0.1,
            64: 0.2,
            128: 0.3,
            256: 0.4,
            512: 0.5,
            1024: 0.6,
        },
        "late_task_within_task_mean_accuracy": {
            32: 0.1,
            64: 0.2,
            128: 0.3,
            256: 0.4,
            512: 0.5,
            1024: 0.6,
        },
    }
    assert classify_batch_size_direction(mean_effects) == "DIRECTION_FLIP_OBSERVED"


def test_paired_batch_bootstrap_is_deterministic_and_reports_effect_and_drift() -> None:
    effect_by_size = {
        32: -0.2,
        64: 0.4,
        128: 0.6,
        256: 1.0,
        512: 1.1,
        1024: 1.2,
    }
    data = _synthetic_values(effect_by_size)
    first = paired_batch_size_comparisons(data, samples=300, seed=7331)
    second = paired_batch_size_comparisons(data, samples=300, seed=7331)

    assert first == second
    assert first["registered_direction_classification"] == "DIRECTION_FLIP_OBSERVED"
    effect = first["co_primary_metrics"]["final_average_accuracy"]["256"]
    assert effect["mean_treatment_effect_pp"] > 0.9
    assert effect["ci95_treatment_effect_pp"][0] <= effect["ci95_treatment_effect_pp"][1]
    assert effect["mean_drift_vs_256_pp"] == 0.0
    assert effect["ci95_drift_vs_256_pp"] == [0.0, 0.0]
    assert set(effect["per_seed"]) == {str(seed) for seed in H1A_EVALBATCH_SEEDS}
