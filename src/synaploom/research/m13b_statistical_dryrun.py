"""M1.3b synthetic-only inferential contract qualification, NOT scientific analysis.

Uses the *proposed* 90-slot registration as a strict inventory and runs the
proposed seed-paired bootstrap on completely fabricated, clearly marked unit
fixtures. Does not read data/model outputs or accept arbitrary outcome files.
Every computed judgment is prefixed SYNTHETIC_ and CANNOT license GPU runs.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import random
import statistics
from typing import Any, Mapping, Sequence

from synaploom.research.m13b_manifest import canonical_json
from synaploom.research.m13b_prereg_candidate import (
    ARMS, CONFIRMATORY_ORDER, METRICS, ORDERS, PROPOSED_SEED_IDS,
    build_m13b_prereg_candidate, verify_m13b_prereg_candidate,
)


KIND = "M13B_STATISTICAL_SYNTHETIC_SELF_TEST_V1"
FIXTURE_KIND = "SYNTHETIC_NONSCIENCE_METRIC_FIXTURE_V1"
ACCURACY_FIELDS = ("final_average_accuracy", "late_task_within_task_mean_accuracy")
SECONDARY_FIELDS = ("average_forgetting", "backward_transfer")
FIELDS = ACCURACY_FIELDS + SECONDARY_FIELDS
RECORD_KEYS = frozenset((
    "slot_key", "order_id", "order_sha256", "seed", "arm",
    "comparison_role", "candidate_sha256", "fixture_kind",
    "status", "actual_optimizer_steps", "metrics",
))
SCENARIOS = ("positive", "negative", "null", "opposed")


def _paired_deltas(
    items: Mapping[str, Mapping[str, Any]],
    order: str, metric: str,
) -> list[float]:
    """Compare two *synthetic* arms using exactly the same ten seed IDs."""
    out = []
    for seed in PROPOSED_SEED_IDS:
        a = items[f"{order}__seed-{seed}__progressive_reserve"]["metrics"][metric]
        b = items[f"{order}__seed-{seed}__fixed_sparse_mlp"]["metrics"][metric]
        out.append(float(a - b))
    return out


def build_synthetic_fixture(scenario: str = "positive") -> dict[str, Any]:
    """Generate toy summary numbers only; no training, no file read, no outcomes."""
    if scenario not in SCENARIOS:
        raise ValueError("unknown synthetic scenario")
    plan = build_m13b_prereg_candidate()
    verify_m13b_prereg_candidate(plan)
    records = []
    for slot in plan["proposed_trajectory_slots"]:
        i = PROPOSED_SEED_IDS.index(slot["seed"])
        order_index = ORDERS.index(slot["order_id"])
        arm = slot["arm"]
        offset = 0.001 * (i - 4.5)
        center_final = 0.40 + 0.005 * order_index + offset
        center_late = 0.30 + 0.004 * order_index + offset
        if scenario == "positive":
            effects = (0.05, 0.04)
        elif scenario == "negative":
            effects = (-0.05, -0.04)
        elif scenario == "null":
            effects = (0.0, 0.0)
        else:
            effects = (0.05, -0.04)
        # Order-sensitivity is intentionally present in this synthetic fixture.
        # Do not silently pool O1/O2 into O0 confirmation.
        if order_index == 1:
            effects = (-effects[0], -effects[1])
        if arm == "progressive_reserve":
            center_final += effects[0] + 0.0005 * (i - 4.5)
            center_late += effects[1] + 0.0004 * (i - 4.5)
        elif arm == "all_recruited":
            center_final += effects[0] * 0.5
            center_late += effects[1] * 0.5
        record = {
            "slot_key": slot["slot_key"],
            "order_id": slot["order_id"],
            "order_sha256": slot["order_sha256"],
            "seed": slot["seed"],
            "arm": arm,
            "comparison_role": slot["comparison_role"],
            "candidate_sha256": plan["record_sha256"],
            "fixture_kind": FIXTURE_KIND,
            "status": "SYNTHETIC_NOT_TRAINED",
            "actual_optimizer_steps": 0,
            "metrics": {
                "final_average_accuracy": center_final,
                "late_task_within_task_mean_accuracy": center_late,
                "average_forgetting": 0.1 + 0.002 * i,
                "backward_transfer": -0.05 + 0.001 * i,
            },
        }
        records.append(record)
    return {
        "fixture_kind": FIXTURE_KIND,
        "scenario": scenario,
        "scientific_claim": "NONE",
        "real_training_executed": False,
        "candidate_sha256": plan["record_sha256"],
        "records": records,
    }


def validate_synthetic_inventory(fixture: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Fail closed for every missing, duplicate, corrupted or nonfinite slot.

    This validates a toy fixture only, never real result provenance. In a
    future approved scientific runner all data/source/model/mask hashes must
    be independently verified in addition to this inventory.
    """
    if not isinstance(fixture, Mapping):
        raise TypeError("fixture must be a mapping")
    if set(fixture) != {
        "fixture_kind", "scenario", "scientific_claim",
        "real_training_executed", "candidate_sha256", "records",
    }:
        raise ValueError("invalid synthetic fixture schema")
    if fixture["fixture_kind"] != FIXTURE_KIND or fixture["scenario"] not in SCENARIOS:
        raise ValueError("only known synthetic test fixtures supported")
    if fixture["scientific_claim"] != "NONE" or fixture["real_training_executed"] is not False:
        raise ValueError("cannot analyze real or science-claimed outcomes")
    plan = build_m13b_prereg_candidate()
    verify_m13b_prereg_candidate(plan)
    if fixture["candidate_sha256"] != plan["record_sha256"]:
        raise ValueError("candidate plan SHA changed")
    records = fixture["records"]
    if not isinstance(records, list) or len(records) != len(plan["proposed_trajectory_slots"]):
        raise ValueError("incomplete or oversized 90-slot outcome inventory")
    expected = {slot["slot_key"]: slot for slot in plan["proposed_trajectory_slots"]}
    found: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise ValueError("unexpected synthetic result record fields")
        key = record["slot_key"]
        if not isinstance(key, str) or key not in expected or key in found:
            raise ValueError("duplicate or unexpected result slot")
        slot = expected[key]
        for field in ("slot_key", "order_id", "order_sha256", "seed",
                      "arm", "comparison_role"):
            if type(record[field]) is not type(slot[field]) or record[field] != slot[field]:
                raise ValueError("record differs from preregistered slot: " + field)
        if (record["candidate_sha256"] != plan["record_sha256"]
                or record["fixture_kind"] != FIXTURE_KIND
                or record["status"] != "SYNTHETIC_NOT_TRAINED"
                or type(record["actual_optimizer_steps"]) is not int
                or record["actual_optimizer_steps"] != 0):
            raise ValueError("unexpected record provenance or claims of training")
        metrics = record["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != set(FIELDS):
            raise ValueError("missing or additional outcome metrics")
        for metric, val in metrics.items():
            if type(val) not in (int, float) or not math.isfinite(float(val)):
                raise ValueError("missing, nonnumeric, or nonfinite metric " + metric)
            if metric in ACCURACY_FIELDS and not 0 <= val <= 1:
                raise ValueError("accuracy outside [0,1]")
            if metric in SECONDARY_FIELDS and not -1 <= val <= 1:
                raise ValueError("secondary metric outside [-1,1]")
        found[key] = record
    if set(found) != set(expected):
        raise ValueError("missing result slots")
    return found


def _draw_paired_bootstrap(
    values: Mapping[str, Sequence[float]],
    *,
    samples: int = 10_000,
    seed: int = 1337,
) -> dict[str, list[float]]:
    """Resample the SAME paired seed IDs across both co-primary endpoints."""
    if type(samples) is not int or samples < 1 or type(seed) is not int:
        raise ValueError("invalid bootstrap settings")
    if set(values) != set(METRICS):
        raise ValueError("unexpected co-primary endpoints")
    n = len(PROPOSED_SEED_IDS)
    for metric in METRICS:
        series = values[metric]
        if len(series) != n or not all(
            type(x) in (int, float) and math.isfinite(float(x)) for x in series
        ):
            raise ValueError("bootstrap requires ten finite paired seed deltas")
    rng = random.Random(seed)
    draws = {metric: [] for metric in METRICS}
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        for metric in METRICS:
            draws[metric].append(statistics.fmean(values[metric][i] for i in indices))
    for metric in METRICS:
        draws[metric].sort()
    # Match existing historical percentile-index convention.
    return {
        metric: [draws[metric][int(0.025 * (samples - 1))],
                 draws[metric][int(0.975 * (samples - 1))]]
        for metric in METRICS
    }


def analyze_synthetic_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Validate full 90 toy slots first; then demonstrate proposed decisions."""
    data = validate_synthetic_inventory(fixture)
    by_order: dict[str, dict[str, list[float]]] = {}
    means_by_order: dict[str, dict[str, float]] = {}
    for order in ORDERS:
        by_order[order] = {
            metric: _paired_deltas(data, order, metric) for metric in METRICS
        }
        means_by_order[order] = {
            metric: statistics.fmean(by_order[order][metric]) for metric in METRICS
        }

    primary = by_order[CONFIRMATORY_ORDER]
    intervals = _draw_paired_bootstrap(primary)
    if all(intervals[m][0] > 0 for m in METRICS):
        outcome = "SYNTHETIC_FAVORABLE"
    elif all(intervals[m][1] < 0 for m in METRICS):
        outcome = "SYNTHETIC_UNFAVORABLE"
    else:
        outcome = "SYNTHETIC_INCONCLUSIVE"

    # Cluster on seed ID, with all O0/O1/O2 deltas carried together.
    # This is a descriptive, not inferential, all-order summary.
    cluster = {
        metric: [
            statistics.fmean(by_order[order][metric][i] for order in ORDERS)
            for i in range(len(PROPOSED_SEED_IDS))
        ] for metric in METRICS
    }
    pooled_descriptive = _draw_paired_bootstrap(cluster, seed=7331)
    outcome_record: dict[str, Any] = {
        "kind": KIND,
        "fixture_kind": FIXTURE_KIND,
        "scenario": fixture["scenario"],
        "science_executable": False,
        "real_training_executed": False,
        "scientific_claim": "NONE",
        "judgment_scope": "SYNTHETIC_UNIT_TEST_ONLY",
        "candidate_sha256": fixture["candidate_sha256"],
        "all_90_synthetic_slots_present": True,
        "real_scientific_outcomes_present": False,
        "seed_count": len(PROPOSED_SEED_IDS),
        "paired_bootstrap_resamples": 10_000,
        "primary_order": CONFIRMATORY_ORDER,
        "primary_contrast": ["progressive_reserve", "fixed_sparse_mlp"],
        "co_primary_metrics": list(METRICS),
        "primary_mean_deltas_fraction": means_by_order[CONFIRMATORY_ORDER],
        "primary_95pct_percentile_intervals_fraction": intervals,
        "synthetic_decision": outcome,
        "descriptive_order_mean_deltas_fraction": means_by_order,
        "descriptive_equal_order_seed_cluster_intervals_fraction": pooled_descriptive,
        "limitations": [
            "All 90 metric values are invented fixtures. No scientific observations were used.",
            "Only one O0 external contrast is modeled. O1 and O2 are descriptive, not confirmatory.",
            "The same ten seed IDs are paired across all orders; order cells are not independent seeds.",
            "Ten-seed percentile bootstrap coverage, metric correlation and multiplicity require independent review.",
            "This module cannot authorize training or interpret actual experiment outputs.",
        ],
    }
    outcome_record["record_sha256"] = sha256(canonical_json(outcome_record)).hexdigest()
    return outcome_record
