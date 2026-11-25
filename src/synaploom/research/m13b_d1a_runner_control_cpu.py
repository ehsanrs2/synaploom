"""D1-A approved-design control plane: offline replay ONLY.

No worker launcher, dataset, real seed draw, filesystem, clock or CUDA access.
A fabricated successful record is not an optimizer or execution witness.
Real execution requires a separately qualified watchdog, durable attempt
journal, independently anchored provenance and explicit manual owner GO.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
from math import prod
from typing import Any

N = 360
ARMS = ("progressive_reserve", "fixed_sparse_mlp")
LIMIT_MS = 30 * 3600 * 1000
TIMEOUT_MS = {ARMS[0]: 300_000, ARMS[1]: 120_000}
# This is a fixture only. Actual termination margin is not owner-approved.
SYNTHETIC_MARGIN_MS = 10_000
STEPS = 8000
METRICS = ("final_average_accuracy", "late_task_within_task_mean_accuracy")
KIND = "M13B_D1A_SYNTHETIC_RUNNER_CONTROL_V1"
TERMINAL = ("COMPLETE", "FAILED", "TIMEOUT", "INTERRUPTED", "INVALID")


def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False, ensure_ascii=True).encode("ascii")).hexdigest()


def _sha(value: Any) -> bool:
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _ms(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 2**63-1:
        raise ValueError("invalid integer millisecond timestamp")
    return value


def synthetic_registry() -> list[dict[str, Any]]:
    """Full shape, intentionally repeated fake seed values; no RNG."""
    return [
        {"slot_id": f"toy-draw-{i:03d}-{arm}", "draw_index": i,
         "synthetic_seed": 13 if i % 3 else 17, "arm": arm, "order": "O0"}
        for i in range(N)
        for arm in (ARMS if i % 2 == 0 else ARMS[::-1])
    ]


def verify_replay(records: list[dict[str, Any]], *,
                  close_ms: int, close_reason: str,
                  expected_detached_digest: str) -> dict[str, Any]:
    """Verify a closed synthetic prefix. All timestamps are fixture declarations.

    Elapsed occupied time is from origin 0 through close, including idle gaps.
    Each start reserves the full attempt timeout AND cleanup margin. A failure
    is terminal; no retries, replacement or execution after failure. No actual
    process termination or external custody is implemented here.
    """
    close = _ms(close_ms)
    if type(records) is not list or len(records) > 2*N:
        raise ValueError("invalid replay record count")
    registry = synthetic_registry()
    previous = 0
    stopped = False
    bad_timing = False
    keys = {"slot_id", "start_ms", "worker_end_ms", "release_ms",
            "state", "synthetic_observation_sha256"}
    for i, record in enumerate(records):
        if type(record) is not dict or set(record) != keys:
            raise ValueError("invalid replay record schema")
        slot = registry[i]
        if stopped or record["slot_id"] != slot["slot_id"]:
            raise ValueError("out-of-order slot, duplicate, retry or after failure")
        start, end, release = (_ms(record[k]) for k in
                               ("start_ms", "worker_end_ms", "release_ms"))
        if not previous <= start <= end <= release <= close:
            raise ValueError("overlap, backward time or close before release")
        timeout = TIMEOUT_MS[slot["arm"]]
        if start + timeout + SYNTHETIC_MARGIN_MS > LIMIT_MS:
            raise ValueError("attempt started without remaining budget reservation")
        state = record["state"]
        if state not in TERMINAL:
            raise ValueError("unknown attempt state")
        observation = record["synthetic_observation_sha256"]
        if state == "COMPLETE":
            if not _sha(observation):
                raise ValueError("completed attempt requires observation digest")
        elif observation is not None:
            raise ValueError("failed attempt cannot supply a success observation")
        # Record violations as nonconfirmatory; never accept extra attempts.
        bad_timing = (end-start > timeout or release-end > SYNTHETIC_MARGIN_MS)
        stopped = state != "COMPLETE" or bad_timing
        previous = release
    if close < previous:
        raise ValueError("close before final release")
    if stopped:
        expected_reason = "FAILED"
    elif len(records) == len(registry):
        expected_reason = "COMPLETE"
    else:
        next_timeout = TIMEOUT_MS[registry[len(records)]["arm"]]
        expected_reason = ("BUDGET_EXHAUSTED"
                           if close + next_timeout + SYNTHETIC_MARGIN_MS > LIMIT_MS
                           else "INTERRUPTED")
    if close_reason != expected_reason:
        raise ValueError("inconsistent close reason")
    envelope = {"kind": KIND, "registry": registry, "records": records,
                "close_ms": close, "close_reason": close_reason}
    if not _sha(expected_detached_digest) or digest(envelope) != expected_detached_digest:
        raise ValueError("detached synthetic replay digest mismatch")
    eligible = expected_reason == "COMPLETE" and close <= LIMIT_MS and not bad_timing
    return {"kind": KIND,
            "status": "SYNTHETIC_COMPLETE" if eligible else "SYNTHETIC_NO_ANALYSIS",
            "observed_records": len(records),
            "not_started_slot_ids": [x["slot_id"] for x in registry[len(records):]],
            "occupied_ms_declared": close,
            "timing_violation": bad_timing or close > LIMIT_MS,
            "synthetic_analysis_eligible": eligible,
            "science_executable": False, "external_digest_authenticated": False}


def _count(x: Any) -> int:
    if type(x) is not int or not 0 <= x <= 1000:
        raise ValueError("expected integer correct count out of 1000")
    return x


def exact_endpoints(observation: dict[str, Any]) -> tuple[Fraction, Fraction]:
    """Synthetic count contract: 55 seen-stage and 200 epoch evaluations.

    Future instrumentation must capture actual integer correct/example counts
    before float32 matrix conversion; do not reconstruct counts from rounded
    summaries. The recorded epoch steps here are NOT actual optimizer hooks.
    """
    keys = {"kind", "slot_id", "stage_correct", "epoch_correct", "epoch_steps"}
    if (type(observation) is not dict or set(observation) != keys
            or observation["kind"] != KIND):
        raise ValueError("synthetic count observation required")
    stages, curves = observation["stage_correct"], observation["epoch_correct"]
    if (type(stages) is not list or len(stages) != 10
            or type(curves) is not list or len(curves) != 10):
        raise ValueError("missing stages or curves")
    for i, (row, curve) in enumerate(zip(stages, curves)):
        if (type(row) is not list or len(row) != i+1
                or type(curve) is not list or len(curve) != 20):
            raise ValueError("incorrect stage triangle or epoch geometry")
        for x in row + curve:
            _count(x)
    steps = observation["epoch_steps"]
    if (type(steps) is not list or any(type(x) is not int for x in steps)
            or steps != list(range(40, STEPS+1, 40))):
        raise ValueError("incomplete or shifted epoch-step declaration")
    return Fraction(sum(stages[-1]), 10_000), Fraction(sum(curves[-2]+curves[-1]), 40_000)


def fixed_stake(deltas: list[Fraction]) -> dict[str, Any]:
    """Exact rational threshold; conservative lower bound on a 1e-6 grid.

    E(m)=product(1+D)/(1+m)^n. Binary search the largest grid m with
    E(m)>=20. The special all-zero-wealth case returns -1. This is a
    one-sided marginal bound, not simultaneous two-sided confidence.
    """
    if (type(deltas) is not list or len(deltas) != N
            or any(type(x) is not Fraction or not -1 <= x <= 1 for x in deltas)):
        raise ValueError("complete 360 bounded rational paired differences required")
    # Accuracy count inputs have bounded denominators; reject unrelated values.
    if any(40_000 % x.denominator != 0 for x in deltas):
        raise ValueError("difference is not on the admitted evaluation-count grid")
    wealth = prod((1+x for x in deltas), start=Fraction(1))
    lo, hi = -1_000_000, 1_000_000
    while lo < hi:
        mid = (lo+hi+1)//2
        if wealth >= 20*(1+Fraction(mid, 1_000_000))**N:
            lo = mid
        else:
            hi = mid-1
    return {"mean_difference": str(sum(deltas, Fraction(0))/N),
            "reject_positive_null": wealth >= 20,
            "exact_e_value": str(wealth),
            "marginal_lower_95_grid_floor_millionths": lo}


def analyze_closed_synthetic(records: list[dict[str, Any]], *,
                             close_ms: int, close_reason: str,
                             expected_detached_digest: str,
                             observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Only full closed fixture; no interim, no real science claim."""
    audit = verify_replay(records, close_ms=close_ms, close_reason=close_reason,
                          expected_detached_digest=expected_detached_digest)
    if not audit["synthetic_analysis_eligible"]:
        raise ValueError("no analysis of incomplete or invalid fixture")
    if type(observations) is not list or len(observations) != 2*N:
        raise ValueError("full observation registry required")
    registry = synthetic_registry()
    pairs: dict[int, dict[str, tuple[Fraction, Fraction]]] = {}
    for slot, rec, obs in zip(registry, records, observations):
        if (type(obs) is not dict or obs.get("slot_id") != slot["slot_id"]
                or digest(obs) != rec["synthetic_observation_sha256"]):
            raise ValueError("observation identity/digest mismatch")
        pairs.setdefault(slot["draw_index"], {})[slot["arm"]] = exact_endpoints(obs)
    results = {name: fixed_stake([pair[ARMS[0]][j]-pair[ARMS[1]][j]
                                 for pair in pairs.values()])
               for j, name in enumerate(METRICS)}
    return {"kind": KIND, "status": "SYNTHETIC_IUT_ONLY",
            "toy_both_reject": all(x["reject_positive_null"] for x in results.values()),
            "toy_endpoints": results, "science_executable": False,
            "scientific_claim": "NONE", "actual_optimizer_steps": 0}
