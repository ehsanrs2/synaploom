from copy import deepcopy
from fractions import Fraction

import pytest

from synaploom.research.m13b_d1a_runner_control_cpu import (
    ARMS, KIND, LIMIT_MS, METRICS, N, STEPS, SYNTHETIC_MARGIN_MS,
    TIMEOUT_MS, analyze_closed_synthetic, digest, exact_endpoints,
    fixed_stake, synthetic_registry, verify_replay,
)


def observation(slot, count=500, late=None):
    return {"kind": KIND, "slot_id": slot["slot_id"],
            "stage_correct": [[count]*(i+1) for i in range(10)],
            "epoch_correct": [[count if i<8 or late is None else late]*20
                              for i in range(10)],
            "epoch_steps": list(range(40, STEPS+1, 40))}


def make_records(observations):
    return [
        {"slot_id": obs["slot_id"], "start_ms": i*1000,
         "worker_end_ms": i*1000+500, "release_ms": (i+1)*1000,
         "state": "COMPLETE", "synthetic_observation_sha256": digest(obs)}
        for i, obs in enumerate(observations)
    ]


def pin(records, close, reason):
    return digest({"kind": KIND, "registry": synthetic_registry(),
                   "records": records, "close_ms": close, "close_reason": reason})


def verify(records, close, reason, root=None):
    return verify_replay(records, close_ms=close, close_reason=reason,
                         expected_detached_digest=pin(records, close, reason)
                         if root is None else root)


@pytest.fixture(scope="module")
def complete():
    obs = [observation(slot, 520 if slot["arm"]==ARMS[0] else 500)
           for slot in synthetic_registry()]
    return make_records(obs), obs


def test_full_720_slot_registry_keeps_duplicate_draws_and_balances_order():
    slots = synthetic_registry()
    assert len(slots) == 720
    assert len({x["slot_id"] for x in slots}) == 720
    assert len({x["synthetic_seed"] for x in slots}) == 2
    assert sum(x["arm"] == ARMS[0] for x in slots[::2]) == 180
    for i in range(N):
        assert slots[2*i]["draw_index"] == slots[2*i+1]["draw_index"] == i
        assert slots[2*i]["synthetic_seed"] == slots[2*i+1]["synthetic_seed"]


def test_complete_exact_iut(complete):
    records, obs = complete
    close = 720_000
    r = analyze_closed_synthetic(records, close_ms=close, close_reason="COMPLETE",
                                 expected_detached_digest=pin(records, close, "COMPLETE"),
                                 observations=obs)
    assert r["toy_both_reject"]
    assert not r["science_executable"]
    assert r["actual_optimizer_steps"] == 0
    for m in METRICS:
        assert r["toy_endpoints"][m]["mean_difference"] == "1/50"
        assert r["toy_endpoints"][m]["reject_positive_null"]


def test_iut_requires_both():
    obs = [observation(s, 520 if s["arm"]==ARMS[0] else 500, late=500)
           for s in synthetic_registry()]
    records = make_records(obs)
    close = 720_000
    r = analyze_closed_synthetic(records, close_ms=close, close_reason="COMPLETE",
                                 expected_detached_digest=pin(records,close,"COMPLETE"),
                                 observations=obs)
    assert r["toy_endpoints"][METRICS[0]]["reject_positive_null"]
    assert not r["toy_endpoints"][METRICS[1]]["reject_positive_null"]
    assert not r["toy_both_reject"]


def test_exact_lower_bound_floor_and_boundary_cases():
    d = Fraction(1,50)
    v = fixed_stake([d]*N)
    w = (1+d)**N
    k = v["marginal_lower_95_grid_floor_millionths"]
    assert 20*(1+Fraction(k,1_000_000))**N <= w
    assert 20*(1+Fraction(k+1,1_000_000))**N > w
    assert fixed_stake([Fraction(0)]*N)["reject_positive_null"] is False
    assert fixed_stake([Fraction(-1)]+[d]*(N-1))[
        "marginal_lower_95_grid_floor_millionths"] == -1_000_000
    assert fixed_stake([Fraction(1)]*N)["reject_positive_null"]


@pytest.mark.parametrize("values", [[Fraction(0)]*359, [0]*360,
                                    [Fraction(1,3)]*360, [Fraction(2)]*360])
def test_invalid_statistical_inputs(values):
    with pytest.raises(ValueError):
        fixed_stake(values)


@pytest.mark.parametrize("change", [
    lambda x: x["stage_correct"].pop(),
    lambda x: x["stage_correct"][0].append(500),
    lambda x: x["stage_correct"][1].__setitem__(0, True),
    lambda x: x["epoch_correct"][-1].pop(),
    lambda x: x["epoch_correct"][0].__setitem__(0, 1001),
    lambda x: x["epoch_steps"].__setitem__(30, 1),
    lambda x: x.update(kind="REAL_SCIENCE"),
])
def test_invalid_count_observation(change):
    x=observation(synthetic_registry()[0])
    change(x)
    with pytest.raises(ValueError):
        exact_endpoints(x)


def test_failure_stops_and_retains_remaining_slots(complete):
    records=deepcopy(complete[0][:2])
    records[1].update(state="FAILED",synthetic_observation_sha256=None)
    a=verify(records, 2000, "FAILED")
    assert a["observed_records"]==2
    assert len(a["not_started_slot_ids"])==718
    assert not a["synthetic_analysis_eligible"]
    with pytest.raises(ValueError,match="after failure"):
        verify(records+[complete[0][2]],3000,"INTERRUPTED")
    with pytest.raises(ValueError,match="no analysis"):
        analyze_closed_synthetic(records,close_ms=2000,close_reason="FAILED",
                                 expected_detached_digest=pin(records,2000,"FAILED"),
                                 observations=[])


def test_budget_reserves_attempt_plus_cleanup_and_counts_idle():
    arm=synthetic_registry()[0]["arm"]
    start=LIMIT_MS-TIMEOUT_MS[arm]-SYNTHETIC_MARGIN_MS
    rec={"slot_id":synthetic_registry()[0]["slot_id"],"start_ms":start,
         "worker_end_ms":start+100,"release_ms":start+200,
         "state":"COMPLETE","synthetic_observation_sha256":"a"*64}
    assert verify([rec], LIMIT_MS, "BUDGET_EXHAUSTED")["occupied_ms_declared"]==LIMIT_MS
    rec["start_ms"]+=1
    with pytest.raises(ValueError,match="remaining budget"):
        verify([rec],LIMIT_MS,"BUDGET_EXHAUSTED")


def test_overrun_retained_as_failure(complete):
    rec=deepcopy(complete[0][0])
    rec.update(worker_end_ms=300_001,release_ms=300_002)
    r=verify([rec],300_002,"FAILED")
    assert r["timing_violation"]
    assert not r["synthetic_analysis_eligible"]


@pytest.mark.parametrize("change", [
    lambda x: x[1].update(start_ms=0),
    lambda x: x[0].update(worker_end_ms=True),
    lambda x: x[0].update(slot_id=x[1]["slot_id"]),
    lambda x: x[1].update(slot_id=x[0]["slot_id"]),
    lambda x: x[0].update(state="FAILED"),
    lambda x: x[0].update(synthetic_observation_sha256=None),
])
def test_invalid_replay_rejected(complete,change):
    records=deepcopy(complete[0][:2])
    change(records)
    with pytest.raises(ValueError):
        verify(records,2000,"INTERRUPTED")


def test_pin_and_observation_tamper(complete):
    records,obs=complete
    root=pin(records,720_000,"COMPLETE")
    changed=deepcopy(records)
    changed[0]["worker_end_ms"]+=1
    with pytest.raises(ValueError,match="detached"):
        verify(changed,720_000,"COMPLETE",root)
    modified=deepcopy(obs)
    modified[0]["stage_correct"][0][0]+=1
    with pytest.raises(ValueError,match="identity/digest"):
        analyze_closed_synthetic(records,close_ms=720_000,close_reason="COMPLETE",
                                 expected_detached_digest=root,observations=modified)


def test_interruption_and_missing_records_are_not_complete(complete):
    records=complete[0][:1]
    r=verify(records,1000,"INTERRUPTED")
    assert not r["synthetic_analysis_eligible"]
    assert len(r["not_started_slot_ids"])==719
    with pytest.raises(ValueError,match="close reason"):
        verify(records,1000,"COMPLETE")
