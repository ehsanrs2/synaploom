"""Pure CPU/no-input contract tests: synthetic ledger and detached digest pin."""
from copy import deepcopy
from hashlib import sha256

import pytest

from synaploom.research.m13b_d2_trust_envelope_cpu import (
    ARMS, BATCH_SIZES, FAILED, KIND, SEEDS, SLOTS, SUCCEEDED,
    build_synthetic_envelope, d2_trust_requirements,
    digest_json, verify_synthetic_envelope,
)

PINS = {
    "source": sha256(b"invented D2 source context").hexdigest(),
    "sample": sha256(b"invented 1000 rows in 4 observed groups").hexdigest(),
    "mask": sha256(b"invented fixed mask bits").hexdigest(),
}
ATTEMPTS = [(SUCCEEDED,12),(FAILED,4),(SUCCEEDED,12),(SUCCEEDED,12)]


def envelope():
    return build_synthetic_envelope(
        independently_expected_source_sha256=PINS["source"],
        independently_expected_sample_layout_sha256=PINS["sample"],
        independently_expected_fixed_mask_sha256=PINS["mask"],
        attempts=ATTEMPTS,
    )


def check(e, *, root=None, **pin_changes):
    pins = PINS | pin_changes
    return verify_synthetic_envelope(
        e, detached_expected_root_sha256=root or digest_json(envelope()),
        expected_source_sha256=pins["source"],
        expected_sample_layout_sha256=pins["sample"],
        expected_fixed_mask_sha256=pins["mask"],
    )


def test_valid_synthetic_only_detached_pin_and_failure_preservation():
    e=envelope()
    original=deepcopy(e)
    result=check(e)
    assert e==original
    assert result["status"]=="DETACHED_PIN_MATCH_FOR_SYNTHETIC_FIXTURE_ONLY"
    assert result["toy_slot_count"]==4
    assert result["toy_failure_count"]==1
    assert not result["owner_signed"]
    assert not result["science_executable"]
    assert len(e["events"])==len(SLOTS)==len(SEEDS)*len(ARMS)
    assert e["declared_eval_batch_sizes"]==list(BATCH_SIZES)
    assert check(e)==result


def test_complete_failure_preserved_even_with_valid_hashes():
    e=envelope()
    assert e["events"][1]["terminal_state"]==FAILED
    assert e["events"][1]["cpu_optimizer_post_steps"]==4
    assert check(e)["toy_failure_count"]==1
    assert check(e)["all_toy_slots_present"]


@pytest.mark.parametrize("change", [
    lambda x: x["events"].pop(),
    lambda x: x["events"].__setitem__(0, x["events"][1]),
    lambda x: x["events"][0].update(cpu_optimizer_post_steps=11),
    lambda x: x["events"][1].update(terminal_state=SUCCEEDED),
    lambda x: x["events"][1].update(cpu_optimizer_post_steps=12),
    lambda x: x["events"][2].update(previous_sha256="0"*64),
    lambda x: x.update(chain_head_sha256="0"*64),
    lambda x: x.update(declared_eval_batch_sizes=[512,488]),
    lambda x: x.update(owner_signed=True),
    lambda x: x.update(science_executable=True),
    lambda x: x.update(data_protected=True),
    lambda x: x.update(expected_seed_registry=[501,502]),
    lambda x: x.update(sciencerun_optimizer_steps=8000),
    lambda x: x.update(extra=True),
])
def test_mutation_fail_closed(change):
    modified=envelope()
    change(modified)
    with pytest.raises(ValueError):
        check(modified)


def test_rehashed_whole_chain_still_rejected_by_detached_root():
    original=envelope()
    malicious=build_synthetic_envelope(
        independently_expected_source_sha256=PINS["source"],
        independently_expected_sample_layout_sha256=PINS["sample"],
        independently_expected_fixed_mask_sha256=PINS["mask"],
        attempts=[(SUCCEEDED,12)]*4,
    )
    assert malicious["chain_head_sha256"]==malicious["events"][-1]["event_sha256"]
    assert check(original)["toy_failure_count"]==1
    with pytest.raises(ValueError,match="detached external root"):
        check(malicious)


def test_separately_expected_source_sample_mask_digests_fail_closed():
    for pin in ("source","sample","mask"):
        with pytest.raises(ValueError, match="separately pinned expectations"):
            check(envelope(),**{pin:"f"*64})


@pytest.mark.parametrize("bad", [
    [(SUCCEEDED,12)],
    [(SUCCEEDED,12),(FAILED,-1),(SUCCEEDED,12),(SUCCEEDED,12)],
    [(SUCCEEDED,12),(FAILED,4),(SUCCEEDED,11),(SUCCEEDED,12)],
    [(SUCCEEDED,12),(FAILED,True),(SUCCEEDED,12),(SUCCEEDED,12)],
    [(SUCCEEDED,12),("UNKNOWN",4),(SUCCEEDED,12),(SUCCEEDED,12)],
])
def test_invalid_builder_registry_fail_closed(bad):
    with pytest.raises(ValueError):
        build_synthetic_envelope(
            independently_expected_source_sha256=PINS["source"],
            independently_expected_sample_layout_sha256=PINS["sample"],
            independently_expected_fixed_mask_sha256=PINS["mask"],attempts=bad,
        )


def test_invalid_external_root_and_rehashed_wrong_root():
    with pytest.raises(ValueError,match="detached root pin"):
        check(envelope(),root="bad")
    with pytest.raises(ValueError,match="detached external root"):
        check(envelope(),root="f"*64)


def test_documented_external_science_gates_remain_unimplemented():
    record=d2_trust_requirements()
    assert record["kind"]==KIND
    assert record["status"]=="PROSPECTIVE_REQUIREMENTS_NOT_IMPLEMENTED"
    assert len(record["requirements"])==10
    assert not record["implementation_proof_for_science"]
    assert not record["owner_signed_protocol"]
    assert not record["independent_witness"]
    assert not record["science_executable"]
