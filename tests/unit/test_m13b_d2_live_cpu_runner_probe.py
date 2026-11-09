"""D2: real runner CPU instrumentation tests use ONLY generated 2-task tensors."""
from __future__ import annotations

from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from synaploom.research.m13b_d2_live_cpu_runner_probe import (
    EXPECTED_STEPS, _check_small_result, _stream, _stream_digest,
    cpu_source_file_sha256, expected_evaluation_events, run_tiny_cpu_probe,
    validate_events,
)


@pytest.mark.parametrize("masked", [False, True])
def test_real_runner_executes_twelve_optimizer_steps_and_all_eval_batches(masked):
    actual=run_tiny_cpu_probe(seed=13,masked=masked)
    assert actual["status"]=="ACTUAL_TINY_SYNTHETIC_CPU_RUNNER_PASS", actual
    assert actual["real_cpu_optimizer_post_step_observations"]==EXPECTED_STEPS
    assert actual["real_cpu_mask_post_step_checks"]==(EXPECTED_STEPS if masked else 0)
    assert actual["observed_eval_forward_batches"]==14
    assert actual["observed_eval_membership_matches_synthetic_ids"] is True
    assert actual["stream_bytes_stable"] is True
    assert actual["source_bytes_stable"] is True
    assert actual["model_state_changed"] is True
    assert actual["caller_pin_checked"] is False
    assert actual["actual_science_optimizer_steps"]==0
    assert actual["source_attested_by_independent_signer"] is False
    assert actual["full_10_task_8000_step_contract_qualified"] is False
    assert actual["science_runner_integration"] is False
    assert actual["science_executable"] is False
    assert actual["scientific_claim"]=="NONE"


def test_source_pin_is_compared_to_loaded_python_file_bytes_not_fixture_metadata():
    ref=cpu_source_file_sha256()
    assert set(ref)=={"continual","evaluation","batch_iterator"}
    assert all(len(v)==64 for v in ref.values())
    actual=run_tiny_cpu_probe(seed=17,expected_source_sha256=ref)
    assert actual["status"]=="ACTUAL_TINY_SYNTHETIC_CPU_RUNNER_PASS", actual
    assert actual["caller_pin_checked"] is True
    wrong=dict(ref)
    wrong["evaluation"]="0"*64
    with pytest.raises(ValueError,match="caller-pinned source"):
        run_tiny_cpu_probe(seed=17,expected_source_sha256=wrong)


def test_exact_synthetic_eval_sample_membership_and_step_witness_schedule():
    expected=expected_evaluation_events()
    assert len(expected)==14
    assert [len(v["sample_ids"]) for v in expected]==[4,3]*7
    assert [v["steps_observed"] for v in expected]==[
        3,3,6,6,6,6,9,9,12,12,12,12,12,12,
    ]
    assert expected[0]["sample_ids"]==[0,1,2,3]
    assert expected[1]["sample_ids"]==[4,5,6]
    assert expected[-2]["sample_ids"]==[100,101,102,103]
    assert expected[-1]["sample_ids"]==[104,105,106]
    validate_events(expected)
    dropped=deepcopy(expected)
    dropped.pop()
    with pytest.raises(ValueError,match="membership"):
        validate_events(dropped)
    wrong_order=deepcopy(expected)
    wrong_order[0]["sample_ids"].reverse()
    with pytest.raises(ValueError,match="membership"):
        validate_events(wrong_order)
    wrong_step=deepcopy(expected)
    wrong_step[0]["steps_observed"]=4
    with pytest.raises(ValueError,match="hook-step"):
        validate_events(wrong_step)


def test_real_runner_failure_is_preserved_not_reclassified_as_success():
    failed=run_tiny_cpu_probe(seed=13,masked=True,fault_after_step=4)
    assert failed["status"]=="SYNTHETIC_CPU_RUNNER_FAILURE_RETAINED",failed
    assert failed["error_type"]=="RuntimeError"
    assert "INJECTED_SYNTHETIC_CPU_FAULT" in failed["error_detail"]
    assert failed["observed_cpu_post_step_calls_before_failure"]==4
    assert not failed["science_executable"]
    assert failed["actual_science_optimizer_steps"]==0


@pytest.mark.parametrize("kwargs",[
    {"seed":501},{"seed":0},{"seed":True},{"seed":13,"masked":"yes"},
    {"seed":13,"fault_after_step":0},{"seed":13,"fault_after_step":13},
    {"seed":13,"fault_after_step":True},
])
def test_no_reserved_science_seeds_or_invalid_probe_controls(kwargs):
    with pytest.raises(ValueError):
        run_tiny_cpu_probe(**kwargs)


def test_generated_stream_is_byte_hashed_and_detects_tampering():
    stream=_stream()
    before=_stream_digest(stream)
    assert before==_stream_digest(_stream())
    with torch.no_grad():
        stream[0].eval_x[0,0]+=1
    assert _stream_digest(stream)!=before


def test_cpu_probe_restores_python_torch_numpy_rng_states():
    pystate=random.getstate()
    torchstate=torch.get_rng_state().clone()
    numpy_state=np.random.get_state()
    actual=run_tiny_cpu_probe(seed=17)
    assert actual["status"]=="ACTUAL_TINY_SYNTHETIC_CPU_RUNNER_PASS",actual
    assert random.getstate()==pystate
    assert torch.equal(torch.get_rng_state(),torchstate)
    new_numpy_state=np.random.get_state()
    assert new_numpy_state[0]==numpy_state[0]
    assert np.array_equal(new_numpy_state[1],numpy_state[1])
    assert new_numpy_state[2:]==numpy_state[2:]
