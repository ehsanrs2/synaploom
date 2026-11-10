"""D2 evidence checks: no CIFAR files, only fabricated 1000-row CPU tensors."""
from __future__ import annotations

import json

import pytest
import torch

from synaploom.training import evaluation as evaluation_module
from synaploom.research.m13b_d2_cpu_evidence import (
    KIND, audit_reference_cpu_evaluation, make_synthetic_reference_task,
    verify_synthetic_attempts, write_synthetic_attempt,
)


def test_actual_1000_synthetic_rows_through_original_evaluator():
    report = audit_reference_cpu_evaluation()
    assert report["kind"] == KIND
    assert report["status"] == "SYNTHETIC_REAL_EVAL_B256_B232_PASS"
    assert report["observed_forward_count"] == 4
    assert report["observed_batch_sizes"] == [256, 256, 256, 232]
    assert report["observed_first_last_ids"] == [
        [0, 255], [256, 511], [512, 767], [768, 999],
    ]
    assert len(set(report["observed_input_sha256"])) == 4
    assert report["all_1000_synthetic_rows_exactly_once"]
    assert report["protected_membership_attested"] is False
    assert report["science_executable"] is False
    assert report["gpu_used"] is False


@pytest.mark.parametrize("mutate", [
    lambda task: task.eval_x.__setitem__((0, 0), -10.0),
    lambda task: task.eval_x.__setitem__((999, 2), 998.0),
    lambda task: task.eval_x.__setitem__((0, 2), 1.0),
    lambda task: task.eval_y.__setitem__(0, 1),
    lambda task: task.train_x.__setitem__((0, 0), 0.33),
])
def test_byte_level_member_or_label_corruption_rejected(mutate):
    task = make_synthetic_reference_task()
    mutate(task)
    with pytest.raises(ValueError, match="mismatched synthetic"):
        audit_reference_cpu_evaluation(task)


def test_actual_evaluator_missing_partial_tail_is_detected(monkeypatch):
    original = evaluation_module.iter_tensor_batches

    def _drop_last(*args, **kwargs):
        batches = list(original(*args, **kwargs))
        yield from batches[:-1]

    monkeypatch.setattr(evaluation_module, "iter_tensor_batches", _drop_last)
    with pytest.raises(ValueError, match="grouped incorrect|missed"):
        audit_reference_cpu_evaluation()


def test_actual_evaluator_shuffled_synthetic_batch_is_detected(monkeypatch):
    original = evaluation_module.iter_tensor_batches

    def _shuffle_first(*args, **kwargs):
        batches = list(original(*args, **kwargs))
        x, y = batches[0]
        batches[0] = (x.flip(0), y.flip(0))
        yield from batches

    monkeypatch.setattr(evaluation_module, "iter_tensor_batches", _shuffle_first)
    with pytest.raises(ValueError, match="grouped incorrect"):
        audit_reference_cpu_evaluation()


def test_write_once_ledger_records_actual_tiny_success_and_failure(tmp_path):
    directory = tmp_path / "synthetic-attempts"
    good = write_synthetic_attempt(directory, attempt_id="success-a")
    bad = write_synthetic_attempt(
        directory, attempt_id="failure-b", fault_after_step=4,
    )
    assert good.exists() and bad.exists()
    success = json.loads(good.read_bytes())
    failure = json.loads(bad.read_bytes())
    assert success["terminal_state"] == "ACTUAL_TINY_SYNTHETIC_CPU_RUNNER_PASS"
    assert success["tiny_cpu_probe"]["real_cpu_optimizer_post_step_observations"] == 12
    assert failure["terminal_state"] == "SYNTHETIC_CPU_RUNNER_FAILURE_RETAINED"
    assert failure["tiny_cpu_probe"]["observed_cpu_post_step_calls_before_failure"] == 4
    assert failure["reference_eval"]["observed_batch_sizes"] == [256, 256, 256, 232]
    audit = verify_synthetic_attempts(
        directory, expected_ids=["success-a", "failure-b"],
    )
    assert audit["attempt_count"] == 2
    assert audit["all_successful"] is False
    assert audit["signed_or_worm_attestation"] is False
    assert audit["scientific_claim"] == "NONE"
    with pytest.raises(FileExistsError):
        write_synthetic_attempt(directory, attempt_id="success-a")
    with pytest.raises(ValueError, match="missing/extra"):
        verify_synthetic_attempts(directory, expected_ids=["success-a"])


def test_write_once_file_tampering_or_incomplete_json_is_detected(tmp_path):
    directory = tmp_path / "synthetic"
    name = write_synthetic_attempt(directory, attempt_id="attempt-c")
    content = json.loads(name.read_bytes())
    content["terminal_state"] = "SYNTHETIC_CPU_RUNNER_FAILURE_RETAINED"
    name.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="digest"):
        verify_synthetic_attempts(directory, expected_ids=["attempt-c"])
    name.write_text("{")
    with pytest.raises(ValueError, match="unreadable|incomplete"):
        verify_synthetic_attempts(directory, expected_ids=["attempt-c"])


def test_missing_attempts_rejected_and_attempt_names_safe(tmp_path):
    for name in ("../unsafe", "/tmp/out", "bad.name", "", "A", "x/child"):
        with pytest.raises(ValueError, match="attempt"):
            write_synthetic_attempt(tmp_path, attempt_id=name)
    for step in (0, 13, True, "4"):
        with pytest.raises(ValueError, match="invalid small-CPU-only injected failure step"):
            write_synthetic_attempt(tmp_path, attempt_id="valid-name",
                                    fault_after_step=step)
    with pytest.raises(ValueError, match="expected"):
        verify_synthetic_attempts(tmp_path, expected_ids=[])
