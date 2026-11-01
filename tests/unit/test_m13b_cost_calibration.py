"""Offline checks of M1.3b train-only CUDA cost calibration design.

Never opens frozen features on CI or executes any optimizer/GPU step.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from synaploom.protocols.h1a import H1A_CLASS_ORDER
from synaploom.research.m13b_cost_calibration import (
    ARMS, CLASS_COUNT_PER_TASK, EPOCHS, FEATURE_SHA256,
    HOLDOUT_PER_CLASS, KIND, MAX_STEPS, SEED, SPLIT_SEED,
    TASK_OPTIONS, TRAIN_PER_CLASS,
    describe_cost_calibration, make_training_partition_calibration_stream,
    run_one_arm_trainonly_cost_calibration,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "experiments" / "scripts" / "m13b_cost_calibration.py"


@pytest.fixture(scope="module")
def fake_training_only_full_inventory():
    # 100 classes × exactly 500 examples, no protected eval partition.
    # Expanded view avoids materializing the full feature bundle for this test.
    labels = torch.arange(100, dtype=torch.long).repeat_interleave(500)
    feats = torch.arange(50_000, dtype=torch.float32).reshape(-1, 1).expand(-1, 512)
    return feats, labels


def test_calibration_plan_bounded_one_arm_and_no_science():
    p = describe_cost_calibration()
    assert p["kind"] == KIND
    assert p["status"] == "MANUAL_ENGINEERING_CALIBRATION_NOT_SCIENCE"
    assert p["science_executable"] is False
    assert p["scientific_claim"] == "NONE"
    assert p["feature_sha256_expected"] == FEATURE_SHA256
    assert p["data_scope"] == "TRUSTED_FROZEN_CIFAR_TRAIN_ONLY"
    assert p["candidate_arms"] == list(ARMS)
    assert p["one_arm_per_manual_invocation"] is True
    assert p["task_options"] == [2, 10]
    assert p["training_examples_per_class"] == 496
    assert p["train_only_holdout_examples_per_class"] == 4
    assert p["optimizer_steps_per_task"] == 39
    assert p["optimizer_steps_at_2_tasks"] == 78
    assert p["optimizer_steps_at_10_tasks"] == MAX_STEPS == 390
    assert p["full_science_optimizer_steps_per_trajectory"] == 8000
    assert p["full_science_evaluation_workload_reproduced"] is False
    assert p["full_science_gpu_time_estimate_produced"] is False
    assert p["confirmation_seeds_used"] is False
    assert p["not_a_strict_walltime_timeout"] is True
    assert SEED not in range(501, 511)
    assert SPLIT_SEED not in range(501, 511)


def test_two_task_calibration_full_real_batch_geometry(fake_training_only_full_inventory):
    x, y = fake_training_only_full_inventory
    prior_state = torch.get_rng_state().clone()
    stream, signature = make_training_partition_calibration_stream(x, y, tasks=2)
    assert torch.equal(prior_state, torch.get_rng_state())
    assert len(signature) == 64
    assert len(stream) == 2
    global_train, global_holdout = set(), set()
    for t, task in enumerate(stream):
        assert task.task_id == t
        assert task.classes == H1A_CLASS_ORDER[t * 10:(t + 1) * 10]
        assert task.train_x.shape == (4960, 512)
        assert task.eval_x.shape == (40, 512)
        assert task.train_y.shape == (4960,)
        assert task.eval_y.shape == (40,)
        ids_train = set(map(int, task.train_x[:, 0].tolist()))
        ids_holdout = set(map(int, task.eval_x[:, 0].tolist()))
        assert len(ids_train) == 4960
        assert len(ids_holdout) == 40
        assert ids_train.isdisjoint(ids_holdout)
        global_train |= ids_train
        global_holdout |= ids_holdout
        for class_id in task.classes:
            assert int((task.train_y == class_id).sum()) == TRAIN_PER_CLASS
            assert int((task.eval_y == class_id).sum()) == HOLDOUT_PER_CLASS
    assert global_train.isdisjoint(global_holdout)
    assert len(global_train) == 9920
    assert len(global_holdout) == 80
    next_stream, next_sig = make_training_partition_calibration_stream(x, y, tasks=2)
    assert signature == next_sig
    assert torch.equal(stream[0].train_x, next_stream[0].train_x)
    assert torch.equal(stream[1].eval_y, next_stream[1].eval_y)


def test_ten_tasks_and_all_classes_never_access_protected_eval(fake_training_only_full_inventory):
    x, y = fake_training_only_full_inventory
    stream, sig = make_training_partition_calibration_stream(x, y, tasks=10)
    assert len(sig) == 64 and len(stream) == 10
    assert stream.num_classes == 100
    assert sum(t.train_y.numel() for t in stream) == 49_600
    assert sum(t.eval_y.numel() for t in stream) == 400
    assert sum(t.train_y.numel() + t.eval_y.numel() for t in stream) == 50_000
    assert {c for t in stream for c in t.classes} == set(range(100))
    assert all(t.train_x.shape[0] == 4960 and t.eval_x.shape[0] == 40
               for t in stream)


@pytest.mark.parametrize("tasks", [0, 1, 3, 11, True, 2.0, "10"])
def test_task_scope_rejects_unregistered_requests(fake_training_only_full_inventory, tasks):
    x, y = fake_training_only_full_inventory
    with pytest.raises(ValueError, match="two or ten"):
        make_training_partition_calibration_stream(x, y, tasks=tasks)


def test_corrupt_labels_or_features_fail_before_training(fake_training_only_full_inventory):
    x, y = fake_training_only_full_inventory
    with pytest.raises(ValueError, match="50k"):
        make_training_partition_calibration_stream(x[:49_999], y, tasks=2)
    with pytest.raises(ValueError, match="500 examples"):
        bad = y.clone()
        bad[0] = 1
        make_training_partition_calibration_stream(x, bad, tasks=2)
    with pytest.raises(ValueError, match="50k"):
        make_training_partition_calibration_stream(x[:, :8], y, tasks=2)


def test_full_gpu_function_rejects_invalid_device_independent_inputs(fake_training_only_full_inventory):
    x, y = fake_training_only_full_inventory
    stream, digest = make_training_partition_calibration_stream(x, y, tasks=2)
    with pytest.raises(ValueError, match="unregistered engineering arm"):
        run_one_arm_trainonly_cost_calibration(
            stream, arm="rigl_like_mlp", tasks=2,
            frozen_sha256=FEATURE_SHA256, split_sha256=digest
        )
    with pytest.raises(ValueError, match="task count mismatch"):
        run_one_arm_trainonly_cost_calibration(
            stream, arm="progressive_reserve", tasks=10,
            frozen_sha256=FEATURE_SHA256, split_sha256=digest
        )
    with pytest.raises(ValueError, match="trusted feature"):
        run_one_arm_trainonly_cost_calibration(
            stream, arm="progressive_reserve", tasks=2,
            frozen_sha256="0" * 64, split_sha256=digest
        )


def _cli(*args):
    return subprocess.run([sys.executable, str(CLI), *args],
                          cwd=ROOT, capture_output=True, text=True, check=False)


def test_safe_describe_only_does_not_open_features():
    x = _cli("--describe-only")
    assert x.returncode == 0, x.stderr
    assert json.loads(x.stdout) == describe_cost_calibration()


@pytest.mark.parametrize("bad", [
    (),
    ("--manual-trainonly-cost-calibration",),
    ("--describe-only", "--arm", "progressive_reserve"),
    ("--manual-trainonly-cost-calibration", "--arm", "unknown"),
    ("--manual-trainonly-cost-calibration", "--tasks", "3"),
    ("--manual-trainonly-cost-calibration", "--tasks", "20"),
    ("--manual-trainonly-cost-calibration", "--epochs", "20"),
    ("--manual-trainonly-cost-calibration", "--seed", "501"),
    ("--manual-trainonly-cost-calibration", "--device", "cuda"),
    ("--manual-trainonly-cost-calibration", "--execute-science"),
    ("--manual-trainonly-cost-calibration", "--all-arms"),
    ("--manual-trainonly-cost-calibration", "--output", "results.json"),
    ("--manual-trainonly-cost-calibration", "--max-hours", "24"),
    ("--manual-trainonly-cost-calibration", "--untrusted-source"),
    ("--manual-trainonly-cost-calibration", "--approve"),
])
def test_no_science_or_implicit_cuda_run_from_cli(bad):
    x = _cli(*bad)
    assert x.returncode != 0
    assert "error:" in x.stderr


def test_existing_output_prevents_real_data_read_or_gpu_run(tmp_path):
    dest = tmp_path / "already-has-evidence"
    dest.mkdir()
    source = tmp_path / "not-a-real-feature-file.pt"
    source.write_bytes(b"not a trusted pickle")
    x = _cli(
        "--manual-trainonly-cost-calibration",
        "--arm", "all_recruited",
        "--tasks", "2",
        "--features", str(source),
        "--output-dir", str(dest),
        "--trusted-feature-source",
        "--acknowledge-engineering-no-science",
    )
    assert x.returncode != 0
    assert "existing evidence" in x.stderr
    assert list(dest.iterdir()) == []
