"""Offline tests of manual M1.3b GPU pilot guards; never run CUDA training."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from synaploom.protocols.h1a import H1A_CLASS_ORDER
from synaploom.research import m13b_trainonly_cuda_pilot as pilot


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "experiments" / "scripts" / "m13b_trainonly_cuda_pilot.py"


def toy_train_partition():
    # Exactly 20 examples/class for the first 20 O0 classes, *train only*.
    first = H1A_CLASS_ORDER[:20]
    labels = torch.tensor([c for c in first for _ in range(20)], dtype=torch.long)
    features = torch.arange(labels.numel(), dtype=torch.float32).unsqueeze(1).expand(-1, 512).clone()
    return features, labels


def test_train_only_holdout_is_disjoint_deterministic_and_not_test_data():
    x, y = toy_train_partition()
    stream, digest = pilot.train_only_two_task_stream(x, y)
    again, same_digest = pilot.train_only_two_task_stream(x, y)
    assert len(stream) == 2
    assert digest == same_digest and len(digest) == 64
    seen_train = set()
    seen_val = set()
    for task_id, task in enumerate(stream):
        assert task.task_id == task_id
        assert task.classes == H1A_CLASS_ORDER[task_id * 10:(task_id + 1) * 10]
        assert task.train_x.shape == (160, 512)
        assert task.eval_x.shape == (40, 512)
        assert torch.equal(task.train_x, again[task_id].train_x)
        assert torch.equal(task.eval_y, again[task_id].eval_y)
        train_idx = set(map(int, task.train_x[:, 0].tolist()))
        val_idx = set(map(int, task.eval_x[:, 0].tolist()))
        assert train_idx.isdisjoint(val_idx)
        assert len(train_idx) == 160 and len(val_idx) == 40
        seen_train |= train_idx
        seen_val |= val_idx
        for c in task.classes:
            assert int((task.train_y == c).sum()) == pilot.TRAIN_PER_CLASS
            assert int((task.eval_y == c).sum()) == pilot.HOLDOUT_PER_CLASS
    assert seen_train.isdisjoint(seen_val)
    assert len(seen_train) == 320 and len(seen_val) == 80
    assert pilot.ENGINEERING_SEED not in range(501, 511)
    assert pilot.SPLIT_SEED not in range(501, 511)


def test_train_only_split_rejects_missing_examples_and_bad_order():
    x, y = toy_train_partition()
    with pytest.raises(ValueError, match="insufficient"):
        pilot.train_only_two_task_stream(x[:-1], y[:-1])
    with pytest.raises(ValueError, match="complete frozen"):
        pilot.train_only_two_task_stream(x, y, class_order=(1, 2))
    with pytest.raises(ValueError, match="dimension"):
        pilot.train_only_two_task_stream(x[:, :8], y)


def test_untrusted_and_unpinned_files_refused_before_torch_load(tmp_path, monkeypatch):
    path = tmp_path / "mismatch.pt"
    path.write_bytes(b"test only invalid bytes")
    invoked = [False]
    def fake_load(*a, **k):
        invoked[0] = True
        raise AssertionError("pickle was reached before trust and pin check")
    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(ValueError, match="trust"):
        pilot.load_trusted_train_only_features(path, trusted_source=False)
    with pytest.raises(ValueError, match="SHA-256"):
        pilot.load_trusted_train_only_features(path, trusted_source=True)
    assert not invoked[0]


def _cli(*args):
    return subprocess.run([sys.executable, str(CLI), *args], cwd=ROOT,
                          capture_output=True, text=True, check=False)


def test_describe_only_is_safe_cpu_and_has_fixed_contract():
    result = _cli("--describe-only")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["device"] == "cuda:0"
    assert plan["scientific_claim"] == "NONE"
    assert plan["training_part_only"] is True
    assert plan["tasks"] == 2 and plan["epochs_per_task"] == 1
    assert plan["train_per_class"] == 16
    assert plan["holdout_per_class"] == 4
    assert len(plan["arms"]) == 3
    assert plan["full_30_or_90_scientific_runs_allowed"] is False
    assert plan["protected_confirmation_seeds_used"] is False


@pytest.mark.parametrize("args", [
    (),
    ("--manual-gpu-trainonly-pilot",),
    ("--manual-gpu-trainonly-pilot", "--execute"),
    ("--manual-gpu-trainonly-pilot", "--seed", "501"),
    ("--manual-gpu-trainonly-pilot", "--device", "cuda"),
    ("--describe-only", "--trusted-feature-source"),
    ("--describe-only", "--features", "/not-present"),
    ("--describe-only", "--output-dir", "/tmp/foo"),
])
def test_manual_cli_rejects_unapproved_execution_and_science_flags(args):
    x = _cli(*args)
    assert x.returncode != 0
    assert "error:" in x.stderr


def test_existing_output_path_is_rejected_before_feature_access(tmp_path):
    occupied = tmp_path / "already-present"
    occupied.mkdir()
    feature = tmp_path / "fake.pt"
    feature.write_bytes(b"not an actual feature file")
    x = _cli("--manual-gpu-trainonly-pilot", "--features", str(feature),
             "--output-dir", str(occupied), "--trusted-feature-source",
             "--acknowledge-no-scientific-claim")
    assert x.returncode != 0
    assert "already exists" in x.stderr
    assert list(occupied.iterdir()) == []


def test_no_gpu_or_missing_file_reject_without_pilot_artifact(tmp_path):
    outdir = tmp_path / "evidence"
    x = _cli("--manual-gpu-trainonly-pilot", "--features", str(tmp_path / "missing"),
             "--output-dir", str(outdir), "--trusted-feature-source",
             "--acknowledge-no-scientific-claim")
    assert x.returncode != 0
    assert not outdir.exists()
