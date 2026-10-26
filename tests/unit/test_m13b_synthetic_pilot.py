"""Bounded toy pilot; verify all three arms without CIFAR or GPU."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from synaploom.research.m13b_synthetic_pilot import (
    ARMS,
    PILOT_KIND,
    RESERVED_CONFIRMATION,
    PILOT_SEED,
    DATA_SEED,
    run_m13b_synthetic_engineering_pilot,
    pilot_canonical_json,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "experiments" / "scripts" / "m13b_synthetic_pilot.py"
FLAGS = ("--synthetic-engineering-pilot", "--acknowledge-engineering-only")


@pytest.fixture(scope="module")
def sample():
    return run_m13b_synthetic_engineering_pilot()


def test_pilot_has_exact_cpu_toy_scope_and_three_arms(sample):
    assert sample["kind"] == PILOT_KIND
    assert sample["schema"] == 1
    assert sample["device"] == "cpu"
    assert sample["source_dataset"] == "synthetic_gaussian_3x2_NOT_CIFAR100"
    assert sample["science_executable"] is False
    assert sample["scientific_claim"] == "NONE"
    assert sample["authorization"] == "SYNTHETIC_CPU_ENGINEERING_ONLY"
    assert sample["historical_h1a_results_unchanged"] is True
    assert sample["protected_confirmation_seeds_untouched"] is True
    assert sample["seed"] == PILOT_SEED == 13
    assert sample["data_seed"] == DATA_SEED == 41
    assert PILOT_SEED not in RESERVED_CONFIRMATION
    assert DATA_SEED not in RESERVED_CONFIRMATION
    assert sample["epochs_per_task"] == 1
    assert len(sample["data_sha256"]) == 64
    assert tuple(sample["arms"]) == ARMS
    for arm, record in sample["arms"].items():
        assert record["status"] == "ENGINEERING_SMOKE_ONLY"
        assert record["steps"] == 6
        assert len(record["model_state_sha256"]) == 64
        assert tuple(record["metrics_synthetic_only"]) == (
            "final_average_accuracy",
            "late_task_within_task_mean_accuracy",
            "average_forgetting",
            "backward_transfer",
        )
        assert all(math.isfinite(v) for v in record["metrics_synthetic_only"].values())
    sparse = sample["arms"]["fixed_sparse_mlp"]
    assert len(sparse["mask_sha256"]) == 64
    assert sparse["body_active_weights"] < sparse["body_allocated_weights"]
    assert sample["arms"]["progressive_reserve"]["mask_sha256"] is None
    assert sample["arms"]["all_recruited"]["mask_sha256"] is None


def test_pilot_is_reproducible_with_pinned_toy_stream_and_seed(sample):
    second = run_m13b_synthetic_engineering_pilot()
    assert pilot_canonical_json(sample) == pilot_canonical_json(second)
    assert sha256(pilot_canonical_json(sample)).hexdigest() == sha256(
        pilot_canonical_json(second)
    ).hexdigest()


def _cli(*args):
    return subprocess.run(
        [sys.executable, str(CLI), *args], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )


def test_cli_executes_only_explicit_toy_and_saves_exclusive_evidence(tmp_path):
    target = tmp_path / "new-toy-evidence"
    outcome = _cli(*FLAGS, "--output-dir", str(target))
    assert outcome.returncode == 0, outcome.stderr
    result = json.loads(outcome.stdout)
    payload = (target / "engineering-pilot.json").read_text()
    assert json.loads(payload)["arms"] == result["arms"]
    pin = (target / "engineering-pilot.sha256").read_text().split()[0]
    assert pin == sha256(pilot_canonical_json({
        k: v for k, v in result.items() if k != "sha256_canonical_json"
    })).hexdigest()
    assert pin == result["sha256_canonical_json"]
    assert len(list(target.iterdir())) == 2

    # Must fail before re-running if destination already exists.
    again = _cli(*FLAGS, "--output-dir", str(target))
    assert again.returncode != 0
    assert (target / "engineering-pilot.json").read_text() == payload


@pytest.mark.parametrize("invalid", [
    (),
    ("--synthetic-engineering-pilot",),
    ("--acknowledge-engineering-only",),
    (*FLAGS, "--device", "cuda"),
    (*FLAGS, "--features", "data/features/cifar100_resnet18_imagenet1k_v1.pt"),
    (*FLAGS, "--seed", "501"),
    (*FLAGS, "--execute"),
])
def test_cli_does_not_accept_scientific_or_gpu_arguments(invalid):
    failure = _cli(*invalid)
    assert failure.returncode != 0
    assert "error:" in failure.stderr
