"""Analytical M1.3b work-accounting tests. No data files, CUDA or training."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from synaploom.research.m13b_stagewise_budget import (
    AUDIT_KIND, _batch_shapes, _fixed_macs, _syn_macs,
    build_m13b_stagewise_audit,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "scripts" / "m13b_stagewise_budget.py"


@pytest.fixture(scope="module")
def audit():
    return build_m13b_stagewise_audit()


def test_frozen_protocol_step_counts_and_exact_tail_batch(audit):
    assert audit["kind"] == AUDIT_KIND
    assert audit["source_protocol"] == "EXP-H1A-002"
    assert audit["science_executable"] is False
    assert audit["scientific_protocol_frozen"] is False
    assert audit["owner_decision_pending"] is True
    assert audit["new_scientific_experiments_run"] is False
    assert audit["feature_file_read"] is False
    assert audit["cuda_used"] is False
    assert audit["actual_gpu_compute_observed"] is None
    assert audit["actual_gpu_memory_observed"] is None
    assert audit["total_tasks"] == 10
    assert audit["total_optimizer_steps_per_arm"] == 8000
    assert audit["total_optimizer_steps_for_3_arms_per_order"] == 24000
    assert audit["total_training_examples_presented_per_arm"] == 1_000_000
    assert _batch_shapes(5000, 128) == (128,) * 39 + (8,)
    assert len(audit["stage_rows"]) == 10
    for i, row in enumerate(audit["stage_rows"]):
        assert row["task_index"] == i
        assert row["samples_per_epoch"] == 5000
        assert row["epochs"] == 20
        assert row["batches_per_epoch"] == 40
        assert row["last_batch_size"] == 8
        assert row["optimizer_steps"] == 800


def test_real_recruitment_delayed_one_optimizer_step_per_task(audit):
    rows = audit["stage_rows"]
    assert [r["progressive_recruited_first_batch"] for r in rows] == [
        4, 4, 5, 6, 7, 8, 9, 10, 11, 12
    ]
    assert [r["progressive_recruited_after_first_step"] for r in rows] == list(
        range(4, 14)
    )
    assert [r["recruitment_after_first_step"] for r in rows] == [
        False
    ] + [True] * 9


def test_all_mac_arms_and_tail_count_with_independent_formula(audit):
    totals = {k: 0 for k in audit["all_task_linear_mac_estimates"]}
    model_shared = {"input_dim": 512, "d": 128, "h": 256, "routing": 128, "k": 4, "classes": 100}
    for row in audit["stage_rows"]:
        t = row["task_index"]
        expected = _syn_macs(
            128, 4 + max(0, t - 1), **model_shared
        ) if t > 0 else _syn_macs(128, 4, **model_shared)
        # All other steps: in each of 20 epochs there are 39 full 128
        # batches and one 8-example tail; task >0 uses the old R only once.
        after = 4 + t
        final_batch_mac = (
            39 * _syn_macs(128, after, **model_shared)
            + _syn_macs(8, after, **model_shared)
        )
        progressive = 20 * final_batch_mac
        if t > 0:
            progressive += expected - _syn_macs(128, after, **model_shared)
        macs = row["linear_mac_estimates"]
        assert macs["progressive_reserve"] == progressive
        assert macs["all_recruited"] == 20 * (
            39 * _syn_macs(128, 16, **model_shared)
            + _syn_macs(8, 16, **model_shared)
        )
        for key, val in macs.items():
            assert val > 0
            totals[key] += val
        assert row["ideal_progressive_vs_fixed_relative_error"] <= 0.10
    assert totals == audit["all_task_linear_mac_estimates"]


def test_equal_physical_capacities_but_not_dense_masked_work(audit):
    assert audit["reference_total_trainable"] == 1172196
    assert audit["fixed_sparse_total_trainable"] == 1172226
    assert audit["physical_trainable_relative_error"] < 0.02
    assert audit["per_task_ideal_relative_error_at_most_10pct"]
    assert audit["whole_run_ideal_progressive_vs_fixed_relative_error"] < 0.10
    sums = audit["all_task_linear_mac_estimates"]
    assert sums["fixed_sparse_dense_kernel_linear_proxy"] > sums["fixed_sparse_ideal"]
    assert sums["progressive_reserve"] <= sums["all_recruited"]
    assert audit["fixed_sparse_active_body_weights"] < audit["fixed_sparse_allocated_body_weights"]
    assert audit["fixed_sparse_body_density"] < 1
    assert audit["resource_units"] == "LINEAR_MAC_ESTIMATE_NOT_GPU_FLOPS"
    assert len(audit["fixed_sparse_mask_sha256"]) == 64


def test_canonical_digest_and_no_cpu_rng_mutation(audit):
    before = torch.get_rng_state().clone()
    next_report = build_m13b_stagewise_audit()
    assert torch.equal(before, torch.get_rng_state())
    assert audit == next_report
    record = {k: v for k, v in audit.items() if k != "record_sha256"}
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("ascii")
    assert sha256(canonical).hexdigest() == audit["record_sha256"]


def _cli(*args: str):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT,
                          capture_output=True, text=True, check=False)


def test_cli_analytical_only_and_exclusive_evidence(tmp_path):
    dest = tmp_path / "stagewise.json"
    executed = _cli("--audit", "--output", str(dest))
    assert executed.returncode == 0, executed.stderr
    report = json.loads(executed.stdout)
    assert len(report["record_sha256"]) == 64
    assert json.loads(dest.read_text()) == report
    assert report["science_executable"] is False
    again = _cli("--audit", "--output", str(dest))
    assert again.returncode != 0
    assert json.loads(dest.read_text()) == report


@pytest.mark.parametrize("bad", [
    (), ("--execute",), ("--audit", "--cuda"), ("--audit", "--seed", "501"),
    ("--audit", "--features", "data/features/cifar100_resnet18_imagenet1k_v1.pt"),
])
def test_cli_rejects_scientific_execution_or_data_paths(bad):
    x = _cli(*bad)
    assert x.returncode != 0
    assert "error:" in x.stderr
