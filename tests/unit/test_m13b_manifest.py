"""M1.3b preview manifest/preflight gates: never launch scientific training."""
from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from synaploom.protocols.h1a_order_sensitivity import EXPECTED_ORDER_FINGERPRINTS
from synaploom.protocols.h1a2 import H1A2_FEATURE_SHA256
from synaploom.research.m13b_manifest import (
    AUTHORIZATION,
    build_m13b_draft_manifest,
    canonical_json,
    verify_manifest_integrity,
    verify_trusted_feature_file,
)


SOURCE = "a" * 40
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "scripts" / "m13b_external_preflight.py"


@pytest.fixture(scope="module")
def proposed():
    return build_m13b_draft_manifest(source_commit=SOURCE)


def test_draft_plan_has_explicit_no_go_and_competing_order_matrices(proposed):
    verify_manifest_integrity(proposed)
    assert proposed["study_id"] == "EXP-M13B-EXTERNAL-001"
    assert proposed["authorization"] == AUTHORIZATION
    assert proposed["status"] == "DRAFT_DESIGN_PREVIEW"
    assert proposed["science_executable"] is False
    assert proposed["run_status"] == "NO_RUNS_EXECUTED"
    assert proposed["dataset"]["feature_validation"] == "NOT_CHECKED"
    assert proposed["dataset"]["feature_sha256_expected"] == H1A2_FEATURE_SHA256
    assert proposed["dataset"]["feature_sha256_verified"] is None
    assert proposed["class_order_fingerprints"] == EXPECTED_ORDER_FINGERPRINTS
    assert proposed["source_commit"] == SOURCE
    assert proposed["source_revision_verified"] is False
    assert len(proposed["unresolved_owner_decisions"]) >= 5

    single = proposed["candidate_matrices"]["O0_ONLY"]
    triple = proposed["candidate_matrices"]["O0_O1_O2"]
    assert single["orders"] == ["O0"]
    assert triple["orders"] == ["O0", "O1", "O2"]
    assert single["seed_ids"] == list(range(501, 511))
    assert triple["seed_ids"] == list(range(501, 511))
    assert single["arms"] == ["progressive_reserve", "all_recruited", "fixed_sparse_mlp"]
    assert single["expected_trajectories"] == 30
    assert triple["expected_trajectories"] == 90
    assert single["paired_blocks"] == 10
    assert triple["paired_blocks"] == 30
    for matrix in (single, triple):
        keys = matrix["trajectory_keys"]
        assert len(keys) == matrix["expected_trajectories"] == len(set(keys))
    assert not proposed["origin"]["training_sample_routing_allowed"]


def test_budget_does_not_claim_hardware_sparse_work_or_science_admission(proposed):
    budget = proposed["fixed_sparse_candidate"]
    assert budget["matched_total_2pct"]
    assert budget["matched_ideal_macs_10pct"]
    assert budget["fidelity_status"] == "CANDIDATE_NOT_SCIENCE_APPROVED"
    assert budget["total_parameter_relative_error"] <= 0.02
    assert budget["ideal_mac_relative_error"] <= 0.10
    assert budget["implementation_dense_linear_macs_per_batch"] > budget["ideal_fixed_sparse_linear_macs_per_batch"]
    assert proposed["budget_thresholds_proposed"]["dense_masked_execution_is_real_sparse_compute"] is False
    assert budget["observed_wall_seconds"] is None
    assert budget["observed_cuda_peak_bytes"] is None
    assert budget["optimizer_state_bytes"] is None


def test_digest_covers_content_and_rejects_authorized_tampering(proposed):
    base = copy.deepcopy(proposed)
    assert len(base["record_sha256"]) == 64
    assert base["record_sha256"] == sha256(
        canonical_json({k: v for k, v in base.items() if k != "record_sha256"})
    ).hexdigest()
    base["candidate_matrices"]["O0_ONLY"]["expected_trajectories"] = 5
    with pytest.raises(ValueError, match="SHA-256"):
        verify_manifest_integrity(base)
    base = copy.deepcopy(proposed)
    base["science_executable"] = True
    base["record_sha256"] = sha256(
        canonical_json({k: v for k, v in base.items() if k != "record_sha256"})
    ).hexdigest()
    with pytest.raises(ValueError, match="draft, nonexecutable"):
        verify_manifest_integrity(base)
    base = copy.deepcopy(proposed)
    base["authorization"] = "GO"
    base["record_sha256"] = sha256(
        canonical_json({k: v for k, v in base.items() if k != "record_sha256"})
    ).hexdigest()
    with pytest.raises(ValueError, match="not authorized"):
        verify_manifest_integrity(base)


def test_manifest_generation_deterministic_and_cpu_rng_state_preserved(proposed):
    old = torch.get_rng_state().clone()
    other = build_m13b_draft_manifest(source_commit=SOURCE)
    assert torch.equal(old, torch.get_rng_state())
    assert proposed == other
    with pytest.raises(ValueError, match="source_commit"):
        build_m13b_draft_manifest(source_commit="not-a-commit")


def test_untrusted_feature_file_refused_before_read(tmp_path, monkeypatch):
    sample = tmp_path / "fake.pt"
    sample.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="trusted feature"):
        verify_trusted_feature_file(sample, trusted_source=False)
    called = [False]

    def unsafe_load(*args, **kwargs):
        called[0] = True
        raise AssertionError("pickle loading attempted")
    monkeypatch.setattr(torch, "load", unsafe_load)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_trusted_feature_file(sample, trusted_source=True)
    assert not called[0]


def _cli(*args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )


def test_cli_dry_run_prints_preview_without_training_or_data_access(tmp_path):
    out = _cli("--dry-run", "--plan-output", str(tmp_path / "draft.json"))
    assert out.returncode == 0, out.stderr
    preview = json.loads(out.stdout)
    verify_manifest_integrity(preview)
    assert preview["dataset"]["feature_validation"] == "NOT_CHECKED"
    assert preview["run_status"] == "NO_RUNS_EXECUTED"
    written = json.loads((tmp_path / "draft.json").read_text())
    assert written == preview

    no_overwrite = _cli("--dry-run", "--plan-output", str(tmp_path / "draft.json"))
    assert no_overwrite.returncode != 0
    assert json.loads((tmp_path / "draft.json").read_text()) == preview


def test_cli_hard_rejects_unapproved_execution_and_bad_option_pairs(tmp_path):
    for args in (
        ("--execute",),
        ("--verify-features",),
        ("--verify-features", "--features", str(tmp_path / "missing.pt")),
        ("--dry-run", "--features", str(tmp_path / "missing.pt")),
        ("--dry-run", "--trusted-source"),
        ("--dry-run", "--verify-features"),
    ):
        result = _cli(*args)
        assert result.returncode != 0, args
    assert not list(tmp_path.iterdir())
