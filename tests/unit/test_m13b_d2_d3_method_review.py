"""CPU-only D2 baseline anatomy and exact D3 statistical review.

Neither suite loads any CIFAR feature bundle, scientific result or CUDA.
Passing engineering checks does NOT constitute independent scientific review.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from synaploom.protocols.h1a2 import h1a2_protocol
from synaploom.research.fixed_sparse_baseline import (
    FixedMaskedLinear, build_h1a_fixed_sparse_feasibility,
)
from synaploom.research.m13b_d3_exact_null_review import (
    CONFIGURATIONS, KIND, LOWER_TAIL, N, _exact_ideal_bootstrap_favorable_probability,
    _iid_positive_count_probability, _resampled_nonpositive_probability,
    build_exact_null_review,
)
from synaploom.research.m13b_manifest import canonical_json


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "scripts" / "m13b_d3_exact_null_review.py"


def test_zero_means_are_exact_and_bounded():
    for case in CONFIGURATIONS.values():
        p, positive, negative = (
            case["p_positive"], case["positive_effect"], case["negative_effect"]
        )
        assert type(p) is Fraction
        assert p * positive + (1 - p) * negative == Fraction(0)
        assert -1 <= negative < 0 < positive <= 1


def test_one_thousand_toy_probabilities_are_not_simulated():
    result = build_exact_null_review()
    assert result["kind"] == KIND
    assert result["science_executable"] is False
    assert result["real_scientific_data_loaded"] is False
    assert result["actual_optimizer_steps"] == 0
    assert result["scientific_claim"] == "NONE"
    assert result["finite_10000_draw_result_claimed"] is False
    assert result["bootstrap_resampling"] == "EXACT_INFINITE_RESAMPLE_LIMIT"
    assert result["paired_seed_count"] == N == 10
    assert "INDEPENDENT_REVIEW_PENDING" in result["status"]
    assert sha256(canonical_json({
        k: v for k, v in result.items() if k != "record_sha256"
    })).hexdigest() == result["record_sha256"]
    assert result == build_exact_null_review()


def test_skewed_bounded_joint_null_exceeds_nominal_by_constructed_event():
    result = build_exact_null_review()["synthetic_null_scenarios"]["skewed_zero_mean"]
    assert result["true_expected_delta_fraction"] == 0.0
    assert result["correlation_both_endpoints"] == 1.0
    assert result["all_positive_sample_probability_exact"] == "3486784401/10000000000"
    assert result["all_positive_sample_probability"] == pytest.approx(0.3486784401)
    assert result["all_positive_sample_bootstrap_ci"] == [0.01, 0.01]
    # BOTH CO-PRIMARY marginal CI lower bounds strictly positive, with exact
    # probability at least 34.86% despite each endpoint's true mean zero.
    assert result["all_positive_sample_probability"] > 0.05
    assert result["ideal_infinite_bootstrap_favorable_positive_counts"] == [10]
    assert result["ideal_infinite_bootstrap_favorable_probability_exact"] == (
        "3486784401/10000000000"
    )


def test_symmetric_binary_null_has_different_exact_ideal_bootstrap_size():
    result = build_exact_null_review()["synthetic_null_scenarios"]["symmetric_zero_mean"]
    assert result["true_expected_delta_fraction"] == 0.0
    assert result["ideal_infinite_bootstrap_favorable_positive_counts"] == [9, 10]
    assert result["ideal_infinite_bootstrap_favorable_probability"] == pytest.approx(
        11 / 1024
    )


@pytest.mark.parametrize("k", list(range(11)))
def test_exact_bootstrap_nonpositive_mass_is_probabilistic(k):
    skew = CONFIGURATIONS["skewed_zero_mean"]
    mass = _resampled_nonpositive_probability(
        k, skew["positive_effect"], skew["negative_effect"]
    )
    assert 0 <= mass <= 1
    if k == 10:
        assert mass == 0
    if k == 0:
        assert mass == 1


def test_binomial_exact_probability_normalization():
    for spec in CONFIGURATIONS.values():
        p = spec["p_positive"]
        assert sum(
            (_iid_positive_count_probability(k, p) for k in range(N + 1)),
            Fraction(0),
        ) == 1
    skew = CONFIGURATIONS["skewed_zero_mean"]
    probability, positive_k = _exact_ideal_bootstrap_favorable_probability(
        skew["p_positive"], skew["positive_effect"], skew["negative_effect"]
    )
    assert positive_k == [10]
    assert probability == Fraction(9, 10)**10
    assert LOWER_TAIL == Fraction(1, 40)


@pytest.mark.parametrize("k,p", [
    (-1, Fraction(1, 2)), (11, Fraction(1, 2)),
    (True, Fraction(1, 2)), (1, Fraction(-1, 2)),
    (1, Fraction(3, 2)), (1, 0.5),
])
def test_invalid_exact_distribution_inputs_fail_closed(k, p):
    with pytest.raises(ValueError):
        _iid_positive_count_probability(k, p)


@pytest.mark.parametrize("args", [
    (), ("--execute",), ("--results", "scientific.json"),
    ("--approve",), ("--freeze",), ("--cuda",),
    ("--seed", "501"), ("--n", "20"), ("--science",),
])
def test_cli_rejects_real_scientific_capabilities(args):
    p = subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=ROOT,
        capture_output=True, text=True, check=False
    )
    assert p.returncode != 0
    assert "error:" in p.stderr


def test_cli_exact_review_and_no_overwrite(tmp_path):
    out = tmp_path / "review.json"
    command = [
        sys.executable, str(SCRIPT), "--exact-synthetic-null-review",
        "--output", str(out),
    ]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout) == build_exact_null_review()
    assert json.loads(out.read_text()) == build_exact_null_review()
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert second.returncode != 0
    assert json.loads(out.read_text()) == build_exact_null_review()


def test_d2_fixed_sparse_architecture_identity_is_still_a_candidate():
    with torch.random.fork_rng(devices=[]):
        protocol = h1a2_protocol()
        model, info = build_h1a_fixed_sparse_feasibility(
            protocol, seed=13, mask_seed=20261026
        )
    model.assert_masks()
    assert info.matched_total_2pct is True
    assert info.matched_ideal_macs_10pct is True
    assert info.total_allocated_trainable == 1_172_226
    assert info.target_synaploom_total_trainable == 1_172_196
    assert info.implementation_dense_linear_macs_per_batch > (
        info.ideal_fixed_sparse_linear_macs_per_batch
    )
    assert info.body_allocated_weights > info.body_active_weights
    assert 0 < info.body_density < 1
    assert isinstance(model.first, FixedMaskedLinear)
    assert isinstance(model.second, FixedMaskedLinear)
    assert all(layer.mask.dtype == torch.bool for layer in model.masked_layers())
    # Source-shape regression only, not independent benchmarking:
    # even when zeros are masked, torch F.linear uses dense matrix operations.
    source = inspect.getsource(FixedMaskedLinear.forward)
    assert "F.linear(" in source
    assert "self.weight * self.mask" in source
    assert "sparse.mm" not in source and "sparse.addmm" not in source
    assert not hasattr(model, "graph_state")
    assert info.optimizer_state_bytes is None
    assert info.observed_cuda_peak_bytes is None
    assert info.observed_wall_seconds is None
