"""Exact CPU-only mathematical D3 bounded-mean e-value stress and safeguards.

Tests do not access scientific outcomes, models, protected CIFAR features or GPU.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from math import log, log1p
import json
from pathlib import Path
import subprocess
import sys

import pytest

from synaploom.research.m13b_manifest import canonical_json
from synaploom.research.m13b_d3_bounded_mean_betting import (
    ALPHA, ALT_N_GRID, CONSTANT_GAIN_GRID, KIND, NULL_N_GRID, NULL_P_GRID,
    THRESHOLD, _exact_betting_threshold, _exact_binomial_upper_tail,
    _smallest_constant_gain_n, _two_point_distribution,
    build_bounded_mean_evalue_review,
    fixed_betting_log_evalue, joint_both_needed_rejects,
    one_sided_betting_rejects,
)


ROOT = Path(__file__).resolve().parents[2]


def test_fixed_bet_math_and_threshold_are_explicit():
    assert ALPHA == Fraction(1, 20)
    assert THRESHOLD == 20
    assert fixed_betting_log_evalue([0.01] * 10) == pytest.approx(
        10 * log1p(0.01)
    )
    assert not one_sided_betting_rejects(fixed_betting_log_evalue([.01] * 10))
    assert not one_sided_betting_rejects(fixed_betting_log_evalue([.05] * 61))
    assert one_sided_betting_rejects(fixed_betting_log_evalue([.05] * 62))
    assert fixed_betting_log_evalue([0.0, -1.0, .3]) == float("-inf")
    assert not one_sided_betting_rejects(float("-inf"))
    assert fixed_betting_log_evalue([0.0, 0.0]) == 0.0


def test_smallest_constant_gain_sample_counts_are_exact():
    expected = {
        Fraction(1, 100): 302,
        Fraction(2, 100): 152,
        Fraction(5, 100): 62,
        Fraction(10, 100): 32,
    }
    assert set(expected) == set(CONSTANT_GAIN_GRID)
    for delta, n in expected.items():
        assert _smallest_constant_gain_n(delta) == n
        assert (1 + delta)**(n-1) < THRESHOLD
        assert (1 + delta)**n >= THRESHOLD


def test_joint_iut_keeps_both_mean_endpoints_and_does_not_split_alpha():
    strong = [.05] * 62
    neutral = [0.0] * 62
    assert joint_both_needed_rejects(strong, strong)
    assert not joint_both_needed_rejects(strong, neutral)
    assert not joint_both_needed_rejects(neutral, strong)
    with pytest.raises(ValueError):
        joint_both_needed_rejects(strong, strong[:-1])
    with pytest.raises(ValueError):
        joint_both_needed_rejects(strong, neutral + [0.0])
    # Validate BOTH inputs even if first endpoint cannot reject.
    with pytest.raises(ValueError):
        joint_both_needed_rejects(neutral, [float("nan")] * 62)


@pytest.mark.parametrize("deltas", [
    (), [], [True], [float("nan")], [float("inf")],
    [-1.01], [1.01], ["0.05"], 0.05, [0.1, -1, float("nan")],
])
def test_reject_invalid_or_incomplete_paired_effects(deltas):
    with pytest.raises(ValueError):
        fixed_betting_log_evalue(deltas)


@pytest.mark.parametrize("value", [
    True, None, float("nan"), float("inf"), "-inf", "1.0",
])
def test_reject_invalid_log_evalue(value):
    with pytest.raises(ValueError):
        one_sided_betting_rejects(value)


def test_exact_two_point_null_is_bounded_and_centered():
    for p in NULL_P_GRID:
        a = Fraction(1, 100)
        b = -a * p / (1-p)
        assert _two_point_distribution(p, a, b) == 0
        assert -1 <= b < 0 < a <= 1
        for n in NULL_N_GRID:
            k = _exact_betting_threshold(n, a, b)
            probability = _exact_binomial_upper_tail(n, k, p)
            assert 0 <= probability <= ALPHA
            assert k <= n+1
            if k <= n:
                assert (1+a)**k * (1+b)**(n-k) >= THRESHOLD
                if k > 0:
                    assert (1+a)**(k-1) * (1+b)**(n-k+1) < THRESHOLD


def test_skewed_null_all_ten_positive_would_fool_bootstrap_but_not_e_test():
    p = Fraction(9, 10)
    positive = Fraction(1, 100)
    negative = Fraction(-9, 100)
    for n in (10, 55, 100):
        assert _exact_betting_threshold(n, positive, negative) == n+1
        assert _exact_binomial_upper_tail(n, n+1, p) == 0
    # At larger n, this test can reject very rare null events without
    # breaking its global <=5% bound.
    k = _exact_betting_threshold(320, positive, negative)
    assert k == 319
    assert float(_exact_binomial_upper_tail(320, k, p)) == pytest.approx(
        8.328300925463859e-14, rel=1e-9, abs=1e-17
    )


def test_exact_single_endpoint_toy_power_not_incorrectly_called_joint():
    p, a, b = Fraction(9, 10), Fraction(6, 100), Fraction(-10, 100)
    assert _two_point_distribution(p, a, b) == Fraction(44, 1000)
    assert _exact_betting_threshold(55, a, b) == 54
    assert float(_exact_binomial_upper_tail(55, 54, p)) == pytest.approx(
        0.021640908246545472, rel=1e-10
    )
    assert _exact_betting_threshold(100, a, b) == 83
    assert float(_exact_binomial_upper_tail(100, 83, p)) == pytest.approx(
        0.989992720737875, rel=1e-10
    )


def test_report_provenance_and_decision_boundary():
    r = build_bounded_mean_evalue_review()
    assert r["kind"] == KIND
    assert r["status"] == "D3_CALIBRATED_BOUNDED_MEAN_TEST_CANDIDATE__NOT_SELECTED"
    assert r["science_executable"] is False
    assert r["training_or_gpu_accessed"] is False
    assert r["real_outcomes_loaded"] is False
    assert r["actual_optimizer_steps"] == 0
    assert r["scientific_claim"] == "NONE"
    assert r["owner_mean_estimand_preserved"] is True
    assert r["method_chosen_for_actual_science"] is None
    assert r["final_seed_count"] is None
    assert r["minimum_mean_effect_owner_approved"] is False
    assert r["joint_power_established"] is False
    assert r["ready_for_science"] is False
    assert r["formal_negative_claim"] == "NOT_DEFINED__DESCRIPTIVE_ONLY"
    assert r["test_family"] == "FIXED_LAMBDA_ONE_BOUNDED_MEAN_EVALUE"
    assert r["primary_order"] == "O0"
    assert len(r["exact_zero_mean_null_stress"]) == len(NULL_P_GRID) * len(NULL_N_GRID)
    assert len(r["all_identical_positive_gain_thresholds"]) == len(CONSTANT_GAIN_GRID)
    assert len(r["exact_positive_mean_toy_power"]) == 2 * len(ALT_N_GRID)
    for cell in r["exact_zero_mean_null_stress"]:
        assert cell["population_mean_exactly_zero"]
        assert cell["exact_false_positive_probability"] <= .05
    assert r["record_sha256"] == sha256(canonical_json({
        k: v for k, v in r.items() if k != "record_sha256"
    })).hexdigest()
    assert build_bounded_mean_evalue_review() == r


@pytest.mark.parametrize("args", [
    (10, Fraction(-1, 100), Fraction(1, 100)),
    (10, Fraction(1, 100), Fraction(1, 100)),
    (1, Fraction(1, 100), Fraction(-1, 100)),
    (321, Fraction(1, 100), Fraction(-1, 100)),
    (True, Fraction(1, 100), Fraction(-1, 100)),
])
def test_invalid_exact_binary_threshold_inputs(args):
    with pytest.raises(ValueError):
        _exact_betting_threshold(*args)


@pytest.mark.parametrize("args", [
    (1, 0, Fraction(1, 2)), (10, -1, Fraction(1, 2)),
    (10, 12, Fraction(1, 2)), (10, 5, .5),
])
def test_invalid_binomial_parameters(args):
    with pytest.raises(ValueError):
        _exact_binomial_upper_tail(*args)


def test_review_cli_is_synthetic_only_and_rejects_execution_flags():
    cmd = [sys.executable, "-m",
           "synaploom.research.m13b_d3_bounded_mean_betting"]
    good = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout) == build_bounded_mean_evalue_review()
    for extra in (["--train"], ["--method", "betting"],
                  ["--seed", "501"], ["--results", "real.json"],
                  ["--gpu"], ["--approve"]):
        bad = subprocess.run(cmd + extra, cwd=ROOT, capture_output=True, text=True)
        assert bad.returncode != 0
