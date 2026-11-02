"""D3 exact mathematical null stress: CPU-only and no science outcome ingest."""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import subprocess
import sys

import pytest

from synaploom.research.m13b_manifest import canonical_json
from synaploom.research.m13b_d3_exact_t_null_stress import (
    ALPHA, KIND, N_GRID, P_GRID, POSITIVE_EFFECT, T_CRITICAL_95,
    binomial_upper_tail, build_exact_t_stress_report,
    exact_one_sided_t_null_rejection,
    student_t_from_binary_count, two_point_zero_mean,
)


ROOT = Path(__file__).resolve().parents[2]


def test_toy_null_population_is_exactly_zero_and_bounded():
    for p in P_GRID:
        pos, neg = two_point_zero_mean(p)
        assert pos == Fraction(1, 100)
        assert -1 <= neg < 0
        assert p * pos + (1-p) * neg == 0
    assert two_point_zero_mean(Fraction(9, 10))[1] == Fraction(-9, 100)
    assert two_point_zero_mean(Fraction(19, 20))[1] == Fraction(-19, 100)


def test_student_t_statistic_for_two_point_sample_matches_manual_formula():
    positive = Fraction(1, 100)
    negative = Fraction(-1, 100)
    assert student_t_from_binary_count(10, 10, positive, negative) == float("inf")
    assert student_t_from_binary_count(10, 0, positive, negative) == float("-inf")
    assert student_t_from_binary_count(10, 5, positive, negative) == pytest.approx(0)
    # Mean is +.006 when eight of ten observations are +.01.
    # Unbiased sample variance is determined by the 8:2 two-point count.
    import math
    result = student_t_from_binary_count(10, 8, positive, negative)
    assert result == pytest.approx(
        (8*.01 + 2*(-.01)) * math.sqrt(9) / (.02 * math.sqrt(16))
    )
    assert result > T_CRITICAL_95[10]


def test_frozen_one_sided_t_critical_grid_and_normal_limit():
    assert tuple(T_CRITICAL_95) == N_GRID
    assert T_CRITICAL_95[10] == pytest.approx(1.833112932656237, abs=1e-12)
    assert T_CRITICAL_95[55] == pytest.approx(1.6735649063521605, abs=1e-12)
    assert all(
        T_CRITICAL_95[N_GRID[i]] > T_CRITICAL_95[N_GRID[i+1]]
        for i in range(len(N_GRID)-1)
    )


def test_exact_binomial_tail_checks_and_ideal_t_error_rates():
    for p in P_GRID:
        for n in N_GRID:
            assert binomial_upper_tail(n, 0, p) == 1
            assert binomial_upper_tail(n, n, p) == p**n
            assert binomial_upper_tail(n, n+1, p) == 0
            rate, first = exact_one_sided_t_null_rejection(n, p)
            assert 0 <= rate <= 1
            assert rate == binomial_upper_tail(n, first, p)
            assert rate >= p**n


@pytest.mark.parametrize("p,n,k,rate", [
    (Fraction(1, 2), 10, 8, 0.0546875),
    (Fraction(4, 5), 10, 10, 0.1073741824),
    (Fraction(9, 10), 10, 10, 0.3486784401),
    (Fraction(9, 10), 40, 39, 0.08047369601447729),
    (Fraction(9, 10), 55, 53, 0.07743387481967053),
    (Fraction(9, 10), 100, 94, 0.11715561543588425),
    (Fraction(19, 20), 55, 54, 0.23188700409521984),
])
def test_frozen_exact_student_t_null_thresholds(p,n,k,rate):
    actual, observed_k = exact_one_sided_t_null_rejection(n, p)
    assert observed_k == k
    assert float(actual) == pytest.approx(rate, abs=1e-11)


def test_report_is_hashed_consistent_and_has_no_method_selection():
    report = build_exact_t_stress_report()
    assert report["kind"] == KIND
    assert report["status"] == "CPU_EXACT_NULL_STRESS_ONLY__D3_METHOD_NOT_SELECTED"
    assert report["science_executable"] is False
    assert report["model_or_features_loaded"] is False
    assert report["real_outcomes_loaded"] is False
    assert report["optimizer_steps_executed"] == 0
    assert report["scientific_claim"] == "NONE"
    assert report["owner_mean_estimand_preserved"] is True
    assert report["approved_paired_seed_count"] is None
    assert report["actual_type_i_for_synaploom"] is None
    assert report["selection_outcome"].startswith("NONE")
    assert report["scenarios"] and len(report["scenarios"]) == 20
    assert report["record_sha256"] == sha256(canonical_json({
        key: val for key,val in report.items() if key != "record_sha256"
    })).hexdigest()
    assert report == build_exact_t_stress_report()
    n10_skew = next(s for s in report["scenarios"] if (
        s["seed_count"] == 10 and s["one_seed_positive_probability"] == .9
    ))
    assert n10_skew["true_mean_difference_fraction"] == 0
    assert n10_skew["null_rejection_probability"] == pytest.approx(.9**10)
    assert n10_skew["exceeds_nominal_005"] is True
    assert n10_skew["both_endpoint_joint_false_favorable_if_identical_deltas"] == (
        n10_skew["null_rejection_probability"]
    )


@pytest.mark.parametrize("p", [
    float(.9), 0, Fraction(0), Fraction(1),
    Fraction(-1, 2), Fraction(3, 2),
])
def test_invalid_null_probability_is_rejected(p):
    with pytest.raises(ValueError):
        two_point_zero_mean(p)


@pytest.mark.parametrize("args", [
    (10, -1), (10, 11), (1, 0), (10, True), (True, 1), ("10", 1),
])
def test_invalid_student_t_inputs_rejected(args):
    pos, neg = two_point_zero_mean(Fraction(9, 10))
    with pytest.raises(ValueError):
        student_t_from_binary_count(*args, pos, neg)


@pytest.mark.parametrize("n,k,p", [
    (10,-1,Fraction(1,2)), (10,12,Fraction(1,2)), (10,1,.5),
    (1,0,Fraction(1,2)), (10,1,Fraction(-1,2)), (10,1,True),
])
def test_invalid_exact_binomial_inputs_rejected(n,k,p):
    with pytest.raises(ValueError):
        binomial_upper_tail(n,k,p)


def test_safe_module_cli_only_emits_fixed_synthetic_review():
    cmd = [sys.executable, "-m",
           "synaploom.research.m13b_d3_exact_t_null_stress"]
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == build_exact_t_stress_report()
    for extra in (["--science"], ["--seed", "501"], ["--results", "real.json"]):
        denied = subprocess.run(cmd+extra, cwd=ROOT, capture_output=True, text=True)
        assert denied.returncode != 0
