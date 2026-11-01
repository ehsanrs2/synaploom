"""No-outcome D3 estimand review: exact binomial and bounded-mean limits.

The mathematical options are NOT accepted scientific methods.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
from math import comb, log, sqrt
from pathlib import Path
import subprocess
import sys

import pytest

from synaploom.research.m13b_manifest import canonical_json
from synaploom.research.m13b_d3_estimand_options import (
    ALPHA, KIND, N, POWER_POSITIVE_PROBABILITIES,
    build_estimand_options_review, hoeffding_mean_margin,
    sign_power, sign_upper_tail,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "scripts" / "m13b_d3_estimand_options.py"


def test_exact_sign_threshold_for_n10_with_fixed_denominator():
    assert N == 10
    assert sign_upper_tail(10) == Fraction(1, 1024)
    assert sign_upper_tail(9) == Fraction(11, 1024)
    assert sign_upper_tail(8) == Fraction(56, 1024)
    assert sign_upper_tail(0) == 1
    assert sign_upper_tail(9) <= ALPHA < sign_upper_tail(8)
    assert all(
        sign_upper_tail(k) > ALPHA for k in range(9)
    )
    assert all(
        sign_upper_tail(k) <= ALPHA for k in (9, 10)
    )


def test_analytic_exact_sign_power_not_confused_with_mean():
    for p in POWER_POSITIVE_PROBABILITIES:
        exact = sum(
            (Fraction(comb(10, k)) * p**k * (1-p)**(10-k)
             for k in range(9, 11)),
            Fraction(0),
        )
        assert sign_power(9, p) == exact
    assert sign_power(9, Fraction(1, 2)) == Fraction(11, 1024)
    assert sign_power(9, Fraction(9, 10)) == Fraction(7360989291, 10000000000)
    assert sign_power(9, Fraction(0)) == 0
    assert sign_power(9, Fraction(1)) == 1


def test_hoeffding_n10_one_sided_95_and_two_direction():
    # For bounded differences in [-1,+1], range width = 2.
    # Pr(Xbar-E[Xbar]>=t) <= exp(-n*t**2/2).
    assert hoeffding_mean_margin() == pytest.approx(
        sqrt(2 * log(20) / 10)
    )
    assert hoeffding_mean_margin(alpha=Fraction(1, 40)) == pytest.approx(
        sqrt(2 * log(40) / 10)
    )
    assert .77 < hoeffding_mean_margin() < .78
    assert .85 < hoeffding_mean_margin(alpha=Fraction(1, 40)) < .86
    assert hoeffding_mean_margin(n=40) < hoeffding_mean_margin(n=10)


@pytest.mark.parametrize("k,n", [
    (-1, 10), (11, 10), (True, 10), (9, 9), (9, 11), ("9", 10),
])
def test_sign_fixed_n_and_invalid_values_rejected(k, n):
    with pytest.raises(ValueError):
        sign_upper_tail(k, n=n)


@pytest.mark.parametrize("p", [0.5, "0.5", Fraction(-1, 2), Fraction(3, 2)])
def test_sign_power_invalid_probability_rejected(p):
    with pytest.raises(ValueError):
        sign_power(9, p)


@pytest.mark.parametrize("args", [
    {"n": True}, {"n": 0}, {"alpha": 0.05},
    {"alpha": Fraction(0)}, {"alpha": Fraction(1)},
    {"lower": -1, "upper": -1}, {"lower": -1.0},
])
def test_hoeffding_invalid_inputs_rejected(args):
    with pytest.raises(ValueError):
        hoeffding_mean_margin(**args)


def test_no_scientific_selection_or_execution_and_integrity():
    r = build_estimand_options_review()
    assert r["kind"] == KIND
    assert r["status"] == "D3_ESTIMAND_SELECTION_PENDING_INDEPENDENT_REVIEW"
    assert r["science_executable"] is False
    assert r["real_data_loaded"] is False
    assert r["optimizer_steps"] == 0
    assert r["scientific_claim"] == "NONE"
    assert r["accepted_method"] is None
    assert r["accepted_estimand"] is None
    assert r["approved_type_i_error_plan"] is None
    assert r["reviewer_signoff"] is False
    assert r["owner_science_run_authorization"] is False
    assert r["candidate_paired_seed_count"] == 10
    assert r["candidate_primary_order"] == "O0"
    assert set(r["alternative_estimands_and_tests"]) == {
        "MEAN_OF_PAIRED_SEED_DIFFERENCES",
        "POSITIVE_SIGN_PREVALENCE",
        "PARAMETRIC_PAIRED_MEAN",
        "SIGN_FLIP_SYMMETRY_TEST",
    }
    for option in r["alternative_estimands_and_tests"].values():
        assert option["decision"].startswith("NOT_SELECTED")
    mean = r["alternative_estimands_and_tests"]["MEAN_OF_PAIRED_SEED_DIFFERENCES"]
    signs = r["alternative_estimands_and_tests"]["POSITIVE_SIGN_PREVALENCE"]
    assert mean["one_sided_hoeffding_lower_bound_radius"] > .77
    assert mean["two_direction_95pct_hoeffding_radius"] > .85
    assert signs["minimal_positive_seed_count_out_of_10"] == 9
    assert signs["at_threshold_exact_one_sided_p"] == "11/1024"
    assert signs["one_fewer_exact_one_sided_p"] == "7/128"
    assert r["record_sha256"] == sha256(canonical_json({
        k: v for k, v in r.items() if k != "record_sha256"
    })).hexdigest()
    assert r == build_estimand_options_review()


def _run_cli(*args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )


def test_cli_read_only_exact_options_and_exclusive_file(tmp_path):
    path = tmp_path / "d3-review.json"
    call = ("--review-estimands", "--output", str(path))
    first = _run_cli(*call)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout) == build_estimand_options_review()
    assert json.loads(path.read_text()) == build_estimand_options_review()
    second = _run_cli(*call)
    assert second.returncode != 0
    assert json.loads(path.read_text()) == build_estimand_options_review()


@pytest.mark.parametrize("args", [
    (), ("--execute",), ("--science",), ("--features", "frozen.pt"),
    ("--results", "real.json"), ("--seed", "501"),
    ("--approve",), ("--freeze",), ("--method", "sign"),
    ("--review-estimands", "--mean-effect", "0.1"),
    ("--review-estimands", "--select-method", "sign"),
])
def test_cli_does_not_accept_execution_data_or_method_selection(args):
    result = _run_cli(*args)
    assert result.returncode != 0
    assert "error:" in result.stderr
