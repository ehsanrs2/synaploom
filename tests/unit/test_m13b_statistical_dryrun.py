"""M1.3b statistical contract tests: fabricated data, NEVER real science."""
from __future__ import annotations

import copy
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from synaploom.protocols.h1a_order_sensitivity import paired_bootstrap_ci
from synaploom.research.m13b_manifest import canonical_json
from synaploom.research.m13b_statistical_dryrun import (
    ACCURACY_FIELDS, FIELDS, KIND, METRICS, ORDERS, PROPOSED_SEED_IDS,
    analyze_synthetic_fixture, build_synthetic_fixture,
    validate_synthetic_inventory,
)


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "experiments" / "scripts" / "m13b_statistical_dryrun.py"


@pytest.fixture(scope="module")
def positive():
    return build_synthetic_fixture("positive")


def test_90_toy_records_complete_by_exact_slot_and_seed(positive):
    data = validate_synthetic_inventory(positive)
    assert len(data) == 90
    assert len(set(data)) == 90
    assert positive["scientific_claim"] == "NONE"
    assert positive["real_training_executed"] is False
    for order in ORDERS:
        rows = [x for x in data.values() if x["order_id"] == order]
        assert len(rows) == 30
        assert {x["seed"] for x in rows} == set(PROPOSED_SEED_IDS)
        for seed in PROPOSED_SEED_IDS:
            block = [x for x in rows if x["seed"] == seed]
            assert {x["arm"] for x in block} == {
                "progressive_reserve", "all_recruited", "fixed_sparse_mlp"
            }
            assert all(x["actual_optimizer_steps"] == 0 for x in block)
            assert all(x["status"] == "SYNTHETIC_NOT_TRAINED" for x in block)
            assert all(set(x["metrics"]) == set(FIELDS) for x in block)


@pytest.mark.parametrize("scenario,expected", [
    ("positive", "SYNTHETIC_FAVORABLE"),
    ("negative", "SYNTHETIC_UNFAVORABLE"),
    ("null", "SYNTHETIC_INCONCLUSIVE"),
    ("opposed", "SYNTHETIC_INCONCLUSIVE"),
])
def test_paired_decision_requires_both_co_primary_endpoints(scenario, expected):
    report = analyze_synthetic_fixture(build_synthetic_fixture(scenario))
    assert report["kind"] == KIND
    assert report["synthetic_decision"] == expected
    assert report["science_executable"] is False
    assert report["real_training_executed"] is False
    assert report["real_scientific_outcomes_present"] is False
    assert report["scientific_claim"] == "NONE"
    assert report["judgment_scope"] == "SYNTHETIC_UNIT_TEST_ONLY"
    assert report["all_90_synthetic_slots_present"] is True
    assert report["seed_count"] == 10
    assert report["paired_bootstrap_resamples"] == 10_000
    assert report["primary_order"] == "O0"
    assert report["primary_contrast"] == ["progressive_reserve", "fixed_sparse_mlp"]
    assert report["co_primary_metrics"] == list(METRICS)
    for metric in METRICS:
        lo, hi = report["primary_95pct_percentile_intervals_fraction"][metric]
        assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi
    if scenario == "positive":
        assert all(report["primary_95pct_percentile_intervals_fraction"][metric][0] > 0
                   for metric in METRICS)
    if scenario == "negative":
        assert all(report["primary_95pct_percentile_intervals_fraction"][metric][1] < 0
                   for metric in METRICS)
    if scenario == "opposed":
        a, b = (report["primary_95pct_percentile_intervals_fraction"][m] for m in METRICS)
        assert a[0] > 0 and b[1] < 0


def test_order_effects_descriptive_and_same_ten_seed_clusters(positive):
    report = analyze_synthetic_fixture(positive)
    by_order = report["descriptive_order_mean_deltas_fraction"]
    assert list(by_order) == list(ORDERS)
    assert by_order["O0"]["final_average_accuracy"] > 0
    assert by_order["O1"]["final_average_accuracy"] < 0
    assert by_order["O2"]["final_average_accuracy"] > 0
    assert set(report["descriptive_equal_order_seed_cluster_intervals_fraction"]) == set(METRICS)
    assert len(report["descriptive_equal_order_seed_cluster_intervals_fraction"][METRICS[0]]) == 2
    assert all("descriptive" in text.lower() for text in report["limitations"]
               if "O1" in text)


def test_primary_bootstrap_matches_frozen_historical_paired_convention(positive):
    items = validate_synthetic_inventory(positive)
    primary_deltas = {
        seed: (
            items[f"O0__seed-{seed}__progressive_reserve"]["metrics"][METRICS[0]]
            - items[f"O0__seed-{seed}__fixed_sparse_mlp"]["metrics"][METRICS[0]]
        ) for seed in PROPOSED_SEED_IDS
    }
    historic_ci = paired_bootstrap_ci(primary_deltas, samples=10_000, seed=1337)
    report = analyze_synthetic_fixture(positive)
    assert report["primary_95pct_percentile_intervals_fraction"][METRICS[0]] == historic_ci


def test_random_record_arrival_order_does_not_change_analysis(positive):
    forward = analyze_synthetic_fixture(positive)
    scrambled = copy.deepcopy(positive)
    scrambled["records"].reverse()
    validate_synthetic_inventory(scrambled)
    reversed_result = analyze_synthetic_fixture(scrambled)
    assert reversed_result == forward
    bare = {k: v for k, v in forward.items() if k != "record_sha256"}
    assert sha256(canonical_json(bare)).hexdigest() == forward["record_sha256"]


@pytest.mark.parametrize("mutation,pattern", [
    (lambda f: f["records"].pop(), "incomplete"),
    (lambda f: f["records"].append(copy.deepcopy(f["records"][-1])), "oversized"),
    (lambda f: f["records"].__setitem__(1, copy.deepcopy(f["records"][0])), "duplicate"),
    (lambda f: f["records"][0].update(seed=999), "preregistered"),
    (lambda f: f["records"][0].update(arm="rigl_like_mlp"), "preregistered"),
    (lambda f: f["records"][0].update(order_sha256="0" * 64), "preregistered"),
    (lambda f: f.update(candidate_sha256="0" * 64), "candidate plan"),
    (lambda f: f["records"][0].update(status="COMPLETE_REAL"), "provenance"),
    (lambda f: f["records"][0].update(actual_optimizer_steps=8000), "provenance"),
    (lambda f: f["records"][0]["metrics"].pop("final_average_accuracy"), "outcome metrics"),
    (lambda f: f["records"][0]["metrics"].update(final_average_accuracy=float("nan")), "nonfinite"),
    (lambda f: f["records"][0]["metrics"].update(final_average_accuracy=float("inf")), "nonfinite"),
    (lambda f: f["records"][0]["metrics"].update(final_average_accuracy=True), "nonnumeric"),
    (lambda f: f["records"][0]["metrics"].update(final_average_accuracy=101.0), "outside"),
    (lambda f: f["records"][0].update(extra_result="posthoc"), "fields"),
    (lambda f: f.update(real_training_executed=True), "science-claimed"),
    (lambda f: f.update(scientific_claim="FAVORABLE"), "science-claimed"),
])
def test_missing_corrupt_fabricated_seed_records_fail_closed(positive, mutation, pattern):
    changed = copy.deepcopy(positive)
    mutation(changed)
    with pytest.raises(ValueError, match=pattern):
        analyze_synthetic_fixture(changed)


def test_bootstrap_does_not_mutate_torch_rng_or_fixture(positive):
    previous = torch.get_rng_state().clone()
    original = copy.deepcopy(positive)
    analyze_synthetic_fixture(positive)
    assert torch.equal(previous, torch.get_rng_state())
    assert positive == original


def _cli(*args):
    return subprocess.run([sys.executable, str(CLI), *args], cwd=ROOT,
                          capture_output=True, text=True, check=False)


def test_cli_synthetic_and_exclusive_new_report_only(tmp_path):
    path = tmp_path / "synthetic.json"
    out = _cli("--synthetic-self-test", "--scenario", "positive",
               "--output", str(path))
    assert out.returncode == 0, out.stderr
    report = json.loads(out.stdout)
    assert report["synthetic_decision"] == "SYNTHETIC_FAVORABLE"
    assert report["real_scientific_outcomes_present"] is False
    assert json.loads(path.read_text(encoding="utf-8")) == report
    conflict = _cli("--synthetic-self-test", "--output", str(path))
    assert conflict.returncode != 0
    assert json.loads(path.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("flags", [
    (),
    ("--execute",),
    ("--synthetic-self-test", "--approve"),
    ("--synthetic-self-test", "--freeze"),
    ("--synthetic-self-test", "--cuda"),
    ("--synthetic-self-test", "--seed", "501"),
    ("--synthetic-self-test", "--results", "real-science.json"),
    ("--synthetic-self-test", "--scenario", "unknown"),
])
def test_cli_rejects_real_science_and_unapproved_options(flags):
    x = _cli(*flags)
    assert x.returncode != 0
    assert "error:" in x.stderr
