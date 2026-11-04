"""CPU-only 2 percentage-point paired-mean D3 feasibility checks; no science."""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
import subprocess
import sys

import pytest

from synaploom.research.m13b_d3_two_pp_planning import (
    ALPHA, KIND, MU, SD_SCENARIOS,
    binomial_half_upper_tail, constant_gain_first_reject_n, costs_for_n,
    first_n_for_marginal_power, least_favorable_two_point_null, report,
    two_point_evalue_power_at_n,
)


def test_owner_mean_2pp_for_both_endpoints_is_planning_only():
    r = report()
    assert r['kind'] == KIND
    assert r['effect_size_of_interest'] == {
        'final_average_accuracy_mean_gain_fraction': .02,
        'late_task_within_task_mean_accuracy_gain_fraction': .02,
        'interpretation': 'SCIENTIFICALLY_MEANINGFUL_EFFECT_FOR_POWER_PLANNING_ONLY',
        'statistical_positive_null_per_endpoint': 'MEAN_DIFFERENCE_LE_ZERO_UNCHANGED',
        'owner_protocol_frozen': False,
    }
    assert r['status'] == 'OWNER_2PP_PLANNING_EFFECT_RECORDED__NO_PROTOCOL_OR_GPU_APPROVAL'
    assert r['method_selected_for_science'] is None
    assert r['final_seed_count'] is None
    assert r['actual_GPU_budget_owner_approved'] is False
    assert r['science_executable'] is False
    assert r['real_science_outcomes_loaded'] is False
    assert r['data_or_model_loaded'] is False
    assert r['gpu_used'] is False
    assert r['optimizer_steps_executed'] == 0
    assert r['scientific_claim'] == 'NONE'
    assert r['reference_candidate']['order'] == 'O0'
    assert r['reference_candidate']['co_primary'] == [
        'final_average_accuracy','late_task_within_task_mean_accuracy'
    ]
    assert r['reference_candidate']['paired_seeds_candidate'] == list(range(501,511))
    raw = json.dumps({k:v for k,v in r.items() if k!='record_sha256'},
                     sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
    assert r['record_sha256'] == sha256(raw).hexdigest()


def test_information_theoretic_all_constant_data_lower_bound_exact():
    assert MU == Fraction(1,50)
    assert ALPHA == Fraction(1,20)
    result = least_favorable_two_point_null(MU)
    p = Fraction(1,1) / (1+MU)
    assert p == Fraction(50,51)
    assert p*MU+(1-p)*(-1) == 0
    assert result['min_n_for_deterministic_rejection_of_all_constant_positive_deltas'] == 152
    assert p**151 > ALPHA >= p**152
    assert result['null_probability_all_positive_at_previous_n'] == pytest.approx(.05027754506537141)
    assert result['null_probability_all_positive_at_first_n'] == pytest.approx(.04929171084840334)
    assert constant_gain_first_reject_n(Fraction(1,100)) == 302
    assert constant_gain_first_reject_n(Fraction(1,50)) == 152
    assert constant_gain_first_reject_n(Fraction(1,20)) == 62


def test_exact_half_binomial_tail_thresholds():
    assert binomial_half_upper_tail(10,0) == 1
    assert binomial_half_upper_tail(10,11) == 0
    assert binomial_half_upper_tail(10,10) == Fraction(1,1024)
    assert binomial_half_upper_tail(10,9) == Fraction(11,1024)
    assert binomial_half_upper_tail(10,5) == Fraction(638,1024)
    assert binomial_half_upper_tail(0,0) == 1


@pytest.mark.parametrize('sd,n80,n90', [
    (Fraction(0),152,152),
    (Fraction(1,50),164,170),
    (Fraction(1,20),191,210),
    (Fraction(1,10),294,358),
])
def test_two_point_positive_mean_power_is_exact_and_only_hypothetical(sd,n80,n90):
    assert sd in SD_SCENARIOS
    # Exact binomial tail probabilities for invented IID two-point
    # paired mean differences: delta=.02 +/- sd at P=.5 each.
    assert first_n_for_marginal_power(sd,Fraction(4,5)) == n80
    assert first_n_for_marginal_power(sd,Fraction(9,10)) == n90
    p80,_ = two_point_evalue_power_at_n(n80,MU,sd)
    p90,_ = two_point_evalue_power_at_n(n90,MU,sd)
    assert p80 >= Fraction(4,5)
    assert p90 >= Fraction(9,10)
    if n80>1:
        assert two_point_evalue_power_at_n(n80-1,MU,sd)[0] < Fraction(4,5)
    if n90>1:
        assert two_point_evalue_power_at_n(n90-1,MU,sd)[0] < Fraction(9,10)
    for n in (10,20):
        p,_ = two_point_evalue_power_at_n(n,MU,sd)
        assert p == 0
    p40,_ = two_point_evalue_power_at_n(40,MU,sd)
    assert p40 < Fraction(1,1000)


def test_joint_power_is_a_range_without_declaring_endpoint_independence():
    r=report()
    assert len(r['invented_two_point_powers'])==4
    for row in r['invented_two_point_powers']:
        assert row['invented_true_paired_mean_each_endpoint']==.02
        assert row['invented_distribution']['paired_difference_support_valid']
        assert row['joint_80pct_under_unknown_endpoint_dependence_sufficient_n'] == (
            row['first_n_with_at_least_90pct_one_endpoint_power']
        )
        for cell in row['selected_n_cells']:
            p=cell['one_endpoint_power']
            assert cell['two_endpoint_joint_power_frechet_lower'] == pytest.approx(max(0,2*p-1))
            assert cell['two_endpoint_joint_power_frechet_upper'] == pytest.approx(p)
            assert cell['two_endpoint_joint_power_frechet_lower'] <= p
            assert cell['two_endpoint_joint_power_frechet_upper'] >= p


def test_gpu_budget_envelopes_do_not_claim_measured_full_science_runtime():
    assert costs_for_n(10)['narrow_O0_two_arm_trajectories']==20
    assert costs_for_n(10)['original_three_arm_three_order_trajectories_at_same_n']==90
    assert costs_for_n(152)['narrow_O0_two_arm_trajectories']==304
    assert costs_for_n(152)['original_three_arm_three_order_trajectories_at_same_n']==1368
    assert costs_for_n(152)['narrow_O0_two_arm_optimizer_steps']==2432000
    assert costs_for_n(152)['hypothetical_minutes_per_complete_trajectory']==[3,10,30]
    assert costs_for_n(152)['narrow_single_gpu_sequential_hours_scenarios']=={
        '3':15.2, '10':50.667, '30':152.0
    }
    assert costs_for_n(152)['time_is_unmeasured_full_science_assumption'] is True
    assert costs_for_n(152)['overhead_retries_and_resource_contention_not_included'] is True
    r=report()
    assert r['historical_training_only_calibration_390_steps_per_trajectory']['original_eval_and_8000_step_full_science_cost'] == 'NOT_MEASURED'
    assert r['historical_training_only_calibration_390_steps_per_trajectory']['cannot_scale_training_only_directly'] is True


@pytest.mark.parametrize('d', [Fraction(0), Fraction(-1,10),Fraction(3,2),.02,True])
def test_invalid_mean_planning_gain_rejected(d):
    with pytest.raises(ValueError):
        constant_gain_first_reject_n(d)
    with pytest.raises(ValueError):
        least_favorable_two_point_null(d)


@pytest.mark.parametrize('n,mu,sd', [
    (0,MU,Fraction(1,50)),
    (False,MU,Fraction(1,50)),
    (801,MU,Fraction(1,50)),
    (10,0,Fraction(1,50)),
    (10,MU,True),
    (10,MU,Fraction(2)),
])
def test_invalid_betting_power_scenario_rejected(n,mu,sd):
    with pytest.raises(ValueError):
        two_point_evalue_power_at_n(n,mu,sd)


def test_cli_rejects_external_inputs_and_only_emits_review_json():
    cmd=[sys.executable,'-m','synaploom.research.m13b_d3_two_pp_planning']
    ok=subprocess.run(cmd,capture_output=True,text=True)
    assert ok.returncode==0, ok.stderr
    assert json.loads(ok.stdout)==report()
    for args in (['--gpu'],['--results','real.json'],['--freeze'],['--seeds','152'],['--run']):
        bad=subprocess.run(cmd+args,capture_output=True,text=True)
        assert bad.returncode!=0
