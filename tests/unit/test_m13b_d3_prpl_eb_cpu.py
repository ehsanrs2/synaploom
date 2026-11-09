"""Synthetic-only D3 check of Waudby-Smith–Ramdas Theorem 2 / Eq. 16."""
import math

import numpy as np
import pytest

from synaploom.research.m13b_d3_prpl_eb_cpu import (
    ALPHA, CAP, LOG_TWO_ALPHA, _prpl_eb_vectorized, _fixed_bet_vectorized,
    _wilson_95, both_needed, build_review, prpl_eb_lower, prpl_eb_rejects,
    prpl_eb_trace,
)


def test_initial_predictable_statistics_and_paper_constants():
    tr=prpl_eb_trace([0.02, -0.03, 0.07])
    assert tr.past_means_x[0] == 0.5
    assert tr.past_variances_x[0] == 0.25
    lam0=min(CAP, math.sqrt(2*math.log(2/ALPHA)/(3*0.25)))
    assert abs(tr.lambdas[0]-lam0)<1e-14
    x1=(1+0.02)/2
    mu1=(0.5+x1)/2
    var1=(0.25+(x1-mu1)**2)/2
    assert abs(tr.past_means_x[1]-mu1)<1e-14
    assert abs(tr.past_variances_x[1]-var1)<1e-14
    assert abs(tr.radius_x - ((LOG_TWO_ALPHA + sum(
        ((d+1)/2-mu)**2*(-math.log1p(-lam)-lam)
        for d,mu,lam in zip([.02,-.03,.07],tr.past_means_x,tr.lambdas)
    )) / sum(tr.lambdas)))<1e-14
    assert abs(tr.lower_d - (2*(tr.weighted_center_x-tr.radius_x)-1))<1e-14


def test_no_future_information_in_predictable_bets():
    # n must be the same for a valid fixed-n policy comparison.
    a=prpl_eb_trace([0.02,-0.03,0.0,0.99,-1.0])
    b=prpl_eb_trace([0.02,-0.03,0.0,-1.0,0.99])
    assert a.lambdas[:4] == b.lambdas[:4]
    assert a.past_means_x[:4] == b.past_means_x[:4]
    assert a.past_variances_x[:4] == b.past_variances_x[:4]


def test_vectorized_matches_independent_scalar_engine_and_reference():
    batch=np.array([[.02,-.03,.07,.01],[-1.,1.,0.,.2],[.05]*4,
                    [-.2,.5,.1,.02]],dtype=np.float64)
    bs=_prpl_eb_vectorized(batch)
    single=[prpl_eb_lower(row.tolist()) for row in batch]
    assert np.allclose(bs,single,atol=1e-12,rtol=0)
    with np.errstate(divide='ignore'):
        fixed=np.log1p(batch).sum(axis=1)>=math.log(20)
    assert np.array_equal(_fixed_bet_vectorized(batch),fixed)


def test_invalid_inputs_fail_closed():
    for bad in ([],[.01],[.01,float('nan')],[.01,float('inf')],
                [.01,-1.01],[.01,1.001],[.01,True],[.01,'0.01']):
        with pytest.raises(ValueError):
            prpl_eb_lower(bad)
    with pytest.raises(ValueError):
        both_needed([.01,.02],[.01,.02,.03])
    with pytest.raises(ValueError):
        both_needed([.01,.02],[float('nan'),.02])
    with pytest.raises(ValueError):
        _prpl_eb_vectorized(np.array([[.0,1.1]]))


def test_constant_gain_2pp_bounded_and_iut_requires_both():
    assert prpl_eb_lower([0.02]*210)<0
    assert prpl_eb_lower([0.02]*600)<0
    assert prpl_eb_lower([0.02]*1000)>0
    assert not both_needed([.02]*1000,[0.0]*1000)
    assert both_needed([.02]*1000,[.02]*1000)


def test_wilson_interval_has_correct_edges():
    lo,hi=_wilson_95(0,4000)
    assert lo==0.0 and hi>0.0
    lo,hi=_wilson_95(4000,4000)
    assert hi==1.0 and lo<1.0
    with pytest.raises(ValueError):
        _wilson_95(2,1)


def test_simulation_deterministic_json_hash_and_nulls():
    r1=build_review(reps=200,n_grid=(40,80))
    r2=build_review(reps=200,n_grid=(40,80))
    assert r1==r2
    assert r1['record_sha256']==r2['record_sha256']
    assert not r1['science_executable']
    assert r1['method_selected'] is None and r1['n_selected'] is None
    assert r1['gpu_used'] is False and not r1['real_seed_accuracy_used']
    for row in r1['constructed_zero_mean_nulls']:
        assert abs(row['constructed_true_mean'])<1e-12
    for row in r1['constructed_2pp_alternatives']:
        assert abs(row['constructed_true_mean']-.02)<1e-12
    for row in r1['constructed_zero_mean_nulls']+r1['constructed_2pp_alternatives']:
        for c in row['cells']:
            assert 0<=c['prpl_eb_probability_mc']<=1
            assert 0<=c['fixed_lambda1_probability_mc']<=1
            assert c['prpl_eb_mc_wilson95'][0]<=c['prpl_eb_probability_mc']<=c['prpl_eb_mc_wilson95'][1]
    for row in r1['constructed_2pp_alternatives']:
        if row['probability_positive']==0.5:
            for c in row['cells']:
                for name in ('prpl_eb','fixed_lambda1'):
                    assert 0<=c[name+'_joint_independent_mc']<=1
                    assert 0<=c[name+'_joint_opposite_signs_mc']<=1
                    assert c[name+'_joint_identical_mc']==c[name+'_probability_mc']


def test_cli_report_does_not_bear_science_authorization():
    report=build_review(reps=80,n_grid=(40,))
    assert report['kind']=='M13B_D3_PRPL_EB_THEOREM2_EQ16_CPU_V1'
    assert report['theorem_alpha_two_sided']==0.05
    assert report['fixed_n_predictable_c']==0.5
    assert report['real_seed_accuracy_used'] is False
    assert report['method_selected'] is None
