"""CPU-only reference tests for Maurer-Pontil fixed-n bounded mean theorem."""
from fractions import Fraction
from math import log, sqrt

import pytest

from synaploom.research.m13b_d3_empirical_bernstein_cpu import (
    ALPHA, N_GRID, _mp_lower_for_count, _probabilities_for_mask,
    build_review, empirical_bernstein_both_needed, empirical_bernstein_lower,
    empirical_bernstein_rejects, two_point_cell,
)


def test_formula_mapping_sample_variance_bessel():
    xs = [-0.1, 0.0, 0.1, 0.2]
    # sample mean=0.05 and unbiased sample variance=1/60.
    n=len(xs)
    expected = .05 - sqrt(2*(1/60)*log(40)/n) - 14*log(40)/(3*(n-1))
    assert abs(empirical_bernstein_lower(xs)-expected)<1e-12


def test_lower_mean_bound_transformation_consistency():
    for xs in ([0.02]*862,[-1.,1.],[-.1,.05,.2,.05],[0.]*10):
        inv=[-x for x in xs]
        # Sign reversal leaves the variance/radius unchanged. Both one-sided
        # lower bounds must add to minus twice the nonnegative radius.
        mean=sum(xs)/len(xs)
        lower=empirical_bernstein_lower(xs)
        other=empirical_bernstein_lower(inv)
        assert abs(lower-other-2*mean)<1e-12
        assert lower+other<=1e-12
        assert isinstance(empirical_bernstein_rejects(xs),bool)


def test_fail_closed_numeric_and_length():
    for bad in ([],[.1], [.1,float('nan')],[.1,float('inf')], [.1,True],
                [.1,'0.1'], [.1,1.001], [-1.001,.1]):
        with pytest.raises(ValueError):
            empirical_bernstein_lower(bad)
    for bad_alpha in (0,-1,1,float('nan'),True,'0.05'):
        with pytest.raises(ValueError):
            empirical_bernstein_lower([0.,.1],bad_alpha)
    with pytest.raises(ValueError):
        empirical_bernstein_both_needed([.01,.02],[.01,.02,.03])
    with pytest.raises(ValueError):
        empirical_bernstein_both_needed([.01,.02],[float('nan'),.02])


def test_all_plus_2pp_zero_variance_threshold():
    assert not empirical_bernstein_rejects([.02]*861)
    assert empirical_bernstein_rejects([.02]*862)
    assert not empirical_bernstein_both_needed([.02]*862,[0.]*862)
    assert empirical_bernstein_both_needed([.02]*862,[.02]*862)


def test_two_point_sufficient_stats_match_materialized_array():
    n,k=40,31
    a,b=Fraction(7,100),Fraction(-3,100)
    ds = [float(a)]*k+[float(b)]*(n-k)
    assert abs(float(_mp_lower_for_count(n,k,a,b))-empirical_bernstein_lower(ds))<1e-12
    assert abs(float(_mp_lower_for_count(n,n-k,a,b))-empirical_bernstein_lower([float(a)]*(n-k) + [float(b)]*k))<1e-12
    assert abs(float(_mp_lower_for_count(n,n-k,a,b))-float(_mp_lower_for_count(n,k,a,b)))>1e-4


def test_exact_probabilities_agree_with_binomial():
    assert _probabilities_for_mask(10,Fraction(1,2),[k>=9 for k in range(11)]) == Fraction(11,1024)
    assert _probabilities_for_mask(10,Fraction(1,2),[False]*11) == 0
    assert _probabilities_for_mask(10,Fraction(1,2),[True]*11) == 1


def test_synthetic_null_size_and_2pp_joint_couplings():
    report = build_review()
    assert report['record_sha256']==build_review()['record_sha256']
    assert report['mp_first_n_rejecting_all_plus_2pp_when_variance_zero']==862
    assert report['pointwise_all_plus_2pp_method_independent_min_n']==152
    assert report['method_selected'] is None and not report['science_executable']
    assert report['gpu_used'] is False and not report['real_outcomes_used']
    assert all(c['empirical_bernstein_probability']<=float(ALPHA)+1e-14
               for row in report['null_synthetic'] for c in row['cells'])
    assert [c['n'] for c in report['alternative_synthetic'][0]['cells']]==list(N_GRID)
    for row in report['alternative_synthetic']:
        if row['positive_probability']==.5:
            for c in row['cells']:
                for key in ('empirical_bernstein','fixed_lambda1'):
                    marginal=c[key+'_probability']
                    assert abs(c[key+'_joint_independent']-marginal*marginal)<1e-14
                    assert 0<=c[key+'_joint_opposite_signs']<=marginal<=1
    sd5=next(row for row in report['alternative_synthetic'] if row['label']=='mean_2pp_sd_5pp')
    n210=next(c for c in sd5['cells'] if c['n']==210)
    assert n210['empirical_bernstein_probability']==0.0
    assert .90<n210['fixed_lambda1_probability']<.91


def test_no_real_results_flags_or_synthetic_positive_winner():
    r=build_review()
    for key in ('real_outcomes_used','model_or_features_loaded','science_executable'):
        assert r[key] is False
    assert r['n_selected'] is None and r['method_selected'] is None
