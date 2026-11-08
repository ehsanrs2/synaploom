"""No benchmark/model/GPU access: synthetically test the D3 bounded-mean mixture."""
import itertools
from fractions import Fraction
from math import log

import pytest

from synaploom.research.m13b_d3_hedged_mean_cpu import (
    ALPHA, BET_GRID, THRESHOLD, _wealth_exact, both_needed, build_review,
    exact_binomial_event, log_evalues, positive_count_threshold, rejects,
)


def test_reject_bad_or_missing_data_failclosed():
    for bad in ([], [float('nan')], [float('inf')], [1.001], [-1.001], [True], ['0.01']):
        with pytest.raises(ValueError):
            log_evalues(bad)
    with pytest.raises(ValueError):
        both_needed([0.02], [0.02,0.02])
    for bad in (float('nan'),float('inf'),True,'1'):
        with pytest.raises(ValueError):
            rejects(bad)


def test_fixed_lambda_reference_and_order_invariance():
    xs = [0.02,-0.05,0.11,0.07]
    m,f=log_evalues(xs)
    assert abs(f-sum(log(1+x) for x in xs))<1e-12
    for perm in itertools.permutations(xs):
        mm,ff=log_evalues(list(perm))
        assert abs(mm-m)<1e-12 and abs(ff-f)<1e-12


def test_mixture_stays_nonnegative_at_full_loss():
    m,f=log_evalues([0.5,-1.0,0.5])
    assert f==float('-inf')
    assert m>float('-inf')


def test_all_plus_2pp_pointwise_152_and_mixture_more_conservative():
    assert not rejects(log_evalues([0.02]*151)[1])
    assert rejects(log_evalues([0.02]*152)[1])
    assert not rejects(log_evalues([0.02]*152)[0])
    assert Fraction(50,51)**151 > ALPHA >= Fraction(50,51)**152


def test_exact_threshold_matches_brute_small_n():
    a,b=Fraction(1,2), Fraction(-1,2)
    for n in (2,5,9,15):
        for mixed in (True,False):
            got=positive_count_threshold(n,a,b,mixed=mixed)
            brute=next((k for k in range(n+1) if _wealth_exact(n,k,a,b,mixed=mixed)>=THRESHOLD),n+1)
            assert got==brute


def test_exact_binomial_joint_extremes_and_iut_logic():
    assert exact_binomial_event(10,Fraction(1,2),9)==Fraction(11,1024)
    assert exact_binomial_event(10,Fraction(1,2),9,1)==0
    assert not both_needed([.5]*10, [-.5]*10)
    assert both_needed([.5]*12, [.5]*12)


def test_documented_synthetic_null_and_joint_power():
    r=build_review()
    assert r['record_sha256']==build_review()['record_sha256']
    assert r['method_selected'] is None and not r['science_executable']
    assert r['pointwise_all_plus_2pp_deterministic_min_n_for_ANY_valid_nonrandomized_test']==152
    for fam in r['null_families']:
        for cell in fam['cells']:
            assert cell['fixed_lambda1_probability']<=float(ALPHA)+1e-14
            assert cell['hedged_mixture_probability']<=float(ALPHA)+1e-14
    for fam in r['alternative_mean_2pp_families']:
        if fam['p_positive']==.5:
            for cell in fam['cells']:
                for prefix in ('fixed_lambda1','hedged_mixture'):
                    assert (0<=cell[prefix+'_joint_power_opposite_signs']
                            <=cell[prefix+'_joint_power_identical']<=1)
                    assert abs(cell[prefix+'_joint_power_independent'] -
                               cell[prefix+'_probability']**2) < 1e-14
