"""M1.3b D3: fixed-n empirical Bernstein, theorem-matched CPU research only.

Maurer & Pontil, COLT 2009, Theorem 4, applied to Z=(D+1)/2 in [0,1]
(and 1-Z for the upper-deviation version). IID paired differences D in
[-1,1]; alpha=.05; n>=2. For sample mean m and unbiased sample variance v
of D, the one-sided LOWER mean bound is

    m - sqrt(2*v*log(2/alpha)/n) - 14*log(2/alpha)/(3*(n-1)).

This is NOT a confidence sequence, not a betting e-value, and not a result
from SynapLoom evaluation. All scenarios below are synthetic two-point toys.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
from math import comb, fsum, isfinite, log, log1p, sqrt
import json
from typing import Sequence

ALPHA = Fraction(1, 20)
E_THRESHOLD = 20
N_GRID = (10, 40, 80, 152, 210, 320, 600, 1000)
MAX_N = 1000


def _validate_deltas(xs: Sequence[float]) -> list[float]:
    if not isinstance(xs, (list, tuple)) or len(xs) < 2:
        raise ValueError('requires at least two paired seed differences')
    out = []
    for v in xs:
        if type(v) not in (float, int) or not isfinite(v) or not -1 <= v <= 1:
            raise ValueError('paired differences must be finite numeric fractions in [-1,1]')
        out.append(float(v))
    return out


def empirical_bernstein_lower(xs: Sequence[float], alpha: float = 0.05) -> float:
    """Maurer-Pontil fixed-n one-sided lower bound for population paired mean.

    Statistical guarantee requires *IID* D in [-1,1], independent of
    how the test method was selected; this function cannot verify IID.
    """
    ds = _validate_deltas(xs)
    if type(alpha) not in (float, int) or not isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be a finite numeric probability in (0,1)')
    n = len(ds)
    mean = fsum(ds)/n
    variance = fsum((d-mean)**2 for d in ds)/(n-1)
    c = log(2/alpha)
    return mean - sqrt(2*variance*c/n) - 14*c/(3*(n-1))


def empirical_bernstein_rejects(xs: Sequence[float]) -> bool:
    return empirical_bernstein_lower(xs) > 0.0


def empirical_bernstein_both_needed(xs: Sequence[float], ys: Sequence[float]) -> bool:
    """IUT: both one-sided lower confidence limits exceed zero (no NaN skipping)."""
    if len(xs) != len(ys):
        raise ValueError('two co-primary metrics must share one paired-seed registry')
    # Evaluate both rather than short-circuit: reject missing/nonfinite on either.
    x_low = empirical_bernstein_lower(xs)
    y_low = empirical_bernstein_lower(ys)
    return x_low > 0.0 and y_low > 0.0


def _decimal_fraction(x: Fraction) -> Decimal:
    return Decimal(x.numerator) / Decimal(x.denominator)


def _mp_lower_for_count(n: int, k: int, a: Fraction, b: Fraction) -> Decimal:
    """High-precision evaluation of exact sufficient statistics of two-point data.

    Decimal uses 60-digit precision for log/sqrt, while sample mean and
    Bessel variance are assembled from rational counts. No IID data loaded.
    """
    if not (type(n) is int and 2 <= n <= MAX_N and type(k) is int and 0 <= k <= n
            and type(a) is Fraction and type(b) is Fraction and -1 <= b < a <= 1):
        raise ValueError('invalid two-point inputs')
    with localcontext() as ctx:
        ctx.prec = 60
        mean = _decimal_fraction((k*a+(n-k)*b)/n)
        var = _decimal_fraction(Fraction(k*(n-k), n*(n-1))*(a-b)**2)
        c = (Decimal(2)/_decimal_fraction(ALPHA)).ln()
        rad = (Decimal(2)*var*c/Decimal(n)).sqrt() + Decimal(14)*c/(Decimal(3)*(n-1))
        return mean-rad


def _fixed_bet_reject_for_count(n: int, k: int, a: Fraction, b: Fraction) -> bool:
    # Independent pre-existing fixed lambda=1 reference, no exponent overflow.
    if (a == -1 and k > 0) or (b == -1 and n-k > 0):
        return False
    return k*log1p(float(a))+(0.0 if n==k else (n-k)*log1p(float(b))) >= log(E_THRESHOLD)


def _probabilities_for_mask(n: int, p: Fraction, qualifies: list[bool]) -> Fraction:
    if not (type(n) is int and n >= 2 and len(qualifies) == n+1
            and type(p) is Fraction and 0 <= p <= 1):
        raise ValueError('invalid probability mask')
    powers_pos = [Fraction(1)]
    powers_neg = [Fraction(1)]
    for _ in range(n):
        powers_pos.append(powers_pos[-1]*p)
        powers_neg.append(powers_neg[-1]*(1-p))
    return sum((Fraction(comb(n,k))*powers_pos[k]*powers_neg[n-k]
                for k in range(n+1) if qualifies[k]), Fraction(0))


def two_point_cell(n: int, p: Fraction, a: Fraction, b: Fraction, *, joint: bool = False) -> dict[str, object]:
    if not (type(n) is int and 2 <= n <= MAX_N and type(p) is Fraction and 0 < p < 1
            and type(a) is Fraction and type(b) is Fraction and -1 <= b < a <= 1):
        raise ValueError('invalid synthetic population')
    mp = [_mp_lower_for_count(n,k,a,b) > 0 for k in range(n+1)]
    fixed = [_fixed_bet_reject_for_count(n,k,a,b) for k in range(n+1)]
    mp_probability = _probabilities_for_mask(n,p,mp)
    fixed_probability = _probabilities_for_mask(n,p,fixed)
    cell: dict[str, object] = {
        'n':n,
        'empirical_bernstein_probability':float(mp_probability),
        'fixed_lambda1_probability':float(fixed_probability),
        'empirical_bernstein_rejecting_k_count':sum(mp),
        'fixed_lambda1_rejecting_k_count':sum(fixed),
    }
    if joint:
        if p!=Fraction(1,2):
            raise ValueError('opposite-sign endpoint coupling requires symmetric p=1/2')
        for name,mask,prob in [('empirical_bernstein',mp,mp_probability),('fixed_lambda1',fixed,fixed_probability)]:
            cell[name+'_joint_identical']=float(prob)
            cell[name+'_joint_independent']=float(prob**2)
            cell[name+'_joint_opposite_signs']=float(_probabilities_for_mask(
                n,p,[mask[k] and mask[n-k] for k in range(n+1)]))
    return cell


def build_review() -> dict[str, object]:
    # Pairs (P(positive), positive, negative). All finite-support, invented.
    null_specs = (
        ('balanced',Fraction(1,2),Fraction(2,100),Fraction(-2,100)),
        ('skew_p80',Fraction(4,5),Fraction(4,100),Fraction(-16,100)),
        ('skew_p90',Fraction(9,10),Fraction(2,100),Fraction(-18,100)),
        ('skew_p96',Fraction(24,25),Fraction(2,100),Fraction(-48,100)),
        ('pointwise_extreme_loss',Fraction(50,51),Fraction(2,100),Fraction(-1)),
    )
    alt_specs = (
        ('mean_2pp_sd_2pp',Fraction(1,2),Fraction(4,100),Fraction(0)),
        ('mean_2pp_sd_5pp',Fraction(1,2),Fraction(7,100),Fraction(-3,100)),
        ('mean_2pp_sd_10pp',Fraction(1,2),Fraction(12,100),Fraction(-8,100)),
        ('mean_2pp_rare_large_loss',Fraction(24,25),Fraction(6,100),Fraction(-94,100)),
    )
    null_rows=[]
    for label,p,a,b in null_specs:
        assert p*a+(1-p)*b == 0
        cells = [two_point_cell(n,p,a,b) for n in N_GRID]
        if any(c['empirical_bernstein_probability']>float(ALPHA)+1e-14 for c in cells):
            raise AssertionError('MP bound toy type-I exceeds .05')
        null_rows.append({'label':label, 'mean_exactly_zero':True, 'positive':float(a),
                          'negative':float(b), 'positive_probability':float(p), 'cells':cells})
    alt_rows=[]
    for label,p,a,b in alt_specs:
        assert p*a+(1-p)*b == Fraction(2,100)
        cells=[two_point_cell(n,p,a,b,joint=p==Fraction(1,2)) for n in N_GRID]
        alt_rows.append({'label':label,'mean_exactly_2pp':True,'positive':float(a),
                         'negative':float(b),'positive_probability':float(p),'cells':cells})
    # At zero sample variance: lower=delta-14 ln(2/alpha)/(3(n-1)).
    with localcontext() as ctx:
        ctx.prec=60
        c=(Decimal(2)/_decimal_fraction(ALPHA)).ln()
        delta=_decimal_fraction(Fraction(2,100))
        n_det=next(n for n in range(2,10000) if delta > Decimal(14)*c/(Decimal(3)*(n-1)))
        assert n_det>152
    result:dict[str,object]={
        'schema':1,
        'kind':'M13B_D3_MP_THEOREM4_FIXEDTIME_CPU_REVIEW_V1',
        'status':'THEOREM_MATCHED_SYNTHETIC_RESEARCH_ONLY__NO_TEST_SELECTED',
        'gpu_used':False,'real_outcomes_used':False,'model_or_features_loaded':False,
        'science_executable':False,'method_selected':None,'n_selected':None,
        'primary_estimand':'MEAN_OF_PAIRED_SEED_DIFFERENCES_ON_O0',
        'co_primary':['final_average_accuracy','late_task_within_task_mean_accuracy'],
        'owner_mean_effect_2pp_is_power_alternative_only':True,
        'one_sided_null_per_metric':'POPULATION_MEAN_LE_ZERO',
        'alpha_per_endpoint':float(ALPHA),
        'both_needed_positive_rule':'IUT_ONE_GLOBAL_POSITIVE_CLAIM',
        'theorem':'Maurer and Pontil (2009), Theorem 4, applied to 1-(D+1)/2; iid bounded paired differences, fixed n>=2',
        'lower_bound_equation':'sample_mean_D - sqrt(2*sample_variance_D*ln(2/alpha)/n) - 14*ln(2/alpha)/(3*(n-1))',
        'exact_iid_assumption_verified_for_actual_training_seeds':False,
        'numerical_calculation':'60 digit decimal theorem threshold on exact two-point count statistics; rational exact binomial event probabilities, rounded for JSON',
        'pointwise_all_plus_2pp_method_independent_min_n':152,
        'mp_first_n_rejecting_all_plus_2pp_when_variance_zero':n_det,
        'null_synthetic':null_rows,'alternative_synthetic':alt_rows,
        'limitations':[
            'This is a fixed-time theorem for IID bounded differences, not a time-uniform confidence sequence or proof of actual seed IID.',
            'The conservative 14 ln(2/alpha)/(3(n-1)) term makes zero-variance +2pp mean detectable by this specific method only at much larger n than 152.',
            'A few exact two-point nulls cannot certify the theorem; the proof and exact mapping matter.',
            'Synthetic two-point probabilities contain no model evidence and cannot select final D3 method.',
            'The 152 pointwise lower bound pertains only to deterministic rejection of a specific all +2pp dataset, not general power requirements.',
            'Joint power uses identical, independent and opposite signs only for balanced p=1/2 toys, not measured endpoint dependence.',
            'No GPU budget, protocol freeze, power guarantee, final n, or scientific execution is approved.',
        ],
        'references':['Maurer and Pontil COLT 2009 original https://www.learningtheory.org/colt2009/papers/012.pdf',
                      'Waudby-Smith and Ramdas JRSS B 2024 https://doi.org/10.1093/jrsssb/qkad009'],
    }
    result['record_sha256']=sha256(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return result


def main() -> int:
    import sys
    if len(sys.argv)!=1:
        raise SystemExit('No file, result, train, source, GPU, or seed-selection flags permitted')
    print(json.dumps(build_review(),ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
