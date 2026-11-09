"""D3 bounded paired-MEAN mixture betting: REVIEW-ONLY, synthetic CPU.

This module does not load benchmark data, features, results, or models. It does
not train or select a science method. For D_i in [-1,1], H0: E[D_i | past]<=0,
K_lambda = product_i(1 + lambda D_i) is a nonnegative supermartingale for
predetermined 0<=lambda<=1. A fixed convex mixture of these K values is an
E-value (E<=1 under H0). Reject at E>=1/alpha at a fixed registered n.

Mixture stakes adapt *implicitly* through wealth weighting; this is a finite
mixture of constant bets, NOT the paper's optimized PrPl/GRAPA methods, NOT a
confidence interval, and NOT a claim of sufficient statistical power.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
from math import comb, fsum, isfinite, log, log1p, exp
from typing import Sequence

ALPHA = Fraction(1, 20)
THRESHOLD = Fraction(20)
BET_GRID = (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1))
N_GRID = (40, 80, 152, 210, 320)


def _deltas(xs: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(xs, (list, tuple)) or not xs:
        raise ValueError("nonempty list/tuple of paired differences required")
    out = []
    for x in xs:
        if type(x) not in (int, float) or not isfinite(float(x)) or not -1 <= x <= 1:
            raise ValueError("paired differences must be finite numeric fractions in [-1,1]")
        out.append(float(x))
    return tuple(out)


def _logsumexp(zs: Sequence[float]) -> float:
    m = max(zs)
    if m == float('-inf'):
        return m
    return m + log(fsum(exp(z-m) for z in zs))


def log_evalues(xs: Sequence[float]) -> tuple[float, float]:
    """Return (log mixed, log lambda=1), with all inputs checked first.

    The method is fixed before outcomes and symmetric in seed order; the two
    co-primary endpoint arrays must be checked/paired by a separate runner.
    """
    ds = _deltas(xs)
    logs = []
    for lam in BET_GRID:
        v = float(lam)
        logs.append(float('-inf') if v == 1 and -1.0 in ds else fsum(log1p(v*d) for d in ds))
    return _logsumexp(logs)-log(len(BET_GRID)), logs[-1]


def rejects(log_e: float) -> bool:
    if type(log_e) not in (float, int) or not (isfinite(float(log_e)) or log_e == float('-inf')):
        raise ValueError("invalid E value")
    return float(log_e) >= log(float(THRESHOLD))


def both_needed(xs: Sequence[float], ys: Sequence[float]) -> bool:
    if len(xs) != len(ys):
        raise ValueError("co-primary arrays must contain the same paired seeds")
    a = log_evalues(xs)[0]
    b = log_evalues(ys)[0]
    return rejects(a) and rejects(b)


def _wealth_exact(n: int, k: int, a: Fraction, b: Fraction, *, mixed: bool) -> Fraction:
    if not (type(n) is int and 1 <= n <= 500 and type(k) is int and 0 <= k <= n and
            type(a) is Fraction and type(b) is Fraction and -1 <= b < a <= 1):
        raise ValueError("invalid exact two-point input")
    grid = BET_GRID if mixed else (Fraction(1),)
    return sum(((1+lam*a)**k * (1+lam*b)**(n-k) for lam in grid), Fraction(0)) / len(grid)


def positive_count_threshold(n: int, a: Fraction, b: Fraction, *, mixed: bool) -> int:
    """Smallest number of positive draws for exact E>=20; n+1 if none."""
    if _wealth_exact(n, n, a, b, mixed=mixed) < THRESHOLD:
        return n+1
    lo, hi = 0, n
    while lo < hi:
        mid = (lo+hi)//2
        if _wealth_exact(n, mid, a, b, mixed=mixed) >= THRESHOLD:
            hi = mid
        else:
            lo = mid+1
    return lo


def exact_binomial_event(n: int, p: Fraction, k_lower: int, k_upper: int | None = None) -> Fraction:
    """Exact finite-n probability, no Monte Carlo or asymptotic approximation."""
    if type(n) is not int or not 0 <= n <= 500 or type(p) is not Fraction or not 0 <= p <= 1:
        raise ValueError("invalid binomial parameters")
    if type(k_lower) is not int or k_upper is not None and type(k_upper) is not int:
        raise ValueError("invalid binomial event")
    high = n if k_upper is None else min(n, k_upper)
    return sum((Fraction(comb(n, k))*p**k*(1-p)**(n-k)
                for k in range(max(0,k_lower), high+1)), Fraction(0))


def _cells_for(a: Fraction, b: Fraction, p: Fraction) -> list[dict[str, object]]:
    assert -1 <= b < a <= 1 and p*a+(1-p)*b <= 1
    out = []
    for n in N_GRID:
        fm = positive_count_threshold(n, a, b, mixed=False)
        hm = positive_count_threshold(n, a, b, mixed=True)
        out.append({
            'n': n, 'fixed_lambda1_k_min': fm if fm<=n else None,
            'hedged_mixture_k_min': hm if hm<=n else None,
            'fixed_lambda1_probability': float(exact_binomial_event(n,p,fm)),
            'hedged_mixture_probability': float(exact_binomial_event(n,p,hm)),
        })
    return out


def build_review() -> dict[str, object]:
    """Deterministic, exact synthetic two-point calibration and joint power."""
    null_families = [
        ('balanced', Fraction(1,2), Fraction(2,100), Fraction(-2,100)),
        ('skew_p80', Fraction(4,5), Fraction(4,100), Fraction(-16,100)),
        ('skew_p90', Fraction(9,10), Fraction(2,100), Fraction(-18,100)),
        ('skew_p96', Fraction(24,25), Fraction(2,100), Fraction(-48,100)),
        ('least_favorable_all_plus_2pp', Fraction(50,51), Fraction(2,100), Fraction(-1)),
    ]
    nulls = []
    for label,p,a,b in null_families:
        assert p*a + (1-p)*b == 0
        cells=_cells_for(a,b,p)
        assert all(c['fixed_lambda1_probability']<=float(ALPHA)+1e-14
                   and c['hedged_mixture_probability']<=float(ALPHA)+1e-14 for c in cells)
        nulls.append({'label':label,'p_positive':float(p),'positive':float(a),
                      'negative':float(b),'mean_exactly_zero':True,'cells':cells})
    alts=[]
    for label,p,a,b in [
        ('mean_2pp_sd_2pp',Fraction(1,2),Fraction(4,100),Fraction(0)),
        ('mean_2pp_sd_5pp',Fraction(1,2),Fraction(7,100),Fraction(-3,100)),
        ('mean_2pp_sd_10pp',Fraction(1,2),Fraction(12,100),Fraction(-8,100)),
        ('mean_2pp_rare_large_loss',Fraction(24,25),Fraction(6,100),Fraction(-94,100)),
    ]:
        assert p*a+(1-p)*b == Fraction(2,100)
        cells=_cells_for(a,b,p)
        if p==Fraction(1,2):
            for c in cells:
                n=c['n']
                for method,kkey in [('fixed_lambda1','fixed_lambda1_k_min'),('hedged_mixture','hedged_mixture_k_min')]:
                    km=c[kkey] if c[kkey] is not None else n+1
                    marginal=exact_binomial_event(n,p,km)
                    c[f'{method}_joint_power_identical'] = float(marginal)
                    c[f'{method}_joint_power_independent'] = float(marginal*marginal)
                    c[f'{method}_joint_power_opposite_signs'] = float(exact_binomial_event(n,p,km,n-km))
        alts.append({'label':label, 'p_positive':float(p),'positive':float(a),'negative':float(b),
                     'mean_exactly_2pp':True,'cells':cells})
    p=Fraction(50,51)
    pointwise=next(n for n in range(1,501) if p**n<=ALPHA)
    assert pointwise==152 and p**151>ALPHA
    out={
        'schema':1,'kind':'M13B_D3_FIXEDTIME_HEDGED_MEAN_CPU_V1',
        'status':'SYNTHETIC_METHOD_FEASIBILITY_ONLY__NO_SCIENCE_SELECTION',
        'science_executable':False,'gpu_used':False,'real_model_or_protected_outcomes_used':False,
        'method_selected':None,'approved_seed_count':None,'owner_budget_hours':None,
        'estimand':'POPULATION_MEAN_OF_PAIRED_SEED_DIFFERENCES_O0',
        'coprimary':['final_average_accuracy','late_task_within_task_mean_accuracy'],
        'alpha_one_sided_per_endpoint':float(ALPHA),'joint_rule':'IUT_BOTH_MIXED_EVALUES_GTE_20',
        'bet_grid':[float(v) for v in BET_GRID], 'bet_weights':'UNIFORM_FIXED_PRE_OUTCOME',
        'type_i_proof':'For each fixed lambda in [0,1], (1+lambda D_i) is nonnegative and has conditional expectation <=1 when conditional E[D_i|past]<=0. Each product is a nonnegative supermartingale. Convex mixing preserves E<=1. Markov P(E>=20)<=0.05. IUT both endpoints controls the union null.',
        'not_claimed':'Not WSR GRAPA/PrPl, not a confidence interval, not guaranteed more powerful than lambda=1.',
        'null_families':nulls,'alternative_mean_2pp_families':alts,
        'pointwise_all_plus_2pp_deterministic_min_n_for_ANY_valid_nonrandomized_test':pointwise,
        'pointwise_scope':'Specific all-+.02 sample under full iid [-1,1] mean<=0 null; NOT universal required n or joint power.',
        'joint_couplings':'Identical=shared signs; independent=independent signs; opposite=second endpoint reverses each first-endpoint sign, valid only for symmetric p=1/2 toys.',
        'limitations':[
            'Independent seeded training-run population/distribution and failure-policy remain unverified.',
            'Tests concern synthetic exact two-point families, not real SynapLoom paired differences.',
            'Mixture guarantees size but power may be lower than fixed bet on low-variance gains; do not select from actual protected outcomes.',
            'Rounding float percentages in report is for display only; exact rational arithmetic is used to decide thresholds and probabilities.',
            'One global IUT positive claim is distinct from simultaneous marginal 95% coverage for two reported intervals.',
            'Old preregistration bootstrap contract must be amended prospectively; no change/freeze approved.',
            'No runner, masked-optimizer qualification, new model training, GPU, external file inputs, or science execution is authorized.',
        ],
        'reference':'Waudby-Smith and Ramdas JRSS B 2024 DOI 10.1093/jrsssb/qkad009 (general betting/mixture framework, not exact algorithm reproduction)'
    }
    out['record_sha256']=sha256(json.dumps(out,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return out


def main() -> int:
    import sys
    if len(sys.argv)!=1:
        raise SystemExit('CPU synthetic review only; no result/file/seed/GPU options')
    print(json.dumps(build_review(),indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
