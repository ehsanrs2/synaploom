"""M1.3b D3 predictable plug-in empirical Bernstein CI: CPU synthetic research.

Theorem 2 & Remark 1, Eq. (16), Waudby-Smith & Ramdas, JRSS B 2024.
For bounded paired differences D in [-1,1], set X=(D+1)/2 in [0,1].
With fixed, preannounced n and alpha=0.05 (two-sided theorem convention),
the predictable bet at time t is:
    lambda_t = min(c, sqrt(2*log(2/alpha)/(n*sigmahat_{t-1}^2))), c=1/2
    muhat_t = (1/2 + sum_{j<=t} X_j)/(t+1)
    sigmahat_t^2 = (1/4 + sum_{j<=t}(X_j-muhat_j)^2)/(t+1).
The closed-form two-sided CS at time n has center sum(lambda*X)/sum(lambda)
and halfwidth [log(2/alpha)+sum((X_t-muhat_{t-1})^2 *
                         (-log(1-lambda_t)-lambda_t))]/sum(lambda).
Its lower endpoint transformed back to D is 2*(center-halfwidth)-1.

Assumptions: X_t has a common conditional mean mu given prior X observations;
iid draws suffice. This is not a proof that real neural training seeds are iid.
No GPU, models, features, protected data, or real outcomes are read by CLI.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

ALPHA = 0.05
CAP = 0.5
N_GRID = (40, 80, 152, 210, 320, 600, 1000)
REPLICATES = 4000
ROOT_SEED = 20261109
LOG_TWO_ALPHA = math.log(2.0 / ALPHA)
LOG_E_THRESHOLD = math.log(1.0 / ALPHA)


@dataclass(frozen=True)
class BoundTrace:
    lower_d: float
    weighted_center_x: float
    radius_x: float
    lambdas: tuple[float, ...]
    past_means_x: tuple[float, ...]
    past_variances_x: tuple[float, ...]


def prpl_eb_trace(deltas: Sequence[float]) -> BoundTrace:
    """Exact mapping to paper's Theorem 2 and fixed-time Eq. (16).

    Inputs are numeric deltas, not actual outcome files. One-sided positive
    inference can conservatively use the lower limit of the two-sided 95% CS.
    """
    if not isinstance(deltas, (list, tuple)) or len(deltas) < 2:
        raise ValueError('requires >=2 fixed-n paired differences')
    xs = []
    for d in deltas:
        if type(d) not in (float, int) or not math.isfinite(d) or d < -1 or d > 1:
            raise ValueError('paired differences must be finite fractions in [-1,1]')
        xs.append((float(d) + 1.0) / 2.0)
    n = len(xs)
    sum_x = 0.0
    sum_var_terms = 0.0
    sum_lam = 0.0
    weighted_x = 0.0
    sum_penalties = 0.0
    lambdas: list[float] = []
    past_means: list[float] = []
    past_variances: list[float] = []
    for index, x in enumerate(xs):
        # index previous observations; i=index+1, denom t-1+1=index+1
        mu_prev = (0.5 + sum_x) / (index + 1)
        var_prev = (0.25 + sum_var_terms) / (index + 1)
        lam = min(CAP, math.sqrt(2.0 * LOG_TWO_ALPHA / (n * var_prev)))
        if not 0 < lam < 1:
            raise AssertionError('non-predictable/out-of-range stake')
        err = x - mu_prev
        correction = -math.log1p(-lam) - lam
        sum_penalties += err * err * correction
        weighted_x += lam * x
        sum_lam += lam
        lambdas.append(lam)
        past_means.append(mu_prev)
        past_variances.append(var_prev)
        sum_x += x
        mu_after = (0.5 + sum_x) / (index + 2)
        sum_var_terms += (x - mu_after) ** 2
    center_x = weighted_x / sum_lam
    radius_x = (LOG_TWO_ALPHA + sum_penalties) / sum_lam
    return BoundTrace(
        lower_d=2 * (center_x - radius_x) - 1,
        weighted_center_x=center_x, radius_x=radius_x,
        lambdas=tuple(lambdas), past_means_x=tuple(past_means),
        past_variances_x=tuple(past_variances),
    )


def prpl_eb_lower(deltas: Sequence[float]) -> float:
    return prpl_eb_trace(deltas).lower_d


def prpl_eb_rejects(deltas: Sequence[float]) -> bool:
    return prpl_eb_lower(deltas) > 0.0


def both_needed(d1: Sequence[float], d2: Sequence[float]) -> bool:
    if len(d1) != len(d2):
        raise ValueError('both endpoints must have the same registered paired seeds')
    l1, l2 = prpl_eb_lower(d1), prpl_eb_lower(d2)
    return l1 > 0 and l2 > 0


def _prpl_eb_vectorized(batch):
    """Independent NumPy batch implementation for synthetic Monte Carlo only."""
    import numpy as np
    if not (isinstance(batch, np.ndarray) and batch.ndim == 2 and batch.shape[1] >= 2
            and np.isfinite(batch).all() and np.all(batch >= -1) and np.all(batch <= 1)):
        raise ValueError('expected finite synthetic 2D array of bounded differences')
    reps, n = batch.shape
    x = (batch + 1.0) / 2.0
    sums = np.zeros(reps)
    var_terms = np.zeros(reps)
    weighted = np.zeros(reps)
    penalties = np.zeros(reps)
    sum_lam = np.zeros(reps)
    for index in range(n):
        mean_prev = (0.5 + sums) / (index + 1)
        variance_prev = (0.25 + var_terms) / (index + 1)
        lam = np.minimum(CAP, np.sqrt(2.0 * LOG_TWO_ALPHA / (n * variance_prev)))
        val = x[:, index]
        err = val - mean_prev
        penalties += err * err * (-np.log1p(-lam) - lam)
        weighted += lam * val
        sum_lam += lam
        sums += val
        mean_after = (0.5 + sums) / (index + 2)
        var_terms += (val - mean_after) ** 2
    return 2.0 * ((weighted - LOG_TWO_ALPHA - penalties) / sum_lam) - 1


def _fixed_bet_vectorized(batch):
    import numpy as np
    with np.errstate(divide='ignore'):
        return np.sum(np.log1p(batch), axis=1) >= LOG_E_THRESHOLD


def _wilson_95(k: int, n: int) -> tuple[float, float]:
    if n < 1 or not 0 <= k <= n:
        raise ValueError('invalid binomial count')
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2*n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z*z/(4*n*n)) / denom
    return (0.0 if k == 0 else max(0.0, center - spread),
            1.0 if k == n else min(1.0, center + spread))


def _sample_batch(rng, p: float, a: float, b: float, reps: int, n: int):
    import numpy as np
    return np.where(rng.random((reps, n)) < p, a, b)


def _cell(spec_index: int, name: str, p: float, positive: float, negative: float,
          n: int, *, joint: bool, reps: int = REPLICATES) -> dict[str, object]:
    import numpy as np
    rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, spec_index, n, reps]))
    batch = _sample_batch(rng, p, positive, negative, reps, n)
    pr = _prpl_eb_vectorized(batch) > 0
    fixed = _fixed_bet_vectorized(batch)
    pr_k = int(pr.sum())
    fixed_k = int(fixed.sum())
    cell: dict[str, object] = {
        'n': n, 'replicates': reps,
        'prpl_eb_rejections': pr_k,
        'prpl_eb_probability_mc': pr_k / reps,
        'prpl_eb_mc_wilson95': list(_wilson_95(pr_k, reps)),
        'fixed_lambda1_rejections': fixed_k,
        'fixed_lambda1_probability_mc': fixed_k / reps,
        'fixed_lambda1_mc_wilson95': list(_wilson_95(fixed_k, reps)),
    }
    if joint:
        if p != 0.5:
            raise ValueError('opposite-sign coupling requires p=0.5')
        signs = batch == positive
        independent = _sample_batch(rng, p, positive, negative, reps, n)
        opposite = np.where(signs, negative, positive)
        for name_method, rejects_one, func in (
            ('prpl_eb', pr, lambda y: _prpl_eb_vectorized(y) > 0),
            ('fixed_lambda1', fixed, _fixed_bet_vectorized),
        ):
            second_ind = func(independent)
            second_opp = func(opposite)
            cell[name_method + '_joint_identical_mc'] = float(rejects_one.mean())
            cell[name_method + '_joint_independent_mc'] = float(np.mean(rejects_one & second_ind))
            cell[name_method + '_joint_opposite_signs_mc'] = float(np.mean(rejects_one & second_opp))
    return cell


def build_review(reps: int = REPLICATES, n_grid: tuple[int, ...] = N_GRID) -> dict[str, object]:
    if type(reps) is not int or not 50 <= reps <= 20000:
        raise ValueError('synthetic Monte Carlo replicates out of bounded range')
    if not isinstance(n_grid, tuple) or not n_grid or any(type(n) is not int or n < 2 or n > 1500 for n in n_grid):
        raise ValueError('invalid bounded synthetic n-grid')
    nulls = (
        ('balanced', 0.5, 0.02, -0.02),
        ('p80', 0.8, 0.04, -0.16),
        ('p90', 0.9, 0.02, -0.18),
        ('p96', 0.96, 0.02, -0.48),
        ('rare_extreme_loss', 50/51, 0.02, -1.0),
    )
    alts = (
        ('mean2pp_sd2pp', 0.5, 0.04, 0.0),
        ('mean2pp_sd5pp', 0.5, 0.07, -0.03),
        ('mean2pp_sd10pp', 0.5, 0.12, -0.08),
        ('mean2pp_rare_loss', 0.96, 0.06, -0.94),
    )
    rows_null = []
    rows_alt = []
    for index, (name, p, a, b) in enumerate(nulls + alts):
        mean = p*a + (1-p)*b
        expected = 0.0 if index < len(nulls) else 0.02
        assert abs(mean - expected) <= 1e-14
        cells = [_cell(index, name, p, a, b, n, joint=index >= len(nulls) and p==0.5, reps=reps) for n in n_grid]
        row = {'scenario':name,'constructed_true_mean':mean,'positive':a,'negative':b,
               'probability_positive':p,'cells':cells}
        if index < len(nulls):
            rows_null.append(row)
        else:
            rows_alt.append(row)
    positive_constant = [
        {'n':n, 'prpl_eb_lower':prpl_eb_lower([0.02]*n),
         'fixed_lambda1_rejects':n*math.log1p(0.02)>=LOG_E_THRESHOLD}
        for n in n_grid
    ]
    # Source identity and no-science contract encoded in deterministic report.
    report: dict[str, object] = {
        'schema':1,'kind':'M13B_D3_PRPL_EB_THEOREM2_EQ16_CPU_V1',
        'status':'CPU_SYNTHETIC_RESEARCH_ONLY__NO_METHOD_OR_N_APPROVAL',
        'science_executable':False,'gpu_used':False,'feature_or_model_loaded':False,
        'real_seed_accuracy_used':False,'method_selected':None,'n_selected':None,
        'claim':'POSITIVE_POPULATION_PAIRED_MEAN_BOTH_ENDPOINTS_O0',
        'scientific_mean_effect_fraction_for_planning_only':0.02,
        'null_per_endpoint':'MEAN_PAIRED_DIFFERENCE_LE_ZERO',
        'one_global_both_needed_iut':True,
        'paper_reference':'Waudby-Smith and Ramdas JRSS B 2024, Theorem 2, Remark 1, Equation (16)',
        'paper_url':'https://doi.org/10.1093/jrsssb/qkad009',
        'theorem_alpha_two_sided':ALPHA,
        'log_two_over_alpha':LOG_TWO_ALPHA,
        'fixed_n_predictable_c':CAP,
        'bet_policy':'lambda_t=min(c,sqrt(2log(2/alpha)/(n sigmahat^2_{t-1})))',
        'past_mean_policy':'muhat_t=(1/2+sum_j<=t X_j)/(t+1), X=(D+1)/2',
        'past_var_policy':'sigmahat_t^2=(1/4+sum_j<=t (X_j-muhat_j)^2)/(t+1)',
        'lower_bound_d':'2*((sum lambda_i X_i - log(2/alpha) - sum (X_i-muhat_i-1)^2*(-log(1-lambda_i)-lambda_i))/sum lambda_i)-1',
        'distribution_validity':'Each X_i in [0,1] and a common conditional mean mu across all i; IID is sufficient but not verified for training seeds.',
        'n_grid':list(n_grid),'replicates':reps,'mc_root_seed':ROOT_SEED,
        'mc_rng':'numpy Generator default_rng SeedSequence([20261109, scenario_index, n, reps]); independent separate synthetic replications',
        'mc_interval':'Wilson 95% interval for estimated synthetic rejection frequency, NOT scientific uncertainty.',
        'constructed_zero_mean_nulls':rows_null,
        'constructed_2pp_alternatives':rows_alt,
        'all_plus_2pp_constant_case':positive_constant,
        'limitations':[
            'Two-sided theorem log(2/alpha) used for one-sided lower-bound test: conservative; no unproven log(1/alpha) replacement.',
            'MC rejection probabilities are estimates with MC error; exact theorem supplies type-I control if assumptions hold.',
            'Synthetic independent/identical/opposite endpoint couplings cannot establish dependence between real metrics.',
            'This does not measure actual paired mean, SD, power, and does not prove IID neural training seed population.',
            'Fixed-n tuning depends only on registered n and past observations; no outcome-based method/seed selection is authorized.',
            'Existing 152 n pointwise restriction applies only to the specific all+2pp dataset under full bounded null, not any test general power.',
            'No training/GPU/protected evaluation/source or protocol changes, no final scientific result or run authorization.',
        ],
    }
    payload=json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    report['record_sha256']=hashlib.sha256(payload).hexdigest()
    return report


def main() -> int:
    import sys
    if len(sys.argv)!=1:
        raise SystemExit('CPU synthetic report only: no file, result, seed, GPU or training flags')
    print(json.dumps(build_review(),sort_keys=True,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
