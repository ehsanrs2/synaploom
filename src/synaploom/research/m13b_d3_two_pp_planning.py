"""CPU-only preregistration planning scenarios for paired mean +2 pp; NO science.

This is a stand-alone mathematical sensitivity study. No data, feature bundle,
model, scientific outcomes, or GPU are read or executed. Not a method choice.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from math import ceil, comb, log, log1p
import json
import sys
from typing import Any

KIND = 'M13B_D3_OWNER_2PP_BUDGET_POWER_SCENARIOS_V1'
MU = Fraction(1, 50)  # +2 percentage points mean in EACH endpoint, planning only
ALPHA = Fraction(1,20)
E_THRESHOLD = 20
SD_SCENARIOS = (Fraction(0), Fraction(1,50), Fraction(1,20), Fraction(1,10))
N_GRID = (10, 20, 40, 80, 152, 200, 300, 400)
MINUTES_PER_TRAJECTORY_SCENARIOS = (3, 10, 30)
FULL_OPTIMIZER_STEPS_PER_TRAJECTORY = 8000
MAX_N_FOR_PROSPECTIVE_PLANNING = 800


def constant_gain_first_reject_n(delta: Fraction) -> int:
    """Exact n threshold for fixed-bet E=(1+delta)^n >= 20."""
    if type(delta) is not Fraction or not 0 < delta <= 1:
        raise ValueError('delta must be rational in (0,1]')
    n = ceil(log(E_THRESHOLD) / log1p(float(delta)))
    while (1+delta)**n < E_THRESHOLD:
        n += 1
    while n > 1 and (1+delta)**(n-1) >= E_THRESHOLD:
        n -= 1
    return n


def least_favorable_two_point_null(delta: Fraction) -> dict[str, Any]:
    """For an all-+delta dataset, construct an exact mean-zero null.

    Under H0, D=+delta with probability 1/(1+delta), otherwise D=-1.
    It has exactly zero mean, satisfies the full accuracy-difference bound,
    and generates the all-positive dataset with probability (1+delta)^-n.
    A deterministic fixed-n size-alpha test that rejects all +delta
    therefore requires n>=ceil(log(1/alpha)/log(1+delta)).
    This is a pointwise worst-null lower bound, NOT a complete n/power bound.
    """
    if type(delta) is not Fraction or not 0 < delta <= 1:
        raise ValueError('delta must be rational in (0,1]')
    p = 1/(1+delta)
    if p*delta+(1-p)*(-1) != 0:
        raise AssertionError('null does not have exactly mean zero')
    n = constant_gain_first_reject_n(delta)
    if not p**(n-1) > ALPHA >= p**n:
        raise AssertionError('minimal rejection count is inconsistent')
    return {
        'positive_difference': float(delta),
        'null_positive_probability': float(p),
        'null_negative_difference': -1,
        'null_population_mean': 0,
        'min_n_for_deterministic_rejection_of_all_constant_positive_deltas': n,
        'null_probability_all_positive_at_previous_n': float(p**(n-1)),
        'null_probability_all_positive_at_first_n': float(p**n),
        'pointwise_lower_bound_not_joint_power': True,
    }


def binomial_half_upper_tail(n: int, k_min: int) -> Fraction:
    """Exact Pr(Binomial(n,1/2)>=k_min) for bounded n."""
    if type(n) is not int or not 0 <= n <= MAX_N_FOR_PROSPECTIVE_PLANNING or type(k_min) is not int:
        raise ValueError('invalid binomial parameters')
    if k_min <= 0:
        return Fraction(1)
    if k_min > n:
        return Fraction(0)
    return Fraction(sum(comb(n, k) for k in range(k_min,n+1)), 2**n)


def two_point_evalue_power_at_n(n: int, mu: Fraction, sigma: Fraction) -> tuple[Fraction, int]:
    """Exact *one endpoint* fixed-lambda=1 e-test power under invented IID +/-sd.

    D=mu+sigma or mu-sigma, each with probability 1/2.
    This is a discrete population with exactly mean mu and SD sigma,
    and is NOT based on any SynapLoom seed differences.
    """
    if type(n) is not int or not 1 <= n <= MAX_N_FOR_PROSPECTIVE_PLANNING:
        raise ValueError('invalid n')
    if type(mu) is not Fraction or type(sigma) is not Fraction or not 0 < mu <= 1 or not 0 <= sigma <= 1:
        raise ValueError('invalid invented effect or sd')
    a, b = mu+sigma,mu-sigma
    if not -1 < b <= a <= 1:
        raise ValueError('toy population outside usable bounded support')
    if sigma == 0:
        k_min = 0 if (1+mu)**n >= E_THRESHOLD else n+1
        return binomial_half_upper_tail(n,k_min), k_min
    log_a, log_b = log1p(float(a)), log1p(float(b))
    threshold = (log(E_THRESHOLD)-n*log_b)/(log_a-log_b)
    k_min = max(0,min(n+1,ceil(threshold)))
    # Hard-check the floating threshold against adjacent events and correct
    # any one-step cancellation at a numerical boundary.
    def qualifies(k: int) -> bool:
        return n*log_b+k*(log_a-log_b)>=log(E_THRESHOLD)
    while k_min <= n and not qualifies(k_min):
        k_min += 1
    while k_min>0 and qualifies(k_min-1):
        k_min -= 1
    return binomial_half_upper_tail(n, k_min),k_min


def first_n_for_marginal_power(sigma: Fraction, target: Fraction) -> int | None:
    """First n in 1..800 attaining target *one-endpoint* toy power.

    No claim that attained power is monotonic at every subsequent n.
    """
    if type(target) is not Fraction or not 0 < target < 1:
        raise ValueError('target must be a rational power target')
    for n in range(1, MAX_N_FOR_PROSPECTIVE_PLANNING+1):
        actual,_ = two_point_evalue_power_at_n(n,MU,sigma)
        if actual>=target:
            return n
    return None


def costs_for_n(n: int) -> dict[str,Any]:
    """Time-only illustrative scenarios, NOT owner RTX 4090 cost estimates."""
    if type(n) is not int or n<1:
        raise ValueError('invalid proposed seed count')
    narrow = 2*n  # proposed external comparison O0 only
    full = 9*n  # original 3 arms x 3 class orders
    return {
        'paired_seed_count': n,
        'narrow_O0_two_arm_trajectories':narrow,
        'original_three_arm_three_order_trajectories_at_same_n':full,
        'narrow_O0_two_arm_optimizer_steps':narrow*FULL_OPTIMIZER_STEPS_PER_TRAJECTORY,
        'full_three_arm_three_order_optimizer_steps':full*FULL_OPTIMIZER_STEPS_PER_TRAJECTORY,
        'hypothetical_minutes_per_complete_trajectory':list(MINUTES_PER_TRAJECTORY_SCENARIOS),
        'narrow_single_gpu_sequential_hours_scenarios':{
            str(m):round(narrow*m/60,3) for m in MINUTES_PER_TRAJECTORY_SCENARIOS
        },
        'full_single_gpu_sequential_hours_scenarios':{
            str(m):round(full*m/60,3) for m in MINUTES_PER_TRAJECTORY_SCENARIOS
        },
        'time_is_unmeasured_full_science_assumption':True,
        'overhead_retries_and_resource_contention_not_included':True,
    }


def report()->dict[str,Any]:
    """No inferential method or sample size is frozen by producing this report."""
    lower = least_favorable_two_point_null(MU)
    powers=[]
    for sd in SD_SCENARIOS:
        n80=first_n_for_marginal_power(sd,Fraction(4,5))
        n90=first_n_for_marginal_power(sd,Fraction(9,10))
        cells=[]
        for n in N_GRID:
            p,k=two_point_evalue_power_at_n(n,MU,sd)
            cells.append({
                'n':n,'positive_count_threshold':None if k>n else k,
                'one_endpoint_power':float(p),
                'two_endpoint_joint_power_frechet_lower':float(max(Fraction(0),2*p-1)),
                'two_endpoint_joint_power_frechet_upper':float(p),
            })
        powers.append({
            'invented_true_paired_mean_each_endpoint':float(MU),
            'invented_paired_sd_each_endpoint':float(sd),
            'invented_distribution':{
                'value_a':float(MU+sd),'value_b':float(MU-sd),'probability_a':0.5,
                'probability_b':0.5,'paired_difference_support_valid':True,
            },
            'first_n_with_at_least_80pct_one_endpoint_power':n80,
            'first_n_with_at_least_90pct_one_endpoint_power':n90,
            'joint_80pct_under_unknown_endpoint_dependence_sufficient_n':n90,
            'sufficiency_scope':'ONLY_THESE_HYPOTHETICAL_IDENTICAL_MARGINAL_DISTRIBUTIONS',
            'selected_n_cells':cells,
        })
    report={
        'kind':KIND,'schema':1,
        'status':'OWNER_2PP_PLANNING_EFFECT_RECORDED__NO_PROTOCOL_OR_GPU_APPROVAL',
        'effect_size_of_interest':{
            'final_average_accuracy_mean_gain_fraction':float(MU),
            'late_task_within_task_mean_accuracy_gain_fraction':float(MU),
            'interpretation':'SCIENTIFICALLY_MEANINGFUL_EFFECT_FOR_POWER_PLANNING_ONLY',
            'statistical_positive_null_per_endpoint':'MEAN_DIFFERENCE_LE_ZERO_UNCHANGED',
            'owner_protocol_frozen':False,
        },
        'scientific_claim':'NONE',
        'real_science_outcomes_loaded':False,
        'data_or_model_loaded':False,
        'optimizer_steps_executed':0,'gpu_used':False,
        'method_selected_for_science':None,'final_seed_count':None,
        'actual_GPU_budget_owner_approved':False,
        'science_executable':False,
        'reference_candidate':{
            'order':'O0','arms':['progressive_reserve','fixed_sparse_mlp'],
            'co_primary':['final_average_accuracy','late_task_within_task_mean_accuracy'],
            'positive_global_rule':'BOTH_VALID_ONE_SIDED_MEAN_TESTS_REJECT',
            'test_examined':'FIXED_LAMBDA_ONE_BOUNDED_MEAN_EVALUE_UNSELECTED',
            'nominal_one_sided_alpha_per_component':float(ALPHA),
            'paired_seeds_candidate':list(range(501,511)),
        },
        'worst_null_pointwise_lower_bound':lower,
        'invented_two_point_powers':powers,
        'hypothetical_single_GPU_sequential_budget_envelopes':[
            costs_for_n(n) for n in (10,20,40,lower['min_n_for_deterministic_rejection_of_all_constant_positive_deltas'])
        ],
        'historical_training_only_calibration_390_steps_per_trajectory':{
            'progressive_reserve_mean_seconds':11.1494,
            'all_recruited_mean_seconds':8.7055,
            'fixed_sparse_mlp_mean_seconds':1.9894,
            'original_eval_and_8000_step_full_science_cost':'NOT_MEASURED',
            'cannot_scale_training_only_directly':True,
        },
        'limitations':[
            'Owner has approved +2 percentage points for planning on both endpoints, NOT method, final protocol, n, budget or GPU execution.',
            'The true power of any method cannot be established from an invented mean and unknown real paired-seed distributions.',
            'Both endpoints share the same 2pp assumed mean and marginal SD within each toy; their dependence remains unknown, so Frechet bounds only.',
            'The first n with marginal power 90pct suffices for a joint lower bound of 80pct only for the assumed toy marginals, not real SynapLoom.',
            'The constant-positive lower bound refers to a deterministic fixed-n test rejecting a specific all-+2pp dataset under the full [-1,1] null class.',
            'All GPU-hours are hypothetical scenarios of 3/10/30 minutes per full trajectory, not measured, not owner-approved and not confidence intervals.',
            'The 390-step train-only engineering calibration is not a 20-epoch protected-evaluation full trajectory cost; no extrapolation is valid.',
            'Reducing from 3 arms x 3 orders to 2 arms x O0 requires a new prospective owner-approved protocol and narrows claim scope.',
            'Fail-closed D2 metric validation is not wired into a science runner and seed IID assumption is not verified.',
            'No science run, final method or change of proposal is authorized by this planning report.',
        ],
    }
    payload=json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
    report['record_sha256']=sha256(payload).hexdigest()
    return report


def main()->int:
    if len(sys.argv)!=1:
        raise SystemExit('Planning only: do not pass result, protocol, GPU or execution flags')
    print(json.dumps(report(),sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
