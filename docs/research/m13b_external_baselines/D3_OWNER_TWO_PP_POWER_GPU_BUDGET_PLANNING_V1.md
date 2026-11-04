# SynapLoom M1.3b / D3 — Owner-selected +2 pp meaningful-effect planning and unmeasured GPU budget

**Date:** 2026-11-04. **Status:** OWNER_2PP_PLANNING_EFFECT_RECORDED / NO TEST SELECTED / NO PROTOCOL FREEZE / NO GPU AUTHORIZATION.

The owner explicitly selected **+2 percentage points** as the *scientifically meaningful mean improvement to use for initial planning* on EACH of the two O0 co-primary endpoints:
- final_average_accuracy;
- late_task_within_task_mean_accuracy.

The comparison remains paired-seed Progressive Reserve minus the in-house Fixed Sparse MLP on the **primary class order O0**. This is a **design effect under hypothetical alternatives**, **not** a newly approved scientific conclusion, an outcome, a minimal null threshold, a selected statistical test or a final power guarantee. The original positive null remains mean difference <=0 separately for each endpoint, with BOTH endpoint tests needed to make a global positive conclusion. The owner has not approved changing the full proposed three-arm × three-order 90-trajectory matrix to O0-only, n or GPU spending.

## A strong mathematical limitation on assumption-light inference

The simplest bounded-mean e-test in merged PR #51 uses E=product(1+D_i), one-sided threshold 20, and has null size <=5% when per-seed paired differences D_i are independent (iid suffices), all within [-1,1], and each has expected mean <=0. It does NOT assume a Gaussian distribution. Its fixed bet is not the only possible valid test, and its power is not automatically adequate.

More importantly, there is a **pointwise worst-null lower bound not specific to this chosen test**. Consider a hypothetical dataset where **every single one** of n seed pairs reports exactly +0.02 accuracy difference. Under the full allowed bounded IID null family, construct

    D = +0.02 with probability 50/51
    D = -1.00 with probability 1/51

This distribution has mean exactly zero, yet produces **the identical all-+0.02 dataset** with probability

    P_null(all +0.02) = (50/51)^n.

For any **nonrandomized fixed-n test that rejects on this specific all-+0.02 dataset**, a universal one-sided 5% type-I guarantee would require (50/51)^n <= 0.05. Exactly:
- at n=151: 0.050277545 > 0.05;
- at n=152: 0.049291711 <= 0.05.

Therefore **n>=152 is necessary if one insists on universally controlled 5% type-I error under only the full [-1,1] support and IID assumptions and wants a deterministic rejection for that particular all-+2 pp observation**. This is an elementary limitation of *any test under those rules*, not merely a flaw of the fixed e-bet. It is **NOT** a universal lower bound for achieving 80% power across every plausible positive-mean alternative, nor an instruction to actually train 152 seeds. It does not directly apply if scientifically justified stronger distributional/support assumptions shrink the null family. Randomized tests have a different pointwise constraint.

This also explains why choosing a tiny n merely to keep the experimental GPU hours affordable is incompatible with strong, unrestricted distribution-free claims about +2 pp average improvement.

## Exact illustrative power: fixed-bet test and invented paired-seed variability

A fully synthetic IID toy alternative sets each endpoint's D_i equal to one of two numbers, with equal probabilities:

    D_i = +0.02 + sigma   with probability 1/2,
    D_i = +0.02 - sigma   with probability 1/2.

Every scenario has population **mean +2 pp**, paired-seed standard deviation **sigma**, and valid accuracy-difference support. It is NOT a measurement or model fit of SynapLoom outcomes.

For each n, the fixed e-test positive-count rejection threshold is obtained from the product E>=20 and the exact single-endpoint power is the corresponding **binomial upper tail**, calculated using integers/rational arithmetic without Monte Carlo. Because the actual dependence of the two accuracy endpoints is unknown, the joint probability of both rejecting is only bounded by **Fréchet**: max(0,p1+p2-1) <= P(both) <= min(p1,p2). With equal marginal toy powers p, a **sufficient** n for 80% *joint* power under any endpoint dependence is one giving **>=90% power per endpoint**. It is sufficient **only under these invented marginals**.

| Invented paired-seed SD | First n with >=80% single-endpoint power | First n with >=90% single-endpoint power, sufficient for >=80% joint power with unknown endpoint dependence |
|---:|---:|---:|
| 0 pp (every seed exactly +2 pp) | **152** | **152** |
| 2 pp (0 or +4 pp equally likely) | **164** | **170** |
| 5 pp (-3 or +7 pp equally likely) | **191** | **210** |
| 10 pp (-8 or +12 pp equally likely) | **294** | **358** |

**Do not present these n values as real SynapLoom sample-size requirements.** They refer to one exceptionally conservative fixed bet and to *specific invented two-point alternatives*, not to measured paired variance or the selected final inference method. More efficient variance-adaptive bounded methods might help with actual variability but cannot eliminate the pointwise worst-null argument without changing its assumptions. Parametric approaches may be less costly only with defensible, prospectively recorded assumptions.

The n in the table is the **first** crossing of a discrete binomial power target. Because finite-n discrete power may oscillate slightly, it must not be interpreted as a monotonic guarantee for every subsequently larger n.

## GPU budget: scenario envelopes, not measurements

The existing full-science design calls for ten tasks × twenty epochs × approximately **8,000 optimizer steps per complete trajectory**, plus original protected evaluation, on a single owner-controlled GPU. Its actual full-science time per model trajectory is **NOT MEASURED**. Historical six owner-run GPU calibrations involved only **390 training-only steps** per run (three arms, two rounds): Progressive Reserve 11.1494 seconds mean, All Recruited 8.7055 seconds, Fixed Sparse 1.9894 seconds. They do not measure the full evaluation workload and **must not** be extrapolated by a naive 8000/390 ratio.

Define a budget scenario **T complete-trajectory minutes** for serial execution on one GPU. Then for n paired seeds:
- O0 only, two-arm primary comparison: **2n complete trajectories**, GPU hours = 2nT/60.
- Original three arms across O0/O1/O2: **9n complete trajectories**, GPU hours = 9nT/60.
- The extra All Recruited arm and O1/O2 robustness are **not silently removed** from the old preregistration: any O0-only plan requires a **new explicitly owner-approved prospective protocol** and tighter claim scope.

Purely hypothetical T=3/10/30 minutes per *complete* trajectory, **not actual measurements or verified upper/lower bounds**:

| Paired seeds | O0 two-arm full trajectories | At T=3 min | At T=10 min | At T=30 min |
|---:|---:|---:|---:|---:|
| 10 | 20 | **1.00 h** | **3.33 h** | **10.00 h** |
| 20 | 40 | **2.00 h** | **6.67 h** | **20.00 h** |
| 40 | 80 | **4.00 h** | **13.33 h** | **40.00 h** |
| 152 | 304 | **15.20 h** | **50.67 h** | **152.00 h** |

For comparison, the **original** 10-seed × 3-arm × 3-order matrix has **90** complete trajectories and would cost 4.5 / 15 / 45 hours under the same invented T=3/10/30 min cases. An n=152 replication of that full matrix would be **1,368 trajectories**, 68.4 / 228 / 684 hours under the same invented assumptions, excluding retry, idle time, startup overhead, and other host contention.

No electricity tariff, cloud GPU rate or absolute financial cost is asserted. These are *GPU-hour scenario envelopes only*; the owner has not specified a GPU-hour spending limit and the fully evaluated workload timing is still unknown. Full-run time and variability require a **separate expressly authorized bounded engineering-only calibration** if an actual budget quote is needed. This planning PR does **not** perform such a calibration or request authorization for it.

## Practical D3 decision table before any protected outcomes

1. **Selected for planning by owner:** +2 pp minimum scientifically important improvement for **each** co-primary mean difference. Keep H0: mu<=0, not mu<=.02. The scientific relevance threshold must not be substituted for the test's null.
2. **Not selected:** actual confirmatory inference method, test assumptions, fixed-seed generating population/independence, final n, joint power target as an owner-frozen requirement, allowed negative-direction claim, failed-run/retry policy and protocol version.
3. **Three valid next paths:**
   - Strict distribution-free mean confirmation: expect potentially substantial paired-seed count; use a qualified method and test-specific joint-power analysis before authorizing GPU.
   - Efficient **assumption-dependent** mean confirmation: justify assumptions without peeking at new confirmatory accuracy and register small-n/skew sensitivity. A naive paired t test is **not** automatically reliable (merged PR #50).
   - Budget-constrained **exploratory** O0 comparison: predeclare n/resources and report paired mean estimates, dispersion, both endpoint values and limitations, **without claiming statistically confirmed superiority** if power/assumptions do not support it.
4. **Owner decisions still required:** a concrete maximum total GPU-hours/wall-clock ceiling and whether the primary ambition is confirmatory or exploratory; these determine the budget constraint but are NOT execution permission. Any O0-only arm/order reduction must be separately approved by prospective amendment.
5. **Remaining D2/D4 gates:** agree preoutcome batch evaluation policy (candidate B=256), complete/raw metric fail-closed validator integrated with real manual runner, seed/feature/model/mask/source hashes, full workload cost measurement, bounded retry and owner explicit run authorization.

## Reproduce without scientific data or GPU

    uv run --locked python -m synaploom.research.m13b_d3_two_pp_planning
    uv run --locked python -m pytest -q tests/unit/test_m13b_d3_two_pp_planning.py

No result, file, feature, experiment, approval or GPU flags are accepted. Exact toy binomial power and the constructed pointwise null are available for transparent independent verification.

Related statistical methodology: [Waudby-Smith and Ramdas, Estimating means of bounded random variables by betting, JRSS B (2024)](https://doi.org/10.1093/jrsssb/qkad009). The code here is **one simple fixed-betting special case**, not the full variance-adaptive approach.

**Disposition:** OWNER_2PP_PLANNING_INPUT_ACCEPTED; EXACT_CONDITIONAL_MATH_FEASIBILITY; TEST_NOT_SELECTED; N_NOT_APPROVED; REAL_FULL_TRAJECTORY_GPU_BUDGET_UNKNOWN; SCIENCE_RUN_NO_GO.
