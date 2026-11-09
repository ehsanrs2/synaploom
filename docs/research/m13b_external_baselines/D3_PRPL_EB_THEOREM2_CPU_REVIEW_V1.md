# M1.3b D3 — predictable plug-in empirical Bernstein (PrPl-EB), theorem-mapped CPU research V1

**Research date:** 2026-11-09. **Status:** `D3_PRPL_EB_CPU_METHOD_REVIEW_READY__NO_FINAL_METHOD`. Starts from merged `main=32f06c42263cc22d8c3e89bbca0cb40015625dad` after [PR #56](https://github.com/ehsanrs2/synaploom/pull/56) (merged; hosted CI PASS), and follows [PR #54](https://github.com/ehsanrs2/synaploom/pull/54) and [PR #55](https://github.com/ehsanrs2/synaploom/pull/55). **No GPU, protected accuracy, feature bundle or model data was used.**

## Scientific target and scope

The owner-selected estimand is the **population mean** of paired, independent training-seed differences `D_i=metric(progressive_reserve,O0,seed_i)-metric(fixed_sparse_mlp,O0,seed_i)`, with **BOTH** co-primary metrics:
- `final_average_accuracy`;
- `late_task_within_task_mean_accuracy`.

The `+0.02` accuracy-fraction gain on **each** metric is a **planning alternative**, not an observed effect and not the null. Positive null per endpoint is `mu_D<=0`; the single formal global positive claim requires **both** one-sided component rejections (intersection-union test/IUT), if each marginal method is valid. Results do **not** imply superiority over RigL/NISPA or causal recruitment-timing attribution.

## Exact source-to-code mapping

Primary source: Waudby-Smith & Ramdas, *Estimating means of bounded random variables by betting*, JRSS B 86(1), 2024, **Theorem 2, Remark 1 and equation (16)**: https://doi.org/10.1093/jrsssb/qkad009 . A publicly available typeset original with the equations is https://rss.org.uk/RSS/media/File-library/Events/Discussion%20meetings/Estimating-means-of-bounded-random-variables-by-betting.pdf (PDF pp. 11–12).

The theorem's original random observations are `X_i in [0,1]`. For our bounded paired differences `D_i in [-1,1]`, set `X_i=(D_i+1)/2`. At planned **fixed n**, alpha=.05 and the paper's default `c=.5` (not tuned to observations), implement its equation (16) with **only past values available at round i**:

    muhat_t = (0.5 + sum_{j<=t} X_j)/(t+1)
    sigmahat_t^2 = (0.25 + sum_{j<=t}(X_j - muhat_j)^2)/(t+1)
    lambda_i = min(0.5, sqrt(2*log(2/alpha)/(n*sigmahat_{i-1}^2)))

The variance component is `v_i=4*(X_i-muhat_{i-1})^2` and `psi_E(lambda)=(-log(1-lambda)-lambda)/4`. Thus Theorem 2's **two-sided** interval for X at t=n has

    weighted_center_X = sum_i lambda_i*X_i / sum_i lambda_i
    radius_X = (log(2/alpha) +
        sum_i (X_i-muhat_{i-1})^2*(-log(1-lambda_i)-lambda_i)) /
        sum_i lambda_i
    lower_D = 2*(weighted_center_X-radius_X)-1

The candidate conservatively declares **one-sided positive mean evidence** only if `lower_D>0`. We deliberately **preserve log(2/alpha) from the two-sided theorem**, rather than silently substituting `log(1/alpha)`. Running intersections across earlier t and order-averaged permutation methods were **not** implemented; a final-time theorem interval is used. This is **PrPl-EB equation (16)**, not GRAPA, a fixed lambda mixture, or the Maurer–Pontil 2009 inequality.

**Validity boundary:** Under the original theorem, every `X_i` must be in [0,1] and have the **same conditional mean** `E[X_i|X_1,...,X_{i-1}]=mu` (IID is sufficient). The theorem supports a valid two-sided 1-alpha interval; taking just its lower endpoint is valid for the positive mean-null, conservatively. Both-needed IUT controls one global positive claim at <=alpha if each component's assumptions hold. This does not imply two simultaneously reported marginal 95% intervals and does not establish IID training-seed randomness or ignore failures.

## CPU implementation and reproducibility

- `src/synaploom/research/m13b_d3_prpl_eb_cpu.py` — scalar method, independent vectorized NumPy batch method for fabricated scenarios, deterministic Monte Carlo report with SHA-256 checksum. CLI is parameterless and contains no path/results/GPU/protected outcome interface.
- `tests/unit/test_m13b_d3_prpl_eb_cpu.py` — 8 CPU-only tests for theorem constants, strictly past-only predictable coefficients, scalar/vector parity, invalid input rejection, IUT behavior, Wilson MC intervals, deterministic seeds/JSON identity, and explicit science NO-GO flags.
- Locally tested exact source/test blobs, **8 passed**. Hosted GitHub Actions exact-head result must be checked separately.

Default report: **4,000 independently generated synthetic replications per cell** with NumPy `SeedSequence([20261109, scenario_index, n, reps])`; fixed n-grid 40, 80, 152, 210, 320, 600, 1000. Synthetic mean-zero nulls: balanced +/-2pp, skew p=.8/.9/.96, and rare catastrophic loss `P(+2pp)=50/51, P(-100pp)=1/51`. Synthetic +2pp mean alternatives: SD2pp `{0,+4pp}`, SD5pp `{-3pp,+7pp}`, SD10pp `{-8pp,+12pp}` all with p=.5, and `96% +6pp / 4% -94pp`. No distribution is claimed to fit actual SynapLoom.

Reported Monte Carlo rejection rates include Wilson 95% **simulation uncertainty intervals**; they are NOT scientific confidence bounds or observed accuracy intervals. The existing fixed `lambda=1` e-bet is a separate valid comparator, unchanged.

## Synthetic power findings (single endpoint; Monte Carlo, 4,000 replications each)

| Constructed mean +2pp IID scenario | Paired n | PrPl-EB MC rejection | Fixed lambda=1 MC rejection |
|---|---:|---:|---:|
| SD2pp (0 / +4pp) | 210 | **0%** | **100%** |
| SD5pp (-3 / +7pp) | 210 | **0%** | **90.6%** |
| SD10pp (-8 / +12pp) | 210 | **0%** | **53.55%** |
| SD5pp (-3 / +7pp) | 600 | **0.575%** | **100%** |
| SD5pp (-3 / +7pp) | 1000 | **99.85%** | **100%** |
| SD10pp (-8 / +12pp) | 1000 | **84.975%** | **99.975%** |
| Rare large loss (96% +6pp, 4% -94pp) | 600 | **8.375%** | **0.25%** |
| Rare large loss (96% +6pp, 4% -94pp) | 1000 | **40.8%** | **0.025%** |

These are **synthetic random estimates**, not exact binomial-power calculations; even `0/4000` means **no rejections observed**, not exactly zero true power. In all 35 listed synthetic null cells, the PrPl-EB Monte Carlo sample had **zero observed false rejections**. This does not establish true size or theorem correctness by simulation; type-I assurance comes from the theorem under its assumptions. At n=1000, invented SD10pp endpoint correlation schemes give PrPl-EB both-needed joint rejection **84.975%** if identical, **73.2%** if independent, and **68.95%** for opposite per-seed signs. Real between-endpoint dependence is unmeasured.

For a **constant observed +2pp on every seed**, this exact selected PrPl-EB lower endpoint remains negative at n=600 but becomes positive by n=1000. This is *method-specific*. The broader nonrandomized full-bounded-null **n>=152 pointwise all+2pp observation rejection restriction** remains logically separate; neither number is a universal required n for real desired joint power.

At the owner-local D4 single-pair engineering rate `165.843s`, extrapolating n1000 for only O0/two arms is ~46.07 GPU-hours (not a measured upper bound; 1.5x contingency still unvalidated). Current original 3-arm ×3-order candidate has not been reduced, so these costs are not a priced approved protocol.

## Method screening decision and next work

The **Theorem-2/Eq-16 implementation is an academically grounded candidate** whose conditional type-I proof is credible, but it is **not a practical winner** in the investigated low-variance, budget-limited +2pp synthetic range: even at n210, its estimated marginal power was zero across the three symmetric alternatives, whereas the earlier fixed-bet did better under those **invented** assumptions. Conversely, the PrPl candidate can outperform the fixed full-stake bet under some severe-tail alternatives. These tradeoffs are sensitive to unknown, actually unobserved paired-seed distributions; no method may be selected after inspecting protected outcomes.

**D3 method screening:** fixed-bet, fixed hedged mixture, Maurer–Pontil bound, and exact PrPl-EB now have documented synthetic screening. **Final method, assumptions, seed-generating population, n, and useful two-endpoint joint power remain PENDING.** To avoid endless uninformative CPU method swaps, the next project work should prioritize **D2's exact-runner/guard dry run on CPU and nonprotected tensors**, followed by prospective D1 scope and D4 total-hour decision. Before a confirmatory GO, the owner still must choose a pre-outcome D3 method/power target with honest assumptions; if infeasible, explicitly label the study *exploratory*.

D2 integration needs frozen B256/B232 grouping/membership, fail-closed 8,000-step and ten-task/20-epoch observations, seed/mask/optimizer/source/feature provenance and failure preservation. **No scientific GPU work, protected outcomes, final statistical method/n, preregistration freeze or execution authorization is granted by this PR.**

## Safe local commands

    uv run --locked python -m pytest -q tests/unit/test_m13b_d3_prpl_eb_cpu.py
    uv run --locked python -m synaploom.research.m13b_d3_prpl_eb_cpu

**Disposition:** `D3_PRPL_EB_SYNTHETIC_METHOD_SCREEN_RECORDED`, `D3_FINAL_METHOD_N_NOT_FROZEN`, `D2_CPU_NEXT`, `SCIENTIFIC_GPU_NO_GO`.
