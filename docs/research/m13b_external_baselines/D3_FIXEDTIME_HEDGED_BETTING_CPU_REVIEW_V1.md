# SynapLoom M1.3b / D3 — fixed-time hedged bounded-mean betting, CPU-only (V1)

**Status:** `D3_SYNTHETIC_METHOD_FEASIBILITY__NOT_SELECTED`. This is a research-only continuation of [Issue #29](https://github.com/ehsanrs2/synaploom/issues/29) and [D1–D4 integrated Draft PR #54](https://github.com/ehsanrs2/synaploom/pull/54), **not** a frozen preregistration, a GPU runner or new scientific accuracy evidence.

## Exact claim and evidence boundary

The owner's estimand is the **population mean of paired training-seed accuracy differences** `D_i = progressive_reserve_i - fixed_sparse_mlp_i` on frozen **O0**, for BOTH `final_average_accuracy` and `late_task_within_task_mean_accuracy`. The **+0.02 accuracy fraction** is a meaningful *alternative* for power planning on each metric; it is **not** the null and not an observed effect. Each one-sided positive null is `E[D] <= 0`. A positive *global* claim requires both valid component rejections (intersection–union rule, IUT). No formal negative-direction claim is defined.

The real seed-difference population, distribution and independence are unknown. One owner-verified D4 engineering cost observation per arm does not provide power, variance, protected evaluation accuracy or runtime tail guarantees. The original 10-seed percentile-bootstrap preregistration is not frozen and has not been silently replaced.

## Candidate mathematical method: fixed convex mixture of betting wealths

For any endpoint and fixed n, under bounded differences `D_i in [-1,1]`, predefine five stakes `lambda = 1/8, 1/4, 1/2, 3/4, 1` and uniform weights `1/5`.

- `K_lambda(n) = PRODUCT_i (1 + lambda * D_i)`
- `E_mix(n) = (1/5) SUM_lambda K_lambda(n)`
- At `alpha=0.05`, reject that one-sided mean-null if `E_mix >= 20`. Reject the *global* positive claim only when BOTH endpoints cross 20.

**Finite-sample guarantee:** If for every i the paired differences satisfy `D_i in [-1,1]` and `E[D_i | D_1,...,D_(i-1)] <= 0` under the null, each `1+lambda*D_i` is nonnegative, conditionally has expected value <=1, and `K_lambda` is a nonnegative supermartingale. A nonnegative precommitted mixture of these is again a supermartingale with initial wealth 1. By Markov's inequality `P_H0(E_mix >= 20) <= 0.05`. If either endpoint null is true, global rejection is a subset of false rejection on that endpoint, so the IUT controls the **single global positive claim** at <=0.05 regardless of between-endpoint dependence, assuming each component null model is valid. This does **not** automatically guarantee simultaneous marginal 95% confidence intervals.

The stake mixture has an implicit posterior/wealth adaptation to the realized distribution, but it is **NOT** a reproduction of the optimized variance-adaptive PrPl empirical-Bernstein or GRAPA procedures in the primary paper; it is **not** a confidence interval and **does not promise greater power** than the earlier `lambda=1` e-test. A fixed equal-weight grid can sacrifice power where the full stake is best.

**Primary method reference:** Waudby-Smith and Ramdas, *Estimating means of bounded random variables by betting*, JRSS B 86(1), 2024: https://doi.org/10.1093/jrsssb/qkad009 . The simple mixture and proof above are self-contained special cases, not a claimed exact paper implementation. Complementary nonasymptotic methodology: Howard et al. 2021, https://doi.org/10.1214/20-AOS1991 .

## CPU-only implementation and exact synthetic evaluation

- `src/synaploom/research/m13b_d3_hedged_mean_cpu.py`: fixed mixed and single-bet log e-values, fail-closed numerical input checks, parameterless deterministic synthetic report. No file/results/model/GPU access.
- `tests/unit/test_m13b_d3_hedged_mean_cpu.py`: 7 synthetic tests verifying input rejection, permutation invariance, `D=-1` behavior, pointwise lower bound, brute-force exact thresholds, both-needed decisions and toy null/joint-power consistency.
- **Execution independently performed in an isolated local Python environment:** `7 passed in 0.18s`. Hosted CI at this PR head remains a separate check. The CLI outputs machine-readable JSON with a `record_sha256` and explicit `science_executable=false` / `method_selected=null` flags.

All generated two-point probabilities are exact `Fraction`/binomial calculations (only display conversion to float); these are **constructed distributions**, never results of SynapLoom training. n grid = 40, 80, 152, 210, 320.

**Null calibration:** Invented IID distributions with mean exactly zero: balanced ±.02; p=0.8 with +.04 / -.16; p=0.9 with +.02 / -.18; p=0.96 with +.02 / -.48; and the extreme-loss `P(+.02)=50/51, P(-1)=1/51`. Across these five nulls and five n values, maximum mixture false-positive probability is approximately **0.177%**, not 5%. That reflects conservative behavior on this grid; **proof-based size** comes from the preceding supermartingale argument, not from the checked cases.

**Exact marginal power at an invented true mean of +2 percentage points:**

| Constructed IID distribution | n | Fixed λ=1 test | Equal-weight mixture |
|---|---:|---:|---:|
| SD=2pp, values {0,+4pp}, equally likely | 210 | 99.996% | 41.803% |
| SD=5pp, values {-3,+7pp}, equally likely | 210 | 90.516% | 36.508% |
| SD=10pp, values {-8,+12pp}, equally likely | 210 | 52.750% | 26.733% |
| 96% +6pp / 4% -94pp | 210 | 2.988% | 7.479% |
| 96% +6pp / 4% -94pp | 320 | 1.101% | 10.448% |

The last two examples demonstrate modest protection against rare extreme negatives; the first three reveal a substantial **loss of power** from uniform mixing. No distribution shown is claimed to represent actual paired SynapLoom training variability. **Do not select a winner** between procedures by actual protected benchmark observations.

**Two-endpoint joint power is not determined by marginal power alone.** For symmetric two-point families, the program computes exact joint BOTH-needed probability under three prespecified synthetic couplings: identical per-seed endpoint signs, independent signs, and exactly opposite signs. At n=210 and invented SD=5pp, the fixed λ=1 rule has approximate joint power **90.52% / 81.93% / 81.03%** across these three couplings. The mixture gives **36.51% / 13.33% / 0%**. Unknown actual dependence requires planned sensitivity / Fréchet bounds, not an independence assumption by default.

**Method-independent pointwise restriction:** Under the unrestricted IID bounded null, the exact mean-zero population `P(D=+.02)=50/51`, `P(D=-1)=1/51` generates the all-+.02 dataset with probability `(50/51)^n`. Thus any **nonrandomized fixed-n size-5% test** that rejects *that exact all-+.02 sample* must have **n >= 152**. This is not a global required sample count for every positive-mean alternative and does not imply 80% joint power. Stronger assumptions can change the null family; variance adaptation **alone** cannot remove this particular pointwise constraint under the full bounded null.

## Research disposition and next D3 gate

**No statistical method or seed count is approved.** This uniform mixture is a useful valid control, **not** the final recommended confirmatory method; its small-gain low-variance power is poor. The next research target is **one exact theorem-to-implementation variance-adaptive fixed-time empirical-Bernstein/PrPl or similarly justified bounded-mean test**, separately audited on adversarial skew/nulls and +2pp joint-power scenarios. A paired t alternative may be separately examined only as explicitly assumption-dependent, including its skew/outlier calibration. Predeclare the training-seed population (and paired RNG mapping), missingness/failure rules, and full source/feature/O0/eval-batch/mask fingerprints before science.

Original candidate's 3-arm/3-order scope and its obsolete percentile bootstrap cannot be silently replaced; D1 external candidate fidelity (RigL/NISPA), D2 batch-composition/Runner fail-closed wiring and D4 owner-selected GPU-hour ceiling remain open. **No new GPU/protected-outcome work or protocol freeze is authorized.**

## Safe local reproduction

```bash
uv run --locked python -m pytest -q tests/unit/test_m13b_d3_hedged_mean_cpu.py
uv run --locked python -m synaploom.research.m13b_d3_hedged_mean_cpu
```

The parameterless report reads no files and has no flags for scientific results or model execution.

**State:** `D3_TWO_POINT_METHOD_TRADEOFF_RECORDED`; `D3_FINAL_INFERENCE_PENDING`; `D3_JOINT_POWER_AND_SEED_N_PENDING`; `SCIENTIFIC_GPU_NO_GO`.
