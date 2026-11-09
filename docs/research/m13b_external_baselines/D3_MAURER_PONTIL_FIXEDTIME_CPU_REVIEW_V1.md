# SynapLoom M1.3b — D3 Maurer–Pontil empirical Bernstein fixed-time mean test (V1)

**Status:** `THEOREM_MATCHED_CPU_RESEARCH_COMPLETE__METHOD_AND_N_UNSELECTED`. This is a research-only continuation of [Issue #29](https://github.com/ehsanrs2/synaploom/issues/29), following merged PRs #54 and #55. No GPU, protected evaluation, trained models or outcomes are used. No method/protocol/n/budget is authorized.

## Scientific estimand and null

For primary order `O0` and each co-primary metric (`final_average_accuracy` and `late_task_within_task_mean_accuracy`), define `D_i = metric(progressive_reserve, seed_i) - metric(fixed_sparse_mlp, seed_i)`. Because accuracy fractions lie in [0,1], `D_i∈[-1,1]`. The estimand is the **population mean of independent paired-seed differences**, not positive-sign prevalence. One-sided null for each endpoint: `E[D]≤0`; positive global claim requires BOTH component rejections (IUT). The owner's +2 percentage points is a *meaningful alternative used only for power planning*, not a null threshold or observed effect.

## Exact reference and theorem-to-code mapping

Original primary source: Maurer & Pontil, *Empirical Bernstein Bounds and Sample Variance Penalization* (COLT 2009), **Theorem 4**, original PDF https://www.learningtheory.org/colt2009/papers/012.pdf (paper p.2, second PDF page). For fixed n≥2 and IID Z in [0,1], the theorem gives

```text
E[Z] - mean(Z) ≤ sqrt(2 * V_n(Z) * log(2/alpha) / n)
                 + 7 * log(2/alpha) / (3*(n-1))
```

with probability ≥1−alpha, where `V_n` is the Bessel-corrected sample variance. Apply **`Z=(1-D)/2`** (not `(1+D)/2` alone) to get an upper deviation bound in D, hence a valid **lower** mean bound:

```text
L_MP(D) = mean(D) - sqrt(2 * var(D, ddof=1) * log(2/alpha) / n)
          - 14 * log(2/alpha)/(3*(n-1))
```

For fixed one-sided alpha=.05, reject that component mean-null when `L_MP>0`. A single BOTH-needed IUT positive global claim controls size ≤.05 **if each marginal IID bounded theorem applies**, irrespective of dependence between the two endpoint arrays. This is not a two-interval simultaneous-95%-coverage guarantee. It is a **fixed-time IID theorem**, not a confidence sequence, betting e-value, or implementation of Waudby-Smith–Ramdas's optimized PrPl/GRAPA method. Numerically different seed integers do not establish IID experimental draws.

## CPU-only synthetic study and test evidence

Code: `src/synaploom/research/m13b_d3_empirical_bernstein_cpu.py`; tests: `tests/unit/test_m13b_d3_empirical_bernstein_cpu.py`. The CLI accepts no arguments, real results, training data, feature paths, GPUs or models. For each invented two-point IID population, mean and sample variance are assembled from exact rational sufficient statistics, theorem decisions are evaluated using 60-digit `Decimal` logarithms and square roots, and binomial event probabilities are enumerated as exact `Fraction` values before rounding solely for JSON presentation. The old fixed-lambda=1 e-bet is retained as a separate comparison, not a preferred science method.

Local isolated CPU tests: **8 passed (4.32 seconds)**, including exact constants/variance, sign reversal, invalid/missing values, n=862 threshold, materialized data vs sufficient statistics, rational binomial tails, null-size and co-primary joint dependence and no-science flags. Hosted CI and PR source review must still be checked at their exact head.

The experiments enumerate **five constructed exact-zero-mean nulls**, including a balanced null, p=.8/.9/.96 skewed nulls and `P(D=+.02)=50/51; P(D=-1)=1/51`, and **four invented +2pp mean alternatives**: symmetric SD2pp/5pp/10pp two-point effects, and 96% +6pp / 4% -94pp extreme tail. Fixed n grid: 10, 40, 80, 152, 210, 320, 600, 1000. These are **not** SynapLoom effects, observed paired variability or power estimates. The largest MP empirical type-I rate across the 40 listed synthetic null cells is ≈**2.51×10⁻⁹**; only the cited theorem, under its assumptions, provides a general validity guarantee.

### Exact synthetic +2pp marginal-power results

| Invented IID paired-difference family | Paired n | Maurer–Pontil empirical Bernstein | Existing fixed lambda=1 |
|---|---:|---:|---:|
| Equally likely 0/+4pp (SD2pp) | 210 | **0%** | **99.996%** |
| Equally likely −3/+7pp (SD5pp) | 210 | **0%** | **90.516%** |
| Equally likely −8/+12pp (SD10pp) | 210 | **0%** | **52.750%** |
| Equally likely −3/+7pp (SD5pp) | 1000 | **16.347%** | ≈100% |
| Equally likely 0/+4pp (SD2pp) | 1000 | **95.316%** | ≈100% |
| 96% +6pp / 4% −94pp | 1000 | **2.699%** | **0.014%** |

Power on **both** endpoints depends on their seed-level dependence. The code additionally reports exact identical-sign, independent-sign and opposite-sign couplings for symmetric two-point alternatives. E.g. for SD5pp at n1000 and independent endpoint signs, MP joint power is only **2.672%**. Fréchet bounds apply when dependence is not specified; independence must not be assumed for real co-primary endpoints.

### Method-specific vs universal lower n

Even in the all-+2pp zero-variance toy, this **specific** MP bound first exceeds zero at **n=862** because its finite-sample penalty is `14 log(40)/(3(n−1))`. This is **not** a universal lower bound on required n. The separate **n≥152 pointwise lower limit for *any* nonrandomized level-.05 fixed-n bounded-mean test to reject the *specific all-+.02 dataset*** still follows from the mean-zero null `P(D=+.02)=50/51, P(D=−1)=1/51`. It does **not** establish required sample size or joint power for all actual alternatives.

D4 cost-only engineering observations sum to **165.843 seconds per two-arm O0 pair**. Under the merely *hypothetical O0 two-arm reduction*, n862 would cost ≈**39.710 GPU-hours** by linear extrapolation of those two single engineering runs, or ≈**59.565 hours** with an **unvalidated** ×1.5 contingency. This is not a P95, whole-science runtime guarantee or owner-approved budget; original three-arm three-order protocol and third-arm costs remain different.

## Research verdict and remaining gates

This theorem-matched empirical-Bernstein method is **valid under its explicit IID bounded fixed-n assumptions** but is **NOT recommended for immediate adoption**: it is exceptionally conservative for +2pp mean changes. The earlier fixed-bet method and merged PR55 fixed-stake mixture have different power/robustness limitations. None is yet selected. Variance sensitivity by itself does not guarantee useful small-n power.

**Next bounded D3 research:** implement/review an exactly prespecified predictable plug-in empirical Bernstein (PrPl-EB) or another theorem-specified, more powerful betting mean test, then test synthetic adversarial nulls, two-endpoint joint power, predeclared seed-population/independence/failures and worst-case rare-loss sensitivity. A Student t benchmark may be labeled *assumption-dependent* only, never substituted for unconditional bounded-null validity. Selection cannot depend on protected scientific outcomes.

**Other open gates:** D2 B256/B232 ordered-group evidence, real manual Runner fail-closed mask/optimizer/step/seed/provenance wiring; D1 comparator/order/claim scope and RigL/NISPA feasibility; D4 owner-selected total GPU-hour cap and separate final-run authorization. **No scientific training or GPU is approved by merging this research PR.**

Safe commands:

```bash
uv run --locked python -m pytest -q tests/unit/test_m13b_d3_empirical_bernstein_cpu.py
uv run --locked python -m synaploom.research.m13b_d3_empirical_bernstein_cpu
```

**State:** `D3_MP_THEOREM4_SYNTHETIC_RESEARCH_REVIEW_READY`; `D3_METHOD_AND_SEED_N_PENDING`; `SCIENCE_GPU_NO_GO`.
