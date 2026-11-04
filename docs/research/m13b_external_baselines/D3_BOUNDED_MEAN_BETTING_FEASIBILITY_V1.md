# M1.3b D3 — A calibrated bounded-mean inference candidate, and its cost/power trade-off

**Disposition: BOUNDED-MEAN TEST MATHEMATICALLY VALID UNDER STATED ASSUMPTIONS; NO APPROVED SCIENTIFIC TEST, n, or GPU RUN.**

This is the bounded, CPU-only design follow-up to [merged PR #50](https://github.com/ehsanrs2/synaploom/pull/50). The owner has chosen **mean paired accuracy improvement** as the primary scientific question and accepted ChatGPT D2/D3 analytical reviews as revision input for a budget-limited internal research program **without mandatory human expert signature**. That policy does **not** waive mathematical validity or require an expensive new research arm.

Earlier D3 tests proved **only negatives**: ten-seed percentile bootstrap and assumption-free use of a paired Student t test cannot guarantee nominal 5% size over all skewed mean-zero seed-effect distributions. Here we construct a positive, elementary counterpoint: an exact **one-sided bounded-mean e-value test** with a transparent finite-sample guarantee, then explicitly examine its severe possible loss of power.

## Primary target and fixed candidate test

For primary class order O0 and each co-primary accuracy endpoint, define the per-seed paired difference:

    D_i = accuracy(progressive_reserve, seed_i) - accuracy(fixed_sparse_mlp, seed_i)

Both accuracy metrics are fractions within [0,1], hence each D_i is within **[-1,1]**. The owner-approved target remains the **population mean of those paired seed differences**, not the percentage of seeds on which Progressive wins. Co-primary endpoints are final_average_accuracy and late_task_within_task_mean_accuracy.

For a **fixed and predeclared** betting factor lambda=1, define

    E_m(n) = PRODUCT_i(1 + D_{i,m}).

At positive-direction per-endpoint alpha=0.05, reject the mean-null when **E_m(n) >= 20** (or log E_m >= log 20).

**Finite-n size proof:** If paired training-seed differences are IID from a fixed population, with each D in [-1,1], and H0_m is E[D_m] <= 0, then each 1+D_m is nonnegative and E[1+D_m] <= 1. Independence yields E[E_m(n)] <= 1. Markov's inequality therefore gives P_H0(E_m(n) >= 20) <= 0.05 **for any finite prespecified n and regardless of the shape/skewness of the paired-difference distribution**. This is a narrow special case of bounded-mean betting/e-value inference; it is **not** an implementation of the full adaptive betting/confidence sequence algorithms.

**Two endpoints:** For a positive global claim, require **both** valid endpoint e-values to be >=20. Under the global union null (at least one true mean nonpositive), the joint rejection event is a subset of the false rejection of that null component. Hence the conjunction controls its global false-positive rate at <=0.05 without assuming the endpoints are independent. This is an intersection-union test. There is **no formal negative-direction claim** in this design; a nonrejection must be "insufficient evidence" rather than "no effect".

**Critical assumptions not yet established:** ten consecutive fixed seed integers are not themselves proof of IID draws from a meaningful distribution of training randomness. If the paired differences vary systematically by seed, are not independent, are selected based on outcomes, or come from several incomparable data-generating regimes, the stated generalization/guarantee may not apply. Trial registration, class-order pairing, source/eval configuration, failed-run handling and the seed-generating process still need prospectively frozen verification.

Reference: Waudby-Smith, I. and Ramdas, A., "Estimating means of bounded random variables by betting", *JRSS Series B* 86(1), 2024, DOI [10.1093/jrsssb/qkad009](https://doi.org/10.1093/jrsssb/qkad009). The simple nonnegative betting factor used here is supported by the general supermartingale framework; we do **not** claim to have ported its optimized procedures.

## Actual CPU-only mathematical evaluation

Code: **src/synaploom/research/m13b_d3_bounded_mean_betting.py**.
Tests: **tests/unit/test_m13b_d3_bounded_mean_betting.py**.

The module accepts no file paths or real results on its CLI and performs no stochastic simulations. It uses rational arithmetic to exactly enumerate binomial event probabilities under explicitly fabricated two-point IID seed distributions, and exact Fraction inequalities to decide which positive counts cross E>=20.

### Exact false-positive calibration in *invented* zero-mean populations

Each toy delta is +0.01 with probability p and negative delta -0.01*p/(1-p) otherwise, so the **true paired mean is exactly zero**. With p=0.90, the negative delta is -0.09; with p=0.95 it is -0.19.

| Manufactured null scenario | n=10 | n=55 | n=100 | n=320 |
|---|---:|---:|---:|---:|
| p(+0.01)=0.90, delta -0.09 otherwise | 0% | 0% | 0% | approximately 8.33e-12% |
| p(+0.01)=0.95, delta -0.19 otherwise | 0% | 0% | 0% | approximately 7.44e-6% |

These exact rates are from **these particular synthetic nulls**; the universal <=5% size guarantee comes from the probability inequality and IID/bounded conditions, **not** from these few table cells. Zero observed false rejections in the toy cases does **not** establish a powerful or useful scientific test.

### Power limitation: even perfectly uniform positive improvements may need many seeds

Suppose hypothetically **every** seed yields *exactly the same* gain delta>0 for a given endpoint. Then E(n)=(1+delta)^n. The first integer n reaching E>=20 is:

| Invented constant gain in EVERY seed | First n with e-value >=20 |
|---|---:|
| +1 percentage point | **302** paired seeds |
| +2 percentage points | **152** paired seeds |
| +5 percentage points | **62** paired seeds |
| +10 percentage points | **32** paired seeds |

These are **mathematical best-case constant-data thresholds**, not estimates of realistic required sample sizes or power. Actual variable deltas, including losses, may require more (or in some distributions fewer) seeds to achieve any particular power. At the originally proposed n=10, this fixed-bet test cannot reject even if every observation equals +10 percentage points.

For a separate hypothetical positive-mean alternative, IID deltas are +0.06 with probability 0.90 and -0.10 with probability 0.10, giving **true mean +0.044 (4.4 percentage points)**. Exact *one-endpoint* rejection probability is **~2.16% at n=55**, and **~99.00% at n=100**. This highly nonlinear change illustrates the need for test-specific power modeling; these probabilities are **not real SynapLoom power**. Global power for two different co-primary endpoints depends on their joint distribution and cannot be inferred by multiplying or copying one-endpoint probabilities without further assumptions.

### Why this is not automatically the final D3 method

The fixed-lambda e-test is mathematically transparent and assumption-light but sacrifices power. More powerful **variance-adaptive betting** or carefully justified alternative mean tests may reduce costs; implementing or selecting them requires verified assumptions, correct finite-sample behavior, and **pre-outcome** selection rather than cherry-picking the method that yields a positive result. A traditional paired t test may be more efficient under a defensible approximate normality model, but that assumption must be explicit and cannot be inferred from the small confirmatory results after unblinding.

The study owner has **not** supplied an approved minimum scientifically relevant gain for either endpoint, a defensible paired-seed variability model, a resource limit for the full 20-epoch protected evaluation workload, or a predeclared final n. Accordingly, none of the numeric values above is an authorized amendment to the current 10-pair / 90-trajectory candidate.

## Budget-aware decision required next — not another speculative GPU experiment

1. **Define meaningful gains:** choose a threshold for scientific importance separately for final_average_accuracy and late_task_within_task_mean_accuracy. The +1/+2/+5/+10 pp cases are **illustrative only** and not owner-approved thresholds.
2. **Choose the inference assumption tier before seeing any protected O0 outcome:**
   - **Strong finite-n control:** retain an explicitly validated bounded-mean e-test or better calibrated variance-adaptive bounded-mean procedure, accept potentially high required n/cost.
   - **Model-based efficiency:** adopt a mean test such as paired t *only* after documenting acceptable distributional assumptions and sensitivity limits; report the assumptions honestly.
   - **Budget-limited exploratory path:** keep a small, bounded run, report paired mean estimates and uncertainty **without** advertising a statistically confirmed alpha=5% positive conclusion if adequate power is not justified.
3. **Determine n and cost with both co-primary endpoints:** prospective and test-specific joint-power planning under plausible mean gains, paired seed variation, endpoint correlation, and a fixed failure policy, then compare to actual full-workload cost. Do not scale 390-step training-only times naively to 8,000-step full evaluation.
4. **Only then** freeze data/features, order, full seed registry, fixed mask, evaluation B=256 policy/Batch sensitivity scope, both co-primary definitions, test and alpha, source SHA, retry limits and an auditable manual runner. Obtain the owner's separate explicit GPU-science execution authorization.

## Safe reproducibility — no outcome or feature input

    uv run --locked python -m synaploom.research.m13b_d3_bounded_mean_betting
    uv run --locked python -m pytest -q tests/unit/test_m13b_d3_bounded_mean_betting.py

All evidence in this PR is synthetic mathematical analysis, not a scientific M1.3b outcome.

**Decision:** BOUNDED_MEAN_EVALUE_SIZE_PROOF_AVAILABLE; ACTUAL_D3_METHOD_PENDING; MINIMUM_EFFECT_PENDING; REALISTIC_SEED_N_PENDING; GPU_NO_GO.
