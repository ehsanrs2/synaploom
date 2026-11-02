# M1.3b D3 — Exact small-sample paired t-test null-calibration stress

**Status:** CPU mathematical stress completed in a research-only candidate; **NO selected inferential method, NO approved n, NO scientific training GO**.

This is a follow-up to [merged PR #49](https://github.com/ehsanrs2/synaploom/pull/49), after the owner adopted the AI D2/D3 analytical recommendations as the internal methodological revision basis without mandating a human signature. This report answers a specific D3 question: **does substituting a one-sided paired Student t test automatically fix the n=10 percentile-bootstrap mean-null problem?** Answer: **no**, not without scientifically defensible distributional assumptions.

## Research target retained

The owner-selected main question is whether the *population mean of paired-seed accuracy differences* is positive for **both** co-primary O0 endpoints:

- final_average_accuracy;
- late_task_within_task_mean_accuracy.

The contrast is Progressive Reserve minus the custom Fixed Sparse MLP. A binary sign-prevalence claim (winning most seeds) is **not** an equivalent target. Original fixed seed IDs 501–510, the complete three-arm three-order 90-trajectory matrix and any later candidate n are still *proposals*, not an approved powered confirmatory design.

## Exact stress method, not a Monte Carlo estimate

Source: **src/synaploom/research/m13b_d3_exact_t_null_stress.py**.
Tests: **tests/unit/test_m13b_d3_exact_t_null_stress.py**.

For each independently drawn *hypothetical* paired-seed difference D:

- with probability p, D = +0.01 accuracy fraction;
- otherwise, D = -0.01*p/(1-p), giving an **exactly zero mean**;
- p is one of 0.5, 0.8, 0.9, 0.95 (all values remain within [-1,1]);
- sample size n is one of 10, 20, 40, 55, 100.

For a sample containing k positive seed differences and n-k negative ones, the usual Student t statistic is determined exactly by k and the two effects. The code uses frozen one-sided alpha=0.05 Student t critical values, degrees of freedom n−1, generated offline with SciPy 1.17.0. No SciPy dependency is added to the project, and no stochastic simulation is run.

Under the standard **zero sample-variance convention**, all n observations equal to +0.01 produce a positive infinite t statistic and are rejected. This convention is **explicit**; it matters when one rare adverse seed outcome has not yet appeared. The probability of each k is an exact Binomial(n,p) rational fraction. Summing over rejected k values yields the exact false-positive frequency for this **particular specified null distribution**.

To illustrate a valid joint-null scenario, the two endpoints have identical paired per-seed differences, hence perfect positive correlation: both corresponding marginal t tests reject together, so the two-endpoint BOTH-needed intersection–union false-favorable probability equals the single-endpoint rate. **This is not an estimate of real endpoint dependence.**

### Selected mathematically exact stress frequencies

These table entries are exact binomial calculations evaluated numerically and rounded for readability, *not measurements of SynapLoom*.

| p(positive +0.01 seed effect) | Negative effect required for mean zero | One-sided t false-positive probability at n=10 | At n=55 |
|---:|---:|---:|---:|
| 0.50 | -0.01 | 5.47% | 5.24% |
| 0.80 | -0.04 | 10.74% | 5.76% |
| 0.90 | -0.09 | **34.87%** | **7.74%** |
| 0.95 | -0.19 | **59.87%** | **23.19%** |

The n=55, p=0.90 case rejects if **at least 53 of 55** seed differences are positive; its exact false-positive probability is approximately **7.7434%**, above the nominal 5% one-sided rate. The p=0.95 case has the n=55 rejection threshold **54 of 55** and probability approximately **23.1887%**. These examples refute a **distribution-free 5% error-control assertion**, even after increasing from ten to 55 paired seeds under some skewed nulls.

The original percentile-bootstrap counterexample from merged PR #42 already showed a 34.86784401% false-favorable **lower bound** in one fabricated n=10 skewed joint null. The present study goes beyond that lower bound by calculating the **actual full paired t-rejection probability** under each toy null using all possible positive counts. It is not a second SynapLoom pilot, a measured error rate, an estimate of real mean gain, or a contradiction of Student-t validity under its *own stated assumptions*.

## Why a conservative bounded-mean method does not simply solve planning

Since every difference between two accuracy fractions lies within [-1,+1], a one-sided Hoeffding bound with independent paired seeds offers a distribution-free control of a mean-null at alpha=0.05. But its radius at n=10 is approximately 0.774 accuracy fraction (77.4 percentage points), and at n=100 approximately 0.245 (24.5 pp). In the constructed family above **no sample mean can exceed +0.01**, so this conservative test never rejects for n<=100. This illustrates a severe power tradeoff; it does not discredit other legitimate mean methods with explicit assumptions or optimized bounded-mean inequalities.

## D3 disposition — what is now known and unknown

**Known with mathematical certainty:**
- Changing from percentile bootstrap to a standard one-sided paired t test **without addressing skew/normality assumptions does not yield general finite-n alpha=0.05 mean inference**.
- Adding seeds does not automatically solve every highly skewed zero-mean test scenario: e.g. exact n=55 examples exceed alpha.
- The both-needed co-primary conjunction does not repair a poorly calibrated marginal test when both endpoints are identical.

**Not established:**
- Whether actual SynapLoom per-seed mean differences behave like any constructed two-point population.
- Whether real paired differences are approximately normal or symmetrical enough for a valid, useful t-based assumption, how correlated the co-primary endpoints are, or their variances.
- Which method offers adequate conditional inference, what minimal accuracy improvement has real scientific relevance, and what seed n is affordable with honest joint power.
- Whether full 20-epoch plus protected evaluation science can be financed within the owner's resource ceiling.

**Lean next D3 decision under the owner AI-advisory policy:**

1. Prospectively define seed population and the **smallest scientifically important mean gain** separately for both co-primary endpoints (keep the numerical choice attributable to the owner, not favorable outcomes).
2. Select a *defensible mean-inference rule* with a stated assumption class. Use synthetic adversarial calibration including the constructed p=0.80/0.90/0.95 nulls, realistic effect/variance alternatives and endpoint dependence; do not switch inference after real results.
3. Pair a separately capped CPU power/size assessment with realistic full-workload GPU cost before choosing n. The known-SD normal scenarios and present null stress are **planning evidence only**. An underpowered budget means an **exploratory study** and honest uncertainty intervals, not a confirmatory proof.
4. Preserve predeclared complete data/failed-run handling, O1/O2 as descriptive orders, realistic GPU budget and a distinct manual execution authorization. A green CI or merged PR is not science GO.

## Safe CPU verification (no real outcome input)

From the repo root:

    uv run --locked python -m synaploom.research.m13b_d3_exact_t_null_stress
    uv run --locked python -m pytest -q tests/unit/test_m13b_d3_exact_t_null_stress.py

The code accepts **no CLI arguments or outcome input paths**, imports no scientific feature bundle, executes zero optimizer steps and emits an auditable synthetic review JSON with a canonical SHA-256. All scenarios and critical values are immutable constants in source.

**Disposition: D3_EXACT_SMALL_N_T_NULL_RISK_DEMONSTRATED; INFERENTIAL_TEST_NOT_APPROVED; SEED_COUNT_NOT_APPROVED; CONFIRMATORY_GPU_NO_GO.**
