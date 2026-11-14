# SynapLoom M1.3b — owner-selected D1 option A, D3/D4 conditional research V1

**Status:** D1_A_OWNER_SELECTED_FOR_DESIGN / D3_METHOD_AND_N_OPEN / D4_BUDGET_OPEN / SCIENCE_GPU_NO_GO.

**Baseline:** GitHub main bd3c17ed08b05b1dd41495160015d09cac59fb6c, following verified merged [PR #62](https://github.com/ehsanrs2/synaploom/pull/62) and its successful [hosted CI](https://github.com/ehsanrs2/synaploom/actions/runs/43005088570). This is a new additive design/research record; it preserves the prior 90-slot preregistration candidate and historical findings without modifying them. **Not a final registered experiment.**

## D1 — owner choice is recorded, not inferred from a PR merge

The owner explicitly approved **option A for design**: **O0 only**, comparing **progressive_reserve** against the **in-house fixed_sparse_mlp**. The owner's approval explicitly **excludes permission to perform GPU execution**.

Consequences: any eventual registered primary paired-seed comparison needs **2n total complete trajectories**, where n is unselected. It addresses only the population mean of paired differences conditional on this fixed O0 order, frozen evaluation rule (batch B256 with final B232), the existing model definitions and the current feature setting. It does **not** generalize to O1/O2, prove order robustness, isolate causal recruitment timing or compare with faithful external RigL/NISPA implementations. A fixed-masked dense-operation MLP is not a physically compute-matched sparse CUDA kernel.

The earlier proposed **10 seeds × 3 arms × 3 orders = 90 trajectories** is historical and unfrozen. Selecting option A for design does NOT silently rewrite that candidate or select n. A prospective formal protocol amendment, complete seed-generating process, paired registry and explicit separate owner authorization are required before results.

## D3 — conditional exact two-point power calculation

The two co-primary endpoints remain final_average_accuracy and late_task_within_task_mean_accuracy. For each future independent training-seed draw i, define D(i,m) = progressive accuracy − fixed-sparse accuracy, in the interval [-1,1]. The estimand is the population **mean D**; it is **not** the fraction of seeds with positive D. The owner's **+0.02 accuracy fraction** is a meaningful **planning alternative on EACH metric**, not the threshold in the null. Each formal one-sided null remains population mean <=0; to claim a single global positive result, valid tests must reject BOTH (intersection–union test). Endpoint dependence affects joint power, not the maximum global type-I probability when both marginal procedures are valid. See [two co-primary endpoints and IUT](https://cloud.r-project.org/web/packages/twoCoprimary/vignettes/overview.html).

**Illustrative candidate only:** fixed stake lambda=1 bounded-mean test with nonnegative E-value equal to the product of (1+D(i,m)), rejecting when E>=20. If the seed draw process satisfies conditional mean <=0 under the null and bounded differences, this is a nonnegative test supermartingale and gives a one-sided size <=.05 test. A fixed n and failure/retry protocol remain unregistered. It does NOT establish seed independence, infer the seed population or by itself implement the required confidence-interval reporting. The method is **NOT SELECTED**.

New research implementation: Python standard-library-only module **src/synaploom/research/m13b_d1a_d3d4_conditional_planning.py** calculates exact rational binomial probabilities for deliberately invented two-point alternatives with **exact true population mean +2pp**. No model, benchmark scores, feature data, scientific seed, GPU or protected outcome is read. For two co-primary endpoints, joint-power *Fréchet bounds* follow solely from the assumed marginal toy powers: lower max(0,2p−1), upper p. The module also reports illustrative identical, independent and opposite per-seed endpoint-indicator couplings for symmetric alternatives. These are **not measured endpoint correlations**.

| Invented paired-seed distribution (exact mean +2pp) | Hypothetical n | Marginal power (fixed lambda1) | Joint-power lower bound for same toy marginals |
|---|---:|---:|---:|
| 50/50 −3pp / +7pp (SD 5pp) | 10 | 0% | 0% |
| same SD5pp | 152 | 40.3923% | 0% |
| same SD5pp | 210 | 90.5155% | 81.0310% |
| same SD5pp | 358 | 99.9965% | 99.9930% |
| 50/50 −8pp / +12pp (SD 10pp) | 210 | 52.7497% | 5.4994% |
| same SD10pp | 358 | 90.6836% | 81.3672% |
| 96% +6pp / 4% −94pp (severe rare losses) | 210 | 2.98797% | at most 2.98797% |

The last line is the critical **tail-risk counterexample**: same +2pp population mean, vastly less power. A scenario yielding 80% joint success at n210 DOES NOT prove the same power in SynapLoom, because paired SD/tails/endpoint correlation remain unmeasured. A historical normal-known-SD proxy suggesting a smaller n is *assumption-dependent*, while the previously identified **n>=152** restriction is only a **pointwise condition** for a nonrandomized, uniformly size-.05, bounded-IID fixed-n test to reject the all-+2pp dataset. Neither is a blanket minimum sample size or an instruction to run.

Other previously reviewed options remain open: mixed-stake bounded e-processes, Maurer–Pontil empirical Bernstein, predictable plug-in empirical Bernstein and explicitly assumption-dependent paired t. The original 10-seed percentile bootstrap has a documented bounded skewed mean-null counterexample and is not a valid default confirmatory test. Statistical method, alpha/reporting details, true seed population/independence mechanism, confidence interval method, planned failure handling, target joint power and final n remain **PENDING**. The earlier **80% joint power** is an illustration, not a new owner approval. Further methodological selection must be outcome-blind and registered before any protected O0 result.

For bounded-mean and alternative betting-confidence methodology see [Waudby-Smith and Ramdas](https://doi.org/10.1093/jrsssb/qkad009) and [Shekhar and Ramdas](https://arxiv.org/abs/2310.01547). These sources do **not** certify the actual SynapLoom execution assumptions.

## D4 — owner engineering observations, not measured future-science cost

The owner previously reported successful local SHA/provenance verification of one 8,000-step *cost-only* engineering observation per arm, using training-derived diagnostic evaluation rather than protected accuracy:

| Arm | Observed engineering wall seconds |
|---|---:|
| progressive_reserve | 130.528300 |
| fixed_sparse_mlp | 35.314361 |
| one two-arm O0 pair total | **165.842661** |

The corresponding reported evidence hashes are 01d908f8a8395689d2df1470759da2c1ef8cca1a64b128d42ef3f8770531b410 and d4f7c98ea08d556395f0be01a01861a5c53bb8831468beddf4e6e90c942105c2. These point observations are NOT independent external runtime attestations, not a distribution of run times, not scientific performance and not equal actual GPU computation. The third arm all_recruited is not part of D1 A; its unknown full-shape cost need not be guessed to price this **two-arm** scenario.

| Hypothetical n | O0 two-arm trajectories | Arithmetic hours at owner engineering rate | Illustrated x1.5 hours, NOT a guarantee |
|---:|---:|---:|---:|
| 10 | 20 | 0.461 | 0.691 |
| 54 | 108 | 2.488 | 3.731 |
| 152 | 304 | 7.002 | 10.503 |
| 210 | 420 | 9.674 | 14.511 |
| 358 | 716 | 16.492 | 24.738 |

The optional 1.5 multiplier is **not a measured upper bound**. Under that hypothetical rate alone, 3/6/12/15/24 GPU-hours would arithmetically accommodate at most about 43/86/173/217/347 paired seeds, respectively. This is **NOT** an approved budget or an execution plan. True protected-evaluation time, temperature/driver changes, preparation, audit time, failed attempts, retries and interrupted work remain unmeasured. A hard runtime cap may stop a study before completing the registered number of seeds; that possibility needs a prospective all-failures-retained policy.

Before science, the owner must separately approve a **finite total GPU-hour cap**, per-attempt timeout, maximum retries and a failure/incompletion response. The final runner must enforce those guardrails and preserve all attempts. None of these decisions is implied by option A.

## Pending decisions and next gate

1. **D1:** selected scope is recorded for design. Prospective 90-to-2n protocol amendment still required; no signed final protocol.
2. **D3:** select inference guarantee/assumptions, joint-power target, preoutcome seed distribution and complete n; do NOT infer actual power from synthetic tables. If no credibly powered confirmatory design fits the owner cap, select **exploratory-only** before results, without a false alpha=.05 success claim.
3. **D4:** obtain explicit owner GPU-hour cap, attempt/retry/timeout policy and uncertainty allowance; actual full-science runtime remains unknown.
4. **D2:** source/feature/order/mask/protected original-row identity, external evidence custody and actual final 8k step runner remain unqualified for scientific execution.

**Nothing in this research records scientific outcomes, selects method/n, freezes a final protocol, uses GPU, or authorizes a future scientific run.**

Safe checks (after PR code review): run the targeted Python test module **tests/unit/test_m13b_d1a_d3d4_conditional_planning.py**, and optionally execute the pure CPU planning module above. New source cannot start training or access datasets. Hosted CI at the exact head is a separate condition.

**Disposition:** D1_A_OWNER_SELECTED_FOR_DESIGN__D3_FIXED_BET_SENSITIVITY_EVALUATED__D4_TWO_ARM_COST_SCENARIOS_EVALUATED__D3_D4_OWNER_DECISIONS_PENDING__SCIENCE_GPU_NO_GO.
