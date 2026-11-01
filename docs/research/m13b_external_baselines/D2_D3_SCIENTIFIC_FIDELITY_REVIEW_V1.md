# M1.3b — D2 comparator fidelity and D3 n=10 inferential validity review

**Status: \`SUBSTANTIVE_INTERNAL_METHOD_REVIEW__INDEPENDENT_SIGNOFF_PENDING\`. Confirmatory science: \`NO_GO\`.**

This review is based on [the merged six-run engineering cost closure, PR #41](https://github.com/ehsanrs2/synaploom/pull/41), the frozen-for-review 90-slot candidate (PR #37), the synthetic statistical contract (PR #38) and the actual research implementation. It introduces a **mathematical finite-n counterexample** to the candidate D3 percentile-bootstrap positive-verdict rule. This is a check on a *hypothetical inference procedure*; it uses no CIFAR outcomes, no real scientific data, no model training, no GPUs and no proprietary provider.

## D3: what the proposed inference contract actually needs to establish

Proposed primary scientific family: one **O0** external contrast \`progressive_reserve - fixed_sparse_mlp\`, paired by ten proposed seed IDs \`501..510\`, for two co-primary accuracy endpoints \`final_average_accuracy\` and \`late_task_within_task_mean_accuracy\` (fractions, both higher better). O1/O2 and all-recruited are predeclared *descriptive*, not extra independent seed trials. The candidate currently draws 10,000 bootstrap resamples of the **same paired seed indices for both endpoints**, computes **two marginal two-sided 95% percentile bootstrap CIs**, and calls \`FAVORABLE\` if **both lower bounds are strictly positive**. It calls \`UNFAVORABLE\` if **both upper bounds are strictly negative** and \`INCONCLUSIVE\` otherwise, subject to complete valid runs.

A conjunction of well-calibrated component tests is an intersection–union test. It need not incur a Bonferroni penalty *solely for requiring both endpoints* under a union null. However, **marginal n=10 bootstrap CIs may not be valid component tests for the stated scientific estimand**. Their 95% nominal level is not a guarantee of 95% true coverage. D3 cannot be frozen on logic alone.

### Constructed zero-mean counterexample: exact arithmetic, no Monte Carlo

\`src/synaploom/research/m13b_d3_exact_null_review.py\` constructs **ten iid hypothetical paired seed differences**. Both co-primary endpoints intentionally share the identical per-seed difference, so they are perfectly positively correlated. This is a valid constructed setting to test the candidate **joint** inference rule, not a claim that real SynapLoom endpoints correlate exactly.

For each seed, under the **true joint null**, draw:

| Synthetic distribution | Per-seed effect | Probability | True mean |
|---|---|---|---|
| Symmetric | +0.01 or −0.01 | 0.5 / 0.5 | exactly 0 |
| Skewed | **+0.01** or **−0.09** | **0.9 / 0.1** | exactly 0 |

Both effects are bounded within [−1,+1], consistent with differences between two accuracy fractions in [0,1]. The skewed distribution has mean \`0.9×0.01 + 0.1×(−0.09) = 0\`. In the skewed case there is probability \`0.9**10 = 0.3486784401\` (**34.86784401%**) that **all ten observed seed differences are +0.01**. Conditional on that event, EVERY paired-bootstrap sample (whether 100, 10,000 or infinitely many draws) is made of +0.01 values; BOTH marginal CIs collapse to \`[+0.01,+0.01]\`. The candidate \`FAVORABLE\` verdict is then falsely triggered **at least 34.87% of the time** under a true joint zero-mean null. This lower bound is **mathematically exact** for the explicitly constructed iid generator. Hence the current candidate **cannot be claimed to control type-I error at 5% over all bounded zero-mean paired-seed distributions**.

The reviewer also computes the exact **infinite-bootstrap-draw percentile limit** by summing Binomial(10,k/10) mass over all possible resampled positive counts, and then averaging over all k observed positive seeds:
- Symmetric null: ideal positive-verdict probability \`11/1024 = 1.07421875%\` (only k=9 or 10).
- Skewed null: ideal positive-verdict probability \`0.9**10 = 34.86784401%\` (only k=10).
- The exact infinite-bootstrap statement is separate from the original RNG=1337, 10,000-draw finite-interval convention. Crucially, the skewed **all-positive lower bound applies even for the actual finite-bootstrap implementation**, with no RNG dependence.

This does **not** show that actual SynapLoom seed effects have the skewed distribution, that the scientific study would have a 34.87% error rate, or that any alternative method is uniformly calibrated. It shows the **existing D3 candidate has no distribution-free 5% type-I guarantee** for a mean-effect scientific claim.

### What an external D3 statistics reviewer must choose *before data*

1. **Target estimand:** Is the primary target mean difference across random seeds, a typical/median seed difference, or a fixed-ten-seed descriptive summary? Define scientifically meaningful inference and what randomness the paired-seed design represents.
2. **Assumptions:** If using one-sided paired t inference, specify approximate normality/finite-sample sensitivity and its fragility to skew/outliers. If using exact paired sign-flip, justify *symmetric* paired differences or algorithm-label exchangeability; a **zero mean alone is not sufficient**. A sign test targets a sign/median-related quantity, **not** necessarily the mean under skew. No post-hoc choice of the most favorable technique.
3. **Two co-primary endpoints:** Lock the global positive alternative ("both positive"), marginal test direction and type-I budget. Define whether \`UNFAVORABLE\` is itself a formal scientific claim with error control or only an exploratory/descriptive label. Avoid secretly turning either endpoint into a separate unadjusted family.
4. **Completeness:** All preregistered O0 seeds, arms, orders and valid finite endpoint records must be present and provenance-checked. Define aborted/failed/nonfinite-run statuses and retries up front; never delete a failed or inconvenient seed or use O1/O2 as retroactive substitutes.
5. **Sensitivity vs power:** Independently quantify test size and power under plausible seed-effect shapes, inter-endpoint correlation, training randomness and candidate effect sizes. With n=10, meaningful robust mean-effect confidence bounds can be very wide; report that limitation instead of claiming a nominal level without justification. More seeds, a revised estimand, or a predeclared sensitivity-only exploratory study may be required.

**Current D3 disposition: \`CANDIDATE_BOOTSTRAP_NOT_DISTRIBUTION_FREE_VALIDATED__INDEPENDENT_METHOD_DECISION_REQUIRED\`.** Do NOT silently replace the candidate inference rule after scientific outcomes are known. This protocol is **not yet frozen**, so a scientifically justified redesign is possible now, before any real confirmatory result exists.

## D2: fixed sparse is not the same computational mechanism

Source inspection in \`src/synaploom/research/fixed_sparse_baseline.py\` shows:

- \`FixedMaskedLinear.forward\`: \`F.linear(x, self.weight * self.mask, self.bias)\`; these are ordinary **dense** PyTorch linear operations on masked weights, not actual sparse matrix kernels.
- \`FixedSparseMLP\`: residual feed-forward body with two fixed masked \`nn.Linear\`-derived layers, \`LayerNorm\`, GELU, a dense head and possibly a dense encoder; the masks are sampled once and do **not** support regrowth or adaptive rewiring.
- \`install_fixed_mask_guard\`: optimizer step post-hook that projects both masked weights and same-shaped optimizer moments back onto frozen zeros. This prevents inactive weight drift but does not make their gradients/backward compute sparse.
- \`build_h1a_fixed_sparse_feasibility\`: matches *physically allocated parameter elements* close to the full capacity of SynapLoom, while sizing the number of **active body weights** with a target based on an **ideal analytical linear-MAC proxy**. Those are two different resource tiers; it is not a guarantee of matched realized CUDA work.

A CPU-only test here independently constructs the *research candidate* using H1a2 settings and verifies the count of **1,172,226** candidate physical parameter elements vs **1,172,196** SynapLoom physical elements, mask density strictly below one, dense-masked forward source signature, ideal-vs-dense proxy distinction and missing real CUDA cost fields in the analytical budget structure. This **does not** certify it as RigL/NISPA or a named paper's faithful baseline. Such admission requires a separate original-paper method-fidelity audit, hyperparameter/training schedule alignment and potentially a different control arm.

The owner-reported **3 arms × 2 repeats** 390-step train-only cost experiment (read-only host checks all PASS) found wall-time means of **11.1494 s** progressive, **8.7055 s** all-recruited and **1.9894 s** fixed sparse. Thus this engineering setup has visibly different runtime costs despite near-equal parameter allocations. Those are *two* unrandomized wall-time observations/arm, not equal GPU-FLOP or full-science measurements. Memory from the caching allocator is not total device memory. No GPU work was performed in this review PR.

**Suggested narrow D2 classification:** \`FIXED_MASKED_MLP_AS_CAPACITY_MATCHED_CANDIDATE\`, with separate ideal MAC and observed cost axes; **not compute-matched**, not a true sparse-kernel baseline, not a faithful external named-method baseline by default. Independent baseline-fidelity reviewer can approve this *specific* capacity comparison or require modifications to the preregistration. The suggestion is **not** an independent reviewer signoff.

## Safe invocation and review status

\`\`\`bash
# Read-only mathematical toy analysis; no feature file or GPU permitted
uv run --locked python experiments/scripts/m13b_d3_exact_null_review.py \
  --exact-synthetic-null-review

# Optional independent output artifact (must not already exist)
uv run --locked python experiments/scripts/m13b_d3_exact_null_review.py \
  --exact-synthetic-null-review \
  --output /tmp/m13b-d3-exact-null-review-v1.json

# CPU-only scientific-method test suite
uv run --locked python -m pytest -q tests/unit/test_m13b_d2_d3_method_review.py
\`\`\`

The analysis' SHA-256 checksum proves only serialization identity, not scientific method validity or owner attestation. No real results or training are accepted by the CLI; no \`--execute\`, \`--features\`, \`--seed\`, \`--real-results\` or \`--approve\` options exist.

**Next gate:** obtain external qualified D2/D3 peer review of **this concrete counterexample and model-anatomy evidence**. Approve or amend the D3 estimand/inference rule and D2 control identity **before** any non-synthetic confirmatory outcomes. Then estimate full-science 20-epoch/real-eval costs and have the owner set a finite GPU-hours/retry budget, source/data/seed/mask-bound final protocol, and a distinct explicit owner-issued real-experiment run authorization.

**No change** to historical \`EXP-H1A-002=INCONCLUSIVE\`, \`ORDER_SENSITIVE\` or \`DIRECTION_FLIP_OBSERVED\`. \`EXP-M13B-EXTERNAL-001\` remains a proposed, **non-executable** study.
