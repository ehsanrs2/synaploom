# M1.3b — Owner-selected mean-paired-accuracy estimand and prospective power planning

**Status: \`OWNER_SELECTED_MEAN_ESTIMAND__D3_TEST_AND_N_NOT_APPROVED\`. Scientific training: \`NO_GO\`.**

Following the merge of [PR #43](https://github.com/ehsanrs2/synaploom/pull/43), the owner explicitly answered **"بله" / yes** to maintaining **improvement in mean accuracy** as the primary scientific goal, recorded in [Issue #29, owner-estimand comment](https://github.com/ehsanrs2/synaploom/issues/29#issuecomment-6557812225). This establishes *research-goal direction*, **not** a statistically valid test, final seed count, minimal scientifically important effect, final protocol freeze or GPU spend authorization.

## Primary target selected; other claims remain secondary

For primary class order O0 and each of the two prespecified co-primary endpoints, define paired seed difference:

\`\`\`
D(seed, endpoint) =
    accuracy(progressive_reserve, O0, seed, endpoint)
  - accuracy(fixed_sparse_mlp, O0, seed, endpoint)
\`\`\`

**Selected estimand:** population mean \`E[D(seed, endpoint)]\` over a **scientifically justified distribution of independent training seeds**. Global positive conclusion requires credible positive mean differences on **both** \`final_average_accuracy\` and \`late_task_within_task_mean_accuracy\`. The original proposed seed IDs \`501..510\` are **still candidates** and do not become a random sample merely by being consecutive, fixed integers. Independent reviewer must specify how seed-level stochasticity, reproducibility, independence and meaningful population target are defined. If the intended target is *these ten specific fixed seeds only*, the interpretation is **finite-registry descriptive** unless additional inferential assumptions are defensible.

**Not selected:** \`P(D>0)>0.5\` (a majority of seeds). A sign-based analysis may only be a separately labeled, predeclared sensitivity for a different estimand. A small positive mean can occur in fewer than half the seeds, and a positive sign majority can coexist with negative mean due to rare large losses. Do not make the primary inference method depend on favorable observed outcome patterns.

The user has **not** selected: distributional model, numerical alpha policy for both directions, minimum effect size of scientific interest, paired-seed population, n=10 adequacy, or the final two-endpoint CI construction.

## Quantitative feasibility with transparent invented assumptions (not measured)

The new offline tool computes a simple **one-sided normal known-SD planning proxy** for one co-primary endpoint:

\`\`\`
power_one_endpoint(n, delta, sigma) ≈
    Φ( delta * sqrt(n) / sigma - Φ⁻¹(0.95) )
\`\`\`

Here \`delta\` is a hypothetical **mean paired accuracy gain**, \`sigma\` a hypothetical **standard deviation of the paired-seed difference**, and 0.05 is a proposed *per-endpoint one-sided positive* alpha. These proxy assumptions are **not** currently established for the n=10 bounded accuracy differences: a true normal distribution is unbounded while accuracy differences lie in [-1,+1], and sigma is unknown. Real paired Student-t inference (estimated sigma) has a different power curve and requires review at small n. Therefore these numbers are **planning approximations, not actual power or an adopted confirmatory test**.

Some illustrative cells from the predeclared mathematical grid (each delta and sigma in accuracy-fraction units):

| Invented true mean gain | Invented paired-seed SD | One-endpoint power at n=10 (normal known-SD proxy) | n for 80% one-endpoint power | n sufficient for ≥80% joint power under worst endpoint dependence *if both have that same marginal proxy* |
|---:|---:|---:|---:|---:|
| +1 pp | 2 pp | ~47.46% | 25 | 35 |
| +2 pp | 5 pp | ~35.20% | 39 | 54 |
| +5 pp | 5 pp | ~93.54% | 7 | 9 |
| +1 pp | 10 pp | ~9.20% | 619 | 857 |

The one-endpoint \`n=10\` power is **NOT joint power**. For endpoint powers \`p1\` and \`p2\`, without assuming independence *between endpoints*, the sharp Fréchet bounds are:

\`\`\`
max(0, p1 + p2 - 1) <= P(both reject) <= min(p1, p2)
\`\`\`

Thus, even when both hypothetical endpoint powers equal 35.2%, the lower bound for global success can be 0%. For a conservative **80% joint power** planning target, it suffices (under the marginal normal proxy) that **each endpoint** has ≥90% power, because the probability either endpoint misses is at most 10%+10%=20%. This union-bound sufficiency is with respect to *unknown endpoint dependence*, **not** an assumption-free guarantee about data generation. The necessary marginal-80% and sufficient marginal-90% sample-size columns are **scenario calculations**, not approved revisions to the 90-slot research design. If the two endpoints have different deltas/SDs, review them separately.

These illustrative scenarios use no H1a/M1.3b measured scientific effects, no learned seed variance and no protected outcomes. Hardware cost-calibration wall times do **not** estimate the variance of paired accuracy deltas. An independent reviewer must propose justified effect sizes/variances from permitted external literature, historical *nonprotected* pilot evidence if legitimate and predeclared, or a preoutcome design rationale before finalizing n.

## Why we cannot silently retain the original bootstrap test

Merged [PR #42](https://github.com/ehsanrs2/synaploom/pull/42) supplies an exact constructed two-point skewed joint-null where **both true mean differences are zero** but the original paired n=10 marginal-percentile-bootstrap both-needed \`FAVORABLE\` rule has an at-least-34.87% false-positive event. That is a mathematical **counterexample to universal 5% size control**, not an estimate of SynapLoom's empirical error probability.

- **A paired t-test** may be an appropriate and efficient mean test if seed-paired differences satisfy justified assumptions; however n=10 exposes sensitivity to skewed effects, outliers and nonnormality. An optional t-based method needs independent calibration/sensitivity under plausible seed distributions.
- **A sign-flip test** requires symmetry or a scientifically justified exchangeability/sharp-null mechanism; mean zero by itself is insufficient.
- **A bounded-mean Hoeffding bound** is distribution-free conditional on independent bounded differences, but at n=10 its one-sided 95% lower-radius is ~0.774 accuracy units (77.4 percentage points), so it is often too conservative to be useful.
- Modern variance-adaptive bounded-mean inference methods exist, but they require implementation review and small-sample power assessment; the present PR does not choose them.
- A both-needed **intersection–union** rule need not independently Bonferroni-correct for requiring both endpoints *if each component test truly controls its size*. This does not rescue miscalibrated percentile bootstrap. Whether \`UNFAVORABLE\` is a separately formal directional claim must be fixed before outcomes.

Relevant methodology sources, as background and not D3 signoff: [OpenStax paired samples and Student-t assumptions](https://openstax.org/books/statistics/pages/10-4-matched-or-paired-samples-optional), [Hoeffding inequality, MathWorld](https://mathworld.wolfram.com/HoeffdingInequality.html), and [Waudby-Smith & Ramdas, confidence intervals for bounded means (JRSS B)](https://doi.org/10.1093/jrsssb/qkad009). These do not themselves establish valid inference for the particular SynapLoom seed population.

## Independent D3/D2 reviewer decision gates

Before approving a final protocol, request an attributed external review with:
1. Scientifically meaningful **minimum paired mean improvement** for *each* co-primary endpoint and whether both expected gains must exceed zero vs a meaningful minimum.
2. A justified **seed-generation population**: can the chosen integers be treated as independent draws, or is this a fixed-registry description? Which randomness is generalizable?
3. Predeclared method for one-sided **mean** inference, its n=10 nominal size/power under plausible skew/outlier/endpoint correlation, and the reason for rejecting or replacing the original percentile bootstrap. Review a sensitivity distribution family before results.
4. Planned **total seed count**, including any necessary preoutcome amendment to 10/90-slot proposal. If n increases, all arms, orders, intended descriptive analyses, resource budget, provenance and SHAs must be amended and independently re-reviewed before training.
5. Predeclared missing/nonfinite/failed seed policy and bounded retries (never drop inconvenient outcomes), two-endpoint global success rule, and distinct interpretation of \`UNFAVORABLE\` if included.
6. **D2 external comparator fidelity:** does the current in-house fixed-masked MLP adequately test the claim of a capacity-matched baseline, with no false claims of actual GPU-compute parity or literature named-method identity?
7. **D4** quantitative full-science GPU-hours/walltime/retry budget, final source/feature/order/mask freeze and separately approved manually invoked real-science runner. No auto-run from merging this review.

## Safe inspection

\`\`\`bash
uv run --locked python experiments/scripts/m13b_d3_mean_power_planning.py \
  --review-mean-power

# Optional immutable-by-exclusive-create review artifact only:
uv run --locked python experiments/scripts/m13b_d3_mean_power_planning.py \
  --review-mean-power \
  --output /tmp/m13b-d3-mean-power-PLANNING-v1.json

uv run --locked python -m pytest -q tests/unit/test_m13b_d3_mean_power_planning.py
\`\`\`

No outcomes input, no seed selection, no method choice flag, no CUDA, no optimizer, no authorizations. Checksums are content-integrity tokens only. The review packet records the user's **specific goal confirmation** while keeping \`independent_statistical_review_complete=false\`, \`method_chosen=null\`, \`ready_for_science=false\` and \`owner_gpu_hours_budget=null\`.

**Decision:** \`MEAN_ESTIMAND_SELECTED\` for scientific design; \`n=10 POWER/TEST NOT VALIDATED\`; \`D2 FIDELITY PENDING\`; \`NO_GO_CONFIRMATORY_GPU_SCIENCE\`.
