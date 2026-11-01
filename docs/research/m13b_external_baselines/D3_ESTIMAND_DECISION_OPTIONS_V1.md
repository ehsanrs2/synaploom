# M1.3b — D3 estimand decision package: mean vs sign vs model-based inference

**Status: \`PRE_OUTCOME_D3_ESTIMAND_REVIEW_ONLY\`; NO-GO for real 30/90-run training.**

This is a follow-up to merged [PR #42](https://github.com/ehsanrs2/synaploom/pull/42), which provided a **mathematically exact constructed counterexample** to distribution-free calibration of the proposed 10-seed percentile-bootstrap decision. This package is a prospective **choice of scientific question**, not an approval of an analysis method, a replacement of the preregistration, or an independent statistical signoff. It uses only mathematical constants and proposed study metadata. No CIFAR-100 data, GPU, actual learner outcomes or protected seed 501..510 was used.

## Before choosing a test, decide what claim we seek

The proposed study samples 10 matched new seeds across 3 arms and each of O0/O1/O2. One O0 external contrast \`progressive_reserve - fixed_sparse_mlp\` has co-primary endpoints \`final_average_accuracy\` and \`late_task_within_task_mean_accuracy\`.

For one endpoint and seed i, define \`D_i = endpoint(progressive, O0, seed_i) - endpoint(fixed_sparse, O0, seed_i)\`, where each endpoint is a fraction in [0,1] and therefore \`D_i ∈ [-1,1]\`. **Pairing means comparing the two algorithms under the same seed ID**. Whether these ten preselected integers can be considered iid random samples from a meaningful distribution of training randomness must be justified; deterministic chosen seeds are not automatically a random sample.

Two distinctly different possible targets:

| Estimand / research question | Suitable candidate analysis | Key price |
|---|---|---|
| **Population mean paired accuracy difference**: "on average across independent random training seeds, does Progressive improve both endpoints?" | A justified one-sided paired t model under a defensible distributional approximation; conservative one-sided bounded-mean Hoeffding as sensitivity. *Original percentile bootstrap needs a demonstrated small-n calibration.* | n=10 and uncertain/skewed seed differences may make mean inference extremely weak |
| **Positive-sign prevalence**: "do more than half of independently drawn seeds improve on both endpoints?" | Exact one-sided binomial **positive-sign** test per endpoint, followed by a two-endpoint both-needed intersection–union conclusion | Tests prevalence of \`D_i>0\`, **not** a positive population mean; rare, large losses may dominate the mean |

Other options, e.g. exact sign-flip of paired differences, assume suitable symmetry or label-exchangeability; **a zero mean alone is not enough**. This packet does not choose among methods by looking at scientific performance. It would be improper to silently change the claim from "mean benefit" to "majority of seeds" merely to make significance easier.

## Distribution-free independent bounded-mean limit at n=10

If the ten paired differences are independent and each lies in [-1,1], Hoeffding's inequality gives a valid conservative **one-sided** lower confidence bound for the average expected difference (and for a common mean if seed draws are iid):

\`\`\`text
P( sample_mean - expected_mean >= t ) <= exp( -n * t² / 2 )
L_one_sided = sample_mean - sqrt(2 * ln(1/alpha) / n)
\`\`\`

With n=10 and alpha=0.05, the margin is **approximately 0.7740 accuracy-fraction units** (77.4 percentage points). A statistically certified positive mean would require an enormous observed mean difference exceeding this threshold; with typical small accuracy gains this conservative proof is uninformative. This does **not** establish that every valid mean-inference method is equally powerless; less conservative alternatives require explicit assumptions, sensitivity analysis and separate validation.

If **both** a positive global claim and a *negative global claim* are to be formally protected at an overall 0.05 directional family level, a simple prospective approach allocates alpha/2=0.025 to each direction. The corresponding bounded-mean margin at n=10 becomes about **0.859 accuracy-fraction units** (85.9 pp). This is distinct from the original symmetrical \`FAVORABLE / UNFAVORABLE\` proposal: its error interpretation and both-direction family must be decided **before** outcomes. Restricting formal claims to positive only, with negative labeled descriptive, is also an option to review.

The positive joint claim has the mathematical intersection–union structure: if at least one of the two true endpoint means is nonpositive, requiring *both* independently valid one-sided component tests to reject controls the global false-positive probability by the size of that nonpositive endpoint's test, regardless of correlation **between endpoints**. This logic relies on component tests really having their claimed size; the current n=10 percentile-bootstrap procedure has no distribution-free 5% guarantee, as PR #42 demonstrates.

## Exact 10-seed positive-sign alternative, with a different estimand

Define \`K = count(D_i > 0)\` across **all ten** independent seed trials. Treat \`D_i == 0\` as **nonpositive**, not as dropped or a newly imputed favorable trial. Under \`H0: P(D>0) <= 0.5\`, \`K\` is stochastically dominated by \`Binomial(10, 1/2)\` if seed signs are independent Bernoulli trials with a common null upper probability; the exact one-sided p-bound is its upper tail:

\`\`\`text
K >= 9: (C(10,9)+C(10,10)) / 2^10 = 11/1024 = 0.0107421875
K >= 8: (C(10,8)+C(10,9)+C(10,10)) / 2^10 = 56/1024 = 0.0546875
\`\`\`

Consequently, at one-sided alpha=0.05, **at least nine positive out of ten** are required *for each co-primary endpoint*. Unlike standard implementations of the median sign test that omit zero differences, the fixed-n positive-prevalence test here regards zeros as nonpositive and does **not** drop seed pairs. The sampling/independence assumption still must be justified, and incomplete or nonfinite runs must fail-closed.

With a hypothetical true fraction of seeds showing positive change \`p=0.9\`, the chance of at least nine positives is only **73.61%** per endpoint; at \`p=0.8\`, **37.58%**; at \`p=0.7\`, **14.93%**. These are **one-endpoint** probabilities. Requiring both endpoints jointly can reduce power further depending on their unknown dependence. These exact binomial calculations are theoretical, not real model outcomes.

For a *formal negative* sign-prevalence conclusion, the reviewer must define a negative-sign null, tie handling and two-direction error allocation in advance. A favorable finding for positive prevalence does **not** imply a mean increase in accuracy, particularly under rare catastrophic losses.

## D2 scientific identity is a separate unresolved decision

The current \`fixed_sparse_mlp\` is a static **fixed-masked residual MLP**, with two masked dense \`F.linear(x, weight*mask)\` body layers, post-step re-projection of masked weights and optimizer moments, fixed mask, dense encoder/head and no structural rewiring. Its **physical allocated trainable parameters** ~1,172,226 closely match the SynapLoom reference's ~1,172,196; the mask's ideal analytical MAC count is a separate accounting tier from actual dense CUDA work. The completed six-run engineering cost means (progressive 11.1494 s, all-recruited 8.7055 s, fixed sparse 1.9894 s) show differences in 390-step timing, **not** matched actual GPU compute in the 8,000-step scientific study.

A reviewer must explicitly determine whether an **in-house static fixed-masked MLP at matched physical capacity** is an adequate scientific comparator for *this precise question*. If the intended claim is superiority over named state-of-the-art sparse continual-learning methods, an **independently fidelity-checked** baseline from the literature (e.g. explicitly admitted and accurately implemented methods) may be needed. The present baseline is **not** automatically RigL/NISPA, a true sparse kernel, or a compute-matched control. Never rename it to imply a literature method.

## Prospective decision form for independent reviewers (not automatically signed)

A reviewer should explicitly return all of:

1. **Estimand:** \`mean_of_paired_seed_differences\` vs \`positive_sign_prevalence\` vs other; scientific rationale, source of randomness, and whether ten fixed seed IDs can support an inferential population claim.
2. **Inference method:** exact one-sided test/CI, paired t with justified assumptions, or an alternative with verified finite-sample calibration. Provide null, alpha, two-endpoint intersection–union decision, positive/negative direction policy, tie and failure rules.
3. **Power and sensitivity:** substantiate small-n skew/outlier risk, approximate or exact size/power with meaningful accuracy-difference scales; decide if n=10 is adequate or the proposal should change before data.
4. **External comparator identity:** is capacity-matched fixed-mask MLP a valid scientific baseline for the claimed research question? Clearly prohibit actual-compute-matched claims, or introduce a properly specified comparator and revised registration.
5. **Formal admission:** attribute the independent reviewer and decision evidence. A model-generated report, PR merge, checksum or owner statement is **not** independent signoff.

Only **after** independent D2/D3 review should an owner consider a finite GPU-hours/walltime/retry budget (D4), a final source/feature/order/mask-hash-bound executable protocol and separately authorized real science runner.

## Safe CPU-only commands

\`\`\`bash
uv run --locked python experiments/scripts/m13b_d3_estimand_options.py --review-estimands
uv run --locked python experiments/scripts/m13b_d3_estimand_options.py \
  --review-estimands --output /tmp/m13b-d3-estimand-options-v1.json
uv run --locked python -m pytest -q tests/unit/test_m13b_d3_estimand_options.py
\`\`\`

Review scripts load **no scientific output**, models, protected features or GPU; there is no \`--select\`, \`--execute\`, \`--approve\` or real-results interface. The emitted record SHA is a serialization checksum, not a scientific signature.

### Mathematical background (for review, not a substitute for peer signoff)

- Hoeffding's inequality for independent bounded variables (mathematical reference): https://mathworld.wolfram.com/HoeffdingInequality.html . A related JMLR discussion is McAllester and Ortiz, *Concentration Inequalities for the Missing Mass and for Histogram Rule Error* (2003), https://jmlr.csail.mit.edu/papers/v4/mcallester03a.html
- NIST, *Sign Test* (classical nonparametric paired sign procedure and binomial exact framework): https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm — note this packet **does not drop zeros**, because it targets a positive-sign probability with a fixed ten-seed denominator.
- U.S. FDA, *Multiple Endpoints in Clinical Trials* (conceptual reference regarding preplanned multiplicity distinctions; clinical-trial guidance is **not directly an ML benchmark standard**): https://www.fda.gov/regulatory-information/search-fda-guidance-documents/multiple-endpoints-clinical-trials

**Decision remains:** \`D3_ESTIMAND_SELECTION_PENDING_INDEPENDENT_REVIEW\`; \`D2_BASELINE_FIDELITY_REVIEW_PENDING\`; \`CONFIRMATORY_SCIENCE_NO_GO\`. Prior H1a \`INCONCLUSIVE\`, \`ORDER_SENSITIVE\` and \`DIRECTION_FLIP_OBSERVED\` are untouched.
