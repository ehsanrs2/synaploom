# M1.3b — Independent D2/D3 methodological review request (pre-outcome)

**Purpose:** obtain independently attributable scientific decisions on **fixed-baseline identity and resource fairness (D2)** and **mean-effect inference, error control, and power (D3)** before freezing a real experiment. This is a request for expert review, **not** an expert review performed by its author.

**Current disposition:** \`INDEPENDENT_REVIEW_REQUEST_READY__D2_D3_UNRESOLVED\`. **No 30/90-run science execution is authorized.**

## Exact primary research question — owner-selected

The owner explicitly chose **average accuracy improvement**, not “wins in most seeds” ([Issue #29 decision](https://github.com/ehsanrs2/synaploom/issues/29#issuecomment-6557812225)). For the proposed primary class order \`O0\` and each endpoint \`m\`, the prospective mean estimand is

\`\`\`text
mu[m] = E_seed[
    accuracy(progressive_reserve, O0, seed, m)
  - accuracy(fixed_sparse_mlp, O0, seed, m)
]
\`\`\`

The two co-primary metrics are \`final_average_accuracy\` and \`late_task_within_task_mean_accuracy\`, measured as fractions in [0,1]. Both mean effects must have valid evidence in the favorable direction before a **global positive** conclusion. The scientific seed population and independence assumptions are **not yet established**. The initial \`501..510\` seed IDs and 10 paired observations are **proposals**, not a statistically powered, approved sample.

The \`all_recruited\` comparator and two prespecified alternate orders \`O1/O2\` are descriptive, not substitute confirmatory data. No after-seeing-results decision may elevate one of these to the main test.

## Current source and engineering record

- **Merged source basis on request creation:** \`main=789340b637994eb459d7e069cfc1de78a8f7afc6\` (merged PR #44). A later approved science protocol would require its own fresh pinned, independently verified execution-host commit SHA and provenance.
- **Design inventory:** [90-slot preregistration candidate](PREREGISTRATION_CANDIDATE_V1.md). Three arms × ten seeds × three orders: **90** *proposed* 8,000-optimizer-step trajectories (720,000 steps if all run). Original full benchmark is ten tasks × 20 epochs with full protected evaluation, not the one-epoch training-only pilot.
- **Resource evidence:** [six-run cost closure](SIX_RUN_COST_CLOSURE_AND_METHOD_REVIEW_V1.md). Owner-local validated 3 arms × 2 manual GPU training-only 390-step runs; mean wall seconds **11.1494 progressive**, **8.7055 all-recruited**, **1.9894 fixed-sparse**, and max PyTorch CUDA allocated **27.50 / 28.26 / 39.69 MiB** respectively. This evidence is **owner-shared terminal output; reviewer has not been given raw on-host \`/tmp\` artifacts** and cannot infer full-science GPU hours, equal compute or learning outcome.
- **D2 anatomy:** [fixed-masked implementation audit](D2_D3_SCIENTIFIC_FIDELITY_REVIEW_V1.md) plus \`src/synaploom/research/fixed_sparse_baseline.py\`. Custom fixed-mask residual MLP, ordinary dense \`F.linear(input, weight*mask)\` kernels, mask projection of weight/optimizer state; physical trainable element count **1,172,226 vs 1,172,196** SynapLoom. **Not established as a faithful RigL/NISPA or actual sparse-kernel/compute-matched baseline**.
- **D3 method failure:** [exact constructed small-n zero-mean counterexample](D2_D3_SCIENTIFIC_FIDELITY_REVIEW_V1.md), source \`src/synaploom/research/m13b_d3_exact_null_review.py\`. For synthetic iid two-point paired deltas +0.01 with probability 0.9, −0.09 with probability 0.1, both true mean endpoint deltas are zero, yet the original *both-needed marginal percentile-bootstrap* positive verdict fires on all-ten-positive samples, a mathematically exact **≥34.86784401% false-favorable event** for that *constructed* distribution. Not an observed SynapLoom error rate.
- **D3 alternatives and power:** [estimand-choice memo](D3_ESTIMAND_DECISION_OPTIONS_V1.md), [owner mean-power memo](OWNER_MEAN_ESTIMAND_AND_POWER_REVIEW_V1.md). The former distinguishes sign prevalence (not the chosen primary scientific target) from positive mean; the latter computes *invented* Gaussian-known-SD design proxies. Example **invented** +2 percentage-point mean difference and 5 percentage-point paired-seed SD: one-endpoint n=10 planning power ~35.2%; n≥39 necessary for 80% joint power under marginal power criterion, n=54 sufficient under two 90%-marginal criterion, **only within this hypothetical model**. Neither n nor minimal important improvement is approved.

## External reviewer D2 — decision questions

The D2 reviewer should be competent to audit sparse-ML implementation, optimization, fairness and continual-learning comparison validity. Provide explicit evidence and answer:

1. **Identity:** Is the candidate appropriately described as an *in-house static fixed-masked residual MLP*? Which actual model ingredients differ from SynapLoom and literature baselines? Does this model test the intended **capacity-matched** hypothesis? If a named published external method is asserted, does its original algorithm, mask/recruitment mechanism, optimizer, initialization, hyperparameters and evaluation protocol actually match? Without that proof, prohibit its use as a “RigL/NISPA/SOTA reproduced” baseline.
2. **Parameter/capacity accounting:** Recompute physical trainable parameter elements, active vs masked weights, optimizer-state and mask-buffer allocation, initialization and seed policy. Is parameter allocation comparison scientifically meaningful under two different mechanisms? If insufficient, specify the minimal redesign.
3. **Computation:** Is the ideal linear-MAC proxy accurately described and separated from dense-masked *actual* forward/backward, optimizer and routing/dispatch work? Which measured cost metrics should accompany accuracy (per-task GPU wall time, CUDA allocated/reserved peaks, memory/optimizer state, and if feasible profiler/energy metrics)? Distinguish **measured engineering runtime** from full protected scientific workload.
4. **Reviewer disposition:** \`ACCEPT_CAPACITY_MATCHED_CANDIDATE\`, \`REVISE_COMPARATOR_OR_ACCOUNTING\`, or \`REJECT_FOR_TARGET_CLAIM\`, plus evidence and conditions. Even an ACCEPT is **D2-only** and does not authorize runs.

## External reviewer D3 — decision questions

A qualified statistical-methodology reviewer should examine both endpoint definitions and per-seed stochasticity and return written answers, *before any original protected confirmatory result is revealed*:

1. **Population and estimand:** Define the random-seed population and sampling algorithm. Are successive fixed integers \`501..510\` scientifically defensible as independent seed draws for generalization? Are per-seed tasks/class orders paired fairly across algorithms? State exactly which expectation is tested.
2. **Minimum meaningful effect:** State and justify the **minimum scientifically important mean difference** separately for both endpoints (e.g. the *illustrative* +1/+2/+5 percentage-point values are NOT owner thresholds). If a strict \`mu>0\` claim is used, distinguish *statistical* from *practical* importance and avoid inflating either.
3. **Primary mean test:** Propose a prespecified valid *one-sided mean-effect* inference for each endpoint, including assumptions and a credible finite-n size assessment. A t-based approach needs small-n skew/outlier sensitivity; exact sign-flip needs symmetry/exchangeability and cannot be justified merely from mean zero; positive-sign tests target a different estimand. The original n=10 percentile bootstrap **is not distribution-free calibrated** as shown above. If the proposed test depends on distributional assumptions, define diagnostics and sensitivity methods that do not permit outcome-driven method switching.
4. **Joint claim and directions:** Define the co-primary both-needed intersection–union rule and the **per-component** α/error guarantee. Distinguish a formal negative claim, which may require its own preallocated directional error budget, from a descriptive unfavorable result; specify multiplicity for any further confirmatory contrasts.
5. **Seed count and power:** Using scientifically defensible, independently sourced *preoutcome* effect-size/variance assumptions, calculate attainable power (both endpoints, realistic dependence), compare n=10 to alternatives, and decide whether the full 90-slot plan stays or must be prospectively amended. If powered mean inference is impossible at acceptable GPU budget, issue a candid **exploratory-only/underpowered** disposition instead of claiming proof.
6. **Missing/failure and selection:** Define finite-timeout/oom/provider/routing failures, no dropping seeds, deterministic restart/retry limits, complete provenance of all planned slots, unblinding order and fail-closed interpretation. Do not reuse O1/O2 or all-recruited after observing O0 failures.
7. **Reviewer disposition:** \`ACCEPT_MEAN_D3_METHOD_WITH_CONDITIONS\`, \`REVISE_METHOD_AND_N\`, or \`REJECT_CONFIRMATORY_POWER\`; state evidence, assumptions and unresolved risks.

## Evidence and acceptance controls

- Two **separately attributable** expert reviews (D2 and D3; one person only if independently qualified in both) with identity, date, source revision, conflict/independence declaration, explicit decision and rationale. The authors of this repository's code, this AI-generated review request, and its owner cannot self-mark an independent check complete merely by merging a PR.
- Use [the blank reviewer response template](INDEPENDENT_D2_D3_REVIEW_RESPONSE_TEMPLATE_V1.md). Do **not** prefill favorable decisions; no GitHub merge/green CI/SHA automatically constitutes reviewer acceptance.
- Reviewer should be given source and original on-host evidence if raw cost/provenance validation is material. This PR contains owner-reported summaries only.
- If any review changes seeds, statistical method, endpoint, baseline or order scope, **revise and re-audit the *whole* prospective preregistration before outcomes**, re-run CPU consistency gates and set a new SHA-bound version. Preserve historical proposal hashes and records without rewriting them as if originally frozen.

## After independent D2/D3 review (not before)

D4 would require an independently checked source/feature/order/mask/fixed-science-plan freeze, a finite measured full **20-epoch plus protected eval** resource and GPU-hours/retry ceiling, qualified manually initiated science runner with fail-closed evidence, and **distinct explicit owner execution approval**. Neither \`PR #45\` merge nor completion of this handoff is a substitute.

**Decision boundary: \`D2_INDEPENDENT_REVIEW_MISSING\`; \`D3_INDEPENDENT_REVIEW_MISSING\`; \`FULL_GPU_SCIENCE_NO_GO\`. No new GPU, science training, protected data read, or research-outcome interpretation was performed for this request.**
