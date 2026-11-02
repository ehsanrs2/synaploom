# M1.3b — Owner research policy: AI-advisory D2/D3 accepted as the amendment basis

**Decision date:** 2026-11-02. **Status:** \`OWNER_ACCEPTS_AI_ADVISORY_FOR_INTERNAL_RESEARCH__TECHNICAL_REVISIONS_REQUIRED\`.

## Why this amendment exists

The project owner has explicitly decided that this **budget-constrained, owner-directed research effort** will **not require a human specialist's independent D2 or D3 signoff as a mandatory progress gate**. The two submitted reviews attributed to *ChatGPT GPT-6* are accepted **as critical analytical inputs and concrete revision requirements**, not as independently verified reviews or as proof of scientific validity.

This is a change to the project's **internal approval/governance procedure**. It is not a mathematical justification for an invalid test, a claim of independent peer review, a completed D2/D3 method acceptance, a protocol freeze, or authorization to execute confirmatory GPU runs.

This document prospectively **supersedes only** the *human-reviewer-is-mandatory* clause in previous D2/D3 handoff docs and issue checklists. It does **not** erase, rewrite or falsify those historical records. Their provenance and the original unsigned review templates remain unchanged. Outside review may still be requested voluntarily or required by a journal, sponsor or subsequent publication process.

## Evidence and attribution

- Parent science issue: [#29](https://github.com/ehsanrs2/synaploom/issues/29).
- D2 intake and scope: [#46](https://github.com/ehsanrs2/synaploom/issues/46). Review submitted to owner: **AI analytical report**, dated Nov 2 2026, suggested disposition \`REVISE_COMPARATOR_OR_ACCOUNTING\`.
- D3 intake and scope: [#47](https://github.com/ehsanrs2/synaploom/issues/47). Review submitted to owner: **AI analytical report**, dated Nov 2 2026, suggested disposition \`REVISE_METHOD_AND_N\`.
- Previous documentation-only external review packet: [merged PR #45](https://github.com/ehsanrs2/synaploom/pull/45).
- Source base at policy amendment start: \`ec5acd7b95a95fa62aa96cabce8834e59e668e41\`.

The AI reviews report source inspection, but the project has not independently ingested/replayed all host-local raw GPU artifacts or real protected scientific results. No external human reviewer has signed off; do **not** label the AI reports "independent human expert approval".

## Unchanged primary scientific objective and honest scope

The selected primary *estimand* remains the paired-seed **mean accuracy difference** \`progressive_reserve - fixed_sparse_mlp\` on predeclared primary order \`O0\`, for **both** \`final_average_accuracy\` and \`late_task_within_task_mean_accuracy\`. A positive global conclusion requires defensible mean-effect inference for **both** endpoints.

**Allowed conditional claim, if the final method and results support it:** Progressive Reserve has a larger mean of these accuracy endpoints **than this particular in-house fixed-masked residual MLP** in the frozen feature/task/order/seed-generating setting.

**Not established by that contrast:** that progressive recruitment timing *causes* any observed difference; superior continual-learning performance versus all published methods; equal actual GPU compute; independent coverage of new datasets, orders or mask topologies; proof of a biological-plasticity/forgetting mechanism.

\`all_recruited\` remains a **descriptive/mechanistic architecture-neighbor control** in the current candidate plan. If owner later wants a causal timing hypothesis, protocol and comparison family need prospective amendment and matched-control key-imprinting ablation, not a retrospective promotion of a favorable result.

## Lean D2 revision gate — no compulsory human signoff

**D2 is not PASS yet.** Required evidence prior to any genuine confirmatory science:

1. **Truthful comparator/resources:** call \`fixed_sparse_mlp\` an in-house *fixed masked residual MLP* implemented using **dense** masked linear operations. Distinguish physically allocated ~1.172M parameters from active weights, ideal linear-MAC estimates and **measured** CUDA time/memory. No actual GPU-compute matching claim.
2. **Frozen eval policy:** define evaluation batch size, data ordering/shuffling, drop-last/padding, task and epoch aggregation and the two endpoint formulas **before** protected outcomes. Because SynapLoom routes from the **batch mean**, evaluate sensitivity to batch composition/size on **synthetic or explicitly non-protected** inputs, report limitations, and specify up front whether a large sensitivity triggers \`NO_GO\` or a restricted descriptive claim. Do not select evaluation batch size based on desired accuracy.
3. **Correctness/integrity:** test fixed mask and optimizer-state guard under the intended training path; verify initialization and schedule differences are disclosed; enforce completeness and finite values for all entries needed to compute confirmatory endpoints rather than silently \`nanmean\`-dropping missing measurements.
4. **Claim boundary:** retain primary mean comparison with fixed sparse as **architecture contrast only**. A separate \`progressive_reserve\` vs \`all_recruited\` and key-imprinting ablation is necessary for a **specific causal recruitment-timing** claim, but is **not mandatory to test the narrower primary architecture contrast** if causality is expressly disclaimed.

Additional Experience Replay/NISPA faithful baseline, matched-GPU-hour training and full causal ablation are **optional follow-on studies**, not automatic expansion of the initial cost-constrained primary protocol.

**Internal D2 closure evidence:** exact source revision, read-only test outputs and hashes, frozen eval-configuration digest, honest resource ledger, written limitation/disposition and recorded **explicit owner acceptance**. No human signature required for this *internal* gate.

## Lean D3 revision gate — no compulsory human signoff

**D3 is not PASS yet.** Required prior to confirmatory outcomes:

1. Keep **population paired-seed means**, not "wins in most seeds". Define seed-generating population, pairing, independence assumptions and sampling/registry; the originally proposed consecutive seed IDs \`501..510\` do not automatically justify population inference.
2. Preserve **two co-primary O0 endpoints** and a one-direction positive intersection–union rule: both corresponding one-sided component tests must validly reject \`mu_m <= 0\`. Component tests must actually control their specified error under stated assumptions. A negative observed effect may be **descriptive**, not a newly formal negative claim unless separate predeclared alpha policy exists.
3. **Original 10-seed percentile bootstrap is not automatically admissible.** A constructed skewed joint mean-zero two-point null yields at least 34.86784401% false favorable events. Replacing it by a paired t-test alone also does **not** guarantee distribution-free 5% error. Select a defensible paired-mean procedure **before** real outcomes: consider proven independent-bounded-mean bounds, or an assumption-dependent t-like test only with explicit justified assumptions and distributional sensitivity. Validate exact code/theorem conventions, n, variance/endpoint dependence and type-I calibration. No post-hoc test selection.
4. **Power and feasible sample size:** define a scientifically important gain **separately** for both endpoints; evaluate two-endpoint joint power across plausible preoutcome effect, paired-SD, skew/outlier and correlation scenarios using the **actual chosen test**. Do not automatically promote illustrative known-SD Gaussian numbers (e.g. n=39/54) or 10 current seeds into approved n. If finite budget cannot support an adequately powered inferential claim, **explicitly designate an exploratory study and prohibit claims of statistical proof**, rather than silently weakening error control.
5. **Fail-closed:** predeclare missing/nonfinite endpoint behavior, failed trajectories, bounded recovery/retry, immutable attempt histories, O1/O2 and \`all_recruited\` descriptive-only uses. No dropping unfavorable seeds, adjusting tests based on observed results or turning multiple orders into independent O0 seed replicates.

**Internal D3 closure evidence:** versioned protocol specifying target/endpoint/test/alpha assumptions, corrected implementation and unit/adversarial synthetic type-I tests, prospective power/seed/budget analysis, complete failure policy, exact immutable test/log/source hashes and **explicit owner acceptance**. No human signature required for this *internal* gate.

## Remaining D4 and owner-run gates

Even when the D2 and D3 internal revisions are accepted:

- Revise and freeze the entire pre-outcome protocol (including seed count, order policy, data/feature/mask fingerprints, endpoints and provenance) and qualify a **manual, fail-closed** science runner.
- Derive a **finite affordable** GPU-hours/wall-time/retry ceiling based on relevant real full-workload evidence, not a naive scaling of two 390-step training-only runs.
- Require **separate explicit owner authorization** of the exact bounded GPU study; owner acceptance of AI reports and a merge of this documentation PR are not execution permission.
- Label future results honestly: \`OWNER_GOVERNED_INTERNAL_RESEARCH\`, **not** \`HUMAN_INDEPENDENTLY_REVIEWED\` or \`EXTERNALLY_VALIDATED\`.

## Present decision

\`\`\`text
D2_OWNER_POLICY: AI_ADVISORY_ACCEPTED__REVISION_REQUIRED
D3_OWNER_POLICY: AI_ADVISORY_ACCEPTED__REVISION_REQUIRED
MANDATORY_INDEPENDENT_HUMAN_SIGNOFF: WAIVED_BY_OWNER_FOR_INTERNAL_RESEARCH
PRIMARY_ESTIMAND: MEAN_PAIRED_SEED_ACCURACY_DIFFERENCE__BOTH_O0_ENDPOINTS
SCIENTIFIC_INFERENCE_METHOD: NOT_APPROVED
FINAL_SEED_COUNT: NOT_APPROVED
D4_GPU_RESOURCE_BUDGET: NOT_APPROVED
CONFIRMATORY_SCIENCE_EXECUTION: NO_GO
\`\`\`
