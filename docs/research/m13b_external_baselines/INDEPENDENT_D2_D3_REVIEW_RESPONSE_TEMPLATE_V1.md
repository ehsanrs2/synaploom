# M1.3b — Independent D2/D3 reviewer response template (BLANK)

**Purpose:** a qualified, external reviewer may complete this template after inspecting the actual source and evidence. **This file is not a review submission, an approval, a digital signature, or consent to train. All decisions default to PENDING.** Duplicate it into a separately named, attributable submitted review; do not overwrite this blank template or earlier historical records.

Read [review request](INDEPENDENT_D2_D3_REVIEW_REQUEST_V1.md) and supporting source documents. The reviewer must **not** infer numerical scientific accuracy outcomes from synthetic simulations or one-epoch GPU cost measurements. The prospective main goal is *mean paired accuracy improvement* on both O0 co-primary endpoints, as selected by the owner, but the final statistical method and baseline remain unapproved.

## Provenance of the review submission — reviewer completes

| Field | Response (blank; reviewer supplies) |
|---|---|
| Review scope | [ ] D2 comparator and actual resources  [ ] D3 statistics and power |
| Reviewer name and affiliation or verifiable professional identity | **TO BE SUPPLIED** |
| Reviewer qualifications relevant to selected scope | **TO BE SUPPLIED** |
| Source commit SHA and source documents read | **TO BE SUPPLIED** |
| Exact local/remote evidence, hashes and code reviewed | **TO BE SUPPLIED** |
| Review date/time and timezone | **TO BE SUPPLIED** |
| Independence/conflict statement (project author, owner, collaborator, funding, previous involvement, or AI assistance) | **TO BE SUPPLIED** |
| Non-public data or real confirmatory outcomes seen? If yes: explicitly disclose and assess blindness | **TO BE SUPPLIED** |
| Relevant methodological sources/standards | **TO BE SUPPLIED** |

## D2 — scientific comparator decision (leave blank if D3-only)

**Disposition (select exactly one only after review):** [ ] \`ACCEPT_CAPACITY_MATCHED_CANDIDATE\` [ ] \`REVISE_COMPARATOR_OR_ACCOUNTING\` [ ] \`REJECT_FOR_TARGET_CLAIM\`.

**Evidence-backed responses:**

1. Scientific claim the comparator can and cannot support, including whether it is a named literature method or an in-house static fixed-mask control: **TO BE SUPPLIED**
2. Architecture, initialization and training mechanism fidelity verified (include file:line, source SHA and/or reproduction evidence): **TO BE SUPPLIED**
3. Trainable physical/active parameters, dense-vs-masked weight accounting, optimizer states and allocation: **TO BE SUPPLIED**
4. Ideal MAC vs executed dense CUDA work and proposed fair reporting tiers (resource equivalence **must not be presumed**): **TO BE SUPPLIED**
5. Full 20-epoch and evaluation workload cost/fairness measurements required beyond existing six 390-step calibrations: **TO BE SUPPLIED**
6. Required redesign, minimum independent tests and re-review criteria, if applicable: **TO BE SUPPLIED**
7. Any unresolved issue that makes D2 science \`NO_GO\`: **TO BE SUPPLIED**

## D3 — mean-effect statistical decision (leave blank if D2-only)

**Disposition (select exactly one only after review):** [ ] \`ACCEPT_MEAN_D3_METHOD_WITH_CONDITIONS\` [ ] \`REVISE_METHOD_AND_N\` [ ] \`REJECT_CONFIRMATORY_POWER\`.

**Evidence-backed responses:**

1. Target expectation \`E[accuracy(progressive)-accuracy(fixed)]\`: population definition, pairing, seed generation and independence assumptions: **TO BE SUPPLIED**
2. Endpoint measurement, unit, direction and minimum **scientifically important mean effect** separately for both endpoints: **TO BE SUPPLIED**
3. Proposed *one-sided mean inference* method, null hypotheses and exact global both-needed success rule (not a sign-prevalence surrogate): **TO BE SUPPLIED**
4. Formal overall type-I/coverage error policy and any separately formal negative-direction hypothesis: **TO BE SUPPLIED**
5. Method assumptions, evidence for plausibility, explicit zero-mean skew/outlier/heteroscedastic sensitivity including implications of the exact bootstrap counterexample: **TO BE SUPPLIED**
6. Preoutcome effect/paired-SD justification, α and joint two-endpoint power under plausible dependence, and **required prospective number of paired seeds**: **TO BE SUPPLIED**
7. Failure, incomplete trajectories, nonfinite endpoints, timeout/retry caps, intentional exclusions and provenance/unblinding policy: **TO BE SUPPLIED**
8. O1/O2 order sensitivity and \`all_recruited\` descriptive-only boundaries: **TO BE SUPPLIED**
9. Required modifications to the 90-slot candidate and frozen protocol/version policy: **TO BE SUPPLIED**
10. Unresolved issue that makes D3 science \`NO_GO\`: **TO BE SUPPLIED**

## Reviewer conclusion and custody

- **Exact evidence supporting decision, including negative findings:** **TO BE SUPPLIED**
- **Mandatory conditions and who verifies them:** **TO BE SUPPLIED**
- **Independent reviewer acknowledgement of scope, provenance and conflicts:** **TO BE SUPPLIED**
- **Separate reviewer-issued signature or trusted attribution, if available:** **TO BE SUPPLIED**
- **Owner acknowledgement AFTER an authentic reviewer submission:** **PENDING**
- **Scientific protocol frozen?** **NO**
- **Owner GPU-hours, walltime and retries approved?** **NO**
- **Scientific execution authorized?** **NO**

**No field in this template can authorize an experiment by itself.** A reviewer may conclude the planned 10-seed, 90-run design is insufficient or the baseline invalid. Preserve unfavorable reviews and record revisions as new versioned artifacts, not changes to prior submitted evidence.
