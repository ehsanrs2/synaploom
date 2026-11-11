# SynapLoom M1.3b / D2 — detached-reference evidence trust contract (V1)

**Status:** `D2_TRUST_THREAT_MODEL_AND_SYNTHETIC_DETACHED_PIN_CPU_REVIEW_ONLY`. **No GPU, no protected feature data, no real science seeds, no training, no owner science GO.**

Research base is the **verified merged** [PR #60](https://github.com/ehsanrs2/synaploom/pull/60), merge/current-`main` commit `ecc5af7eab984249233ff79296dd7ff93d9fc153`; its exact-head [hosted CI #42821072373](https://github.com/ehsanrs2/synaploom/actions/runs/42821072373) was SUCCESS. This PR contains **new isolated source, tests and design documentation only**. It does not replace, modify or enable `run_continual_experiment`.

## The critical distinction

Earlier D2 work established (a) fabricated complete-looking 8,000-step result schema checks, (b) *real 12-step* CPU Runner execution on tiny synthetic data, (c) *real evaluator* B256/B232 processing of **1,000 invented** rows and (d) locally exclusive file creation, hashes and failure preservation. These are useful software-path and reproducibility checks, **not trusted scientific provenance**.

An actor able to rewrite a writable JSON attempt and recompute its internal SHA-256 can potentially make it self-consistent again. A linked sequence of hashes alone does not prevent this; it needs an independently preserved **expected root digest**, whose source and custody are authenticated separately. In addition, a source hash displayed inside an execution's own results is self-declared, even if computed from its imported file. True source integrity must check **externally pinned** source identifiers and actual byte content *outside* the untrusted result producer.

The current implementation deliberately demonstrates **only** the digest equality component. It does not implement a signature, identity verification, external witness, trusted execution environment, a signing key, or immutable/WORM evidence custody. A caller can fabricate both the record and its alleged 'detached' root. A PASS is therefore **not** an authoritative scientific certification.

## New CPU-only code

`src/synaploom/research/m13b_d2_trust_envelope_cpu.py` provides:

- A strictly fixed **toy** registry with engineering seeds **13/17**, two comparison arms and four O0 slots, all labeled synthetic. It accepts only terminal states `TINY_SYNTHETIC_CPU_SUCCEEDED` and `TINY_SYNTHETIC_CPU_FAILED`, and at most 12 toy optimizer post-hook observations. A nominal success must declare exactly 12.
- Each event includes its position, registered slot, terminal status, observed toy step count and preceding digest. It produces a deterministic SHA-256-linked event chain and envelope root, using canonical JSON with `allow_nan=False`.
- `verify_synthetic_envelope` requires **four separately supplied reference values**: detached expected envelope-root SHA, expected source SHA, expected synthetic 1,000-row membership/layout SHA and expected fixed-mask SHA. It also requires all four slots in exact order, correct failure states and step limits, exact chain linkage, expected B256+256+256+232 *declared* shape, and all explicit science-denial flags.
- A rehashed ledger where a failure was rewritten into a successful terminal state fails against the old **detached root** even when all internal chain entries have been rehashed. A changed declared source, sample or mask SHA fails against the **separately passed expectations**. The external values in the toy tests are invented, not sourced from a trusted filesystem or signer.
- `d2_trust_requirements()` returns a descriptive, still-unimplemented 10-domain prospective contract for actual science.

`tests/unit/test_m13b_d2_trust_envelope_cpu.py` supplies **25 isolated CPU tests PASS** (no torch import needed): positive toy preservation; missing, duplicated, reordered, altered success/failure events; shifted step counters, wrong links, incorrect batch geometry, source/data/mask mismatches, forged owner/go flags, malformed roots, and attacker-rehashed valid internal chain with wrong detached reference. All tests use fabricated digests derived from non-scientific literal strings. The implementation and tests were tested as an actual `synaploom.research` package import in local CPU Python 3.13.5; hosted CI must confirm the committed PR head separately.

## Required separation of trust and policy

| Evidence layer | What is required for a prospective **real science** wrapper | Status now |
|---|---|---|
| **1 — Pre-outcome owner scope** | D1 arm/order and seed-generating population, D3 test/multiplicity/n/power plan, D4 bounded GPU-hours, retry/failure rules, explicit owner signatures; frozen *before outcomes* | **NOT APPROVED** |
| **2 — Executable source** | Independently fetched/checked Git commit/tree, dependency lock/installed package identities, verified runtime imports, runtime/model/optimizer source checks, signed/pinned environment record | **NOT IMPLEMENTED** |
| **3 — Dataset identity** | Exact byte SHA of the trusted bundle **before safe deserialization**, validated tensor shapes/dtypes, labels, per-task ordered original row IDs and all evaluation B256/B232 group-membership digest traces | **NOT IMPLEMENTED** |
| **4 — Runtime behavior** | Actual 8,000 optimizer steps per trajectory, 40 per epoch, 10 tasks ×20 epochs, per-step mask/optimizer state, fail-closed partial/failure/interrupt records, no silent seed exclusions | **NOT IMPLEMENTED** |
| **5 — Custody/witness** | External pre-run manifest commitment, independently anchored event-root digests, immutable or separately witnessed attempt ledger, owner/reviewer role separation, explicit postrun signature verification | **NOT IMPLEMENTED** |
| **6 — Inference authorization** | Recorded independent statistical review, explicit owner decision on selection and paired-seed registration, verifier rejection of mismatched/replaced analysis and unknown post-selection | **NOT APPROVED** |

The **original proposal with 10 seeds and percentile bootstrap is not silently approved**; the reviewed D3 alternatives still lack an accepted final power-qualified plan. Merging a schema or a CPU integrity checker does not authorize either methodology or execution.

## Outstanding choices / threat model

- **Who is trusted?** Propose separate roles: owner declares pre-outcome protocol, independent witness stores a protocol/source/data/registry digest before any science, executor only produces attempts, independent auditor checks all receipts. Same-machine owner and witness are weaker because both can be rolled back at once; no strong external witness exists in this PR.
- **What are the protected data?** Evaluation inputs/labels and outcome summaries must not be examined while selecting batch size, test method or seeds. Actual row provenance needs deterministic original source indices and feature/label hashes—not a fabricated sample-ID feature column.
- **What is a failure?** An incomplete or interrupted seed is retained as an explicit terminal state and triggers the predeclared failure policy; it is not dropped or reassigned to a fresh seed based on the resulting accuracies.
- **What is an acceptable comparison?** In-house fixed_sparse_mlp is not RigL/NISPA; hardware-equivalent actual computation has not been established. Legacy batch-mean routing is composition-sensitive, so the exact membership and batch geometry must be frozen.
- **What is a future science GO?** A **separate explicit owner command** after audited preregistration, trusted source/feature hashes, valid D3 inference, complete D1/D4 choices and full engineering verifier qualification. This PR has no API or CLI capable of performing such a run.

## Reproduce safely

```bash
uv run --locked python -m pytest -q tests/unit/test_m13b_d2_trust_envelope_cpu.py
uv run --locked python -m pytest -q \
  tests/unit/test_m13b_d2_eval_guard.py \
  tests/unit/test_m13b_d2_runner_output_contract_cpu.py \
  tests/unit/test_m13b_d2_live_cpu_runner_probe.py \
  tests/unit/test_m13b_d2_cpu_evidence.py \
  tests/unit/test_m13b_d2_trust_envelope_cpu.py
```

**Next after review:** explicitly resolve D1 science scope, D3 mean-difference test/n with paired-seed and joint power assumptions, and D4 GPU budget; implement and independently review a protected-data-agnostic full science **pre-run** verifier only after the prospective trust-role/owner-decisions are accepted.

**Disposition:** `D2_EXTERNAL_PIN_CONCEPT_CPU_PASS_LOCAL`, `D2_TRUSTED_SCIENCE_ATTESTATION_NOT_IMPLEMENTED`, `D1_D3_D4_PENDING`, `SCIENCE_GPU_NO_GO`.
