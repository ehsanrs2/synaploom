# SynapLoom M1.3b — D2 strict CPU Runner output-boundary contract (V1)

**Review date:** 2026-11-09. **Status:** `D2_CPU_OUTPUT_CONTRACT_REVIEW_ONLY` / `RUNNER_SCIENCE_NOT_INTEGRATED` / `SCIENCE_GPU_NO_GO`.

Based on verified merged PR #57 (`main=9b0c30657d237a23d2e85896dce195ac5997261c`, Actions success). This is an **additive CPU-only research gate** following [Issue #29](https://github.com/ehsanrs2/synaploom/issues/29); no source edits to the production trainer, protected-evaluation loader, model architecture, or historical evidence.

## Why this matters

The existing `m13b_d2_eval_guard.validate_co_primary_observation` validates a complete 10×10 accuracy lower triangle and ten 20-epoch learning curves; it accepts any strictly increasing global-step labels. In the actual H1a2 reference shape, **each task has 5000 TRAIN examples** and minibatch B128, so there are **ceil(5000/128)=40 optimizer steps per epoch**; 10 tasks ×20 epochs ×40 = **8000 exact steps**. The stricter output-boundary contract rejects a subtly shortened/shifted run whose recorded steps only increase, and checks additional run-level fields before a future scientific result could be admitted.

The original `run_continual_experiment` returns a `ContinualRunResult` whose real `to_dict()` schema also includes evaluation loss matrix, train loss history, stream task manifest, structural history, timings, `global_steps`, configuration and metadata. A *fabricated* `ContinualRunResult` object (NOT an actually trained one) is passed through **the same real serializer** and the existing D2 endpoint guard, so the boundary test covers the actual data shape without consuming GPU or real accuracy.

## Additive components

- `src/synaploom/research/m13b_d2_runner_output_contract_cpu.py`: `validate_shape_and_policy` and `validate_synthetic_registry` (only synthetic fixtures, explicit `scientific_claim=NONE`), plus the reference evaluation *position* digest.
- `tests/unit/test_m13b_d2_runner_output_contract_cpu.py`: actual result serialization roundtrip through `ContinualRunResult.to_dict`, source evaluation-batch iterator on 1000 synthetic positions, exact-step mutation/fail-closed tests, complete 2-seed ×2-arm synthetic slot registry, and a tiny CPU `FixedSparseMLP` SGD momentum/post-step mask-hook test.

**Important trust boundary:** Even when all checks pass, metadata fields such as source commit, feature SHA, order SHA and mask SHA are supplied **by the synthetic fixture itself**, and checked only against expected literals. The SHA over its JSON is a self-contained content checksum, not an independent trusted machine/source/feature/mask attestation. The existing optimizer hook is tested for inactive weights and momentum state on a small CPU model, **not** asserted installed and verified for every optimizer step of a final full-shape runner.

### Proposed exact invariants

| Dimension | Strict CPU-only contract |
|---|---|
| Evaluation grouping | Ordered B256, **256+256+256+232** for 1000 positions; SHA-256 over the **position-range representation only** (not protected feature membership) |
| Co-primary correctness | Reuse standalone completeness guard: ten 10×10 stage/task matrix rows, explicit null future cells; ten 20-epoch learning curves; independently recomputed two primary summaries |
| Step identity | Every declared epoch checkpoint exactly `(stage*20 + epoch)*40` (epoch is 1-based); final `global_steps==8000` |
| Loss/integrity | Required 10×10 evaluation loss matrix with finite nonnegative lower triangle/null upper triangle, ten-element train loss/timing/diagnostic series, ten ordered structural task snapshots |
| Stream identity | Frozen O0 ten classes per task, task IDs 0–9, 5000 training examples and 1000 evaluation examples each |
| Synthetic slot identity | Explicit CPU device, two *engineering-only* seeds 13/17 (not candidate science 501–510), two declared arm names, one record per slot and zero missing/duplicate records; no retries |
| Metadata | `SYNTHETIC_CPU_D2_RUNNER_FIXTURE_V1`, `actual_optimizer_steps=0`, `protected_eval_accessed=false`, no CUDA allocation or evaluation-size sweep, no scientific claim |
| Fixed sparse | Independently *supplied* mask fingerprint must equal fixture-declared fingerprint; separate CPU test exercises true `install_fixed_mask_guard` on SGD momentum and deliberate corruption |

**Position digest is not an actual sample-membership digest.** Because the legacy Progressive router computes batch-mean queries, real protected evaluation identity also depends on the exact ordered feature/label tensors. This report does *not* claim to bind those tensors; a true science runner must compute and attest their identity prior to evaluation. No evaluation batch size may be changed after inspecting protected accuracy to pick a favorable result.

## Failure policy and negative evidence

The strict CPU validator raises `ValueError` on missing or shifted optimizer steps, missing/invalid stage accuracies or losses, altered stream/order, bad mask digest, unapproved training/evaluation configuration, nonfinite numbers, unregistered seed/arm, duplicated slot, partial paired matrix, synthetic failure/attempt re-labeling or unauthorized GPU/protected markers. It cannot establish that an upstream model actually trained those 8000 steps just because its JSON claims to. Such evidence must come from separately trusted run-process provenance, optimizer instrumentation, pinned binaries/files, immutable evidence root and failure/attempt audit.

Never drop failed or nonfinite seed pairs to obtain a favorable estimate. A real failure must remain a separate terminal outcome; this synthetic contract **requires all selected slots** but does not prescribe an unapproved scientific retry policy.

## Remaining D2 technical gates

1. Integrate a post-run fail-closed gate **inside a separately authorized future manual science wrapper**, with independent source checkout/feature content hashes and optimizer-step instrumentation. This PR does not expose a science runner or approve a command.
2. Pin actual ordered sample/batch membership and B256/B232 composition against reference evaluator behavior; CPU nonprotected/train-derived batch-companion sensitivity probes are still recommended.
3. Preserve fixed-mask identity and optimizer state across task boundaries and fail closed if the post-step hook is absent, mutates, or becomes detached on resume. The tiny CPU test cannot certify future full-length training.
4. Freeze D1 arms/orders, a final D3 inference method/seed-generating distribution/power ambition, and D4 total GPU-hour cap before protected results or a science GO.
5. Register and preserve all complete/partial/failure attempts with immutable evidence and cryptographic provenance; matching JSON fields alone does not attest the source that emitted them.

**Not authorized**: model training, GPU time, real protected evaluations, 8,000-step scientific runs, use of proposed science seeds, final protocol freeze, method/n selection, or conversion of a CPU shape check into proof of fairness/effect.

## Reproduce safely (CPU)

```bash
uv run --locked python -m pytest -q \
  tests/unit/test_m13b_d2_eval_guard.py \
  tests/unit/test_m13b_d2_runner_output_contract_cpu.py
```

**State:** `D2_CPU_BOUNDARY_GUARDS_READY_FOR_REVIEW`; `D2_REAL_RUNNER_INTEGRITY_OPEN`; `D3_FINAL_METHOD_PENDING`; `D1_SCOPE_PENDING`; `D4_TOTAL_GPU_BUDGET_PENDING`; `SCIENCE_GPU_NO_GO`.
