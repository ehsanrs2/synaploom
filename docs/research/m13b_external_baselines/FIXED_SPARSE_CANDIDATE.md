# M1.3b Fixed Sparse MLP — candidate feasibility adapter v1

**Status: EXPERIMENTAL / NOT FROZEN / NOT ADMITTED TO A SCIENTIFIC COMPARISON.** Tracks [Issue #29](https://github.com/ehsanrs2/synaploom/issues/29). Based on merged [PR #30](https://github.com/ehsanrs2/synaploom/pull/30).

## What is implemented

- Separate `src/synaploom/research/fixed_sparse_baseline.py`, without edits to production `SynapLoomModel`, `TrainingEngine`, `SparseRouter` or original H1a protocols/results.
- `FixedSparseMLP`: identical high-level encoder → LayerNorm → residual two-layer GELU body → head to existing `DenseControlModel`. Only body `Linear` weight matrices are masked; bias, encoder, normalization and head stay dense.
- Each body weight mask is drawn once from a locally seeded CPU generator with **exact specified active count** and registered as a persistent boolean buffer. No recruitment or mask rewiring. The two masks have distinct seeds. Selection is independent of training/evaluation data, labels, metrics and model initialization seed.
- Forward multiplies the dense weight tensor by its frozen mask; inactive-gradient entries are algebraically zero. A separately installed optimizer post-step hook projects masked weights and shape-matched momentum/Adam state back to zero after `optimizer.step`. It verifies the optimizer covers exactly the model parameter set. The hook **must** stay installed for the entire run, including after any optimizer reconstruction; never assume a checkpoint automatically reattaches it.
- The baseline emits `DenseOutput` (logits and representation), so the existing class-incremental training/evaluation harness can consume it. Only a tiny synthetic engineering smoke is covered. **No CIFAR-100 scientific training occurred.**

## Candidate budget derivation — not a frozen scientific arm

`build_h1a_fixed_sparse_feasibility(h1a2_protocol(), seed=..., mask_seed=...)` creates an ephemeral configured SynapLoom reference (full physical N=16 parameter pool, H1a initial recruited R=4, K=4), then:

1. Choose one residual MLP hidden width nearest to reference **total physically allocated trainable parameters**, including the 16 dormant+active microcircuits and all routing tensors, using existing `dense_hidden_width_for_target`.
2. Compute the reference's **ideal linear MAC count at B128**, with the existing `estimate_synaploom_linear_macs_per_batch`. This counts reference encoder, router-query/projection/candidate scoring, K experts and classifier but explicitly excludes nonlinearities, normalization, data movement and real kernel overhead.
3. Subtract the candidate MLP's dense encoder and head MACs, then allocate the remaining *ideal* MAC target across the two masked body matrices (fixed total number of nonzero weights, divided evenly between equal-sized matrices). Freeze the resulting masks by deterministic seed. This is an arithmetic count match — not training-throughput or GPU compute parity.
4. Record: physical trainable count; exact active body weights and density; ideal sparse MACs; **dense masked kernel MACs actually used by this PyTorch implementation**; mask buffer bytes; SHA-256 over named, shaped mask bit arrays; resource errors relative to the initial SynapLoom reference. Training optimizer memory, actual wall time and CUDA peak are `None` until **measured**.
5. Mark proposed numerical review gates `total error <= 2%` and `ideal linear MAC error <= 10%`, but no scientific fidelity/promotion claim is made even if both hold. These thresholds were design proposals in `FAIRNESS.md`, not observed outcomes. Verify against the complete current H1a reference configuration before any final freeze.

### Important budget distinction

- **Allocated:** Every parameter tensor element exists physically in `FixedSparseMLP`, even if its connection is zeroed by a mask.
- **Ideal mathematical sparse work:** Count active nonzero weights plus dense shared layers; this figure can numerically match SynapLoom's ideal top-k MACs.
- **Actual compute:** `F.linear(input, weight * mask, bias)` **still runs a dense matrix multiplication** and adds mask-multiply overhead. The `implementation_dense_linear_macs_per_batch` field is a lower-bound-style linear-layer arithmetic description, NOT measured FLOPs or real GPU efficiency.
- Optimizer state and mask storage are extra allocation; report them separately. No density target is justified scientifically yet; this candidate uses derived counts rather than claiming a universal optimal mask.
- `fixed_small` from the old H1a protocol is **not** this external unstructured fixed sparse MLP.

## Qualification and safety gates

Proposed tests in `tests/unit/test_fixed_sparse_baseline.py` check:
- exact H1a-shaped resource ledgers and distinctions;
- stable masks and deterministic hash across fixed mask seed, independent of model initialization;
- source and trained mask integrity, masked gradient inactivity and SGD-momentum/AdamW post-step projection;
- strict buffer `state_dict` roundtrip and changed-mask rejection;
- compatibility with seen-class masking and a tiny synthetic class-incremental harness;
- no mutation of production default model/router/engine code.

No CIFAR features, no GPU/profiler, no final accuracy/forgetting metrics or confirmation seeds are used. Source/historical protocol artifacts remain unchanged.

## Before any owner-approved M1.3b scientific run

1. Review and freeze total-allocated versus ideal-work matching, including model initial R=4 vs final R=13 and optimizer state/temporary buffers. Decide whether O0-only or preregistered multi-order O1/O2 robustness is mandatory.
2. Define a fair baseline name, mask distribution and exact seeds, optimizer/hook state restart policy, source/feature/class-order/stream provenance, budget report and non-finite fail-fast behavior.
3. Add a **separate** versioned runner with `--dry-run` and strict immutable manifest that refuses unexpected seeds/variants/output paths; do not adapt `EXP-H1A-002`.
4. Execute only tiny bounded hardware feasibility tests first; inspect real GPU wall/memory and potential masked-kernel slowdown. Run confirmatory training only after separate protocol freeze and explicit owner authorization.
5. Keep RigL/NISPA conditional. No original-method fidelity claims are made by this Fixed Sparse adapter.

**Research decision:** GO for review and bounded CI feasibility only. NO-GO for scientific baseline admission, GPU performance equivalence, TrainingEngine sample routing or resuming historical scientific checkpoints.
