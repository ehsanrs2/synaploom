# Trusted, pinned model-only checkpoint loader — research v1

**Status: DRAFT / integration gate only / no training or production loader.**

This R&D stage follows the merged metadata-only preflight ([PR #25](https://github.com/ehsanrs2/synaploom/pull/25)), without editing existing `src/synaploom/training/checkpoint.py`, the training engine, the default router, or historical experiment files.

## Bounded supported operation

`load_trusted_routing_model_checkpoint(path, model, requested=RoutingSpec(...), expected_sha256=..., trusted_source=True, intent="replay"| "counterfactual", allow_counterfactual=...)`:

1. Requires explicit trust in checkpoint **origin** and an independently pinned, lowercase SHA-256. Hash equality is **not** proof that an arbitrary file is trustworthy.
2. Reads the file once (default 512 MiB bound), checks the digest, then deserializes **those exact bytes** using `torch.load(weights_only=False)`. **WARNING:** This executes Python pickle; never pass internet/untrusted checkpoint data. An independently trusted checksum and trusted origin are strict prerequisites, not a sandbox or defense from malicious pickle.
3. Strictly checks envelope fields, model state_dict names, shapes, dtypes and dense layouts, then invokes the previously merged `preflight_routing_extra` before live model changes. Confirms serialized structural version agrees with declared routing metadata.
4. Loads into an independent deep-copied *shadow* model, validates graph invariants and computes its canonical state SHA before touching the live model.
5. Applies **only model state**, validates it, checks the resulting canonical state hash, and attempts rollback to an immutable previous-state snapshot if an ordinary commit exception occurs.
6. Returns versioned result: pinned checkpoint SHA/size, global step, semantic routing preflight, exact loaded state digest and three explicit `False` flags for optimizer/controller/RNG restoration.

A missing `extra.routing` is legacy batch for **replay**. Sample evaluation of legacy checkpoints requires explicit counterfactual authorization and a separately isolated output/run manifest established by a future orchestration layer. Sample-trained checkpoint recovery is forbidden. Caller owns modes and transactions.

## Explicit non-goals and limitations

- **Model-only inference/diagnostic loading**. If a checkpoint envelope includes optimizer/controller/RNG, they are never restored. No optimizer steps, training resumption, utilization updates or structural controller operations.
- The source may contain an optimizer/controller payload, but this adapter only validates their basic mapping shape. It does **not** establish their compatibility or safely restore them.
- Rollback is **best-effort** after the shadow preflight. This is NOT an atomic transaction across all devices, CUDA asynchronous errors, hardware failures, signals or concurrent model mutations. Concurrent callers must be externally excluded.
- The file is already in the trusted local filesystem. No remote downloading, file signature PKI or independent attestation of the expected digest is offered.
- A mismatch between legacy checkpoint architecture and current model may not be detectable if it preserves all tensor shapes. Historical metadata-free checkpoints cannot yield exact training semantics from their parameter bytes alone.
- Model hash equality does not validate training/evaluation semantic provenance. The adapter does not create the immutable run manifest or output directories required for counterfactual experiments.
- No CUDA qualified runs or representative performance promotion are claimed; the path stages a deep copy and model snapshot and can be expensive. Memory ceiling (file bytes only) does not cap intermediate PyTorch allocations.
- Test execution status must be established by a *new* GitHub Actions run. Never attribute the 114 earlier CPU test passes to this PR.

## Proposed verification

```sh
uv run --locked pytest -vv tests/unit/test_trusted_routing_checkpoint.py
uv run --locked pytest -vv
```

For eventual true training/resume parity: design a separate approval-gated transactional loader validating and rolling back optimizer/controller/RNG envelopes and rechecking model/feature/config/routing provenance, plus CPU/CUDA failures, stale transactions and corrupted/unknown payload handling. Do not promote this model-only R&D loader as production or as a training-resume contract.
