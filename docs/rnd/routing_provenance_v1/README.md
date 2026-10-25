# Routing checkpoint semantic preflight — research v1

**Status: DRAFT / NOT A LOADER.** Tracks [#21](https://github.com/ehsanrs2/synaploom/issues/21). Based on [opt-in session PR #24](https://github.com/ehsanrs2/synaploom/pull/24). No production checkpoint/model/router/training files are changed.

## Narrow purpose

`src/synaploom/research/routing_provenance.py` provides:

- `make_routing_metadata(model, spec, implementation_revision=..., precision_policy=...)` — a version-1 dictionary suitable for a *future* trusted writer's checkpoint `extra["routing"]`. Stage 1 always records `training_granularity="batch"` and records exact config topology, structural version and routing semantics.
- `preflight_routing_extra(extra, model=..., requested=..., intent="replay", allow_counterfactual=False)` — read-only semantics validation over a mapping that the caller *already possesses*. Returns `RoutingPreflightV1` for audit/provenance. It does not open files or call `torch.load`.

A legacy checkpoint with no routing metadata is assigned `batch / legacy / legacy_topk` only for default replay. Explicit `sample` evaluation against the same legacy model state is a **counterfactual** and requires `intent="counterfactual", allow_counterfactual=True` plus separate output/provenance at the future orchestration layer. Replay with any routing mismatch fails closed. `row_reference` versus `grouped` switching is currently treated as a semantic change unless explicitly counterfactual; no automated parity-based relaxation.

Supported Stage-1 topology is `q_max=0, message_rounds=0` for sample routes, with no implicit graph support. Unknown schema versions, malformed metadata, topology mismatches and sample-trained provenance are rejected before any model change.

## Critical limitations

1. **No checkpoint reader, serializer, loader or metadata-aware resume adapter is implemented here.** Existing production `save_checkpoint/load_checkpoint` remain unchanged. A metadata dictionary is not cryptographic proof of checkpoint bytes.
2. This preflight cannot detect a malicious/mismatched state_dict, checkpoint bytes changed after validation, incompatible optimizer/controller envelopes, or stale structural changes at load time. A future loader must validate the immutable checkpoint payload and routing semantics **before** touching parameters, optimizer, controller or RNG; check payload tensor keys/shapes and atomicity/failure semantics as separately designed.
3. Existing `torch.load(weights_only=False)` is a pickle-based trusted-file interface. Do not call it on untrusted downloaded checkpoints; do not treat this metadata check as a security boundary.
4. Structural-version field is recorded, not compared with the *current* graph-state version before load (an uninitialized target model may legitimately have another version). A future loader must compare it against the checkpoint's serialized state and verify the loaded model state.
5. `implementation_revision` and `precision_policy` are caller-declared identifiers, **not independently attested hashes**. Future reproducible evaluation requires source commit, model/feature/checkpoint/config digests and a separate immutable run manifest.
6. No auto-rollback or training transition support is claimed. Persistent mode/RNG/input identity and exception restoration need explicit transaction tests when the loader/session adapter is introduced.
7. This is compatible with opt-in inference/diagnostic exploration only; does **not** authorize a sample-trained resume, TrainingEngine integration or any historical H1a verdict change.

## Reproduction on qualified environments

```bash
uv run --locked pytest -vv tests/unit/test_routing_provenance.py
uv run --locked pytest -vv
```

GitHub Actions must execute actual test steps; CPU-skipped CUDA variants cannot qualify GPU numerical behavior. Do not claim these commands ran until the workflow reports them.

## Remaining acceptance gates

See `docs/rnd/routing_integration_design_v1/GATES.md`. Full checkpoint envelope compatibility, strict load/save, optimizer/controller/RNG roundtrip, actual pre-mutation loader rejection, missing/stale/unknown metadata handling and counterfactual run-provenance/output isolation remain **NO-GO**. Five-process representative performance gates and CPU/CUDA parity for promoted code are also outstanding.

Do not merge before review and explicit acceptance of the narrow read-only scope.
