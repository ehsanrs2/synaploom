# Full-envelope shadow audit — research v1

**Scope: SHADOW AUDIT ONLY. No live restore, no training resume, no production loader.**

This follows merged PR #26. It evaluates the saved model, optimizer, controller and RNG envelope on **disconnected shadow objects**, with no optimizer steps or updates to the live system.

## Research audit contract

The opt-in function `audit_trusted_routing_envelope(...)`:

1. Requires independently trusted local checkpoint origin and an expected pinned SHA-256. Takes a bounded snapshot of its bytes and unpickles **only those verified bytes**. `torch.load(weights_only=False)` executes Python pickle: never pass untrusted checkpoint data. A SHA-256 digest is not a trust/authorization boundary.
2. Calls the existing strict model envelope checks and routing provenance preflight. **Only legacy batch routing is allowed**, and training/sample resume remains unapproved.
3. Requires model optimizer parameters to be linked exactly once; checks saved optimizer parameter-group count, parameters and hyperparameters against expected target configuration.
4. Requires controller checkpoint state to have exactly `last_applied_step` with a step bounded by `global_step`. External event loggers are explicitly unsupported.
5. Validates saved Python and torch CPU RNG states on **isolated RNG generators**, with NumPy state validated in an isolated `RandomState` if included. CUDA RNG entries are structurally checked but not replayed on hosted CPU CI.
6. Deep-copies `(model, optimizer, controller)` as one graph to retain cloned optimizer-to-model parameter identities, then loads the envelope into the clones. Compares every saved optimizer group/state value to the cloned loaded state, checks controller state and GraphState invariants, and computes canonical model digest.
7. Returns an audit result, with `live_state_restored=False` and `resume_authorized=False`. This is not a checkpoint application operation.

## Remaining correctness boundaries

**Controller state is incomplete for equivalent scientific resume.** The current `StructuralController.state_dict()` captures only `last_applied_step`, not recruitment, edge, consolidation policies, current observation or event logging configuration. Shadow roundtrip is *not* proof of policy-equivalent continuation.

This audit also does not attest checkpoint publisher or Python pickle safety, scheduler/scaler state, feature/config/source digests, other controller side-effects or CUDA PRNG replay. Pinned checkpoint bytes do not independently certify experiment provenance. Model, optimizer, controller and global RNG are **never** restored on a live process by this component.

## Test scope

`tests/unit/test_routing_envelope_audit.py` covers synthetic Adam/SGD momentum optimizer state roundtrips, read-only live invariance, state/parameter links, policy limitations, bad metadata, malformed optimizer groups/learning rates, invalid RNG/controller envelopes and trust/SHA/byte limits. Test execution must be verified against a new GitHub Actions run; prior CI successes do not validate this new code.

## Gate before further work

Record a frozen policy-identity contract, provenance manifest and optimizer/scheduler/controller compatibility matrix; only then design a separately reviewed **disposable** full resume/transaction experiment. CPU-only CI does not validate CUDA; production or training promotion remains NO-GO. No changes to default model, router, engine, training checkpoint code, existing experiment artifacts or historical verdicts.
