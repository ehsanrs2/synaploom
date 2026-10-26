# Controller-policy provenance contract v1 — research only

**Status: DRAFT / pure policy-identity preflight. No checkpoint loading, training resume or production change.**

Merged [PR #27](https://github.com/ehsanrs2/synaploom/pull/27) showed an essential limitation: `StructuralController.state_dict()` only saves `last_applied_step`. Two controllers with different recruitment schedules can have the same serialized state. The new `controller_policy_provenance.py` makes such policy drift detectable **without modifying the controller or the production checkpoint format**.

## Contract

`snapshot_controller_policy(controller)` returns a JSON-compatible V1 record with:

- Exact allowlisted Recruitment policy: `NoRecruitment`, `FixedScheduleRecruitment` (sorted steps), or `TaskBoundaryRecruitment`.
- Exact allowlisted EdgeInit policy: `NoEdges`, `ConnectToSelectedTopR` (top_r), or `SeededRandomR` (top_r, seed).
- Exact allowlisted Consolidation policy: `NoConsolidation` or `TaskBoundaryTopImportance` (top_n).
- Finite `imprint_noise_std` and `edge_gate_logit`, schema and kind.

Canonical SHA-256 uses sorted-key, compact, ASCII JSON with NaN disabled. The caller may write the record and independently pinned digest into a future authenticated checkpoint/run sidecar. `verify_controller_policy(record, controller, expected_sha256=...)` verifies record identity, independently pinned digest, and **exact policy equality** against the target controller. It returns `resume_authorized=False` even on a match.

Unknown/custom policy subclasses and non-null event-loggers fail closed; they require a separate versioned contract. The identity record deliberately does **not** cover `last_applied_step`, prior controller observation, event-log cursor, optimizer, scheduler, RNG, dataset position or structural graph. These must be checked under separate contracts. Full resume and any routing-TrainingEngine integration remain **NO-GO**.

## Threat and compatibility limits

A policy SHA-256 only detects a mismatch relative to an **independently trusted** expected digest; it does not authenticate a checkpoint publisher or make unsafe pickle files safe. A user-supplied rehashed malicious record cannot be trusted merely because its digest matches. The target-policy equality check remains independent of the digest. Source revision, model/feature/config hashes, stream offset and event-log cursor still require a frozen run-manifest design.

No existing checkpoint is rewritten; missing policy metadata is **not** silently filled in or treated as valid scientific resume. Current historical H1a conclusions remain INCONCLUSIVE, ORDER_SENSITIVE and DIRECTION_FLIP_OBSERVED respectively.

## Tests / next gate

`tests/unit/test_controller_policy_provenance.py` checks stable hashes, all built-in M1 policies, no-op policies, seed/top-r/top-n/noise/gate/step perturbations, duplicate controller state hiding policy drift, unknown policies, event logging, malformed/unpinned records, and RNG/controller-state nonmutation.

GitHub Actions must actually execute the suite; CUDA and training-transition equivalence are **not claimed**. After review, the next research gate is an immutable **run/feature/config/source + policy + stream/cursor** provenance manifest and separate disposable replay proof before any live restore or TrainingEngine adapter.

No training, GPU benchmark, production modification, PR merge or external push outside GitHub work is authorized by this contract.
