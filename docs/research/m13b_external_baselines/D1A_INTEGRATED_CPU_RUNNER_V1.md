# M1.3b D1-A — integrated full-registry CPU runner V1

Status: IMPLEMENTED_FOR_CPU_QUALIFICATION / SCIENCE_GPU_NO_GO. Base main 3b5a3a3c4a90a4330c531c71c09ac9896d91cf8d; PR #66 merged and exact-head CI #46111964963 success.

## Integrated path

The new module m13b_d1a_integrated_cpu_runner composes the existing exact 720-slot synthetic registry, exclusive fsync journal and closed count-based statistical analyzer. It uses actual independent Linux CPU child processes executing only a fixed, stdlib-only artificial output program. There is no model, training, real seed draw, protected feature access, arbitrary worker command, CUDA or science entry point. Each child emits an invented count observation, independently compared against the parent reference. This is intentionally a constant +2pp fixture, not evidence of SynapLoom power or accuracy.

One fresh directory, one cooperative lock, one monotonic origin, serial slots. Every START is exclusive and synced before spawn, followed by observation (success only), terminal and closure. Links plus the observation digest bind the exact inventory. The full success path has 720 actual process launches, 720 START, 720 terminal, 720 observation files, manifest, lock and close (2163 entries). No retry, resume, replacement or partial statistical result. A failure at slot 2 leaves two complete and 717 not-started slots. Separate fixture budgets of 5s per worker, 1s cleanup and 180s total are engineering-only; actual approved science parameters (360 pairs, 30 GPU hours, 300/120s, no retry) are unchanged.

Inspection validates all observations and the complete ordered journal. A complete unpinned session returns no analysis. Only a supplied matching detached journal head and complete valid closure allow the synthetic exact IUT analyzer. The detached pin is caller-held, not authenticated; deriving the replay digest after verifying the journal is a local composition check, not external attestation. Files are readable to the same account; no real outcome secrecy is claimed.

## Live-worker parent-death handling

The fixed nonforking child installs Linux PR_SET_PDEATHSIG(SIGKILL) via libc, then checks the expected parent PID to cover the ordinary parent-death race before signal registration. The test synchronizes on a live sleeping child after registration; kills its supervisor; uses a pidfd to observe that exact child exit (not a reused numeric PID); then verifies dangling journal and no retry. The journal conservatively marks release unknown despite the independent test observation. No arbitrary process-tree or GPU cleanup claim. This mechanism assumes a single-threaded supervisor and Linux semantics; descendants, kernel hangs, supervisor thread loss and cgroup lifecycle require separate qualification. The pidfd test skips where the APIs are unavailable.

## Validation

Tests exercise the actual full 720 process launch path, exact synthetic IUT after closure and matching pin, wrong pin, nonzero exit, timeout, insufficient budget, corrupted observation/start/close, unexpected records, invalid fixture faults, and supervisor death with a live worker. Hosted CI must validate the exact commit; no new local or hosted pass claimed before results. Full fixture can take tens of seconds, but does not execute optimizer steps or require torch for the new modules themselves.

## Remaining real-science admission work

- Do not expand the toy-control layer indefinitely: next phase should audit actual training/evaluation source, independent seed population and RNG offsets, fixed mask identity, actual ordered sample/label and batch bytes, 8000 actual optimizer updates and sealed exact count path. Bind these to the final executable contract, without loading protected features now.
- Actual process tree isolation and independent occupied-GPU watchdog, close/fsync/cleanup accounting, real resource release and shutdown margin remain unqualified. Spawn, storage sync and scheduler stalls can overrun a nominal bound. Local cooperative locks do not prevent competing campaigns in another directory or by another process.
- Need externally authenticated source/manifest/attempt roots, preoutcome commitments, sealed results and access controls. Local hashes and same-account fixtures are not a witness. Final protocol freeze and separate manual GPU GO remain mandatory.

No real seed draw, protected outcomes, feature files, GPU, actual training or scientific claim. Historical protocol/results and owner-approved numeric choices unchanged.
