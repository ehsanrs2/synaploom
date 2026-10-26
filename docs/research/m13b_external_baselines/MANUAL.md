# M1.3b — Review, qualification and manual execution runbook

**DRAFT / NO SCIENTIFIC EXECUTION.** This describes work to approve, not work that has been executed. The documented M1.3b runner, RigL and NISPA adapters **do not yet exist** in SynapLoom. Do not paste hypothetical commands into a terminal.

## Decisions needed before final protocol freeze

1. Confirm whether the new external comparator study is **O0-only with explicit order sensitivity as a limitation**, or whether the new study must include the already frozen O1/O2 orders. Recommendation: establish a bounded O0 primary comparison then a predeclared, separately accounted multi-order robustness block *before inspecting O0 results*; no cherry picking or retroactive selective runs.
2. Approve the proposed 501–510 paired seed block, comparison family and bootstrap decision rule, including how to account for multiple external comparisons. A newly favorable external effect is not a change to the historic H1a verdict.
3. Approve the exact fixed-sparse MLP architecture/mask density, active vs allocated parameter ledger and evaluation restrictions before implementing it. It is **not** the existing `fixed_small` expert baseline.
4. Decide whether RigL/NISPA satisfy their method-fidelity and data-validation constraints; if not, mark them NO-GO without obstructing a qualified fixed-sparse study. Only admit them under new, accurately named contracts.
5. Approve numerical stability/calibration, finite checks, training-time optimizer policy, dataset/training-only validation split (if any), failure policy and pilot acceptance. Never select LR or masks using held-out evaluation.
6. Approve immutable source/feature/config/stream order/cursor/event-log provenance and whether checkpoint/restart is required in this comparison. Current checkpoint tools are research-only and do not authorize training resume.
7. After independent design review, explicitly freeze the protocol into a **separate final file**. Do not rename this proposal to `FROZEN` merely because CI passes.

## Verification available now (offline / no GPU)

The existing, already implemented H1a anchor dry run can be manually inspected *if desired*; it does not execute training and does not read the feature artifact:

```sh
uv run --locked python experiments/scripts/m1_3_h1a_002.py --dry-run
```

GitHub CI automatically runs the existing unit/smoke suite on the PR. This draft adds documentation only, so successful CI means regression checks on unchanged code, **not external baseline correctness**. No GPU data or local model artifact is available through this PR.

The draft's own review checks:
- paths and hash `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47` match prior frozen H1a artifacts;
- original `docs/M1_3_PROTOCOL.md`, `docs/H1A_002_PROTOCOL.md`, `docs/EXPERIMENT_LOG.md` and prior results are untouched;
- no code modifies `TrainingEngine`, Router default, optimizer update or training checkpoint API;
- all base methods use the same task stream and disjoint confirmatory/pilot seed sets;
- external baseline method labels cannot be assigned without a reviewable fidelity record;
- the fairness ledger names physical allocation, ideal sparse MACs, actual masked compute, optimizer/rewiring overhead, wall and memory separately.

## Proposed follow-up PR sequence (GitHub-first)

**PR A — fixed sparse contract/adapter.** Choose mask shape/density by budget and freeze the algorithm's parameter ledger before accuracy is observable. Implement a stand-alone `fixed_sparse_mlp` adapter with tests for masks, dormant zeros, optimizer effects, seen-logit masking, seed identity, and serialization. CI CPU only; no default-model changes.

**PR B — reproducibility + preflight.** Implement an immutable M1.3b manifest, `--dry-run` with exact seed × order × variant count, hash checks and code provenance. The runner should refuse any unapproved variants or overrides, and refuse existing result directories. Mark `INVALID` on integrity/numeric failures, preserving partial evidence rather than cherry-picking valid seeds.

**PR C — external fidelity conditional.** First audit official RigL/NISPA sources and independent reviewer checklists; implement only methods that pass F0–F2. Do not use an approximate rewire scheme under an authentic method label.

**Owner-only pilot after approved code.** Use bounded synthetic and train-only task smoke and, only if needed, manual GPU qualification; record wall, memory and numerical validity with no final test outcome tuning. No performance/training on GitHub-hosted CPU alone qualifies GPU science.

**Final scientific study** only after owner approves the frozen protocol and exact manual `uv run --locked ...` command from a verified runner. Record every predeclared arm/seed/order and negative/failure outcome; no background GPU execution.

## Review / stop decision

| Question | Current |
|---|---|
| Existing H1a causal control and mechanics available? | YES; historic result INCONCLUSIVE |
| Frozen 512-D feature bundle present and SHA-checked on GPU host? | UNKNOWN — local preflight required before science |
| External fixed-sparse architecture/mask/budget frozen? | NO |
| Original RigL behavior reproduced in this harness? | NO |
| Original NISPA behavior reproduced with leakage-free validation? | NO |
| Full new multi-control protocol frozen and owner-approved? | NO |
| New scientific runner and paired results produced? | NO |
| Allowed to execute external scientific training now? | **NO-GO** |

This design-only work is deliberately scoped to restore focus to the project hypothesis, rather than expanding checkpoint engineering indefinitely.
