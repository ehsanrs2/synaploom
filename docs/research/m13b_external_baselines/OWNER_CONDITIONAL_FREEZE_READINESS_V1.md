# M1.3b — Conditional owner design direction and protocol-freeze readiness

**Decision: `OWNER_DIRECTION_ACCEPTED__SCIENTIFIC_FREEZE_BLOCKED`. Scientific execution: `NO_GO`.**

Prepared after the user's 2026-11-01 reply, **«ادغام انجام شد و با پیشنهادت برای مرحله بعد موافقم»** (the owner agrees with the recommended next phase). That endorsement is a direction to prepare the scientific protocol and review D1–D4. It **does not** independently prove methodological validity, authorize 90 RTX 4090 jobs, set a GPU-hours spending limit or create an authenticated digital signature. The owner-specific decision and its limits were recorded in [Issue #29, comment 6553724926](https://github.com/ehsanrs2/synaploom/issues/29#issuecomment-6553724926).

**Important GitHub correction:** At the time this document was prepared, [PR #38](https://github.com/ehsanrs2/synaploom/pull/38) was **OPEN** (`draft=false`, `merged=false`, `merged_at=null`), despite the user's statement that it was merged. Main remained at `7957a06e9858ce7a4e54c41313e1a079422fa477`. This branch is based on that latest verified `main`. Do not mark PR #38 merged based on user language alone, and do not merge it on the user's behalf unless specifically authorized.

## Approved direction, pending gates

| Decision | User-endorsed direction | Still unresolved before a scientific freeze |
|---|---|---|
| **D1** Scope | `O0` primary external test (30 trajectories); `O1/O2` predeclared descriptive robustness (60) | Independent reviewer verifies hypothesis-scope and interpretation |
| **D2** Resource honesty | Report physical trainable elements, ideal sparse/selected MAC proxies and **measured actual GPU work separately** | Fixed-sparse method fidelity; execution-work/optimizer-state/memory measurement protocol; no claim of equal real GPU compute |
| **D3** Inference | One primary O0 `progressive_reserve – fixed_sparse_mlp` comparison with both co-primary accuracy endpoints | Independent statistical assessment of small n=10, percentile bootstrap CI coverage, joint criterion, multiplicity and missing-seed policy; not automatically accepted by owner direction |
| **D4** Run matrix | 3 admitted arms, 10 paired proposed seeds `501..510`, O0/O1/O2 total **90 projected trajectories / 720,000 optimizer steps** | Actual wall-time and GPU-hour budget, failure/restart cap, source/data attestations, separate manual execution authorization |

Both the total physical parameter matching (~0.0026% difference) and cumulative *ideal* linear MAC proximity from PR #36 are theoretical resource/accounting evidence. The Fixed Sparse implementation still uses **dense CUDA matrix multiplies** and showed larger peak CUDA memory in the tiny host engineering pilot. Similar theoretical MACs **cannot establish** equal training throughput, similar optimizer states or scientific fairness in an actual 10-task run. Do not claim a favorable learning effect absent full confirmatory evidence.

The proposed **per-seed pair bootstrap** samples the same seed IDs for both co-primary endpoints; all-recruited and O1/O2 remain descriptive. An independently reviewed statistical analysis must still define when the whole protocol is invalid due to failed/missing/nonfinite runs and how all negative outcomes are preserved. No prospective accuracy-dependent pruning, tuning or selective O1/O2 execution.

## Machine-readable fail-closed readiness packet

`src/synaploom/research/m13b_freeze_readiness.py` produces a canonical JSON proposal with the exact PR #37 candidate fingerprint, named owner-design endorsement, proposed 90 slots and explicit missing gates:

- PR #38 status must be rechecked at the final review.
- **Independent** n=10 statistical review (coverage/type-I risk, multiple endpoints).
- Independent fixed-sparse model-identity, mechanism/fidelity and actual-resource audit.
- Explicit owner-set finite GPU-hours, maximum wall hours and failure/retry budget.
- Final pinned clean source commit, running host environment, trusted exact feature bytes, class orders and fixed-sparse mask.
- Independently reviewed **final** immutable protocol hash, and a separate, tested, owner-manually-invoked training runner.
- Separate explicit owner instruction permitting the full GPU scientific experiment.

The packet is deliberately **incapable of transitioning to GO**. Even editing and recomputing its SHA to set `ready_for_scientific_execution=true` fails an independent regenerated-template check. Its SHA-256 is a consistency hash, not a cryptographic owner signature.

## Sensitivity of the n=10 statistical proposal — synthetic illustrations

`src/synaploom/research/m13b_smalln_method_sensitivity.py` implements a **bounded, synthetic Monte Carlo demonstration** of the paired percentile CI classification rule using fake correlated Gaussian paired deltas: ten paired seeds, two endpoints, known synthetic correlation 0.7 and standard deviation 0.12. Scenarios: both true null, one null with one large positive endpoint, both positive and opposed signs.

The default **120 independent toy datasets per scenario, 400 bootstrap draws per dataset** is intentionally lightweight to support engineering-method discussion. It is **not the proposed 10,000-draw actual inference**, **not empirical calibration from real SynapLoom effects**, and is too small to establish exact type-I error/coverage or statistical power. The Monte Carlo error and assumptions must be reviewed independently before freezing D3. No actual training data, model summaries, or GPU are read. Positive/negative/inconclusive tags from the simulation are always prefixed `SYNTHETIC_` and cannot become scientific results.

## Safe, offline commands

```bash
# Create machine-readable conditional decision packet (no training or data access)
uv run --locked python experiments/scripts/m13b_freeze_review.py --readiness

# Optionally preserve a brand-new review file; exclusive, never overwrite
uv run --locked python experiments/scripts/m13b_freeze_review.py --readiness \
  --output /tmp/m13b-conditional-owner-readiness-v1.json

# Verify the packet matches source-derived non-executable template
uv run --locked python experiments/scripts/m13b_freeze_review.py \
  --verify-readiness /tmp/m13b-conditional-owner-readiness-v1.json

# Synthetic Gaussian n=10/bootstrap statistical sensitivity demonstration
uv run --locked python experiments/scripts/m13b_freeze_review.py \
  --synthetic-method-review
```

No `--execute`, `--approve`, `--freeze`, `--features` or `--gpu` options exist. The result and numerical Monte Carlo summary remain **review material only**. If PR #38 subsequently merges, re-evaluate the final source/code hash rather than silently changing this decision audit trail.

## Minimum next gate — no unnecessary infrastructure drift

1. Obtain independent statistician/scientific reviewer sign-off on D3 and the fixed-sparse resource/method audit on D2.
2. Obtain actual bounded cost information for a full *one-trajectory* CUDA feasibility check using a nonconfirmatory seed, without using protected eval outcomes for tuning.
3. Owner specifies a quantitative GPU-hours and failure/retry budget. If observed cost is too high, amend and **re-review** the study design *before* confirmatory outcomes; never post-hoc shrink to favorable subsets.
4. After final review and separate execution approval, create a new frozen SHA-bound protocol and a narrowly bounded manually invoked real scientific runner. The existing research-only candidates and this packet are not that runner.

**Historical experiments unchanged:** `EXP-H1A-002=INCONCLUSIVE`, `EXP-H1A-ORDER-001=ORDER_SENSITIVE`, `EXP-H1A-EVALBATCH-001=DIRECTION_FLIP_OBSERVED`.
