# M1.3b — Owner decision gate for external fixed-sparse science

**STATUS: PROPOSED DECISION / NOT FROZEN / NOT SCIENCE-EXECUTABLE.**

Follows merged [PR #34](https://github.com/ehsanrs2/synaploom/pull/34) and owner-supplied successful host GPU feasibility evidence (see [GPU_PILOT_EVIDENCE.md](GPU_PILOT_EVIDENCE.md), tracked [Issue #29](https://github.com/ehsanrs2/synaploom/issues/29)). The purpose of this document is to resolve the **scientific design**, not to build more checkpoint machinery or claim the GPU pilot proves SynapLoom superiority. Original H1a results remain `INCONCLUSIVE`, `ORDER_SENSITIVE`, and `DIRECTION_FLIP_OBSERVED`.

## Proposed experimental design — requires explicit owner and independent reviewer decision

| Dimension | Preferred PRE-REGISTRATION proposal | Alternative / unresolved |
|---|---|---|
| Primary order | Frozen `O0` from `H1A_CLASS_ORDER` | Do not choose based on new outcomes |
| Robustness orders | **Predeclare O1 and O2 now**, as secondary mandatory order-sensitivity evaluations, not extra confirmation of O0 | O0-only can reduce cost but must disclose historical order sensitivity and not imply robustness |
| Pairing | 10 new, proposed seeds `501..510`, all arms matched within order and seed | Actual seed set still requires approval before freeze |
| Arms | `progressive_reserve`, `all_recruited`, `fixed_sparse_mlp` | RigL/NISPA **not admitted**; replay is a different resource condition |
| Primary external contrast | `progressive_reserve - fixed_sparse_mlp` **on O0**, as its own new scientific family | `all_recruited` is internal reference/descriptive for this external test; any inferential use requires separate multiplicity plan |
| Co-primary metrics | Final average accuracy and late-task within-task mean accuracy, both higher better | Forgetting/BWT and per-task curves secondary; loss-of-plasticity requires independent comparator |
| Statistical proposal | 10,000 seed-paired bootstrap resamples (RNG=1337), percentile 95% CIs. `FAVORABLE` only if **both lower CI bounds >0**, `UNFAVORABLE` only if **both upper CI bounds <0**, else `INCONCLUSIVE`; any failed seed is not silently dropped | Independent review of type-I error, family definition, and the consequences of selection/multiplicity required |
| Robustness evaluation | O1/O2 show all seeds and both co-primary effects by order and a seed-cluster pooled descriptive interval; interpret direction changes explicitly | No post-hoc cherry picking, no treating 30 seed×order records as 30 independent random seeds |
| Proposed matrix | **O0: 30** training trajectories; **O1/O2: 60** additional → **90** total if robustness is approved | The draft Manifest currently offers both 30 and 90 schedules; selecting one is not authorization |
| Resource comparability | Physical trainable parameter count error ≤2% and ideal linear MAC count error ≤10% are *theory/accounting tiers*, not measured computational equality | Actual forward/backward GPU execution, dense-mask overhead, optimizer state, wall, memory and allocation dynamics must be separately disclosed |
| Dataset | Exact pinned CIFAR-100 512-D frozen ResNet18 feature bundle, 10×10 tasks, seen-class masking, batch-level SynapLoom routing, no replay | Training-only GPU engineering pilot is separate, does **not** license altering the protected evaluation partition or routing semantics |
| Optimizer anchor | SGD LR 0.01, momentum .9, weight decay 5e-4, 20 epochs/task, train B128, eval B256, no scheduler | External method-specific tuning would require a separately predeclared fidelity/fairness tier |
| Execution | New, immutable experimental ID and isolated result root, append-only run audit, hash-bound source/feature/config/class-order/seed/mask, bounded fail-fast/restart policy | NO authorized scientific runner or owner-approved command exists yet |

## Why the successful hardware pilot is NOT the experiment

The 2026-10-27 owner report confirms the limited 12-step GPU integration pilot exited successfully with matching canonical result SHA and full three-arm completion. That is enough to proceed to protocol review, not enough to:
- establish GPU speed rankings from single unrandomized subsecond samples;
- claim an equal-compute comparison merely because physical parameter counts are near identical;
- prove low forgetting or sustained plasticity (only two tasks and 16 train items per class);
- say an external fixed-mask MLP is a RigL/NISPA reproduction;
- qualify precise training/data-order determinism over 20 epochs and 10 tasks;
- infer behavior of the full pretrained-feature set's protected evaluation partition from a small train-only holdout.

## Freeze gates — all must be true BEFORE the first confirmatory trajectory

1. Independent review of exact fixed-sparse architecture, mask generation/seed, lifetime of optimizer post-step hook, total and active model budget, reference estimator assumptions (initial recruited vs task-progressive recruited) and dense-masked actual work.
2. Approval of O0 primary and O1/O2 robustness scope and costs, paired seed IDs, run ordering/randomization and all reporting including adverse results; register one final canonical protocol and publish its SHA **before outcomes**.
3. Source commit, environment/CUDA/PyTorch versions, exact feature SHA, O0/O1/O2 fingerprints, metadata, fixed mask SHA and validation of fully disjoint train/test usage pinned in a reviewed run manifest.
4. No unreviewed method-specific optimization, no validation on protected eval labels, no use of prior confirmatory seeds for engineering/pilot tuning, no silent outcome-dependent exclusion, no pre-existing output directory clobbering.
5. Numerically finite, deterministic, step-by-step audit, CPU unit CI + owner-controlled bounded GPU dry/smoke qualification, explicit rollback/failure preservation and canonical per-run report.
6. A **new**, narrow, separately reviewed science execution runner that accepts only a verified frozen protocol and owner manual execution authorization. The current `m13b_external_preflight.py`, synthetic pilot and train-only CUDA pilot are **not** that runner.

## Decision questions for owner review

- **D1 — scope:** accept *O0 primary plus predeclared O1/O2 mandatory robustness* (recommended), or choose *O0 only* and explicitly limit the generality of conclusions?
- **D2 — budget:** accept separate physical-capacity-matched (≤2%) and **ideal** MAC-matched (≤10%) reporting tiers with actual GPU work/latency/memory as separately reported resource axes, or require a stronger measured-work match before promotion?
- **D3 — statistics:** accept *one* confirmatory external comparison with two co-primary metrics and proposed paired bootstrap decision rule; keep the internal all-recruited contrast and order robustness descriptive pending independent multiplicity review?
- **D4 — resources:** agree to run 10 new paired seeds across all admitted arms after a bounded, owner-reviewed short non-confirmatory qualification, with the total 90 trajectories only if the O1/O2 block is pre-approved?

All are currently **PENDING**; PR merge itself must NOT mark these approved. The owner may approve or modify the protocol in a separate attributed freeze decision, which then becomes the immutable execution contract.

**Scientific status: `DESIGN_READY_FOR_OWNER_REVIEW`; `NO_GO_CONFIRMATORY_GPU_EXECUTION`.**
