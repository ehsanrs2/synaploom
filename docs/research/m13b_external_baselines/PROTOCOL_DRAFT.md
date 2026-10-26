# M1.3b — External sparse/continual baseline comparison DESIGN DRAFT

**Status: DRAFT_FOR_REVIEW / NOT A FROZEN PREREGISTRATION / NO RUN AUTHORIZATION.**
**Date:** 2026-10-26. **Tracks:** [Issue #29](https://github.com/ehsanrs2/synaploom/issues/29).
**Parent protocol:** `docs/H1A_002_PROTOCOL.md` and original `docs/M1_3_PROTOCOL.md`. They remain immutable historical records.

## A. Scientific question and scope

Can progressively recruited reserve microcircuits offer a meaningful continual-learning stability/plasticity advantage under equal **total** capacity and transparent **executed** compute, against scientifically faithful fixed and dynamic sparse controls?

**Do not** call this H1b (adaptive need-trigger), a graph-edge/consolidation study, multimodal learning, end-to-end CIFAR-100 or a sample-routing-training experiment. M1.3b tests external controls for **H1a-like controlled growth** only.

Historical verdicts are frozen: `EXP-H1A-001=INVALID_NUMERICAL_INSTABILITY`; `EXP-H1A-002=INCONCLUSIVE`; `EXP-H1A-ORDER-001=ORDER_SENSITIVE`; `EXP-H1A-EVALBATCH-001=DIRECTION_FLIP_OBSERVED`. Neither favorable isolated estimates nor additional control methods retroactively change those verdicts.

## B. Frozen historical anchor, not a new result

- Input: frozen CIFAR-100 ImageNet ResNet-18 (`IMAGENET1K_V1`) **512-dimensional** features, SHA-256 `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47`.
- Task stream: exact order `H1A_CLASS_ORDER` from `src/synaploom/protocols/h1a.py`, 10 tasks x 10 classes, no task ID at inference, 100-class classifier, seen-class masking.
- Existing SynapLoom treatment: `progressive_reserve` (N=16, 4 initially eligible, top-k=4; one recruitment after first optimizer step on each subsequent task, final 13 recruited).
- Internal causal control: `all_recruited` (same parameter tensors/N=16/top-k=4, 16 eligible immediately). Also `fixed_small`, `dense_active_proxy`, `dense_total` descriptive controls.
- Optimizer for existing anchor: SGD LR **0.01** (stability-only calibration; do **not** reuse original 0.03), momentum 0.9, weight decay 5e-4; no scheduler; 20 epochs/task; training batch 128; evaluation batch 256; no replay/consolidation/edges/rounds; 10 paired seeds 201–210 in `EXP-H1A-002`.
- Historical outcomes (treatment minus internal causal control): final average accuracy **+0.069 pp**, CI [+0.036,+0.103]; late-task within-task accuracy **−0.041 pp**, CI [−0.171,+0.087]; combined `INCONCLUSIVE`. Approx. **88.5%** forgetting in both routed groups, under this particular frozen-feature, no-replay scenario.

The primary-run bundle is historically documented but **not uploaded here from the host**. Its presence at `data/features/cifar100_resnet18_imagenet1k_v1.pt` and SHA must be verified by a future **local preflight**, not inferred from GitHub. GitHub CI is CPU-only and cannot establish benchmark CUDA availability.

## C. Proposed new study identifiers and data firewall

- Proposed new identifier: `EXP-M13B-EXTERNAL-001`; cannot be executed before owner explicitly approves a frozen final revision and the protocol ID is registered.
- Proposed 10 **new** paired seeds: `501..510`, disjoint from all earlier primary/calibration/order/eval-batch cohorts. Freeze exactly once before outcome access.
- Reuse historical class order as a **fixed comparability anchor**. This does not establish order robustness; an independent follow-up needs a separately frozen design.
- Retain PyTorch/Python/CUDA/GPU versions, feature artifact hash, per-run RNG seeds, CUDA deterministic/TF32 policy, and class-order hash in an immutable run manifest.
- Evaluation batch size 256 and deterministic ordering for legacy batch-routed SynapLoom, with no sample routing substituted during training or evaluation. Explicitly label any cross-architecture batch sensitivity as a limitation; optional batch-size sensitivity is a separate descriptive report.
- Feasibility/sanity checks must not observe held-out accuracy/forgetting. No tuning after looking at confirmatory results.
- All historical run IDs and data directories remain append-only and separate.

## D. Admission sequence (must be evaluated in order)

**D0: Method fidelity.** Review the original paper, official code commit/version/license, mechanism and optimizer dynamics. An imitation that omits the defining step cannot be called RigL/NISPA. Allowed outcome: `GO_METHOD`, `CONDITIONAL` or `NO_GO_FIDELITY`. Require a reviewer-signed evidence artifact (not just green CI).

**D1: Resource and dataset feasibility.** Ensure legal/reproducible access to the exact frozen feature bundle and original baseline code; validate environment/accelerator support, expected memory and runs. `DATA_UNAVAILABLE`/ `DEP_UNAVAILABLE` blocks any science run; avoid silent downloadable replacements.

**D2: Budget matching.** Record **separate** views: (a) physically allocated trainable parameters including dormant reserve/masks, (b) reachable/routable/active parameter proxy, (c) per-batch executed nonzero parameters/MAC estimates, (d) training-step and rewiring overhead, (e) real wall time, GPU peak allocated and reserved memory. Do not report two methods as compute-matched merely because their trainable parameter counts agree. Proposed target gates for a *matched tier*: total parameter count within 2% and measured/estimated executed MACs within 10% at matched batch size; if infeasible, classify as separate unmatched-resource tier rather than silently relax. Specify sparse-kernel reality and dense-mask arithmetic separately.

**D3: Numerical and training semantics.** Validate clean finite logs and gradient/inactive-weight updates for synthetic steps, correct 100-way masking, stream boundaries, optimizer scheduling/weight decay and deterministic evaluation; route/replay/no-replay assumptions held fixed. CPU unit/CI plus bounded GPU smoke are needed; one smoke does not qualify scientific fidelity.

**D4: Provenance and owner approval.** Pin source commit, exact parameters/budget ledger, resolved config/dataset/class-order/feature hashes, all seeds/ordering, evaluation rule, artifacts and fail-fast policy. Owner approves the final frozen protocol and manual execution command. **This draft is not that approval.**

Any failure blocks promotion. Reviewers must preserve `GO`/`NO_GO` for each baseline independently, not substitute an easier method behind the same name.

## E. Proposed statistical reporting (subject to freeze)

The confirmatory *treatment vs all-recruited* H1a comparison retains the historical co-primary metrics and sign conventions, without modifying the old conclusion. A first **external** comparison against an admitted fixed sparse control is a separate family reported as paired treatment minus fixed sparse:

1. final average accuracy — higher is better;
2. late-task within-task mean accuracy — higher is better.

For ten new paired seeds, report every seed, mean paired delta, **paired bootstrap 95% CI** (predeclare 10,000 resamples and RNG seed 1337), variants' means/stds. Combined verdict may be `FAVORABLE` only when both CIs are strictly positive, `UNFAVORABLE` only when both strictly negative, otherwise `INCONCLUSIVE`; numerical or data failure is `INVALID`. This rule reuses H1a logic but is a **new**, explicitly named external comparison, not H1a-002 reanalysis.

RigL/NISPA, if later admitted, form **additional exploratory comparison families** unless a multiplicity-controlled confirmatory plan is approved prior to results. Do not select the winning baseline post hoc. Secondary metrics: average forgetting (lower better), BWT, stage-by-task matrix, per-task learning curves, routing/sparsity histograms, recruitment/rewiring counts, actual train-step FLOPs, task latency, wall and GPU memory. **Forgetting and loss of plasticity are different constructs**: a fresh-network late-task reference is necessary for any stronger H6 plasticity interpretation.

Stop all scientific runs and mark the new experiment INVALID on NaN/Inf, missing/mismatched feature hashes, invalid masks, budget fraud, missing variants/seeds or corrupted outputs. Do not compute favorable pooled CIs from partial survivors.

## F. No-go boundaries

- No `RoutingSession` sample route in `TrainingEngine`; its `[B,K]` trace is not compatible with the legacy `[K]` trainer.
- No automatic adaptive novelty claim (H1b) or consolidation (H2) claim.
- No automatic performance optimization promotion, CUDA benefit claim or science runs from this PR.
- No replacing the frozen feature protocol with an end-to-end encoder or altered class ordering under the same ID.
- No reclassifying prior results; keep all negative/inconclusive evidence.
- No code/API change to production default router, controller, checkpoint or scientific training engine.

## G. Decision at this draft stage

**GO** for offline method-fidelity audit and bounded feasibility contract implementation. **CONDITIONAL** for a fixed sparse comparator after its concrete allocation/mask/budget contract exists. **NO-GO** for full external scientific experiment until D0–D4, independent review and explicit owner freeze. RigL and NISPA are not yet admitted and must not be labeled qualified.

Supporting files: `FAIRNESS.md`, `FIDELITY.md`, `MANUAL.md`. This document is a proposal for owner review, **not a completed preregistration or a result**.
