# Roadmap

## M0 — Foundation

Goal: make the project reproducible before implementation.

- [x] Project name selected: SynapLoom
- [x] Repository created
- [x] Architecture baseline reviewed
- [x] Research hypotheses formalized
- [x] Project-state memory established
- [x] Experiment log template established
- [x] Merge foundation documentation
- [x] Decide Python/PyTorch package skeleton and configuration approach
- [x] Record mathematical/complexity framing
- [x] Freeze M1 implementation specification

Exit criterion: another researcher can read the repository and explain the exact v0.1 hypothesis, non-goals, evaluation plan, and M1 implementation contract.

## M1 — Single-modality structural core

### M1.0 — execution/state machinery

Implemented:
- [x] fixed total microcircuit capacity
- [x] DORMANT / PLASTIC / CONSOLIDATED states
- [x] recruited-vs-executed distinction
- [x] batch-level sparse router
- [x] bounded O(N q_max) edge-slot representation
- [x] graph executor / bounded message passing
- [x] trace and structural accounting
- [x] checkpoint/restore of model, optimizer, structural tensors, and RNG state
- [x] invariant unit tests
- [x] append-only structural event logger
- [x] pull-request test workflow

Local M1.0 validation: **10/10 unit tests passed** on 2026-10-05.

### M1.1 — controlled structural adaptation

Implemented:
- [x] StructuralController with observe / propose / apply
- [x] fixed-schedule recruitment
- [x] task-boundary recruitment
- [x] norm-calibrated context-imprinted routing keys
- [x] bounded context-linked edge initialization
- [x] deterministic random edge-init baseline
- [x] task-boundary top-importance consolidation
- [x] node-owned gradient protection for consolidated capacity
- [x] node-importance statistics
- [x] controller checkpoint state
- [x] mutation-safe training engine
- [x] controller event emission
- [x] integration test and standalone controlled-growth smoke script

Automatic novelty detection remains M2.

### M1.2 — continual evaluation harness and controls

Implemented:
- [x] task/stream abstraction
- [x] synthetic multi-cluster continual stream
- [x] SplitMNIST-equivalent CI smoke plus optional real SplitMNIST adapter/script
- [x] evaluation matrix after every task
- [x] average accuracy / forgetting / backward-transfer / late-task-learning metrics
- [x] structural histories: recruited/executed nodes, edges, routing selection distribution
- [x] dense control model using the same harness
- [x] fixed-capacity routed control with no structural growth
- [x] deterministic run config + task-stream manifest in result artifacts

The M1.2 metrics validate the evaluation machinery. The dense control is not yet a budget-matched scientific baseline, and late-task immediate accuracy is only a proxy for plasticity.

Exit criterion: the measurement pipeline can compare controlled growth against non-growth controls on a continual stream without changing the training/evaluation harness.

### M1.3 — first fair H1 benchmark

#### M1.3a — controlled reserve recruitment (H1a)

Implemented/preregistered:
- [x] frozen CIFAR-100 ResNet-18 feature pipeline
- [x] canonical 10-experience / 10-class mechanism-isolation setup
- [x] explicit frozen class order
- [x] seen-class logit masking
- [x] progressive-reserve treatment
- [x] all-recruited exact routed control
- [x] fixed-small routed control
- [x] dense active-parameter proxy
- [x] dense total-parameter match
- [x] 10 paired seeds
- [x] within-task learning curves
- [x] parameter/MAC/wall-time/VRAM reporting hooks
- [x] paired bootstrap comparison
- [x] full run provenance
- [x] execute primary 10-seed run — invalidated by numerical instability
- [x] record invalid result in EXPERIMENT_LOG
- [x] add numerical fail-fast diagnostics
- [x] run EXP-H1A-STABILITY-001 on independent calibration seeds — lr=0.01 selected with 15/15 finite runs
- [x] preregister EXP-H1A-002 with fresh seeds and calibrated lr=0.01
- [x] execute EXP-H1A-002 and record the paired scientific result — 50/50 numerically valid runs; preregistered overall verdict INCONCLUSIVE
- [x] preregister EXP-H1A-ORDER-001 class-order sensitivity: three fixed orders x 10 paired seeds x two primary variants
- [x] implement sensitivity runner and protocol/unit tests with exact order fingerprints and 60-run manifest validation — PR #15; local validation pending
- [x] run local tests and dry-run, then merge PR #15 — 43 tests passed and the dry-run confirmed the frozen 60-run matrix
- [x] execute EXP-H1A-ORDER-001 and record results — 60/60 numerically valid runs; registered classification ORDER_SENSITIVE; EXP-H1A-002 remains INCONCLUSIVE
- [x] preregister EXP-H1A-EVALBATCH-001 evaluation batch-size sensitivity: fixed O0 class order, fresh paired seeds 401–410, sizes 32/64/128/256/512/1024, all measured on the same in-memory model state
- [x] implement the multi-batch evaluation runner, protocol helpers, canonical state digest, and tests — implementation PR #18; local validation pending
- [ ] run local tests and the dry-run; validate the registered 20/120 matrix and state-hash invariance checks
- [ ] execute EXP-H1A-EVALBATCH-001 (20 training trajectories / 120 run-size summaries) and record results without changing EXP-H1A-002's primary verdict

#### M1.3b — external sparse/continual controls

After the internal causal result:
- [ ] fixed sparse baseline
- [ ] RigL-style dynamic sparse baseline where implementation fidelity is acceptable
- [ ] NISPA-style sparse continual baseline where feasible
- [ ] replay baseline as a separate replay-enabled resource condition
- [ ] end-to-end CIFAR-100 replication if H1a is promising

Exit criterion: H1a has a preregistered paired result and H1 can be evaluated against stronger external controls without hiding resource differences.

## M2 — Adaptive plasticity and stability

Add:
- calibrated automatic growth triggers
- pruning/recycling
- consolidation ablations
- explicit loss-of-plasticity evaluation
- candidate-search policies if needed

Exit criterion: determine whether adaptive structural control improves late-task learnability and/or forgetting.

## M3 — Multimodal pathways

Add a second modality and modality-private regions.

Initial dataset may use AV-MNIST for debugging, but conclusions must not rely on AV-MNIST alone.

Compare:
- early fusion
- late fusion
- fixed middle fusion
- SynapLoom private pathways

Exit criterion: H4 tested under missing/noisy modality conditions.

## M4 — Adaptive association

Allow selected cross-modal structural changes.

Add:
- candidate edge policy
- evidence-based association activation
- cross-modal topology logging
- subgraph ablations

Exit criterion: H5/H7 tested without assuming that co-occurrence equals semantic association.

## M5 — Stronger continual multimodal evaluation

Move to realistic audio-visual or multimodal continual-learning benchmarks.

Measure:
- average accuracy
- forgetting
- forward/backward transfer
- late-task plasticity
- compute
- active/recruited capacity
- modality reliance/grounding where possible

Exit criterion: architecture survives a realistic continual multimodal setting.

## M6 — Biological extensions, one at a time

Only after M1-M5:

- dendritic microcircuits
- predictive/local learning
- neuromodulatory signals
- episodic replay/consolidation
- asynchronous/time-scale dynamics
- spiking implementation

Each extension requires its own ablation and may be rejected.

## Go / no-go discipline

A milestone should not automatically lead to the next mechanism.

If the core structural hypotheses fail against strong baselines under fair budgets, document the negative result and revise or stop that line before adding complexity.
