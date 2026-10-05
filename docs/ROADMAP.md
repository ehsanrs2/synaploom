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

Implement:
- scheduled/task-boundary recruitment policies
- context-imprinted routing key for recruited nodes
- bounded edge initialization
- optional task-boundary consolidation
- structural event log

Smoke-test on synthetic data and a simple continual stream.

Candidate first scientific benchmark: Split CIFAR-100 class-incremental.

Required controls/baselines:
- dense
- fixed-capacity routed model without growth
- fixed sparse
- dynamic sparse where feasible
- standard continual-learning baseline(s)

Exit criterion: H1 can be evaluated fairly under explicitly reported total-capacity and executed-capacity budgets.

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
