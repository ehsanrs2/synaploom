# Roadmap

## M0 — Foundation

Goal: make the project reproducible before implementation.

- [x] Project name selected: SynapLoom
- [x] Repository created
- [x] Architecture baseline reviewed
- [x] Research hypotheses formalized
- [x] Project-state memory established
- [x] Experiment log template established
- [ ] Merge foundation documentation
- [ ] Decide Python/PyTorch package skeleton and configuration system

Exit criterion: another researcher can read the repository and explain the exact v0.1 hypothesis, non-goals, and evaluation plan.

## M1 — Single-modality structural core

Build the smallest system capable of testing reserve structural recruitment.

Components:
- fixed supergraph
- MicroCircuit
- node state
- sparse router
- graph execution
- structural controller
- consolidation manager
- event logging/checkpointing

Start with synthetic/debug datasets, then a non-trivial continual benchmark.

Required baselines:
- dense
- fixed sparse
- dynamic sparse
- standard continual-learning baseline(s)

Exit criterion: H1 can be evaluated fairly under matched parameter/compute budgets.

## M2 — Plasticity and stability

Add:
- calibrated growth triggers
- pruning/recycling
- consolidation ablations
- explicit loss-of-plasticity evaluation

Exit criterion: determine whether structural adaptation improves late-task learnability and/or forgetting.

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

Move to realistic audio-visual or multimodal continual-learning benchmarks such as CavRL-like settings and modality/task-shift scenarios.

Measure:
- average accuracy
- forgetting
- forward/backward transfer
- late-task plasticity
- compute
- active capacity
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
