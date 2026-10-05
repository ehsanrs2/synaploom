# PROJECT_STATE — SynapLoom

> This file is the authoritative project memory. Update it in every PR that changes architecture, experiment status, accepted assumptions, or the next milestone.

**Last updated:** 2026-10-05  
**Repository:** ehsanrs2/synaploom  
**Current stage:** M0 — Foundation  
**Research status:** architecture defined; no implementation results yet

## 1. Project objective

Investigate a brain-inspired adaptive neural graph in which pre-existing reserve capacity can be recruited, routed, consolidated, pruned, and associated across modalities during continual learning.

Primary target problems:
- continual learning
- loss of plasticity
- catastrophic forgetting
- multimodal specialization and association
- conditional/sparse computation

## 2. Current architecture baseline

Accepted for v0.1:

- fixed typed supergraph rather than runtime allocation of new PyTorch modules
- microcircuit-level graph nodes, not biological-neuron-level nodes
- node states: DORMANT, PLASTIC, CONSOLIDATED
- modality-private pathways
- sparse conditional top-k routing
- structural updates on a slower schedule than gradient updates
- backpropagation through active paths for the first prototype
- growth means activating reserve capacity
- pruning means deactivating low-utility capacity
- lightweight/shared edge transforms rather than full dense matrix per possible edge
- growth triggers calibrated from persistent signals, not one arbitrary weighted novelty score
- cross-modal association requires more than temporal co-occurrence

## 3. Explicit non-claims

We do NOT currently claim:

- SynapLoom is the first dynamic/brain-inspired graph network
- knowledge is proven to be stored as subgraphs
- the architecture is biologically faithful
- sparse routing automatically reduces wall-clock time
- reserve capacity is memory-free
- multimodal private pathways are necessarily superior
- structural growth solves catastrophic forgetting
- local learning is required

All are empirical questions.

## 4. Architecture corrections made during foundation review

1. Dynamic topology will initially be represented as masks over a fixed supergraph.
2. Removed the extra STABLE node state from the MVP.
3. Rejected per-edge full dense matrices as the default because of quadratic scaling.
4. Rejected raw co-occurrence as sufficient cross-modal association evidence.
5. Rejected a single hand-weighted novelty scalar as the default growth trigger.
6. Separated loss of plasticity from catastrophic forgetting in the evaluation plan.
7. Deferred dendrites, spiking, neuromodulation, and local predictive learning.
8. Split validation into single-modality structural learning first, multimodal claims later.

## 5. Evidence motivating the direction

Relevant prior work already covers important components:

- NISPA: sparse continual learning with stable paths and rewiring
- RigL: dynamic sparse topology under fixed parameter/compute constraints
- PathNet: reusable pathways through a larger network
- Progressive Neural Networks: transfer through expandable protected capacity
- modular/MoE routing: conditional specialization, with known routing-collapse risks
- continual backpropagation: continual learning can suffer loss of plasticity separate from forgetting
- multimodal continual-learning work such as CavRL: audio-visual forgetting is a real evaluation setting

Therefore novelty must come from a validated combination and new behavior/results, not from naming known ingredients.

## 6. No experimental results yet

There are currently:

- no training runs
- no benchmark results
- no accuracy claims
- no compute claims
- no ablation results

Any future result must be added to docs/EXPERIMENT_LOG.md and summarized here when it affects project direction.

## 7. Immediate next milestone

**M1 scaffold: Single-modality structural core**

Planned components:

- GraphState / supergraph representation
- MicroCircuit
- SparseRouter
- StructuralPlasticityController
- ConsolidationManager
- metrics/event logger
- exact checkpoint/restore of topology and controller state

Before large experiments, build unit tests that prove:
- dormant nodes do not execute
- routing respects top-k
- topology changes are reproducible
- structural state survives checkpoints
- active parameter/node/edge counts are correct

## 8. Open design questions

These are unresolved and must not silently become assumptions:

- learned router vs fixed/random router as default
- exact lightweight edge operator
- batch-level vs sample-level routing for the MVP
- structural update interval
- growth calibration method
- consolidation importance estimator
- whether node recycling should be core or an ablation
- first non-trivial continual dataset after debugging
- exact fairness protocol for matched compute vs matched total parameters

## 9. Documentation rule

For every merged milestone PR:

1. Update this file.
2. Add architecture decisions to docs/ARCHITECTURE.md if needed.
3. Add all experiment commands/results to docs/EXPERIMENT_LOG.md.
4. Update docs/ROADMAP.md checkboxes.
5. State the next action explicitly.

The repository, not chat history, is the source of truth.
