# SynapLoom

**A Brain-Inspired Adaptive Graph Architecture for Continual Multimodal Learning**

SynapLoom is a research project exploring whether a neural system can learn continually by reorganizing and recruiting pre-existing capacity — changing active pathways and structural state in addition to ordinary weights.

> Project status: **M1 design complete; implementation not started**. No experimental performance claims yet.

## Core idea

The initial architecture uses fixed total microcircuit capacity with dynamic structural state.

SynapLoom distinguishes:

- **dormant capacity** — present but not available to normal routing,
- **recruited capacity** — available to routing,
- **executed capacity** — the bounded top-k microcircuits actually run for a batch.

Learning may eventually:

- update weights in executed plastic circuits,
- recruit dormant circuits,
- activate or prune bounded structural connections,
- consolidate important capacity,
- route different inputs and modalities through different sparse pathways.

The first implementation deliberately uses ordinary backpropagation through the executed path. Dendritic computation, predictive/local learning, neuromodulation, replay, and spiking are later research extensions.

## Research question

Under a fixed total capacity budget, can reserve-capacity recruitment, sparse routing, and consolidation improve continual-learning stability/plasticity trade-offs — and later improve multimodal continual learning — compared with strong dense, sparse, routing, and continual-learning baselines?

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — formal v0.1 architecture and constraints
- [M1 design](docs/M1_DESIGN.md) — implementation-ready contract for the structural core
- [Theory notes](docs/THEORY.md) — mathematical framing, complexity targets, and open questions
- [Research hypotheses](docs/RESEARCH_HYPOTHESES.md) — falsifiable hypotheses and baselines
- [Roadmap](docs/ROADMAP.md) — milestone plan
- [Experiment log](docs/EXPERIMENT_LOG.md) — reproducible experimental record
- [PROJECT_STATE](PROJECT_STATE.md) — authoritative project memory and current state

## Current principles

1. Brain-inspired, not a brain simulation.
2. Test one mechanism at a time.
3. Structural claims require strong ablations.
4. Match budgets fairly when comparing models.
5. Measure forgetting and loss of plasticity separately.
6. Do not equate multimodal co-occurrence with semantic association.
7. Separate recruited capacity from executed compute.
8. Avoid dense topology materialization.
9. The repository is the source of truth for project state.

## Current milestone

The next implementation step is **M1.0 — execution and state machinery**:

- package/config scaffold
- GraphState
- MicroCircuit
- batch-level SparseRouter
- bounded edge slots
- GraphExecutor
- ForwardTrace
- event logging
- checkpoint/restore
- invariant tests

## Prior-work anchors

SynapLoom is informed by, and must be distinguished experimentally from, work on dynamic sparse training, continual-learning rewiring, reusable pathways, modular routing, and plasticity maintenance, including:

- [NISPA](https://arxiv.org/abs/2206.09117)
- [RigL](https://arxiv.org/abs/1911.11134)
- [PathNet](https://arxiv.org/abs/1701.08734)
- [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)
- [Loss of plasticity in deep continual learning](https://doi.org/10.1038/s41586-024-07711-7)

No novelty claim is made solely from combining these ideas; novelty must be demonstrated through a precise method and empirical results.
