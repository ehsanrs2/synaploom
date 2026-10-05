# SynapLoom

**A Brain-Inspired Adaptive Graph Architecture for Continual Multimodal Learning**

SynapLoom is a research project exploring whether a neural system can learn continually by reorganizing and recruiting pre-existing capacity — changing active pathways and structural state in addition to ordinary weights.

> Project status: **M0 / architecture foundation**. No experimental performance claims yet.

## Core idea

The initial architecture is a **fixed typed supergraph with dynamic active topology**.

Instead of allocating new neural modules at runtime, SynapLoom starts with active and dormant microcircuits. Learning may:

- update weights in active plastic circuits,
- recruit dormant circuits,
- activate or prune allowed connections,
- consolidate important capacity,
- route different inputs and modalities through different sparse pathways.

The first implementation deliberately uses ordinary backpropagation through the active path. Dendritic computation, predictive/local learning, neuromodulation, replay, and spiking are later research extensions rather than requirements of the MVP.

## Research question

Under a fixed total capacity budget, can reserve-capacity recruitment, sparse routing, and consolidation improve continual-learning stability/plasticity trade-offs — and later improve multimodal continual learning — compared with strong dense, sparse, routing, and continual-learning baselines?

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — formal v0.1 architecture and constraints
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
7. The repository is the source of truth for project state.

## Current milestone

**M0 — Foundation**

The architecture and hypotheses are being frozen before code implementation.

Next milestone: **M1 — single-modality structural core**, including the fixed supergraph, microcircuits, sparse router, structural plasticity controller, consolidation manager, checkpointing, and structural metrics.

## Prior-work anchors

SynapLoom is informed by, and must be distinguished experimentally from, work on dynamic sparse training, continual-learning rewiring, reusable pathways, modular routing, and plasticity maintenance, including:

- [NISPA](https://arxiv.org/abs/2206.09117)
- [RigL](https://arxiv.org/abs/1911.11134)
- [PathNet](https://arxiv.org/abs/1701.08734)
- [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)
- [Loss of plasticity in deep continual learning](https://doi.org/10.1038/s41586-024-07711-7)

No novelty claim is made solely from combining these ideas; novelty must be demonstrated through a precise method and empirical results.
