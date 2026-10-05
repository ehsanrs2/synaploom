# SynapLoom

**A Brain-Inspired Adaptive Graph Architecture for Continual Multimodal Learning**

SynapLoom is a research project exploring whether a neural system can learn continually by reorganizing and recruiting pre-existing capacity — changing active pathways and structural state in addition to ordinary weights.

> Project status: **M1.3 H1a protocol and benchmark machinery ready; primary scientific run pending**. No scientific performance claims yet.

## Development environment

SynapLoom uses **uv** as the project package and environment manager.

```bash
# Create/sync .venv from pyproject.toml + uv.lock
uv sync

# Run the test suite in the managed environment
uv run pytest

# Install the optional vision stack used by CIFAR-100 feature extraction
uv sync --extra vision
```

The normal workflow does not require manually activating `.venv`; prefer `uv run ...`.

Dependency changes should go through uv so `pyproject.toml` and `uv.lock` remain synchronized:

```bash
uv add <package>
uv add --dev <package>
uv add --optional vision <package>
uv lock
```

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
- [M1.3 H1a protocol](docs/M1_3_PROTOCOL.md) — preregistered first scientific reserve-recruitment test
- [H1a execution runbook](docs/H1A_RUNBOOK.md) — feature extraction, preflight, execution and integrity checks
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
10. uv + uv.lock are the source of truth for the Python environment.

## Current milestone

**M1.0** provides the fixed-capacity sparse execution machinery. **M1.1** adds controlled structural adaptation: scheduled/task-boundary recruitment, norm-calibrated context imprinting, bounded edge initialization, consolidation policies, importance statistics, controller checkpointing, and a mutation-safe training engine.

**M1.2** adds one model-agnostic continual pipeline for controlled growth, routed no-growth, and dense controls: deterministic task streams, the stage-by-task evaluation matrix, continual metrics, structural histories, and JSON artifacts. CI uses a deterministic synthetic class-incremental smoke; a real SplitMNIST adapter is optional.

**M1.3a** is now preregistered as a mechanism-isolation test on frozen CIFAR-100 ResNet-18 features. The primary paired comparison is progressive reserve recruitment vs an all-recruited routed control with the same total capacity and top-k execution.

The next action is to extract the frozen feature bundle, pass the mandatory H1a preflight, then execute the frozen 10-seed protocol and record the result. External RigL/NISPA-style comparisons and end-to-end CIFAR-100 are follow-up validation, not hidden tuning steps.

## Prior-work anchors

SynapLoom is informed by, and must be distinguished experimentally from, work on dynamic sparse training, continual-learning rewiring, reusable pathways, modular routing, and plasticity maintenance, including:

- [NISPA](https://arxiv.org/abs/2206.09117)
- [RigL](https://arxiv.org/abs/1911.11134)
- [PathNet](https://arxiv.org/abs/1701.08734)
- [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)
- [Loss of plasticity in deep continual learning](https://doi.org/10.1038/s41586-024-07711-7)

No novelty claim is made solely from combining these ideas; novelty must be demonstrated through a precise method and empirical results.
