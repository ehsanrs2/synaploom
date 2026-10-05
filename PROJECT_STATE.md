# PROJECT_STATE — SynapLoom

> This file is the authoritative project memory. Update it in every PR that changes architecture, experiment status, accepted assumptions, or the next milestone.

**Last updated:** 2026-10-05  
**Repository:** ehsanrs2/synaploom  
**Current stage:** M1.3 H1a protocol and benchmark machinery implemented in PR; scientific execution pending  
**Research status:** M1.0-M1.2 merged; H1a is preregistered as a controlled reserve-recruitment test; no scientific benchmark result yet

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

- fixed total microcircuit capacity
- microcircuit-level graph nodes, not biological-neuron-level nodes
- node states: DORMANT, PLASTIC, CONSOLIDATED
- recruited structural capacity is separate from runtime executed capacity
- modality-private pathways in later multimodal milestones
- sparse conditional top-k routing
- M1 uses batch-level routing
- structural updates happen on a slower schedule than gradient updates
- backpropagation through executed paths for the first prototype
- growth means recruiting pre-existing dormant capacity
- normal runtime routing scores recruited nodes only
- dormant recruitment is controlled by the structural controller
- logical supergraph is represented with bounded edge slots, not a dense N x N adjacency matrix
- default edge transform is scalar gate + shared relation projection
- automatic calibrated novelty growth remains M2
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
- recruited-capacity growth will be sublinear
- graph topology will converge

All are empirical/theoretical questions.

## 4. Foundation architecture corrections

1. Dynamic topology is represented within fixed total model capacity.
2. Removed the extra STABLE node state from the MVP.
3. Rejected per-edge full dense matrices as the default.
4. Rejected raw co-occurrence as sufficient cross-modal association evidence.
5. Rejected a single hand-weighted novelty scalar as the default growth trigger.
6. Separated loss of plasticity from catastrophic forgetting.
7. Deferred dendrites, spiking, neuromodulation, and local predictive learning.
8. Split validation into single-modality structural learning first, multimodal claims later.

## 5. M1 design decisions

1. Separate recruited nodes from executed nodes.
2. Keep executed expert count k bounded even when recruited capacity grows.
3. Use batch-level top-k routing first; per-sample dispatch is deferred.
4. Score only recruited nodes during normal routing.
5. Treat dormant nodes as exchangeable before recruitment in M1.
6. Recruit a deterministic free node and imprint its routing key from triggering context.
7. Materialize topology as O(N q_max) bounded edge slots.
8. Use scalar edge gates plus a shared relation projection by default.
9. Use synchronous bounded message-passing rounds.
10. Mutate structural state only after optimizer.step under no_grad.
11. Implement policy interfaces with scheduled/task-boundary policies before adaptive novelty policies.
12. Apply consolidation with per-node gradient scaling rather than dynamic optimizer groups; node-owned parameters include the microcircuit, routing-key row, and outgoing edge gates. Shared relation projections remain shared/plastic.
13. Use typed Python dataclasses as canonical configuration; YAML may supply run configs.
14. Use a src/ package layout with unit/integration tests.
15. Treat Split CIFAR-100 class-incremental as the current first scientific benchmark candidate after toy/smoke tests.
16. Context-imprinted routing keys preserve the triggering context direction but are norm-calibrated to the median recruited-key norm.
17. `task_boundary=True` means the task ends after the current train step: old capacity is consolidated first, then new capacity may be recruited for the following task.
18. Structural events live in a neutral module so the controller/control plane does not depend on the training package import graph.

## 6. Mathematical/computational position

SynapLoom is currently viewed as a:

- mixed discrete-continuous optimization problem
- non-convex online learner
- hybrid dynamical system with multiple timescales
- dynamic sparse graph under capacity constraints

Target M1 forward cost, excluding encoder/head, is approximately:

O(B A d + B k d h + K B k q d)

for recruited nodes A, executed top-k nodes k, representation width d, microcircuit hidden width h, bounded degree q, and K message rounds.

Total microcircuit parameter memory remains approximately O(N d h); dormant capacity is not memory-free.

See docs/THEORY.md for assumptions and open questions.

## 7. M1.0 implementation status

Implemented:

- typed GraphConfig / ModelConfig
- fixed-capacity GraphState with registered structural buffers
- DORMANT / PLASTIC / CONSOLIDATED state transitions
- trainable routing keys
- batch-level top-k SparseRouter over recruited nodes only
- residual MicroCircuit experts
- bounded q_max edge slots
- scalar edge gates + shared relation projections
- bounded synchronous message passing
- ModelOutput / ForwardTrace
- structural utilization tracking
- consolidation gradient scaling for microcircuits
- total/recruited/executed capacity accounting
- checkpoint/restore for model, optimizer, step, extra metadata, and RNG state
- append-only JSONL structural event logger
- pytest package configuration and GitHub Actions test workflow

Implementation choice recorded for message passing:

- one initial expert pass;
- each message-passing round adds mean incoming messages to the node state;
- the same residual microcircuit is then applied again;
- no explicit message alpha parameter is used in M1.0.

Local validation on 2026-10-05:

- 10 unit tests passed
- PyTorch 2.10.0+cpu
- pytest 9.0.2

The GitHub CI workflow will independently validate the repository branch/PR.

## 8. M1.1 implementation status

Implemented in the current milestone PR:

- StructuralController with observe / propose / apply separation
- deterministic lowest-free-node recruitment
- FixedScheduleRecruitment and TaskBoundaryRecruitment
- NoRecruitment control policy
- context-imprinted routing keys with norm calibration
- bounded ConnectToSelectedTopR edge initialization
- deterministic SeededRandomR edge-init baseline
- NoEdges control policy
- TaskBoundaryTopImportance consolidation
- NoConsolidation control policy
- node-importance EMA from mean |parameter * gradient| over node-owned parameters
- node-owned consolidation for microcircuit parameters, routing keys, and outgoing edge gates
- mutation-safe TrainingEngine
- controller checkpoint/restore state
- controller-integrated structural event logging
- end-to-end controlled-growth integration test
- standalone controlled-growth smoke script executed by CI

Validation on GitHub Actions (Python 3.11, CPU-only PyTorch):

- 17 pytest tests passed
- controlled-growth smoke passed
- smoke structural outcome: recruited nodes 2 -> 5, active edges 0 -> 3, structural version 0 -> 6
- smoke loss values are engineering diagnostics only and are not interpreted as research results

Automatic novelty-driven recruitment remains explicitly outside M1.1.

## 9. M1.2 implementation status

Implemented in the current milestone PR:

- immutable tensor-backed TensorTask / ContinualStream abstraction
- deterministic seeded batch iteration
- deterministic Gaussian class-incremental engineering stream
- optional real SplitMNIST adapter with lazy torchvision dependency
- common evaluation-after-each-task accuracy/loss matrix
- average accuracy, final accuracy, forgetting, backward transfer, diagonal learning accuracy, and late-task learning proxy
- per-stage structural history: recruited nodes, active edges, executed-node mean, routing selection entropy/share/counts
- DenseControlModel through the same continual harness
- routed SynapLoom no-growth control through the same harness
- controlled-growth SynapLoom through the existing TrainingEngine
- JSON result artifacts with resolved continual config and task-stream manifest
- deterministic synthetic continual smoke in CI
- optional real SplitMNIST functional smoke script

Measurement semantics:

- unseen future tasks are NaN in the standard matrix unless explicitly requested;
- task_boundary is emitted only between tasks by default, avoiding unused final recruitment;
- structural snapshots are post-stage/post-boundary state, while routing selection statistics summarize training on that stage;
- late_task_learning_accuracy is an immediate post-learning accuracy proxy, not by itself a loss-of-plasticity claim;
- the M1.2 dense control validates pipeline comparability but is not yet parameter/compute matched;
- because M1 uses batch-level routing, routed predictions are batch-context dependent; evaluation order and eval_batch_size are therefore part of the frozen protocol until sample-level routing is tested.

## 10. M1.3 H1a protocol status

The first scientific test is now split into H1a and H1b.

H1a asks whether controlled progressive reserve availability helps under equal total capacity and equal top-k expert execution. H1b asks whether the model can detect persistent need for recruitment automatically; H1b remains M2.

H1a is frozen as a mechanism-isolation experiment:

- CIFAR-100 frozen ImageNet ResNet-18 features
- 10 experiences x 10 classes
- fixed explicit class order
- 100-way classifier with unseen logits masked
- no replay
- no consolidation
- q_max=0 and message_rounds=0
- N=16, initial recruited=4, top-k=4
- one recruitment after the first optimization step of each new task after task 0
- 20 epochs/task, SGD lr=0.03, momentum=0.9, weight decay=5e-4
- 10 paired model seeds
- fixed eval_batch_size=256 because routing is batch-level

Primary causal comparison:

progressive_reserve vs all_recruited

Both use the same total parameter tensors and the same top-k expert execution count.

Additional controls:

- fixed_small
- dense_active_proxy
- dense_total

Outputs include per-seed results, paired bootstrap CIs, full continual matrices, within-task learning curves, structural histories, parameter-budget views, linear-MAC estimates, wall time, peak CUDA memory, commit/environment metadata, and feature-bundle provenance.

See docs/M1_3_PROTOCOL.md.

## 11. No scientific experimental results yet

There are currently:

- no continual-learning benchmark results
- no accuracy/forgetting claims
- no compute-efficiency claims
- no ablation conclusions
- no evidence yet that controlled growth improves learning

Engineering smoke tests validate mechanics only. Research results must be recorded in docs/EXPERIMENT_LOG.md.

## 12. Immediate next milestone

**Execute M1.3a — H1a primary paired run**

Execution safeguards are now specified in `docs/H1A_RUNBOOK.md`:

- feature extraction writes SHA-256 provenance;
- a mandatory preflight validates dimensions, finiteness, class ids/counts and metadata;
- the benchmark independently reruns the same audit before any seed;
- every run records the exact feature-bundle SHA-256.

Next actions:

1. Extract the frozen CIFAR-100 ResNet-18 feature bundle.
2. Run the mandatory H1a preflight and preserve its output.
3. Run the preregistered 10-seed benchmark.
4. Verify all expected artifacts/provenance and paired comparisons.
5. Record results in docs/EXPERIMENT_LOG.md.
6. Only after inspecting the preregistered H1a result, decide whether to:
   - proceed to H1a sensitivity / end-to-end replication;
   - add M1.3b external sparse baselines;
   - revise the mechanism before M2.

No adaptive novelty trigger is allowed into H1a.

## 13. Open design questions

Still unresolved:

- exact default d, h, N, k, q_max, K
- exact load-balancing loss
- consolidation coefficient and importance EMA hyperparameters
- first fair dynamic-sparse baseline implementation for M1.3
- class-order and eval-batch-size sensitivity protocol after the primary H1a run
- exact fairness reporting table
- when to introduce sample-level or hierarchical routing

## 14. Environment and dependency management

Accepted project tooling:

- uv is the official Python package/environment manager.
- `pyproject.toml` declares project, optional and development dependencies.
- `uv.lock` is committed and is the reproducible dependency resolution.
- `.python-version` pins the default development interpreter.
- use `uv sync` / `uv sync --locked` instead of pip-based project installation;
- use `uv run` for tests, smoke scripts and experiments;
- dependency changes should use `uv add` / `uv remove` and update `uv.lock`;
- CI uses `astral-sh/setup-uv`.

## 15. Documentation rule

For every merged milestone PR:

1. Update this file.
2. Update architecture/design docs if a decision changes.
3. Add all experiment commands/results to docs/EXPERIMENT_LOG.md.
4. Update docs/ROADMAP.md checkboxes.
5. State the next action explicitly.

The repository, not chat history, is the source of truth.
