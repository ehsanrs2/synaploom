# PROJECT_STATE — SynapLoom

> This file is the authoritative project memory. Update it in every PR that changes architecture, experiment status, accepted assumptions, or the next milestone.

**Last updated:** 2026-10-12  
**Repository:** ehsanrs2/synaploom  
**Current stage:** EXP-H1A-EVALBATCH-001 evaluation-batch-size sensitivity protocol preregistered; runner and tests pending; study not run  
**Research status:** EXP-H1A-002 remains the frozen primary result with verdict INCONCLUSIVE; EXP-H1A-ORDER-001 is COMPLETE_NUMERICALLY_VALID / ORDER_SENSITIVE. EXP-H1A-EVALBATCH-001 will evaluate six batch sizes at each in-memory training checkpoint, using the same model state across sizes. Prior artifact sets do not contain model checkpoints, so this is a new paired training cohort and not a retrospective re-evaluation of the historical weights.

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
- 20 epochs/task, SGD lr=0.01 (selected by numerical-validity-only calibration), momentum=0.9, weight decay=5e-4
- 10 paired model seeds: 201..210 for EXP-H1A-002
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

## 11. EXP-H1A-001 execution status

EXP-H1A-001 was executed on 2026-10-07 at commit `188b32ab663301e8486657bf8661d40d58514159`.

Artifact integrity passed, but numerical validity did not:

- progressive_reserve: 2/10 finite, 8/10 NaN/Inf
- all_recruited: 0/10 finite, 10/10 NaN/Inf
- fixed_small: 7/10 finite, 3/10 NaN/Inf
- dense_active_proxy: 1/10 finite, 9/10 NaN/Inf
- dense_total: 1/10 finite, 9/10 NaN/Inf

Therefore:

```text
EXP-H1A-001 = INVALID_NUMERICAL_INSTABILITY
H1a = NOT_EVALUATED
```

Do not interpret the aggregate accuracy/forgetting deltas from that run.

## 12. EXP-H1A-STABILITY-001 result

Calibration completed at commit `7d8622fdb47bdb79fce109c64066b854711bafb2`.

- lr=0.03: 1/15 stable
- lr=0.01: 15/15 stable
- lr=0.003: not run because the stop rule selected the first fully stable candidate
- selected learning rate: **0.01**

Selection used numerical validity only. Scientific metrics from the calibration were not used.

## 13. EXP-H1A-002 scientific result

EXP-H1A-002 completed on 2026-10-11 at commit `7b301224151882d677403e1ea7a91d4b26b1291c`.

- Execution/artifact validity: PASS — all 50 expected runs completed; the 50 per-run JSON files had the expected variant/seed metadata and no non-finite serialized values.
- Feature bundle SHA-256 matched the frozen artifact: `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47`.
- Environment: Python 3.11.15, PyTorch 2.14.1+cu130, CUDA 13.0, NVIDIA GeForce RTX 4090.
- Primary comparison: `progressive_reserve - all_recruited`.
- Final average accuracy delta: +0.069 percentage points; paired-bootstrap 95% CI [+0.036, +0.103] pp.
- Late-task within-task mean accuracy delta: -0.041 percentage points; paired-bootstrap 95% CI [-0.171, +0.087] pp.
- Preregistered overall verdict: **INCONCLUSIVE**. The final-average-accuracy co-primary is favorable, but the late-task co-primary is inconclusive, so the two-outcome rule is not satisfied.

H1a is **not established** by this experiment. Do not relabel the outcome as favorable or change the frozen primary protocol to improve the result after seeing it. The separately preregistered EXP-H1A-ORDER-001 class-order sensitivity study has since completed; it has its own ORDER_SENSITIVE classification and does not change the frozen primary verdict. See section 14 and `docs/EXPERIMENT_LOG.md` for its execution and limitations.

See `docs/EXPERIMENT_LOG.md` for per-variant summaries and paired seed-level deltas.

## 14. EXP-H1A-ORDER-001 — class-order sensitivity result

The frozen protocol is documented in `docs/H1A_CLASS_ORDER_SENSITIVITY_PROTOCOL.md`. The runner and tests were merged in [PR #15](https://github.com/ehsanrs2/synaploom/pull/15); local validation passed (43 tests, plus the registered dry-run).

Execution completed on 2026-10-11 at merge commit `4f572068892093124ada78023c1d4bba6eb9f331`.

- Integrity: **PASS** — all 60 registered order × seed × variant runs are present, with no non-finite numeric values in the per-run JSON files.
- Feature bundle SHA-256 matched the frozen artifact: `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47`.
- Environment: Python 3.11.15, PyTorch 2.14.1+cu130, CUDA 13.0, NVIDIA GeForce RTX 4090.
- Run matrix: orders O0/O1/O2 × seeds 301–310 × `progressive_reserve`/`all_recruited` = 60 runs.
- Analysis: 10,000 paired-bootstrap resamples per order; the pooled CI uses 10,000 seed-cluster bootstrap resamples and equal order weighting.

All deltas below are `progressive_reserve - all_recruited` in percentage points (pp).

| Class order | Final average accuracy Δ (95% CI), pp | Late-task within-task mean accuracy Δ (95% CI), pp |
|---|---:|---:|
| O0 | -0.0010 [-0.0820, +0.0660] | +0.0363 [-0.0825, +0.1658] |
| O1 | -0.0140 [-0.0470, +0.0230] | -0.0283 [-0.1060, +0.0583] |
| O2 | -0.0330 [-0.1240, +0.0390] | -0.0325 [-0.1460, +0.0728] |
| Equal-order pooled, seed-cluster bootstrap | -0.0160 [-0.0457, +0.0170] | -0.0082 [-0.0738, +0.0538] |

Registered classification: **ORDER_SENSITIVE**. The late-task point estimate is positive for O0 and negative for O1/O2, satisfying the frozen directional-heterogeneity rule. Both pooled co-primary confidence intervals include zero. This classification does **not** establish a statistically significant order-by-treatment interaction or show a robust benefit for progressive reserve recruitment.

Descriptive secondary metric deltas (not used by the registered classifier), also in pp:

| Class order | Average forgetting Δ | Backward transfer Δ |
|---|---:|---:|
| O0 | +0.0244 | -0.0244 |
| O1 | -0.0344 | +0.0344 |
| O2 | -0.0056 | +0.0056 |
| Equal-order pooled | -0.0052 | +0.0052 |

Artifacts were generated under `runs/m1_3_h1a_order_sensitivity/`; the aggregate records the 60-run manifest, environment, feature provenance, per-order paired results, and seed-cluster pooled results.

**Primary-result guardrail:** EXP-H1A-002 remains the frozen primary experiment with verdict **INCONCLUSIVE**. EXP-H1A-ORDER-001 is a separate sensitivity result and cannot retroactively change that verdict.

**Next action:** preregister evaluation-batch-size sensitivity, evaluating identical trained model states at multiple evaluation batch sizes to isolate batch-context effects from training differences.

## 15. Open design questions

Still unresolved:

- exact default d, h, N, k, q_max, K
- exact load-balancing loss
- consolidation coefficient and importance EMA hyperparameters
- first fair dynamic-sparse baseline implementation for M1.3
- implement and locally validate the EXP-H1A-EVALBATCH-001 multi-size evaluation runner; execute the 20 registered seed/variant training trajectories and record the batch-size sensitivity without changing EXP-H1A-002's primary verdict
- exact fairness reporting table
- when to introduce sample-level or hierarchical routing

## 16. Environment and dependency management

Accepted project tooling:

- uv is the official Python package/environment manager.
- `pyproject.toml` declares project, optional and development dependencies.
- `uv.lock` is committed and is the reproducible dependency resolution.
- `.python-version` pins the default development interpreter.
- use `uv sync` / `uv sync --locked` instead of pip-based project installation;
- use `uv run` for tests, smoke scripts and experiments;
- dependency changes should use `uv add` / `uv remove` and update `uv.lock`;
- CI uses `astral-sh/setup-uv`.

## 17. Documentation rule

For every merged milestone PR:

1. Update this file.
2. Update architecture/design docs if a decision changes.
3. Add all experiment commands/results to docs/EXPERIMENT_LOG.md.
4. Update docs/ROADMAP.md checkboxes.
5. State the next action explicitly.

The repository, not chat history, is the source of truth.
