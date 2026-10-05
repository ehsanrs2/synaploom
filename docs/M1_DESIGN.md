# M1 Design Specification — Single-Modality Structural Core

**Status:** M1.0 + M1.1 + M1.2 implemented; fair H1 evaluation pending  
**Milestone:** M1  
**Date:** 2026-10-05

## 1. Purpose

M1 builds the smallest SynapLoom system that can test reserve-capacity recruitment without introducing multimodal learning, biological neuron models, local learning rules, or complicated automatic novelty detection.

M1 is primarily an architecture/control-plane milestone.

## 2. Scope split

### M1.0 — execution and state machinery

Implement:

- fixed total microcircuit capacity
- structural states
- recruited-vs-executed distinction
- batch-level sparse routing
- bounded sparse edge representation
- synchronous message passing
- task head
- event tracing
- exact checkpoint/restore

### M1.1 — controlled structural adaptation

Add:

- deterministic/scheduled recruitment policy
- context imprinting for newly recruited routing keys
- optional task-boundary consolidation
- controlled edge creation for recruited nodes
- structural metrics

Automatic calibrated novelty/growth detection remains M2.

This separation lets us validate mechanics before evaluating a noisy adaptive controller.

## 3. Key state distinction

A node can be:

- DORMANT: not eligible for routing, not executed
- PLASTIC: recruited and eligible for routing, full plasticity
- CONSOLIDATED: recruited and eligible for routing, reduced plasticity

Separately, each forward pass produces an execution mask.

Therefore:

DORMANT/PLASTIC/CONSOLIDATED = long-timescale structural state

selected/not-selected = short-timescale runtime state

A PLASTIC or CONSOLIDATED node may be recruited but not executed for a particular batch.

## 4. Fixed-capacity sparse topology

Do not store a dense N x N adjacency matrix.

Use bounded edge slots:

- node capacity N
- max outgoing structural slots q_max per node
- each slot stores destination, enabled flag, relation id, gate/statistics

Conceptually all admissible node pairs may exist in the logical supergraph, but only O(N q_max) topology state is materialized.

For M1, q_max is small and fixed by config.

## 5. Batch-level routing for the MVP

M1 uses **batch-level top-k routing**.

Reason:

- simpler deterministic execution
- easier real compute sparsity in Python/PyTorch
- easier checkpoint/reproducibility
- avoids grouped per-example expert dispatch before the research mechanism is validated

For encoded samples z_b:

q_b = W_q z_b

The batch routing query is:

q = mean_b(q_b)

For each recruited node i with routing key k_i:

score_i = <q, k_i> / sqrt(d_r)

Only recruited nodes are scored.

Select:

X = TopK(score, k)

Apply softmax only across the selected scores to produce routing weights.

Top-k indices are discrete. Gradients flow through selected gates/keys/query but not through the index operation.

Per-example routing is a later ablation/optimization.

## 6. Recruitment policy

Dormant nodes do not need to be continuously scored.

In M1, dormant nodes are considered exchangeable prior to recruitment.

A recruitment event:

1. chooses the lowest/free deterministic dormant node id (or another deterministic policy);
2. changes state DORMANT -> PLASTIC;
3. initializes its routing key using the triggering batch context q, optionally with seeded small noise;
4. creates a bounded number of structural edge slots according to the configured edge-init policy;
5. logs the event.

This avoids an O(N) or O(N^2) dormant candidate search and gives a newly recruited node an immediate affinity for the context that caused recruitment.

Random-key initialization is retained as an ablation.

## 7. MicroCircuit

Default M1 microcircuit:

input d
-> LayerNorm
-> Linear(d, h)
-> activation
-> Linear(h, d)
-> residual output

The exact activation is configurable; GELU is a reasonable default.

All N microcircuits exist at model construction time.

Only the k selected modules are invoked in the normal forward path.

Dormant modules must not receive gradients.

## 8. Message passing

M1 uses a small fixed number K of synchronous rounds.

For executed node i:

h_i^0 = MicroCircuit_i(z)

For each active structural edge i -> j whose endpoints are both in the execution set:

m_ij = gate_ij * T_relation(h_i)

Aggregate messages at j with sum/mean normalization.

Then update:

h_j^{r+1} = h_j^r + alpha * MicroCircuit_j(LN(h_j^r + aggregate_j))

Implementation may use one initial microcircuit pass plus lightweight update blocks if invoking the full expert K+1 times is too expensive. The implementation must record which variant is used.

M1 initially has one intra-region relation type.

## 9. Edge operator

Default M1 edge operator:

**scalar gate + shared relation projection**

No dense parameter matrix per edge.

For a single relation:

message = gate_edge * W_rel h_source

W_rel is shared by all edges of that relation.

A pure identity+gate mode must be available as a lower-cost ablation.

## 10. Output aggregation

After message passing, combine executed node states using router weights:

h_out = sum_{i in X} alpha_i h_i

Then:

logits = Head(h_out)

The initial head is task-agnostic/class-incremental unless an experiment explicitly declares a task-aware setting.

## 11. Structural controller interface

The controller does not mutate topology during forward/backward.

Mutation boundary:

forward
-> loss
-> backward
-> consolidation gradient scaling
-> optimizer.step
-> metric/statistic update
-> optional structural event under torch.no_grad()
-> next step

This ordering is an invariant.

Suggested interface:

StructuralController.observe(trace, loss_stats)
StructuralController.should_update(global_step)
StructuralController.propose(graph_state, stats) -> StructuralPlan
StructuralController.apply(graph_state, plan) -> list[StructuralEvent]

The proposal and application phases are separated so plans can be tested/logged before mutation.

## 12. M1 structural policies

Implement policy interfaces, with simple policies first:

### RecruitmentPolicy
- NoRecruitment
- FixedScheduleRecruitment
- TaskBoundaryRecruitment

### EdgeInitPolicy
- NoEdges
- ConnectToSelectedTopR
- SeededRandomR

### ConsolidationPolicy
- NoConsolidation
- TaskBoundaryTopImportance

Adaptive novelty-driven recruitment is M2.

## 13. Consolidation implementation

Importance statistic for M1:

I(theta) <- beta I(theta) + (1-beta) |theta * grad(theta)|

Aggregate parameter importance to node importance.

A consolidated node receives a plasticity multiplier:

0 < p_cons < 1

Rather than dynamically changing optimizer parameter groups, apply per-node gradient scaling after backward and before optimizer.step.

Dormant/unselected nodes should normally have grad=None.

## 14. Router anti-collapse

Track per-node:

- selection count
- routing weight mass
- utilization EMA

M1 may include a small load-balancing auxiliary loss.

The exact formula must be replaceable and independently disableable.

Tests/metrics must detect:

- one-node domination
- never-selected recruited nodes
- severe utilization imbalance

## 15. Data flow object

Forward should return structured output, not only logits.

Suggested conceptual schema:

ModelOutput:
- logits
- representation
- aux_loss
- trace

ForwardTrace:
- recruited_node_ids
- selected_node_ids
- router_scores/weights
- executed_edge_ids/count
- structural version
- optional per-node activations/statistics

Trace tensors used only for diagnostics should be detach-able to avoid accidental graph retention.

## 16. Graph state

Structural state should be represented by registered tensors/buffers so checkpointing is exact.

Minimum state:

- node_state[N]
- node_age[N]
- node_utilization_ema[N]
- node_importance_ema[N]
- node_plasticity[N]
- routing_keys[N, d_r]
- edge_dst[N, q_max]
- edge_enabled[N, q_max]
- edge_relation[N, q_max]
- edge_gate[N, q_max]
- edge_utilization_ema[N, q_max]
- structural_version
- counters required by policies

Trainable quantities remain Parameters; persistent non-trainable state remains buffers/controller state.

## 17. Checkpoint contract

A resumable checkpoint must include:

- model state_dict
- optimizer state_dict
- scheduler state if present
- controller/policy state
- global step
- task/stream position
- experiment config snapshot/hash
- Python RNG state
- NumPy RNG state
- torch CPU RNG state
- torch CUDA RNG state(s)
- structural event sequence or event-log cursor

For exact-resume tests, use deterministic data order and initially num_workers=0.

## 18. Package layout

Proposed source layout:

src/synaploom/
  __init__.py
  config.py
  model.py
  core/
    states.py
    graph_state.py
    microcircuit.py
    router.py
    edge_ops.py
    executor.py
    outputs.py
  plasticity/
    controller.py
    policies.py
    consolidation.py
    statistics.py
  training/
    engine.py
    checkpoint.py
    metrics.py
    events.py
  data/
    streams.py

experiments/
  configs/
  baselines/
  scripts/

tests/
  unit/
  integration/

docs/

This is a proposed implementation skeleton, not a public API guarantee.

## 19. Configuration

Use typed Python dataclasses as the canonical configuration schema.

Human-readable experiment configurations may be YAML, loaded into validated dataclasses.

Avoid introducing Hydra in M1 unless configuration composition becomes painful.

Every run writes the resolved configuration into its output directory.

Suggested config groups:

- model
- graph
- router
- structural policy
- consolidation
- optimizer
- stream/dataset
- evaluation
- reproducibility

## 20. Required unit tests before experiments

### Structural invariants
- dormant node is never selected
- dormant node does not execute
- DORMANT -> PLASTIC transition is deterministic
- invalid transitions are rejected
- selected count is <= k
- out-degree never exceeds q_max
- edges never point outside valid node ids
- edge execution requires both endpoints to be in the execution set

### Gradient invariants
- unselected node has no parameter gradients
- dormant node has no gradients
- consolidated node gradient norm is scaled as configured
- selected router gates receive gradients

### Checkpoint invariants
- graph structural tensors restore bit-for-bit
- router selections match after restore for the same batch/RNG state
- next structural event matches after restore

### Accounting invariants
- recruited-node count correct
- executed-node count correct
- active-edge count correct
- parameter accounting correct

## 21. Smoke experiments

Before continual-learning claims:

1. identity/toy classification to prove routing and gradients;
2. synthetic multi-cluster data to see whether different recruited experts specialize;
3. SplitMNIST or equivalent as a functional continual-learning smoke test.

No research claim should be based on these.

## 22. First scientific benchmark

Initial candidate:

**Split CIFAR-100, class-incremental**, with task boundaries available to the training controller for controlled M1 policies but not provided as an inference feature.

Final protocol must define:

- number of tasks/classes per task
- encoder capacity
- total and executed parameter budgets
- replay/no-replay condition
- augmentation
- number of seeds
- evaluation matrix after every task

This choice may be revised before experiments if a stronger fairness argument emerges.

## 23. Baseline fairness

SynapLoom needs multiple fairness views:

### Equal total capacity
Same total parameter budget including dormant reserve.

### Equal executed capacity
Comparable parameters/FLOPs executed per batch.

### Equal wall-clock
Optional practical comparison after implementation is optimized.

A result is not sufficient if SynapLoom wins only because it owns much more total capacity without that fact being reported.

## 24. M1 exit criteria

M1 is complete when:

1. structural execution passes all invariant tests;
2. checkpoint/resume is deterministic in the controlled test;
3. a scheduled recruitment run completes end-to-end;
4. recruited and executed capacity are measured separately;
5. dense/fixed-routing control models can run through the same training harness;
6. at least one continual smoke experiment verifies the measurement pipeline;
7. no automatic novelty, multimodal, or biological-extension claims are mixed into the result.

M1 produces trustworthy machinery. M2 asks whether adaptive structural control is actually useful.


## 25. M1.0 implementation record

Implemented on 2026-10-05.

The current executor variant is:

1. run each selected microcircuit once on the encoded batch;
2. for each message round, aggregate mean incoming messages from selected-endpoint edges;
3. add the aggregate to the current node state;
4. run the same residual microcircuit again;
5. aggregate final selected node states using router weights.

This is the allowed full-expert-per-round variant. M1.0 does not use a separate explicit message alpha coefficient.

Validation at implementation time:

- 10 local unit tests passed;
- tests cover structural transitions, bounded edge slots, recruited-only routing, sparse router gradients, non-execution/non-gradient of unselected microcircuits, selected-endpoint edge execution, consolidation gradient scaling, structural/parameter accounting, checkpoint restoration, and append-only event logs;
- a GitHub Actions pytest workflow is included for independent PR validation.

Not yet implemented:

- StructuralController;
- scheduled/task-boundary policy interfaces;
- controller-integrated event emission;
- context-imprinted automatic recruitment;
- end-to-end training engine;
- automatic novelty/growth detection.

The first four items are M1.1. Automatic novelty remains M2.


## 26. M1.1 implementation record

Implemented on 2026-10-05 in the controlled-adaptation milestone.

### Controller boundary

The implemented control flow is:

forward
-> task loss
-> backward
-> node-importance update
-> node-owned consolidation gradient scaling
-> optimizer.step
-> utilization update
-> StructuralObservation
-> StructuralController.observe
-> propose StructuralPlan
-> apply plan under no_grad

Structural mutation never occurs during forward/backward.

### Recruitment policies

Implemented:

- NoRecruitment
- FixedScheduleRecruitment
- TaskBoundaryRecruitment

The controller deterministically chooses the lowest-ID dormant node when recruitment is requested.

### Context imprinting

The new node's routing key uses the direction of the triggering batch routing context. Its norm is rescaled to the median norm of currently recruited routing keys.

This avoids a raw context vector winning routing simply because of scale. Optional seeded noise can be added and re-normalized.

### Edge initialization

Implemented:

- NoEdges
- ConnectToSelectedTopR
- SeededRandomR

All policies respect q_max. SeededRandomR derives its deterministic permutation from seed, structural step, and new node id.

### Consolidation

Implemented:

- NoConsolidation
- TaskBoundaryTopImportance

Node importance is an EMA of mean |parameter * gradient| over parameters owned by the node.

Consolidation scales gradients for:

- the node microcircuit;
- the node routing-key row;
- the node outgoing edge-gate row.

Shared relation projections remain globally shared/plastic in M1 because assigning their gradient to one node would be ambiguous.

### Task-boundary semantics

For TrainingEngine, task_boundary=True means the current training step is the final step of the current task.

At that boundary:

1. the step is trained normally;
2. importance/utilization are updated;
3. selected old capacity may be consolidated;
4. new dormant capacity may then be recruited for the following task.

Therefore a newly recruited node is not immediately consolidated at the same boundary.

### Checkpointing

Controller state currently stores the last applied structural step. M1.1 policies are otherwise deterministic/stateless from their explicit configuration, so this prevents duplicate structural mutation after resume.

### Import architecture

StructuralEvent and JsonlEventLogger live in synaploom.events. synaploom.training.events remains a compatibility re-export.

This keeps the structural controller independent from the training package and avoids a controller -> training -> engine -> controller import cycle.

### Validation scope

The milestone includes unit/integration tests and an end-to-end scheduled-growth smoke script in CI.

Final M1.1 CI validation:

- 17 pytest tests passed;
- scheduled smoke recruited nodes 2 -> 5 at steps 3, 6, and 9;
- three bounded structural edges were added;
- structural version advanced from 0 to 6;
- pytest and the standalone smoke script both completed successfully.

These validate mechanism correctness only. The smoke loss values are not research metrics and do not support accuracy, forgetting, transfer, or compute claims.

### Next design step

M1.2 builds the continual evaluation harness and no-growth controls before any adaptive novelty policy is introduced.


## 27. M1.2 implementation record

Implemented on 2026-10-05.

### Continual stream contract

M1.2 introduces immutable tensor-backed tasks and ordered streams. Each task declares:

- task id;
- global class ids;
- train tensors;
- evaluation tensors.

The harness is class-incremental by default: labels remain global and no task id is supplied as an inference feature.

Training batches are generated deterministically from explicit seeds and streamed rather than materialized as a list.

### Evaluation matrix

After training stage t, the harness evaluates every seen task j and records:

R[t, j] = accuracy on task j after learning through task t.

Future unseen cells remain NaN by default. An opt-in mode can evaluate future tasks for diagnostics, but standard continual metrics use learned-stage semantics.

Because the M1 router is batch-level, predictions can depend on the other examples in the evaluation batch. M1.2 therefore makes evaluation order deterministic and records eval_batch_size in the run config. M1.3 must freeze this batching protocol and should include a sensitivity check before interpreting small accuracy differences.

Derived metrics currently include:

- average accuracy by stage;
- final average accuracy;
- forgetting;
- backward transfer;
- diagonal immediate-learning accuracy;
- late-task immediate-learning accuracy.

The final metric is a proxy only. A stronger loss-of-plasticity claim requires learning-curve/fresh-reference comparisons in later milestones.

### Structural history

For routed models, each training stage records:

- structural version;
- recruited-node count;
- active-edge count;
- mean executed nodes per batch;
- routing selection entropy;
- maximum selection share;
- node selection counts.

The structural state is sampled after the stage boundary action. Selection statistics summarize the batches used to learn that stage.

### Task-boundary convention

By default, task_boundary=True is sent only when another task follows.

This prevents the controlled recruitment policy from adding unused reserve capacity after the final task merely because the experiment ended.

### Controls

M1.2 runs three model modes through the same continual harness:

1. controlled-growth SynapLoom via TrainingEngine;
2. routed SynapLoom without a structural controller;
3. DenseControlModel.

The dense model exists to validate common measurement/control flow. It is not yet parameter- or compute-matched; fair matching is an M1.3 requirement.

### Streams and smoke validation

The core package provides a deterministic Gaussian multi-cluster class-incremental stream for offline engineering validation.

A real SplitMNIST adapter is also provided behind an optional torchvision dependency. CI does not download MNIST.

The CI smoke compares growth, no-growth, and dense execution paths without imposing any accuracy threshold. Its purpose is to verify the pipeline and structural trajectory, not to select a winning model.

### Artifact provenance

A saved continual result includes:

- accuracy matrix;
- evaluation-loss matrix;
- train-loss history;
- structural history;
- global step count;
- resolved ContinualConfig;
- task ids/classes and train/eval counts;
- optional caller metadata.

M1.3 must extend experiment-level provenance with the full model/baseline config and hardware/commit metadata used for scientific runs.
