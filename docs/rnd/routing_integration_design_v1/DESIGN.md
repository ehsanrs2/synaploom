# Controlled routing integration design v1

Status: design for a later opt-in implementation; NOT an installed interface.
Research baseline: fec8aed6c826df1ebb869af496608501d384f186, preserving qualified
work from 2f5ebe7526d34845f45a520b574227ac71e769ca. Production defaults stay frozen.

## Interface and rollout boundary

Propose a plain Python RoutingSession that references an existing SynapLoomModel;
it does not register/copy parameters or become a second nn.Module checkpoint root.
The model's constructor, config, forward default, SparseRouter and GraphExecutor
remain unchanged. Only an explicit session call selects alternative semantics.

```python
# Proposed API, not executable in this package:
session = RoutingSession(model, RoutingSpec())  # batch / legacy executor
legacy_output = session.forward(x)             # delegates model(x) exactly once
sample = RoutingSession(model, RoutingSpec(
    granularity="sample", execution="grouped", tie_policy="node_id_ascending_v1"))
sample_output = sample.forward(x, purpose="inference", update_statistics=False)
```

| granularity | execution | tie policy | result | eligibility |
|---|---|---|---|---|
| batch (default) | legacy (default) | legacy_topk | existing ModelOutput/ForwardTrace unchanged | all existing production configurations |
| sample | row_reference | node_id_ascending_v1 | SampleRoutingOutputV1 | H1a only |
| sample | grouped | node_id_ascending_v1 | SampleRoutingOutputV1 | H1a only |

Reject invalid combinations, unknown values and unsupported graphs before forward;
never infer a route from batch size, checkpoint absence, performance, or device.
A grouped session may internally use a qualified sample singleton fallback when
B=1; this is an execution choice within the same declared sample semantics. It
must not call production topk as a shortcut. Record requested execution and actual
path in external run provenance. Do not auto-switch batch/sample granularity.

Stage 1 scope is immutable-state inference plus explicitly requested diagnostic
backward. Both sample modes reject update_statistics=True and TrainingEngine/
continual-training attachment before any mutation or optimizer action. Existing
research SampleOutput is not yet this interface and is not a trainer drop-in.
A future capability preflight must enforce this; duck typing is insufficient.
Normal model eval/train modes remain caller-owned; routing forward must not toggle
them. Future inference adapters must restore mixed child modes even on exceptions.

## Trace and accounting contract

Legacy call returns the existing production result unchanged. Sample result is a
separate versioned type, never disguised as the one-dimensional ForwardTrace:

- logits [B,C], representation [B,D], scalar aux_loss=0, all on model device;
- selected_node_ids [B,K], router_weights [B,K], routing_context [B,router_dim];
- recruited_node_ids [R], structural_version, executed_edge_count=0;
- unique_executed_ids [U], effective_expert_rows=B*K, actual_expert_calls;
- schema_version=1, granularity, requested execution, actual execution, tie policy.

No sample-to-batch flattening, mean-query routing, or silently duplicated node
parameters. Keep traces on device until an explicit detached export. Production
metrics.py counts selected_node_ids.numel() and indexes per-node counts using
.tolist(); it cannot consume [B,K]. New accounting must report distinct expert
union U, parameter budgets for that union, per-row K and expert-row work B*K.
Allocated parameters, recruited parameters and executed parameters are separate.
Reference execution calls B*K experts; grouped calls U experts on selected rows.
Only the reference and grouped sample routes are semantic parity baselines;
production batch routing is a singleton oracle on non-ties and a different-batch
semantics/cost control otherwise.

## Checkpoints and provenance

All three routes share the same underlying parameter names/shapes. Save/load and
model_state_sha256 always target the underlying model, never the registered
research wrappers (their state_dict names have a model. prefix). Strict loading,
optimizer parameter identity/order and graph-state buffers must remain unchanged.
Do not add routing state/weights/buffers to the frozen model state_dict.

Existing checkpoint.py stores model/optimizer/controller/RNG and extra. A future
opt-in envelope uses extra.routing={schema:1, granularity, execution, tie_policy,
training_granularity, topology, implementation_revision, precision_policy}. Also
record feature/checkpoint/config hashes, structural version, session configuration
and source commit in the run manifest; hash identity alone does not bind semantics.
Do not rewrite old checkpoints to add metadata. Plain state_dict files remain
strict architecture loads with a separate immutable provenance sidecar.

Missing routing metadata means legacy batch for default/replay only. Explicit
sample evaluation of a legacy checkpoint is an allowed counterfactual experiment
with a new output directory and explicit override provenance, never a replay or
historical result replacement. Validate metadata/config/topology before loading
into a live model; fail on unknown versions, training-mode mismatches or topology
inconsistency. A future sample-trained checkpoint must not silently resume legacy
training. Changing row_reference/grouped execution can be allowed only after parity
qualification; changing batch/sample training granularity is a new experiment.

The default legacy loader/replay stays unchanged. Proposed metadata validation
belongs to the new session/loader adapter; no current loader is modified here.
Checkpoint envelope, strict load, RNG restore and optimizer/controller round-trip
are mandatory integration tests, not yet implemented by these design probes.

## Graph-state and mutation boundaries

Sample support requires q_max=0, message_rounds=0 at construction AND every forward.
No general graph, edge/message-passing, mixed-precision, compile or distributed
support is implied. Re-read recruited IDs each forward; never cache selection or
parameter bytes. GraphState.validate remains enabled. Dormant nodes are ineligible;
consolidated nodes remain eligible with unchanged forward parameters.

Recruitment/consolidation may occur between complete forward/backward/optimizer
transactions only. No mutation during forward, between forward and backward, or
concurrently from another session. Stage 1 owner serializes access; a future
adapter checks a structural-version token at transaction boundaries and fails
closed on stale dispatch. An optimization that caches static metadata must bind
it to device/dtype/topology and structural_version, invalidate after strict load,
.to(), recruitment/consolidation and topology changes, and revalidate. No cache is
needed for the initial implementation. Per-call validation is not a concurrency lock.

Evaluation and diagnostic backward do not update utilization, age, importance,
parameters or structural buffers. Full hash equality guards every diagnostic
transaction, including failures; input identity and RNG/mode preservation are
separate checks because state_dict hashes omit gradients, RNG and module modes.

## Training compatibility — explicitly blocked for Stage 1

TrainingEngine reads output.trace [K], calls update_utilization, updates node
importance, scales protected gradients, steps optimizer and constructs a
StructuralObservation whose IDs/weights are tuples of scalars. Continual logging
also iterates one-dimensional IDs. Passing [B,K] would break semantics or accounting.
No controller or statistics path is silently reused for sample routing.

A later training adapter requires a separately frozen contract. Proposed utilization
summary is u[n]=(1/B)*sum_b sum_j 1[ids[b,j]=n]*w[b,j], followed by exactly one existing
EMA decay/update and one recruited-node age increment per optimizer step. This
reduces to legacy singleton weights and preserves total mass 1; it does not preserve
batch-router learning trajectories. Controller observations need an explicit schema
carrying per-row queries plus the node-weight summary; no batch-mean query is fed
back into prediction routing. H1a no-edge policies must be explicit. Importance
uses actual accumulated gradients; protected-gradient scaling and both gradient
checks stay in their existing order. Nonfinite checks, optimizer and post-step
parameter checks cannot be skipped. This proposal is not training authorization.

Diagnostic gradient tests must retain None on never-executed expert parameters,
zero unselected routing-key rows, correct shared parameter/input gradients and
consolidated node scaling. Optimizer weight decay/momentum can update a nonexecuted
key row even with zero gradient because keys are one tensor; do not promise
unselected parameter immutability after arbitrary optimizers. That requires an
explicit future optimizer policy and tests. No optimizer is run in this phase.

## Exact ties and numerical determinism

Legacy means the existing torch.topk behavior, without stable re-sorting. Sample
node_id_ascending_v1 means stable descending computed scores, then ascending node
ID at exact equality. Never add epsilon to scores or use a near-tie threshold.
Singleton sample delegates the sample reference and preserves this tie rule.

Float32 tests use CPU atol=rtol=1e-6, CUDA/cross-device 1e-5, backward 2e-5. Selected
IDs and fixture predictions must be identical; tolerances never excuse a different
expert. Exact state hashes are required within-device for read-only transactions.
Batched GEMM shape/rounding can reorder nearly tied scores and argmax boundaries;
no universal cross-device/companion guarantee for adversarial near ties is made.
Record score margins; classify such disagreements, never count them as a pass.
TF32 is off for current qualification. Other precision/determinism policies require
fresh qualification, including deterministic-algorithm enforcement or an explicit
fail-closed unsupported-kernel result. Do not weaken global determinism settings.

## Decision and risks

GO to implement an inference/diagnostic-only opt-in session behind the gates in
GATES.md. NO-GO for production default changes, TrainingEngine integration,
automatic optimization promotion, scientific training or historical reinterpretation.
The session/trace/checkpoint adapters are future work, not delivered runtime APIs.

Primary risks are trace-shape misuse, semantic provenance absent from model hashes,
near-tie numerical instability, stale/concurrent structure, batch-union memory/cost,
optimizer row-state effects and false claims from one-process synthetic timings.
Singleton fallback is promising; row-index precomputation and slot restoration are
research candidates, not selected automatically. See PERFORMANCE.md for measured
negative results and per-candidate promotion limits. Historical EXP-H1A-002 remains
INCONCLUSIVE; EXP-H1A-ORDER-001 ORDER_SENSITIVE; EXP-H1A-EVALBATCH-001 DIRECTION_FLIP_OBSERVED.
