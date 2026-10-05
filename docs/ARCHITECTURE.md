# SynapLoom Architecture

**Status:** architecture baseline v0.1  
**Date:** 2026-10-05  
**Scope:** research prototype, not a claim of biological fidelity

## 1. Research goal

SynapLoom studies whether a neural system with pre-existing reserve capacity can learn continually by changing which modules and connections are active, while preserving modality-specific pathways and selectively consolidating useful knowledge.

The central engineering hypothesis is:

> Learning should be able to modify topology, routing, parameter values, and plasticity state — not only parameter values.

SynapLoom is brain-inspired, not a brain simulation.

## 2. Core abstraction

The implementation uses a **fixed typed supergraph**:

G* = (V, E*)

At time t, the effective computational graph is induced by masks and state:

G_t = (V_active(t), E_active(t))

No new PyTorch modules need to be allocated during training in v0.1. "Growth" means activating previously dormant capacity inside the fixed supergraph. "Pruning" means deactivating capacity.

This preserves the scientific idea of structural plasticity while keeping optimizer state, checkpointing, batching, reproducibility, and GPU execution manageable.

## 3. Granularity

A graph node is **not an individual biological neuron** in v0.1. A node is a small trainable microcircuit/expert.

Hierarchy:

- Region
- Microcircuit node
- Conventional artificial units inside the microcircuit

Individual dendritic or spiking computation is explicitly deferred.

## 4. Regions

Initial regions:

- Modality-private regions, e.g. vision and audio
- Association region
- Optional memory/context region
- Output/task head
- Reserve capacity exists inside each eligible region

Encoders are modality-specific. Modalities are not forced into an early common representation.

## 5. Node state

Each microcircuit has one structural state:

- DORMANT — exists in the supergraph but is excluded from active computation
- PLASTIC — active and freely trainable
- CONSOLIDATED — active but protected by reduced plasticity

We intentionally remove a separate STABLE state from v0.1. Three states are sufficient to test the main hypothesis without introducing unnecessary state transitions.

Suggested runtime metadata per node:

- utilization EMA
- activation frequency
- gradient/importance statistic
- age since activation
- plasticity coefficient
- region/type
- routing key or routing parameters

## 6. Edge representation

Edges are typed and directed.

An edge contains:

- active/inactive mask
- source and destination node IDs
- relation type
- importance/utilization statistics
- plasticity coefficient
- optional lightweight transfer parameters

Important implementation constraint: v0.1 must **not** allocate an independent dense d x d matrix for every possible node pair. That would turn structural flexibility into quadratic parameter growth.

Preferred edge operators for the first prototype:

1. identity + scalar gate, or
2. shared relation-specific projection, or
3. low-rank adapter on selected relations

Full per-edge matrices are a later ablation, not the default.

## 7. Sparse conditional routing

Each region exposes multiple candidate microcircuits but only top-k nodes are active for a sample or batch.

Conceptually:

score_i = router(context, node_i)
active = TopK(score, k)

Routing goals:

- sparse computation
- specialization
- reuse for similar inputs
- limited overlap between unrelated inputs
- availability of unused reserve capacity

Known failure modes include routing collapse, expert starvation, and unstable specialization. Therefore v0.1 includes:

- deterministic warm-up
- utilization monitoring
- a load-balancing regularizer or equivalent anti-collapse mechanism
- fixed/random routing as an ablation baseline

## 8. Forward computation

High-level flow:

1. modality-specific encoding
2. modality-private sparse routing
3. typed message passing across active nodes
4. optional association-region routing
5. task head / predictive head
6. loss computation
7. backpropagation only through the active differentiable path

v0.1 keeps ordinary backpropagation. Local learning and predictive-coding-style credit assignment are separate future experiments so architecture effects are not confounded with learning-rule effects.

## 9. Multimodal organization

For modalities m in M:

x_m -> Encoder_m -> PrivateRegion_m

Private regions may connect to an association region through typed cross-modal edges.

A key hypothesis is that shared concepts can emerge as reusable distributed subgraphs rather than requiring all modalities to be fused at one fixed layer.

However, **temporal co-occurrence alone is not sufficient evidence for a semantic edge**. Cross-modal growth must be supported by an objective signal such as:

- persistent co-activation, plus
- measurable predictive/task improvement, alignment improvement, or reduced representation error

This guardrail prevents spurious associations from being treated as knowledge.

## 10. Structural plasticity controller

Structural updates happen on a slower clock than gradient updates.

Timescales:

activation/routing << weight updates << structural updates << consolidation

The controller can:

- activate a dormant node
- activate an allowed dormant edge
- prune a low-utility edge
- return an unused plastic node to dormant state
- reduce plasticity of important nodes/edges

Structural actions are discrete and logged.

## 11. Growth trigger

The previous idea of one arbitrary weighted novelty scalar is rejected for v0.1.

Instead, the controller observes a vector of signals, for example:

- task or predictive residual
- model uncertainty
- routing saturation / low available capacity
- representation novelty
- persistent gradient conflict

Each signal is normalized against a moving historical distribution.

A structural growth event requires a **persistent condition across multiple structural windows**, for example:

- performance residual remains above a calibrated quantile, AND
- current active capacity is saturated or routing diversity is inadequate, AND
- reserve capacity is available

Thresholds should be quantile- or validation-calibrated rather than chosen as unnormalized absolute constants.

## 12. Consolidation

Useful capacity should become harder, not necessarily impossible, to overwrite.

Initial importance proxy:

importance(theta) = EMA(|theta * grad(theta)|)

A consolidated component receives a lower effective learning rate or gradient multiplier.

Consolidation must be compared with standard continual-learning regularization baselines; it is not assumed to be beneficial.

## 13. Pruning and recycling

An edge/node becomes a pruning candidate when it has persistently low:

- utilization
- contribution
- importance

Pruning is conservative and happens only at structural update boundaries.

A later experiment may reinitialize/recycle low-utility units to maintain plasticity, motivated by continual-backprop results showing that loss of plasticity is distinct from catastrophic forgetting.

## 14. What is deliberately NOT in v0.1

The following are research extensions, not foundational requirements:

- spiking neurons
- biological neuron simulation
- dendritic compartments
- global neuromodulatory learning rules
- fully local learning
- predictive coding as a replacement for backprop
- generative replay
- dynamic runtime allocation of new modules
- arbitrary graph rewiring at every optimization step

These may be introduced only after the structural core has a clean baseline.

## 15. Invariants

The first implementation should enforce:

1. Total supergraph capacity is fixed during a run.
2. Dormant capacity consumes parameter memory but should not consume normal forward FLOPs.
3. Active node/edge counts are measurable at every evaluation point.
4. Structural changes are deterministic given seed, data order, and controller state.
5. Every growth/prune/consolidation event is logged.
6. Structural state is checkpointable and exactly restorable.
7. No claim of parameter or compute efficiency is made until actual FLOPs/wall-time/memory are measured.

## 16. Initial objective

The first experiments optimize task loss plus only the minimum regularizers needed for routing stability and capacity discipline.

Avoid a large multi-term biologically motivated objective in v0.1.

Candidate:

L = L_task + lambda_load L_load + lambda_capacity L_capacity + optional lambda_stability L_stability

Predictive, cross-modal, replay, and local-learning losses are introduced one at a time through ablations.

## 17. Scientific positioning

SynapLoom overlaps with several existing families:

- dynamic sparse training (e.g. RigL)
- sparse continual learning and rewiring (e.g. NISPA)
- reusable pathways (PathNet)
- progressive/expandable continual networks
- modular and mixture-of-experts routing
- multimodal continual learning
- plasticity-maintenance methods

The project does **not** currently claim that any individual mechanism is novel. The research question is whether the specific combination of fixed reserve capacity, structural recruitment, consolidation, modality-private routing, and learned cross-modal association produces measurable advantages.

## 18. References / anchors

- NISPA: https://arxiv.org/abs/2206.09117
- RigL: https://arxiv.org/abs/1911.11134
- PathNet: https://arxiv.org/abs/1701.08734
- Progressive Neural Networks: https://arxiv.org/abs/1606.04671
- Loss of plasticity in deep continual learning: https://doi.org/10.1038/s41586-024-07711-7
- Dendritic ANN work: https://arxiv.org/abs/2404.03708
- CavRL continual audio-visual benchmark: https://doi.org/10.1016/j.knosys.2024.112513

## 19. Architecture question for v0.1

The first falsifiable question is intentionally narrow:

> Under a fixed total parameter budget, does reserve-capacity recruitment plus sparse routing and consolidation improve continual-learning plasticity/forgetting trade-offs compared with dense, fixed-sparse, dynamic-sparse, and replay/regularization baselines?

Multimodal claims are tested only after the structural single-modality core is validated.


## 20. M1 execution refinement

The M1 implementation distinguishes **recruited structural capacity** from **runtime executed capacity**.

Let:

- R_t be all non-dormant nodes available to routing;
- X_t be the nodes selected for the current batch.

Then:

X_t subseteq R_t

and M1 keeps |X_t| <= k even as |R_t| grows.

This allows the hypothesis that knowledge capacity can increase through specialization while per-batch expert execution remains bounded.

Dormant nodes are not scored by the normal runtime router. They enter the routing pool only through a slower structural recruitment event.

### Sparse topology representation

The logical supergraph must not imply a materialized dense adjacency matrix.

Implementation uses a fixed number q_max of outgoing edge slots per node. Edge endpoints and enabled state are structural data, while edge transforms are shared/lightweight.

Therefore topology memory targets:

O(N q_max)

rather than O(N^2).

M1 uses batch-level top-k routing and bounded synchronous message passing. Per-example expert dispatch and hierarchical routing are deferred until the core behavior is validated.
