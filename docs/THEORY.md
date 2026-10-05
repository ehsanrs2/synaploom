# SynapLoom Theory Notes

**Status:** formalization baseline, not a theorem claim  
**Date:** 2026-10-05

This document separates what is already mathematically defined from what is only a future theoretical question.

## 1. Mathematical object

Let the total microcircuit capacity be a fixed set:

V = {1, ..., N}

Each node has a structural state:

s_i(t) in {DORMANT, PLASTIC, CONSOLIDATED}

Define the recruited set:

R_t = {i : s_i(t) != DORMANT}

and the runtime execution set for a batch:

X_t subseteq R_t

with:

|X_t| <= k

This distinction is fundamental:

- recruited capacity may grow over time;
- executed capacity per batch may remain fixed.

Therefore learning can increase representational specialization without forcing per-example compute to grow at the same rate.

## 2. Topology

Conceptually, SynapLoom has an admissible directed supergraph:

G* = (V, E*)

The implementation must not materialize a dense |V| x |V| edge mask.

Instead, each node has at most q_max outgoing edge slots. A slot stores:

- enabled flag
- destination node id
- relation type
- scalar gate / lightweight edge state

Thus the materialized topology state is bounded by:

O(N q_max)

even when the logical admissibility relation is much larger.

The runtime edge set is:

E_t^exec = {(i,j) : i,j in X_t and an enabled structural edge i -> j exists}

so:

|E_t^exec| <= k q_max

## 3. Parameters and mixed optimization

Let theta denote ordinary differentiable parameters and z denote discrete structural variables.

A generic objective is:

min_{theta, z}  sum_t L_t(theta_t, z_t)
                 + lambda_c C(z_t)
                 + lambda_delta D(z_t, z_{t-1})
                 + lambda_s S(theta_t, z_t)

subject to capacity constraints.

This is a mixed discrete-continuous, non-convex online optimization problem.

SynapLoom does not attempt exact global optimization over structures.

## 4. Multi-timescale view

The design intentionally separates timescales:

routing/activation << gradient updates << structural updates << consolidation decisions

This suggests a hybrid dynamical-system interpretation:

state_t = (hidden state, parameters, structural state, utilization statistics, controller state)

with continuous/differentiable updates interleaved with discrete structural transitions.

A future simplified model may be analyzed using two-timescale stochastic approximation, but no convergence claim is made for the full deep system.

## 5. Online-learning view

The environment may be non-stationary:

P_t(X,Y) != P_{t+1}(X,Y)

Therefore static empirical-risk minimization is not a complete theoretical description.

Possible future quantities include:

- static regret for fixed environments
- dynamic regret under changing comparators
- path variation of the environment
- learning speed on late tasks
- structural adaptation cost

For the full non-convex discrete system these quantities will initially be empirical, not theorem-backed.

## 6. Structural-search complexity

Exact search for an optimal k-node subset among N nodes is combinatorial:

C(N,k)

and unrestricted edge-set search is also combinatorial.

Therefore SynapLoom explicitly rejects global topology search.

The controller uses local/bounded structural actions:

- recruit a bounded number of dormant nodes
- add a bounded number of edges
- prune a bounded number of low-utility edges
- consolidate selected recruited nodes

Per-update structural work should remain polynomial and preferably near-linear in recruited capacity.

## 7. Target forward complexity

Define:

- B: batch size
- N: total node capacity
- A = |R_t|: recruited nodes
- k = |X_t|: executed nodes per batch
- d: representation width
- h: hidden width inside a microcircuit
- q: executed average out-degree, q <= q_max
- K: number of synchronous message-passing rounds

### Router

The runtime router scores recruited nodes only, not dormant nodes.

With a d-dimensional query/key score:

T_router = O(B A d)

plus top-k selection, approximately O(B A log k) or implementation-dependent equivalent.

Dormant nodes are considered only by the slower structural controller.

### Microcircuits

If each executed microcircuit is an MLP d -> h -> d:

T_micro = O(B k d h)

up to constant factors.

### Message passing

With bounded edge degree:

|E_exec| <= k q

For scalar-gated/shared lightweight messages:

T_message = O(K B k q d)

If a shared dense relation projection d x d is used, it should be applied once per source/relation where possible, yielding roughly:

O(K B k d^2 + K B k q d)

rather than one d x d multiplication per edge.

### Total target

Ignoring the modality encoder/head:

T_step ~= O(B A d + B k d h + K B k q d)

for the scalar/shared-lightweight edge variant.

If k, q, K remain bounded while A grows, expert execution remains bounded but router cost grows with A.

This makes routing scalability a future concern.

## 8. Structural-update complexity

Let S be the number of optimization steps between structural updates.

M1 avoids expensive dormant-node search: dormant units are initially exchangeable, so a deterministic free node can be recruited and initialized from the triggering context.

Recruitment selection can therefore be O(1) to choose a slot plus O(d) initialization.

Selecting r neighbors among A recruited nodes using similarity can be:

O(A d)

for a recruitment event.

Amortized over S steps:

O(A d / S)

which should be small relative to normal training for sufficiently large S.

Future candidate-search methods may use bounded candidate sets of size c:

O(A c)

rather than all-pairs O(A^2).

## 9. Memory complexity

Microcircuit parameter memory exists for dormant capacity:

O(N d h)

Therefore SynapLoom v0.1 is primarily a **conditional-compute architecture**, not automatically a parameter-memory sparse architecture.

Topology state with bounded edge slots is:

O(N q_max)

Router keys/state are approximately:

O(N d)

Worst-case optimizer memory can eventually approach the total parameter memory after all nodes have been used, although optimizer state may be lazily created in practical implementations.

We must separately report:

- total parameters
- recruited parameters
- executed parameters per batch
- actual FLOPs
- wall-clock time
- peak VRAM

## 10. Capacity-growth questions

Important empirical/theoretical functions include:

A(T) = recruited nodes after T tasks/episodes

E(T) = active structural edges after T tasks/episodes

We do not yet know whether useful growth is:

- linear
- sublinear
- saturating
- unstable/oscillatory

A particularly interesting result would be sublinear capacity growth with retained performance, but this is a hypothesis, not an expectation to bake into the implementation.

## 11. Graph-theoretic observables

Potential structural metrics:

- recruited-node count
- edge count and degree distribution
- component count
- path length
- clustering
- modularity
- task/concept path overlap
- edge turnover
- node reuse frequency

Spectral metrics may be considered later, especially if message passing becomes deep enough for oversmoothing/oversquashing to matter.

## 12. Information-theoretic direction

A useful future formulation is structural utility:

structural_gain = task/prediction improvement - lambda * structural_cost

An edge/node should ideally be recruited only when its expected information/utility gain exceeds its complexity cost.

This can connect SynapLoom to minimum-description-length or information-bottleneck ideas.

For multimodal association, future criteria may approximate conditional mutual information or predictive gain rather than treating temporal co-occurrence as semantic evidence.

No mutual-information estimator is selected for M1.

## 13. Open theoretical questions

These are intentionally unanswered:

1. Does the structural process converge?
2. Is convergence desirable in a non-stationary environment?
3. What is the optimal recruited-to-executed capacity ratio?
4. What growth law A(T) emerges?
5. Can a simplified convex/linear SynapLoom admit regret bounds?
6. Can a stability/plasticity bound be derived?
7. When does topology reuse imply positive transfer?
8. Can structural utility be submodular or approximately submodular under useful assumptions?
9. How should routing complexity scale when A becomes very large?
10. Is hierarchical routing necessary for asymptotic scalability?

## 14. Theory-development strategy

Do not try to prove theorems directly for the full deep system first.

Preferred sequence:

1. implement and measure the full empirical system;
2. identify stable mechanisms and quantities;
3. define a simplified linear/convex or finite-expert model;
4. analyze one mechanism at a time;
5. only then attempt regret, stability, or capacity bounds.

The theory should explain observed behavior rather than retroactively decorate an arbitrary architecture.
