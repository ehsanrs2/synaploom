# Research Hypotheses

**Status:** working hypotheses; none are established results.

SynapLoom should be developed by falsifying hypotheses, not by adding biologically inspired mechanisms until the model becomes impossible to interpret.

## H1 — Reserve structural recruitment

**Hypothesis:** Under a fixed total parameter budget, keeping some capacity dormant and recruiting it only when persistent need is detected improves the stability/plasticity trade-off over always-active dense or fixed-sparse networks.

### H1a — controlled recruitment

M1.3 first isolates the reserve-availability mechanism: progressive recruitment follows a preregistered task-start schedule and is compared primarily with an all-recruited routed model having the same total capacity and top-k execution.

H1a does **not** test whether the network knows when it should grow.

### H1b — adaptive recruitment trigger

M2 tests whether persistent need/novelty can trigger recruitment automatically without task-start scheduling.

Full H1 requires evidence from both the mechanism and the trigger.

**Compare against:**
- dense network with equal total parameter count
- dense network with equal active parameter count
- fixed sparse network
- RigL-style dynamic sparse training
- NISPA-style sparse continual learning where feasible

**Falsified if:** gains disappear after equalizing compute/active parameters, or recruitment merely behaves like hidden parameter over-provisioning.

## H2 — Structural state helps continual learning

**Hypothesis:** PLASTIC -> CONSOLIDATED state transitions reduce forgetting without preventing learning of new tasks.

**Compare against:**
- no consolidation
- EWC/SI-style regularization
- replay
- freeze-after-task variants

**Falsified if:** consolidation reduces new-task learning more than it protects old knowledge, or standard regularization dominates it consistently.

## H3 — Sparse routing creates reusable specialization

**Hypothesis:** Conditional top-k routing learns partially overlapping subnetworks such that similar inputs reuse more capacity than unrelated inputs.

**Measure:**
- active-set overlap vs representation/input similarity
- expert/node utilization distribution
- routing entropy
- collapse/starvation rate
- transfer speed to related vs unrelated tasks

**Falsified if:** routing is effectively random, collapses to a few nodes, or specialization does not improve transfer/generalization.

## H4 — Modality-private pathways are useful

**Hypothesis:** Maintaining modality-private capacity before learned association produces better robustness and continual adaptation than unconditional early fusion under the same budget.

**Compare against:**
- early fusion
- late fusion
- fixed middle fusion
- modality-private + fixed association
- modality-private + adaptive association

**Measure:**
- normal accuracy
- missing-modality robustness
- noisy-modality robustness
- modality-specific forgetting
- cross-modal transfer

**Falsified if:** fixed fusion is consistently better after fair tuning and budget matching.

## H5 — Cross-modal topology can carry useful association

**Hypothesis:** Allowing selected cross-modal edges to activate based on persistent evidence yields reusable associations that improve multimodal tasks and transfer.

**Important:** co-occurrence alone is not treated as semantic evidence.

**Falsified if:** learned topology mostly tracks spurious co-occurrence or offers no benefit over a fixed association graph.

## H6 — Structural adaptation helps preserve plasticity

**Hypothesis:** recycling/recruiting low-utility capacity can mitigate the loss of plasticity observed in long continual streams.

This is distinct from catastrophic forgetting.

**Measure:**
- learning speed on late tasks
- performance relative to a fresh network on the same late task
- dormant-unit fraction
- representation rank/diversity
- forgetting separately

**Falsified if:** late-task learnability still degrades similarly to standard training.

## H7 — Knowledge-as-subgraph is a testable representation hypothesis

"Knowledge is a subgraph" is not assumed as fact.

**Operational test:** after training, identify stable active subgraphs for concepts/tasks and test whether:
- ablation selectively harms the associated concept/task,
- reuse predicts positive transfer,
- topology is reproducible across seeds beyond chance,
- subgraphs remain meaningful under controlled perturbations.

If these tests fail, the language should remain "conditional computational pathways" rather than "knowledge subgraphs."

## Order of testing

1. H1
2. H2/H3
3. H6
4. H4
5. H5
6. H7

Dendritic computation, predictive coding, and biologically local learning are deliberately outside the initial hypothesis stack.
