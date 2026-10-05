# M1.3 H1 Benchmark Protocol

**Status:** frozen protocol scaffold; scientific runs not yet executed  
**Date:** 2026-10-05

## 1. Question

M1.3 tests the narrow hypothesis:

> Under the same total model capacity, does gradual recruitment of dormant microcircuits improve continual class-incremental learning relative to no-growth routing controls?

This is **H1a: recruitment-only**. It does not test graph message passing, consolidation, replay, automatic novelty, pruning, or multimodal learning.

## 2. Dataset

Primary benchmark:

- CIFAR-100
- 10 sequential tasks
- 10 disjoint classes per task
- global 100-class output space
- task id is not provided to the model at inference
- one deterministic shuffled class order shared by all model conditions and model seeds

The 10-step / 10-classes-per-step Split CIFAR-100 setting is common in continual-learning literature. We do not compare raw accuracy numbers to papers unless preprocessing, backbone, replay budget, task order, and evaluation protocol are actually matched.

## 3. Primary protocol v1

- 5 model seeds
- 10 epochs per task
- AdamW
- fixed learning rate and weight decay across conditions
- fixed CIFAR-100 normalization
- no replay
- no stochastic data augmentation
- message_rounds = 0
- no edge creation
- no consolidation
- no final-task recruitment
- deterministic batch ordering per seed
- fixed eval batch size

No augmentation is deliberate: M1.3 v1 is a controlled mechanism test, not a state-of-the-art CIFAR recipe.

## 4. SynapLoom shape

Default:

- total reserve nodes N = 12
- initially recruited A0 = 2
- executed top-k = 2
- one node recruited at each task boundary
- d_model = 128
- microcircuit hidden width = 256
- router_dim = 128
- q_max = 0
- message passing disabled

After 9 task transitions, the growth condition has recruited 11 of 12 nodes. One reserve node remains unused.

## 5. Required conditions

### A. synaploom_growth

- starts with 2 recruited nodes
- recruits one dormant node between tasks
- no edges
- no consolidation

### B. routed_no_growth

- identical architecture and total reserve parameter capacity
- starts with 2 recruited nodes
- never recruits more nodes

This isolates the value of access to additional reserve capacity.

### C. routed_all_recruited

- same total SynapLoom parameter capacity
- all reserve nodes are recruitable/routable from the beginning
- still executes only top-k

This asks whether staged recruitment itself matters versus exposing all experts immediately.

### D. dense_total_match

Dense residual control whose total parameter count is numerically matched as closely as possible to the full SynapLoom parameter count.

### E. dense_executed_match

Dense residual control whose parameter count is numerically matched as closely as possible to SynapLoom's estimated per-batch executed parameter footprint.

These two dense views prevent a single ambiguous notion of "same size".

## 6. Shared image encoder

All primary conditions use the same encoder family:

3x32x32
-> Conv(32)
-> pool
-> Conv(64)
-> pool
-> Conv(96)
-> global average pool
-> projection to d_model

Weights are independently initialized per model condition using the same seed.

The encoder remains trainable. Therefore encoder forgetting is a shared source of interference across conditions.

## 7. Metrics

Primary:

- final average accuracy

Required secondary:

- average forgetting
- backward transfer
- accuracy immediately after each task
- pre-task accuracy
- per-task learning gain:
  G_t = post_train_accuracy_t - pre_train_accuracy_t
- late-task learning gain
- recruited capacity trajectory
- routing utilization/selection distribution
- wall-clock time
- peak CUDA memory when applicable
- total parameter count

The M1.3 learning-gain metric is stronger than the M1.2 diagonal-accuracy proxy, but it is still not a complete theorem or definitive measurement of biological/plasticity loss.

## 8. Batch-level router caveat

The current router uses one query per batch. Therefore a sample's prediction can depend on the other samples in the batch.

For M1.3:

- evaluation order is deterministic;
- eval_batch_size is frozen;
- all routed conditions use the same batching;
- small differences must not be over-interpreted without a batch-composition sensitivity check.

A later sample-level routing ablation is explicitly separate.

## 9. Replay track

Replay is excluded from the primary H1a comparison.

A later reference track may add ER/DER++-style replay, but replay memory must be reported separately from model parameters. Otherwise parameter capacity and stored-example memory are incomparable resources.

## 10. Sparse-training reference

RigL-style dynamic sparse training is relevant as a structural-compute reference, but its sparse FLOP accounting and topology updates are not identical to SynapLoom reserve recruitment.

If added, report:

- nominal sparsity
- active parameter count
- training/inference FLOPs
- actual wall time

Do not infer real speedup from nominal sparsity alone.

## 11. Decision rule

Do not accept H1 from one metric or one seed.

Evidence supporting recruitment requires a consistent multi-seed advantage over both:

- routed_no_growth, and
- routed_all_recruited

under transparent capacity reporting.

If the advantage disappears against all-recruited routing, the useful mechanism may be capacity availability rather than developmental recruitment.

If growth loses, preserve the negative result and inspect routing collapse, encoder interference, and task-wise learning gain before adding more mechanisms.

## 12. External anchors

- RigL: Evci et al., "Rigging the Lottery: Making All Tickets Winners", ICML 2020.
- NISPA: Gurbuz & Dovrolis, "NISPA: Neuro-Inspired Stability-Plasticity Adaptation for Continual Learning in Sparse Networks", ICML 2022.
- Split CIFAR-100 is commonly evaluated as 10 tasks x 10 classes and sometimes 20 tasks x 5 classes in continual-learning studies.

These anchors motivate controls; they do not establish SynapLoom novelty or expected superiority.
