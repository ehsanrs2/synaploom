# M1.3 H1a Protocol — Controlled Reserve Recruitment

**Status:** frozen protocol before scientific execution  
**Date:** 2026-10-05  
**Hypothesis slice:** H1a — controlled progressive reserve recruitment

## 1. Why H1a instead of full H1

The original H1 requires recruitment when persistent need is detected.

M1 does not yet contain an adaptive need detector. Therefore M1.3 must not claim to test full H1.

M1.3 tests the narrower causal question:

> Under equal total model capacity and equal top-k expert execution, does progressively making reserve microcircuits available improve continual learning relative to making the entire expert pool routable from the beginning?

Adaptive trigger quality is H1b and belongs to M2.

## 2. Mechanism-isolation benchmark

Primary benchmark:

- dataset: CIFAR-100
- scenario: class-incremental
- 10 experiences
- 10 new classes per experience
- no task id at inference
- no replay
- no consolidation
- no structural edges
- no message passing
- no pruning/recycling
- frozen image representation

The 10 x 10 split matches the common SplitCIFAR100 benchmark convention used by Avalanche.

The longer 20 x 5-class protocol used in plasticity-focused work is reserved for H6, where the question is explicitly long-horizon loss of plasticity.

## 3. Frozen representation

Images are transformed by torchvision's:

ResNet18_Weights.IMAGENET1K_V1.transforms()

and passed through ImageNet-pretrained ResNet-18 with the final classifier replaced by Identity.

The resulting 512-dimensional feature vector is stored once and reused by every seed and model variant.

Reason:

- all variants receive bit-identical representations;
- image-encoder forgetting cannot confound the first structural test;
- the benchmark is cheap enough for paired multi-seed runs;
- the test isolates reserve recruitment/routing.

This is not an end-to-end SynapLoom claim.

A positive H1a result must later survive an end-to-end representation-learning setting.

## 4. Frozen class order

The primary run uses this exact order, generated once with torch.randperm(100) and seed 20261005:

1. [75, 59, 1, 39, 93, 87, 37, 51, 7, 34]
2. [21, 41, 19, 86, 27, 89, 44, 0, 20, 14]
3. [49, 91, 30, 69, 23, 70, 62, 9, 29, 76]
4. [6, 15, 71, 32, 96, 95, 24, 99, 50, 66]
5. [80, 88, 53, 98, 68, 10, 97, 18, 73, 61]
6. [4, 45, 40, 72, 74, 63, 31, 28, 5, 92]
7. [90, 25, 3, 38, 48, 79, 94, 35, 65, 58]
8. [13, 60, 47, 54, 57, 85, 82, 42, 83, 43]
9. [56, 8, 17, 2, 84, 78, 16, 36, 67, 64]
10. [46, 81, 52, 55, 77, 12, 22, 33, 26, 11]

The class order is held fixed across model seeds to make the primary comparison paired.

Class-order sensitivity is a follow-up, not a hidden tuning dimension.

## 5. Seen-class evaluation

The classifier has 100 output units from the start.

During task t, logits for classes not yet introduced are masked before cross-entropy and prediction.

Thus evaluation is among all classes seen so far, not among only the current task classes.

No task label is provided at inference.

## 6. SynapLoom core configuration

- feature input: 512
- d_model: 128
- microcircuit hidden width: 256
- total microcircuits N: 16
- initially recruited: 4
- top-k executed: 4
- router dimension: 128
- q_max: 0
- message rounds: 0

The q_max/message choices deliberately remove graph communication from H1a.

H1a asks about reserve availability and sparse routing only.

## 7. Variants

### A. progressive_reserve — primary treatment

- N = 16 total microcircuits
- 4 recruited initially
- top-k = 4
- one dormant microcircuit recruited at the first batch of each new experience after experience 0
- newly recruited routing key is context-imprinted using that new task's first batch
- the structural mutation occurs after that first optimization step, preserving the M1 mutation boundary

Final recruited count after 10 tasks: 13.

### B. all_recruited — primary causal control

- same N = 16
- all 16 routable from the beginning
- same top-k = 4
- same parameter tensors and initialization seed
- no structural growth

This is the most important control because total capacity and expert execution count are equal to progressive_reserve.

### C. fixed_small

- N = 16 physically exists
- only the initial 4 are ever routable
- top-k = 4
- no recruitment

This tests whether any benefit merely comes from having more usable capacity later.

### D. dense_active_proxy

A dense residual MLP whose hidden width is selected to make its total trainable parameter count as close as possible to SynapLoom's initial active-parameter proxy.

This is an approximate budget view, not the primary causal control.

### E. dense_total

A dense residual MLP whose total trainable parameter count is matched as closely as possible to total SynapLoom trainable parameters, including dormant reserve.

## 8. Training protocol

- optimizer: SGD
- learning rate: 0.03
- momentum: 0.9
- weight decay: 5e-4
- epochs per experience: 20
- batch size: 128
- evaluation batch size: 256
- replay: none
- LR scheduler: none
- seeds: 0,1,2,3,4,5,6,7,8,9

No model-specific hyperparameter tuning is permitted in the primary run.

If the shared optimization setting is clearly broken in a preflight, it must be changed before any primary result is inspected, and the change must be recorded here.

## 9. Evaluation batching caveat

M1 uses batch-level routing.

Therefore routed predictions can depend on the composition of the evaluation batch.

For the primary run:

- evaluation order is deterministic;
- eval_batch_size is fixed at 256;
- the same ordering/batching is used for all routed variants.

Before interpreting a small H1a effect, run a sensitivity check with alternative deterministic evaluation batch sizes.

## 10. Primary outcomes

The primary paired comparison is:

progressive_reserve - all_recruited

using identical seed and class order.

Co-primary outcomes:

1. final average accuracy;
2. late-task within-task mean accuracy.

Secondary outcomes:

- average forgetting;
- backward transfer;
- diagonal post-task accuracy;
- full stage-by-task accuracy matrix;
- within-task learning curves;
- routing entropy / max selection share;
- recruited capacity trajectory;
- train/eval wall time;
- peak CUDA memory;
- transparent linear-MAC estimate.

## 11. Statistical reporting

The primary run uses 10 paired seeds.

For each scalar metric report:

- every seed value;
- paired seed delta;
- mean delta;
- bootstrap 95% CI of the paired mean delta;
- per-variant mean and standard deviation.

Interpretation:

- CI fully above zero on a higher-is-better primary metric: evidence favorable to H1a;
- CI fully below zero: evidence against H1a in this setting;
- CI crossing zero: inconclusive.

For forgetting, lower is better and the sign must be interpreted accordingly.

No single seed can promote a claim into PROJECT_STATE.

## 12. Budget reporting

Report separately:

- total trainable parameters;
- total microcircuit-pool parameters;
- per-microcircuit parameters;
- executed top-k microcircuit parameters;
- recruited routing-key parameters;
- router candidate count;
- initial and final estimated linear MACs per batch;
- measured wall time;
- peak CUDA memory.

Sparse execution is not assumed to imply wall-clock speedup.

## 13. External baselines and M1.3b

H1a's internal causal comparison comes first.

After it is mechanically validated, M1.3b should add stronger external sparse/continual controls where implementation fidelity can be defended, especially:

- a fixed sparse baseline;
- RigL-style dynamic sparse training;
- NISPA-style stability/plasticity rewiring;
- a standard replay baseline as a separate replay-enabled condition.

NISPA is particularly relevant because it preserves stable sparse paths and rewires plastic capacity under fixed density.

Replay is not mixed into the primary no-replay H1a comparison because that would test a different resource assumption.

## 14. What this experiment can and cannot say

A positive result supports:

- controlled reserve availability can be useful under the frozen-feature protocol.

It does not establish:

- that the automatic growth trigger works;
- that graph edges are useful;
- that consolidation is useful;
- that end-to-end image learning improves;
- that SynapLoom is more biologically realistic;
- that dormant reserve is computationally free.

A negative result means the current controlled-recruitment mechanism did not help under this isolation protocol. It should trigger diagnosis before adding biological complexity.

## 15. Commands

Feature extraction:

python experiments/scripts/extract_cifar100_resnet18_features.py --download

Primary benchmark:

python experiments/scripts/m1_3_h1a_benchmark.py \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt \
  --device cuda

Run artifacts are written under runs/m1_3_h1a by default.

## 16. Literature anchors

- SplitCIFAR100 benchmark convention: https://avalanche.continualai.org/
- Loss of plasticity in deep continual learning (Nature, 2024): https://doi.org/10.1038/s41586-024-07711-7
- NISPA (ICML 2022): https://proceedings.mlr.press/v162/gurbuz22a.html
- RigL: https://arxiv.org/abs/1911.11134
