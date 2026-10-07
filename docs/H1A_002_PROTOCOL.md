# EXP-H1A-002 — Controlled Reserve Recruitment Rerun

**Status:** preregistered / not run  
**Date:** 2026-10-07  
**Hypothesis:** H1a — controlled progressive reserve recruitment

## 1. Why a new experiment ID

EXP-H1A-001 is permanently recorded as `INVALID_NUMERICAL_INSTABILITY`.

EXP-H1A-STABILITY-001 then calibrated only numerical stability and selected a shared learning rate of **0.01**. Accuracy, forgetting, transfer and routing metrics from the calibration were not used for selection.

EXP-H1A-002 is therefore a fresh scientific test, not a relabeling of EXP-H1A-001.

## 2. Frozen scientific question

Under equal total model capacity and equal top-k expert execution, does progressively making reserve microcircuits available improve continual learning relative to making the entire expert pool routable from the beginning?

Primary causal comparison:

```text
progressive_reserve - all_recruited
```

## 3. Data and representation

Unchanged from EXP-H1A-001:

- CIFAR-100;
- frozen torchvision ResNet-18 `IMAGENET1K_V1` features;
- exact feature artifact SHA-256:
  `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47`;
- 512-dimensional feature vector;
- exact frozen 10 x 10 class order from `H1A_CLASS_ORDER`;
- seen-class logit masking;
- no task id at inference.

## 4. Model/mechanism configuration

Unchanged:

- feature input = 512;
- d_model = 128;
- d_hidden = 256;
- total microcircuits N = 16;
- initially recruited = 4;
- top-k = 4;
- q_max = 0;
- message rounds = 0;
- no replay;
- no consolidation;
- no structural edges;
- no pruning/recycling.

Variants remain exactly:

1. `progressive_reserve`
2. `all_recruited`
3. `fixed_small`
4. `dense_active_proxy`
5. `dense_total`

## 5. Optimizer

The only scientific-protocol change relative to EXP-H1A-001 is the shared learning rate selected by the preregistered numerical calibration:

- optimizer = SGD;
- **learning rate = 0.01**;
- momentum = 0.9;
- weight decay = 5e-4;
- scheduler = none;
- epochs/task = 20;
- batch size = 128;
- eval batch size = 256.

Reason for 0.01:

`EXP-H1A-STABILITY-001` evaluated LR candidates in descending order using numerical validity only:

- 0.03: 1/15 stable, 14/15 numerical failures;
- **0.01: 15/15 stable, 0/15 failures**;
- 0.003: not executed because the preregistered stop rule selected the first fully stable candidate.

No accuracy-based optimizer choice was performed.

## 6. Fresh primary seeds

Use exactly these 10 paired seeds:

```text
201, 202, 203, 204, 205, 206, 207, 208, 209, 210
```

They are disjoint from:

- invalid EXP-H1A-001 primary seeds 0..9;
- calibration seeds 101,102,103.

The class order remains fixed so the treatment/control comparison stays paired.

## 7. Numerical validity gate

The fail-fast checks introduced after EXP-H1A-001 remain active.

If any primary run produces a detected NaN/Inf:

1. abort EXP-H1A-002;
2. write the failure artifact;
3. mark EXP-H1A-002 invalid;
4. do not interpret partial aggregate results.

## 8. Co-primary outcomes

Unchanged from the original preregistration:

1. final average accuracy;
2. late-task within-task mean accuracy.

Secondary outcomes:

- average forgetting;
- backward transfer;
- diagonal task accuracy;
- full stage-by-task matrix;
- within-task learning curves;
- routing/structural histories;
- parameter budget;
- linear-MAC estimates;
- wall time;
- peak CUDA memory.

## 9. Statistical reporting

Use 10 paired seeds.

For each scalar comparison report:

- per-seed treatment/control values;
- per-seed delta;
- mean delta;
- paired bootstrap 95% CI of the mean delta;
- per-variant mean/std.

For higher-is-better primary metrics:

- CI strictly above zero → favorable;
- CI strictly below zero → unfavorable;
- CI containing/touching zero → inconclusive.

### Overall preregistered H1a classification

- **FAVORABLE** only if both co-primary metrics are favorable;
- **UNFAVORABLE** only if both co-primary metrics are unfavorable;
- **INCONCLUSIVE** otherwise;
- **INVALID** if numerical/artifact validity fails.

This combined rule is fixed before EXP-H1A-002 results are inspected.

## 10. Batch-routing caveat

Batch-level routing remains unchanged.

Evaluation order and `eval_batch_size=256` are frozen.

If the result is favorable but small, or otherwise sensitive-looking, evaluation-batch-size/class-order sensitivity remains a follow-up and cannot alter the primary result.

## 11. Command

After merging the preregistration PR:

```bash
git checkout main
git pull origin main
uv sync --locked --extra vision

mkdir -p runs/m1_3_h1a_002

uv run --locked python experiments/scripts/m1_3_h1a_002.py \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt \
  --output runs/m1_3_h1a_002 \
  --device cuda \
  2>&1 | tee runs/m1_3_h1a_002/console.log
```

Expected scientific runs:

```text
10 seeds x 5 variants = 50 runs
```

Do not add CLI seed/variant overrides to the primary command.
