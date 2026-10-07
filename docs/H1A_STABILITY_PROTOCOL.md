# EXP-H1A-STABILITY-001 — Numerical Stability Calibration

**Status:** preregistered / not run  
**Date:** 2026-10-07  
**Purpose:** choose a numerically stable shared learning rate after EXP-H1A-001 was invalidated by NaN/Inf divergence.

## 1. This is not an H1 hypothesis test

This experiment must not be used to claim that reserve recruitment helps or hurts continual learning.

Scientific accuracy, forgetting, backward transfer, routing quality and late-task performance are ignored for selection.

The only question is:

> Which candidate shared learning rate, if any, completes the full H1a training protocol across independent calibration seeds and all five model variants without any detected NaN or Inf?

## 2. Frozen data and model protocol

Keep unchanged from EXP-H1A-001:

- exact frozen CIFAR-100 ResNet-18 feature artifact;
- feature SHA-256:
  `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47`;
- exact class order;
- 10 experiences x 10 classes;
- seen-class masking;
- N=16, initial recruited=4, top-k=4;
- d_model=128, d_hidden=256;
- no replay;
- no consolidation;
- no edges/message passing;
- 20 epochs per task;
- batch size 128;
- eval batch size 256;
- SGD momentum 0.9;
- weight decay 5e-4;
- all five H1a variants.

Only the learning rate and calibration seeds differ.

## 3. Calibration seeds

Use exactly:

```text
101, 102, 103
```

These are disjoint from the invalid primary seeds 0..9.

## 4. Candidate learning rates

Evaluate in this exact descending order:

```text
0.03
0.01
0.003
```

Each candidate is tested on:

```text
3 seeds x 5 variants = 15 runs
```

## 5. Numerical fail-fast contract

Training/evaluation aborts immediately on the first detected non-finite value at any of these boundaries:

- model logits / auxiliary loss / routed weights or context;
- scalar training loss;
- gradients after backward;
- gradients after structural protection;
- parameters after optimizer.step;
- evaluation logits;
- evaluation loss.

Failure artifacts record:

- phase;
- tensor category;
- offending parameter when applicable;
- stage/task;
- epoch;
- batch index;
- global step;
- count of non-finite values;
- largest finite magnitude still observable.

## 6. Selection rule

For each learning rate, all 15 seed x variant runs must complete with no detected NaN or Inf.

Evaluate learning rates in descending order and select the **highest** candidate satisfying:

```text
stable_runs = 15
failed_runs = 0
```

Stop after the first passing candidate.

Accuracy and all scientific metrics are ignored for this decision.

If no candidate passes, this experiment selects **no learning rate**. Do not add gradient clipping inside this experiment. Open a new stability experiment with its own ID and preregistration.

## 7. Command

```bash
uv run --locked python experiments/scripts/m1_3_h1a_stability.py \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt \
  --output runs/m1_3_h1a_stability \
  --device cuda \
  2>&1 | tee runs/m1_3_h1a_stability/console.log
```

## 8. Follow-up

If one learning rate passes, use that value to preregister a new scientific experiment, **EXP-H1A-002**.

EXP-H1A-002 should use fresh primary model seeds rather than reclassifying EXP-H1A-001, whose scientific conclusion remains NOT_EVALUATED.

Do not modify EXP-H1A-001 artifacts or reinterpret its aggregate statistics after this calibration.
