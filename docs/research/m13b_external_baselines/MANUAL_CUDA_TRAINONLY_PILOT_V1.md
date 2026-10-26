# M1.3b — bounded MANUAL CUDA train-only feasibility pilot v1

**Draft; no scientific run authorization.** Follows merged [PR #33](https://github.com/ehsanrs2/synaploom/pull/33). This is not `EXP-M13B-EXTERNAL-001`, not a new benchmark outcome, not a 30/90-trajectory runner and not a RigL/NISPA implementation. It is an optional, **explicitly user-invoked** RTX/CUDA systems integration pilot.

## What it does (strict limits)

Uses the HISTORICALLY FROZEN feature bundle `data/features/cifar100_resnet18_imagenet1k_v1.pt` **only after** trusted origin acknowledgement and verification that its bytes match SHA-256 `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47`. It takes a bounded snapshot (512 MiB max) and unpickles **the same pinned bytes** with `torch.load(weights_only=False)`; Python pickle is unsafe for unknown origins, and SHA identity by itself does not establish trust.

It returns **only** `train_features` and `train_targets`. The original CIFAR evaluation tensors are present in the file, but are **never used to train, tune or compute the pilot's diagnostic outcomes**. From the first 20 frozen O0 classes, it deterministically samples from the original train partition 16 examples per class for tiny training and four *different train examples* for held-out engineering diagnostics. Train/holdout indices are disjoint, reproducible and fingerprinted. This is **not** the benchmark evaluation set or a method-fidelity validation partition for NISPA.

- Exactly two tasks × ten classes, feature dimension 512 and 100-class classifier.
- Exactly one epoch/task, H1a SGD LR .01, momentum .9, decay 5e-4, training batch 128, evaluation batch 256.
- Only three arms: `progressive_reserve`, `all_recruited`, `fixed_sparse_mlp`, seeded by **engineering seed 13** and fixed-mask seed 17; no confirmatory seeds 501–510.
- **Four optimizer steps per arm**, 12 steps in total (160 train examples/task with B128 → 2 minibatches per task), not 10 tasks × 20 epochs.
- The fixed sparse model must pass its 2% total physical parameter and 10% ideal linear-MAC candidate matching gates. Its actual implementation still executes dense masked GEMMs.
- Per-arm CUDA-synchronized wall time, GPU peak allocation/reservation, physical model parameter count, fixed-mask fingerprint, finite train-partition-only diagnostic holdout metrics and model-state SHA are recorded. These diagnostics must NEVER be pooled with historical H1a or final M1.3b scientific metrics.

CUDA is **not** executed in hosted GitHub Actions; the new offline CI tests check only class split determinism, disjointness, hash gates and refusal of unauthorized flags.

## Read-only description (safe anywhere)

```bash
uv run --locked python experiments/scripts/m13b_trainonly_cuda_pilot.py --describe-only
```

This command doesn't read any feature bytes, touch CUDA or train.

## Manual RTX 4090 command — **only if you authorize/run it locally**

```bash
uv run --locked python experiments/scripts/m13b_trainonly_cuda_pilot.py \
  --manual-gpu-trainonly-pilot \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt \
  --trusted-feature-source \
  --acknowledge-no-scientific-claim \
  --output-dir /tmp/m13b-cuda-trainonly-v1-once
```

This exact command is provided for **future manual owner execution after PR review**, not executed by ChatGPT and not automatically triggered by PR merge. It requires CUDA, a real trusted local file and a NEW output directory. It writes `intent.json` first, per-arm progress JSON when each arm finishes, `pilot.json` and a canonical result SHA on success, or `failure.json` on an ordinary error; previous directories are never overwritten. No result is written into existing H1a experiment directories. Hardware faults, process termination and malicious pickle can defeat these safeguards; they are not a security boundary.

The CLI deliberately has no `--execute`, `--seed`, `--class-order`, `--device`, `--epochs`, `--arm`, `--features-from-web` or real scientific training option. It does not alter `m13b_external_preflight.py` and does not integrate Sample-level Routing into TrainingEngine.

## Decisions after pilot, before scientific studies

1. Record actual pinned feature presence, driver/CUDA/PyTorch version, memory and timing only from **real** host evidence.
2. Review the fixed sparse total-capacity versus executed-work and dense masking overhead on actual GPU, acknowledging that equal estimated sparse MACs are not equal hardware work.
3. Independently choose and freeze O0-only versus the predeclared O0/O1/O2 robustness family, seeds, comparisons/multiplicity, method fidelity and desired scientific effect size **before** any new confirmatory outcomes are observed.
4. Separately approve and implement a full scientific runner with frozen experiment ID, output provenance, failure handling and all required ordered runs. No negative result may be hidden or overwritten.

**GO: review of bounded host feasibility code. NO-GO: automatic CUDA training, formal M1.3b science, claims of improved forgetting/plasticity or reinterpretation of older H1a outcomes.**
