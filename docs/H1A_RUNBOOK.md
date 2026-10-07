# EXP-H1A-001 Execution Runbook

**Status:** historical runbook — EXP-H1A-001 executed on 2026-10-07 and later invalidated by numerical instability  
**Protocol:** `docs/M1_3_PROTOCOL.md`

This runbook preserves the execution procedure used for EXP-H1A-001. Do **not** rerun or reinterpret EXP-H1A-001 in place. Its status is `INVALID_NUMERICAL_INSTABILITY` and H1a remains `NOT_EVALUATED`.

The current next experiment is `EXP-H1A-STABILITY-001`; see `docs/H1A_STABILITY_PROTOCOL.md`.

## 1. Environment

Use a clean environment with the repository checked out at the execution commit.

Sync the locked project environment plus the optional vision dependencies:

```bash
uv sync --locked --extra vision
```

SynapLoom uses uv; do not create a parallel pip-managed experiment environment.

Record:

```bash
git rev-parse HEAD
uv --version
uv run --locked python --version
uv run --locked python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

The benchmark runner also records Git/PyTorch/GPU metadata automatically.

## 2. Extract the frozen feature artifact

```bash
uv run --locked python experiments/scripts/extract_cifar100_resnet18_features.py \
  --root data/cifar100 \
  --output data/features/cifar100_resnet18_imagenet1k_v1.pt \
  --download \
  --device cuda
```

The extractor writes:

- the `.pt` feature bundle;
- a neighboring `.pt.sha256` checksum file;
- metadata including Torch/Torchvision versions and the transform representation.

Do not regenerate the feature file between variants or seeds.

## 3. Mandatory preflight

Run:

```bash
uv run --locked python experiments/scripts/m1_3_h1a_preflight.py \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt
```

The preflight verifies:

- exactly 50,000 train examples;
- exactly 10,000 evaluation examples;
- feature dimension 512;
- finite floating-point features;
- global class ids 0..99;
- exactly 500 train examples per class;
- exactly 100 evaluation examples per class;
- expected ResNet-18 / IMAGENET1K_V1 metadata;
- exact SHA-256 of the artifact;
- frozen H1a class order is a permutation of CIFAR-100.

**Do not execute the primary benchmark unless the preflight prints `"status": "ready"`.**

Save the preflight stdout with the run artifacts.

## 4. Execute the preregistered paired run

Run all preregistered variants and seeds:

```bash
uv run --locked python experiments/scripts/m1_3_h1a_benchmark.py \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt \
  --output runs/m1_3_h1a \
  --device cuda \
  | tee runs/m1_3_h1a/console.log
```

Do not change seeds, variants, optimizer, class order, epochs, or model dimensions after inspecting results.

Primary comparison:

```text
progressive_reserve - all_recruited
```

Co-primary outcomes:

- final average accuracy;
- late-task within-task mean accuracy.

## 5. Expected artifact layout

```text
runs/m1_3_h1a/
  progressive_reserve__seed-0.json
  ...
  all_recruited__seed-9.json
  fixed_small__seed-*.json
  dense_active_proxy__seed-*.json
  dense_total__seed-*.json
  aggregate.json
  console.log
```

Each JSON run records:

- protocol and class order;
- seed and variant;
- feature-bundle metadata;
- feature-bundle SHA-256 and audit;
- Git/Python/PyTorch/GPU provenance;
- initial/final budget information where applicable;
- continual matrices and learning curves;
- structural/routing histories;
- wall-time and peak CUDA memory.

## 6. Integrity check after completion

Before interpretation:

1. confirm all 50 expected per-seed result files exist;
2. confirm all result files report the same feature-bundle SHA-256;
3. confirm all 10 seeds exist for all five variants;
4. confirm `aggregate.json` contains the paired `progressive_minus_all_recruited` comparison;
5. confirm no run terminated early;
6. archive the feature checksum and aggregate artifact.

If any condition fails, classify the run as incomplete rather than interpreting partial results.

## 7. Result registration

Only after the integrity check:

- update `docs/EXPERIMENT_LOG.md`;
- replace the pending commit with the actual execution commit;
- record hardware and feature SHA-256;
- record per-seed values and paired bootstrap CIs;
- classify H1a as favorable, unfavorable, or inconclusive under the preregistered decision rule;
- update `PROJECT_STATE.md`.

Do not tune the protocol in place after seeing the primary result. Follow-up changes belong to a new experiment ID.
