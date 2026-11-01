# M1.3b — one-arm bounded train-only CUDA cost calibration v1

**Status: RESEARCH ENGINEERING PREPARATION ONLY / scientific execution NO-GO.** This step follows the owner's agreement to stop adding generic approval scaffolding and concentrate on meaningful **methodology review, measured GPU cost, and eventual owner-controlled scientific runner**. It is based on the verified main commit `7957a06e9858ce7a4e54c41313e1a079422fa477` and does **not** depend on the still-open [PR #38](https://github.com/ehsanrs2/synaploom/pull/38) and [PR #39](https://github.com/ehsanrs2/synaploom/pull/39). Tracks [Issue #29](https://github.com/ehsanrs2/synaploom/issues/29).

## What is actually implemented

A new manual owner-operated measurement in `m13b_cost_calibration.py` **runs existing H1a SGD/TrainingEngine** for **one** of three explicitly named engineering arms: `progressive_reserve`, `all_recruited`, `fixed_sparse_mlp`. No production TrainingEngine or router changes are made.

- **Input:** the exact historically frozen 512-D CIFAR-100 ResNet18 feature file, **only after** explicit owner trust acknowledgement and original pinned SHA-256 verification **before** unsafe `torch.load(weights_only=False)` deserializes the verified byte snapshot. SHA identity alone does not establish independent source trust. The loader returns *only* original `train_features` and `train_targets`; original benchmark `eval_features`/`eval_targets` are never used for training, diagnostics or cost.
- **Split:** from the original 500 **training** examples/class, choose 496 examples for training and four **disjoint original-training** examples/class for tiny disposable numerical diagnostics; using local CPU RNG with split seed 41. Each task has 10 classes, 4,960 train and 40 held-out original-training diagnostic examples, feature dimension 512. No record of diagnostic accuracy or forgetting is saved.
- **Tasks:** explicitly choose **2 tasks** (78 total optimizer steps for ONE arm) or **10 tasks** (390 total optimizer steps for ONE arm). Exactly **one epoch/task**, training B128 with **38 full batches and one size-96 tail** per task. This exercises nearly the full batch geometry (39 vs frozen scientific 40 steps/task) and, for 10 tasks, the full task-order recruitment R=4→13. It does **not** exercise the 20-epoch science training dynamics.
- **Methods:** only existing H1a `build_h1a_variant`, unchanged `run_continual_experiment`, and existing fixed sparse masked MLP+optimizer mask guard. SGD remains LR .01, momentum .9, weight_decay 5e-4, eval B256 and legacy batch-level routing. Engineering seed 13, class sampling seed 41, fixed mask seed 17; NO proposed confirmation seeds 501..510, different from proposed final science mask seed.
- **GPU output:** single-arm CUDA-synchronized total wall seconds, separate training seconds per task, diagnostic evaluation seconds per stage, peak CUDA allocated and reserved bytes, physical parameter count, initial/final structural recruitment, model and mask SHA, PyTorch/CUDA/GPU provenance and trusted original feature file/split digests. It exposes **no numerical learning accuracy/forgetting metrics**.
- **Evidence:** a NEW, exclusively-created output directory contains `intent.json` before reading data, `cost-observation.json` and a canonical SHA record on success, or `failure.json` on ordinary Python errors. Existing output paths are never overwritten.
- **Hard bounds:** at most 390 optimizer steps **per manually invoked arm**; the CLI has no seed, epoch, full-run, science, automatic three-arm, real evaluation-data, experimental class-order or full-run option. The workload is bounded, **not an OS-enforced hard wall-clock or energy timeout**; Ctrl-C/kill or a kernel/host failure can leave an `intent.json` without `failure.json`, which must be reported as incomplete.

**Important difference from the 2026-10-27 Pilot:** That earlier smoke did 2 tasks × 2 minibatches = 4 steps per arm, too short for reliable timing. This calibration uses 39 batches per task and, optionally, 10 tasks, to improve observability of training step costs and recruitment. It still measures **only 40 small train-derived diagnostic eval examples/task** vs the protected full science evaluation's 1,000 examples/task, and only 1 vs 20 epochs. Neither a single-arm wall time, nor multiplying this timing by 20×90 yields a defensible scientific total GPU-hour budget. The actual masked-dense PyTorch implementation is not sparse CUDA work. GPU clocks, first-run initialization, thermal state, kernel launches, stage-dependent routing and repeated samples can all affect time. Mark measured vs projected quantities clearly.

## Offline command: safe to run anywhere (does NOT access CUDA or data)

```bash
uv run --locked python experiments/scripts/m13b_cost_calibration.py --describe-only
```

## Suggested phased, **explicit owner-initiated** host measurement

A first **78-step** host cost observation on only one engineering arm:

```bash
uv run --locked python experiments/scripts/m13b_cost_calibration.py \
  --manual-trainonly-cost-calibration \
  --arm progressive_reserve \
  --tasks 2 \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt \
  --trusted-feature-source \
  --acknowledge-engineering-no-science \
  --output-dir /tmp/m13b-cost-progressive-2tasks-once
```

After verifying evidence, optionally use a **different new output directory** and `--tasks 10` for a 390-step per-arm measurement. Run each alternative arm only with its own explicit command and new output directory, to detect overhead vs fixed sparse. No automation or multi-arm launch is supplied. This PR does **not** initiate either command.

## Scientific review and cost decision after measurements

- First verify full evidence integrity, **per-stage training time** and GPU allocated/reserved memory for each arm, with explicit warm-up/first-run bias and no selective result omission.
- Assess the actual training/diagnostic evaluation work differences against proposed 8,000 steps/arm and full 1,000-sample/task protected evaluation workload. If a credible full-run GPU-time bound cannot be established, do not freeze D4.
- Independent method review must decide whether the external fixed-sparse MLP is a scientifically meaningful physically allocated capacity comparator when masked dense matmuls may cost more than ideal nonzero MAC arithmetic. Honest parameter-matched vs theoretical-vs-actual work tiers are essential.
- Independent statistical review must examine the two-endpoint n=10 bootstrap percentile accuracy/coverage and multiplicity; failure/missing results cannot be dropped after inspecting outcomes.
- Only then seek the owner's explicit finite GPU budget/abort-retry cap and separate frozen scientifically approved protocol. Later science runner must verify final git HEAD, trusted feature data, class-order/mask SHA and source commitments; **this calibration is not that runner**.

**No autonomous GPU execution, no claim of SynapLoom scientific benefit, no historical result edits, and no implicit acceptance of PRs #38/#39.**
