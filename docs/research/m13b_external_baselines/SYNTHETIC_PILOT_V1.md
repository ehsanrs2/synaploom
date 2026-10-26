# M1.3b synthetic three-arm engineering pilot v1

**Status: ENGINEERING-ONLY / not a scientific study.** After merged [PR #32](https://github.com/ehsanrs2/synaploom/pull/32), this adds one deliberately bounded executable synthetic-CPU pilot. The real M1.3b `EXP-M13B-EXTERNAL-001` protocol remains **UNFROZEN and NO-GO for training**.

## Precisely allowed work

The pilot uses only `make_gaussian_cluster_stream` with:
- 3 tasks, 2 toy classes/task, feature dimension 8, train 4 examples/class and eval 2 examples/class;
- one epoch/task, batch size 4, eval batch 4, fixed engineering model seed 13, data seed 41 and fixed mask seed 17;
- three *exact* arms: `progressive_reserve`, `all_recruited` and `fixed_sparse_mlp`. No control is newly relabeled as RigL/NISPA.
- original H1a builder for SynapLoom arms, the existing fixed-sparse candidate builder and its mandatory optimizer mask post-step hook, and the existing continual training/evaluation harness; no production modifications;
- CPU execution only, one thread during the pilot, with finite diagnostic task-summary metrics and state/mask digests. Inactive fixed-sparse weights and optimizer moments are checked after the run.

This is a **smoke of a miniature engineering fixture**, not an assessment of continual-learning performance: the synthetic Gaussian task is unrelated to real frozen CIFAR-100 ResNet features, has only six classes, and has neither the frozen 10×10 task schedule nor the approved paired 501–510 seeds. Accuracy/forgetting fields explicitly use `metrics_synthetic_only` and `scientific_claim=NONE`. A favorable numeric toy result would never authorize or predict H1a success.

## Exact user-invoked command

```bash
# Run three tiny CPU toy arms and print canonical JSON digest; no file is written.
uv run --locked python experiments/scripts/m13b_synthetic_pilot.py \
  --synthetic-engineering-pilot --acknowledge-engineering-only

# Optional unique artifact destination. Directory MUST NOT already exist.
uv run --locked python experiments/scripts/m13b_synthetic_pilot.py \
  --synthetic-engineering-pilot --acknowledge-engineering-only \
  --output-dir /tmp/m13b-synthetic-pilot-once
```

The CLI has **no** GPU, `--execute`, scientific seed, model override, feature-path, class-order or CIFAR argument. It refuses existing output directories before starting work. Without `--output-dir`, output is printed only. With a new directory, it writes `engineering-pilot.json` and a canonical-JSON SHA-256 record; these hashes detect content changes but do not authenticate source, user authorization or scientific provenance.

This script is **different from** `m13b_external_preflight.py`, which remains non-executable and still handles the trusted local frozen feature SHA audit separately. No main/legacy H1a protocol or scientific results are overwritten.

## CI and future gates

New tests verify the three fixed arms, synthetic class/task/seed/data contract, mask immutability, nonfinite-output rejection, deterministic repeated CPU results, emitted hash and exclusive evidence destination, and fail-closed rejection of unapproved CLI flags. GitHub-hosted CPU CI and smoke cannot qualify RTX 4090 GPU arithmetic or benchmark science.

**Next meaningful scientific gate:** review the M1.3b proposed 30 O0 or 90 O0/O1/O2 runs, finalize fair resource tiers and multiplicity *before seeing new results*, freeze the owner-approved study protocol, independently verify the real trusted feature file on the executing host, then add an explicitly approved scientific runner with run-directory collision/partial-failure records. Prefer time-boxed qualification and proceed promptly to the real experiment rather than indefinitely extending checkpoint machinery.

**No-CIFAR, no-GPU, no-training-run-in-background, no-scientific-claim boundary.**
