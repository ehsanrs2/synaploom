# M1.3b draft manifest, dry-run and feature preflight v1

**Status: DESIGN PREVIEW ONLY / SCIENCE EXECUTION NO-GO.** Research-only continuation after merged [PR #31](https://github.com/ehsanrs2/synaploom/pull/31). This is an **implemented planning CLI**, not a full scientific training runner and **not a frozen preregistration**. No existing H1a experiments, `TrainingEngine`, default routing or checkpoints are changed.

## Implemented commands (safe, read-only by default)

```sh
# No feature access, optimizer steps or training; prints proposed design matrix.
uv run --locked python experiments/scripts/m13b_external_preflight.py --dry-run

# Optional: write a brand-new draft plan; refuses to overwrite existing file.
uv run --locked python experiments/scripts/m13b_external_preflight.py --dry-run \
  --plan-output /tmp/m13b-draft.json

# Explicit owner-side preflight of an independently trusted *local* feature artifact.
# This reads the feature file and invokes PyTorch pickle only AFTER checking
# the historically pinned feature SHA-256. DO NOT use downloaded/unknown files.
uv run --locked python experiments/scripts/m13b_external_preflight.py \
  --verify-features \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt \
  --trusted-source
```

**No `--execute` flag exists.** The CLI is deliberately incapable of starting a scientific trajectory, even if someone supplies an output-plan path. `--dry-run` refuses `--features` and `--trusted-source` options, and thus cannot read the feature file. `--verify-features` requires both path and explicit trust declaration. A locally computed checksum is *byte identity*, not publisher authentication; `torch.load(weights_only=False)` can execute pickle, so the origin must be independently trusted, and no web-downloaded checkpoint may be loaded merely because its self-supplied hash matches.

The `--plan-output` path is only an optional **draft JSON evidence file**, never a training-result directory. Exclusive creation prevents overwriting existing evidence. Parent directory must already exist. No file is created when `--plan-output` is absent.

## Proposed matrix, not approved schedule

The manifest presents **both mutually exclusive order-scope options** rather than silently choosing:
- `O0_ONLY`: 1 order × 10 proposed fresh paired seeds `501..510` × 3 arms = **30 trajectories**, 10 paired blocks.
- `O0_O1_O2`: 3 frozen orders × the same 10 proposed paired seeds × the same 3 arms = **90 trajectories**, 30 paired blocks.

Arms in this *preview*: `progressive_reserve`, `all_recruited`, `fixed_sparse_mlp`. The two O1/O2 class orders are imported from the existing frozen sensitivity protocol; SHA fingerprints are validated against the registered `O0/O1/O2` digests. **RigL and NISPA are NOT admitted** or scheduled and replay remains an independent resource condition.

The original H1a-002 SGD policy is recorded as an **anchor**: LR=0.01, momentum=.9, decay=5e-4, B128 train, B256 eval, 20 epochs/task, 10×10 Split CIFAR-100 with frozen 512-D ImageNet ResNet18 features and seen-class output masking. Historical `EXP-H1A-002=INCONCLUSIVE`, `ORDER_SENSITIVE` and `DIRECTION_FLIP_OBSERVED` remain unchanged.

## Provenance and limits

`build_m13b_draft_manifest()` writes deterministic JSON-compatible fields:
- expected feature SHA `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47` with **NOT_CHECKED** marker until trusted local audit;
- source Git HEAD if a valid SHA is discoverable, but `source_revision_verified=False` and **no code-signature or artifact-origin attestation**;
- frozen class-order SHA-256 IDs and literal proposed seed/arm order;
- resource candidate's mask digest, physical count, body density, **ideal** sparse linear MAC estimate and **actual dense-mask implementation** linear MAC estimate separately;
- two design-only budget gates (≤2% total parameters, ≤10% ideal linear MAC), **not GPU parity**;
- co-primary candidate statistical rules, unresolved decisions, explicit `science_executable=False`, `authorization=NO_GO_DRAFT_NOT_FROZEN`, `run_status=NO_RUNS_EXECUTED`;
- content-based `record_sha256` of canonical JSON excluding the checksum itself, to detect accidental changes. This is **not** independent approval or a signing key.

`verify_trusted_feature_file` checks the exact frozen file SHA *before* the existing feature-metadata audit, then rechecks SHA to detect ordinary concurrent byte changes. It **does not run** on GitHub hosted CI because the original feature bundle is not available. Passing dry-run CI cannot imply the owner's GPU host has the feature file.

A manifest field containing `feature_sha256_verified` means only the trusted local file audit succeeded; it does **not** change execution authorization. Scientific runs need a separately approved and frozen protocol, explicit owner authorization, a real training runner, strict output-directory collision guards, and CUDA/wall/memory/method-fidelity qualification.

## Acceptance and next goal

The new `tests/unit/test_m13b_manifest.py` checks matrix cardinalities, O0/O1/O2 digests, budget views, deterministic RNG-safe preview, canonical manifest tamper detection, invalid source revisions, refusal of untrusted/mismatched feature bytes *before pickle*, CLI rejection of `--execute`, file collision, and no implicit feature reads.

1. Review new tests and actual GitHub Actions CPU steps, keeping CUDA skipped counts separate.
2. Choose O0-only vs O0/O1/O2, proposed 501–510 seed block, mask seed/distribution and matching-resource tier **before any new result**.
3. Verify the real feature file on the owner-controlled host only once it is relevant; never guess based on its historical path.
4. After owner freezes a scientific protocol, create a separate real runner with fail-fast numerical checks, immutable identity/cursor/output tracking, and strict authorization before any GPU pilot.

**GO for offline preflight/manifest review; NO-GO for scientific execution and model-performance claims.**
