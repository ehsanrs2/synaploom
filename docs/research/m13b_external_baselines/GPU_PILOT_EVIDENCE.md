# M1.3b — Owner-reported RTX 4090 train-only pilot evidence

**Evidence date: 2026-10-27 (owner transcript). Classification: `GPU_TRAIN_ONLY_ENGINEERING_COMPLETE`, not scientific training results.**

This audit record transcribes the owner's shell outputs after merging PR #34. The assistant did **not** receive or independently read `/tmp/m13b-cuda-trainonly-v1-once/pilot.json`, `arm-*.json`, or `pilot.canonical-sha256` from the owner's machine. Therefore the reported checks below are **owner-executed validation output**, not separately re-performed remote artifact validation. Recording it in GitHub does not turn it into independently attested evidence. Original result files and historical verdicts have not been modified.

## Exact reported validation

```text
SHA256_MATCH: True
GPU: NVIDIA GeForce RTX 4090
PyTorch: 2.14.1+cu130
CUDA: 13.0
FEATURE_SHA256: c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47
TRAIN_ONLY_SPLIT: 654174acb7b263534d4428d82a3801f95801a67dafdc3d339ddccfda458a96d5
ARMS_COMPLETE: True
FAILURE_FILE: False
EVIDENCE_VALIDATION: PASS
```

The owner confirmed each of three arm status fields equals `GPU_TRAIN_ONLY_ENGINEERING_ONLY`, every arm has four optimizer steps, all arm wall times are positive, the protected CIFAR evaluation partition flag is false, and the report declares `scientific_claim=NONE`. These are checks of the returned report and its canonical JSON hash; the SHA is a consistency check rather than an independent signature of source/authenticity.

## Per-arm observations as reported

| Arm | Steps | Elapsed wall (s) | Peak CUDA allocated (MiB) | Peak CUDA reserved (MiB) | Physically trainable parameters |
|---|---:|---:|---:|---:|---:|
| `progressive_reserve` | 4 | 0.314 | 25.23 | 30.00 | 1,172,196 |
| `all_recruited` | 4 | 0.132 | 25.23 | 30.00 | 1,172,196 |
| `fixed_sparse_mlp` | 4 | 0.027 | 39.49 | 66.00 | 1,172,226 |

The three model SHA-256 values reported by the owner:
- `progressive_reserve`: `0695f9c30d05c328a75fd4567baa5ddfa8f65aad6454c30d2ecb3a8d3e8c71cb`
- `all_recruited`: `f6861ee091d3f51314dcaacee6a1f89f6247abdb653b202d644825dbbedd99eb`
- `fixed_sparse_mlp`: `14c3fd85b3b41b7bb38a77434628e10dc4eaaa789d24070168d00ba02d607515`

Fixed-sparse mask fingerprint: `05c273126ac38b992eef60b98170f0d8af7b1ee81ee5af0594acb28187e7e461`. The other two arms report `mask_sha256: None`.

## Interpretation and measurement limits

- Fixed-sparse MLP has **30 more physical trainable parameters** than SynapLoom (approximately **0.00256%** difference): the preliminary physically allocated weight-budget match is excellent for this tiny pilot. **This does not imply** comparable active capacity, ideal MACs, actual forward/backward FLOPs, activation bandwidth or memory use.
- Peak CUDA allocated memory for fixed sparse is **14.26 MiB / about 56.5% greater** than either SynapLoom arm, and peak reserved is **36.0 MiB more** (66 versus 30 MiB). Optimizer state, dense masked matmul, CUDA allocator behavior and activation patterns could affect these totals; this record does **not** identify a causal mechanism.
- Recorded wall times are extremely short: 0.027, 0.132 and 0.314 seconds. The actual source executes the arms in the order `progressive_reserve → all_recruited → fixed_sparse_mlp`; the sorted JSON output displayed by the owner differs from run order. Warm-up/caching, step timings, variation and evaluation overhead are not isolated or randomized. **Do not claim a robust fixed-sparse speed advantage or an 11.6× practical speedup** from these observations.
- Two training tasks, one epoch/task, only 16 training examples + four train-derived diagnostic-holdout examples/class. The original protected eval partition was not used, nor were confirmatory seeds 501–510. This does not qualify the full CIFAR-100 science runner, loss-of-plasticity hypothesis or reduced forgetting.
- The trusted input bytes matched the historically frozen feature SHA by the pilot's gate, according to the user-transcribed report. No independent host file audit or signature is provided by this GitHub document.

## Decision and next gate

**`GO_BOUNDED_GPU_INTEGRATION`** for this version's owner-reported three-arm train-only systems pilot: the interface, artifact checksum check, CUDA-device reports and numeric completion criteria succeeded. **`NO_GO_M13B_CONFIRMATORY_SCIENCE`** until [OWNER_DECISION_GATE.md](OWNER_DECISION_GATE.md) is reviewed, an independently frozen protocol and full execution runner are approved, and fair-resource/algorithm fidelity gates are satisfied.

An optional dedicated engineering-only GPU performance study would need multiple controlled warm-up/measurement repetitions, randomized or balanced arm order, isolated end-to-end vs per-step timings, recorded torch/CUDA/hardware environment and explicit accounting for dense masked compute. **No such repeated study has been performed or authorized.**
