# SynapLoom M1.3b / D2 — actual synthetic B256/B232 evidence and failure retention (V1)

**Review status:** `D2_SYNTHETIC_EVAL_BYTES_AND_CPU_LEDGER_REVIEW_ONLY`. Based on merged [PR #58](https://github.com/ehsanrs2/synaploom/pull/58) (merge `8fdad73c79ea7f81d298d93d2e0bd5c3fb769fdd`) and merged [PR #59](https://github.com/ehsanrs2/synaploom/pull/59) (merge `28311fc6cdb84b36efac66e43cc61a848f9f3683`), whose GitHub Actions were both successful. No model-science experiment, confirmatory seed, protected evaluation or GPU is authorized by this additive PR.

## What changes

Two independent synthetic research sub-tests complement the earlier D2 work.

**1. Actual B256/B232 evaluation membership.** `src/synaploom/research/m13b_d2_cpu_evidence.py` creates **exactly 1,000** entirely artificial two-class examples, each with one-hot class features and a unique integral float32 sample ID in a separate channel. It uses the actual `evaluate_task` function and `iter_tensor_batches` path with `batch_size=256`, `shuffle=False`, and a partial final batch. A model-forward witness records the actual input bytes and ordered sample IDs in each of four groups: **[0..255], [256..511], [512..767], [768..999]**. Every batch's SHA-256 is compared with independently sliced input tensors; the complete train/eval tensor bytes and labels are additionally compared to a regenerated known synthetic fixture. Missing tail, shuffled members, corrupt features/IDs and mismatched labels are rejected in synthetic unit tests.

This establishes that **the actual reference evaluator processes these artificial inputs in the expected four groups**. It does not establish that the real protected CIFAR-100 feature rows, label hashes, class-order sample indices or training sample selection are trustworthy. The artificial identity channel is a toy control and is never added to the scientific model or training data. The SHA values are content integrity checks, not signatures.

**2. Retained attempt evidence, including failed attempts.** The same module calls merged-PR59's `run_tiny_cpu_probe(seed=13, masked=True)` (12 actual observed CPU optimizer steps across two tiny tasks), with optional bounded synthetic failure injection. Each invocation writes a complete JSON record containing the real small CPU optimizer witness and separate 1,000-item evaluation identity witness. Files are created using Python's exclusive `xb` mode, flushed and `fsync`ed, preventing accidental silent reuse of the same attempt ID. The verifier demands an exact, nonempty expected list: missing, duplicate, unexpected or corrupted records are rejected, and injected failures remain failures rather than being filtered out or labeled successes.

**Security boundary:** A directory writable by its owner is neither WORM/immutable nor protected against deletion, file replacement or a malicious actor recomputing a self-contained checksum. We have NOT implemented owner signatures, trusted hardware attestation, external witnessing, host wall-clock evidence, git-tree attestation, protected dataset access, or a production science runner. A file created exclusively and then checksum-verified is a useful **local engineering ledger**, not independently reliable scientific provenance.

### Safety restrictions

- No CLI accepts arbitrary dataset paths, model checkpoints, scientific seeds 501–510 or GPU flags; the file-writing API only calls the pre-existing tiny 12-step CPU probe with seed 13, never an 8,000-step study.
- Each record and output states `science_executable=false`, `scientific_claim=NONE`, `gpu_used=false` and `protected_eval_accessed=false`.
- The reference-evaluation subtest is a separate forward-only exercise on generated tensors; **its 1,000 examples are not a protected evaluation dataset**.
- No production optimizer/evaluator or historical scientific artifacts are modified. The new source is additive under `research/`.

### Reviewable code and negative cases

- `src/synaploom/research/m13b_d2_cpu_evidence.py`
- `tests/unit/test_m13b_d2_cpu_evidence.py`

The tests assert true B256/B232 forward groups, expected SHA correspondence, synthetic mutation rejection, simulated omission of the trailing batch and injected reordering through the evaluator's iterator, unique attempt IDs, retained injected failure at step 4, evidence integrity, corruption and incomplete JSON rejection, safe attempt ID validation and no scientific authority.

**CI is a separate condition:** do not mark this stage qualified until the exact PR head's hosted test suite succeeds.

### D2 scope left before any protected result

1. Prospectively pin the actual class-order O0 **protected** sample identifiers/feature and label contents, the data-loader version and true evaluation-group membership. A generated toy byte digest cannot substitute for this.
2. Bind a final manual Runner to an externally verified Git tree, trusted feature-bundle checksum, fixed-mask identity, 8,000 actual optimizer-step witnesses, per-attempt complete failure ledger and agreed retry policy, with reliable evidence custody. Even a full-shape CPU demonstration would not suffice to attest an eventual GPU run.
3. Before science, obtain explicit owner decisions on **D1** architecture/order/generalization and RigL/NISPA feasibility, **D3** final mean-based inference/paired-seed count and joint power assumptions, and **D4** bounded GPU-hour budget. The original candidate ten-seed percentile bootstrap must not silently become the approved method.

**Next decision after review:** D2 has enough positive *small synthetic* path coverage to motivate a **prospective D2 evidence trust-model/specification** rather than treating every new toy check as a science GO. Draft the actual run-manifest/identity/owner-attestation schema with independent reviewer gates before touching protected outcomes.

**Disposition:** `D2_SYNTHETIC_FULL_B256_EVALUATOR_CHECK_RECORDED`; `D2_LOCAL_EXCLUSIVE_ATTEMPT_LEDGER_RECORDED`; `D2_TRUSTED_SOURCE_DATA_SCIENCE_RUNNER_NOT_READY`; `D1_D3_D4_OPEN`; `SCIENCE_GPU_NO_GO`.

Safe CPU commands:

```bash
uv run --locked python -m pytest -q tests/unit/test_m13b_d2_cpu_evidence.py
uv run --locked python -m pytest -q \
  tests/unit/test_m13b_d2_eval_guard.py \
  tests/unit/test_m13b_d2_runner_output_contract_cpu.py \
  tests/unit/test_m13b_d2_live_cpu_runner_probe.py \
  tests/unit/test_m13b_d2_cpu_evidence.py
```
