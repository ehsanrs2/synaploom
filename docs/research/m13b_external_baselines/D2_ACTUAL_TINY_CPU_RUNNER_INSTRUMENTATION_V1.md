# SynapLoom M1.3b — D2 tiny live CPU Runner instrumentation (V1)

**Scope:** a *stacked* review-only PR on top of [Draft PR #58](https://github.com/ehsanrs2/synaploom/pull/58) at `f68fddd0feb160633f49741cf0849f8400fa5cda`. PR #58 passed its hosted CI but was not merged at creation. This work makes no production-science path changes. **Scientific GPU execution: NO-GO.**

## Question answered

Can the **actual existing** `run_continual_experiment` code be run entirely on generated CPU tensors, while independent hooks observe real optimizer call counts and the exact synthetic membership/order of evaluation batches, rather than trusting only a report claiming `global_steps=8000`?

**Implementation:** `src/synaploom/research/m13b_d2_live_cpu_runner_probe.py` and `tests/unit/test_m13b_d2_live_cpu_runner_probe.py`.

This is deliberately a *two-task, two-epoch* live software-path probe, **not** the ten-task, 20-epoch, 8,000-optimizer-step protected experiment, and **not** even a full-size CPU engineering replication.

## Fixed tiny CPU fixture

- Two class-incremental synthetic tasks with `classes=(0,1)` and `(2,3)`, **8 training** examples/task and **7 evaluation** examples/task.
- Five numeric input dimensions: four one-hot fabricated class dimensions and a fifth, synthetic **example identity channel**. No CIFAR-100 or protected features are loaded.
- `epochs_per_task=2`, `batch_size=3`, `eval_batch_size=4`, `device=cpu`, no future-task evaluation, mask logits to seen classes, evaluate current task at every epoch.
- Thus `ceil(8/3)=3` optimizer calls per task/epoch; **2×2×3=12 actual optimizer calls**, separately counted by a registered `optimizer.register_step_post_hook`, and checked against `result.global_steps` and each epoch checkpoint.
- Seven distinct evaluation *passes* (four current-task epochs, three stage/seen-task evaluations), each with a 4-example batch and 3-example final partial batch: **14 actual forward batch events**, observed in the **model's real pre-forward hook** and checked against the exact synthetic identity sequence and step-hook clock.
- Two small models cover the real common `nn.Linear`-output branch and a real `FixedSparseMLP`-output branch. The masked case attaches existing `install_fixed_mask_guard` **before** the audit hook and verifies mask identity, inactive weights and optimizer momentum state after **every observed optimizer step**.
- Training uses generated data but calls the *real* `run_continual_experiment`, evaluator, tensor iterator, optimizer, and `ContinualRunResult.to_dict`; this is materially different from the fabricated full-shape output in #58. Actual model parameters are checked for changes across the run; no inference about architecture superiority follows.
- A fault-injection switch available only in the Python test API stops after a requested real optimizer post-hook observation. The result retains `SYNTHETIC_CPU_RUNNER_FAILURE_RETAINED`, the exception class/message and completed-hook count, and never produces a PASS on an interrupted run. The hook handles are removed in a `finally` block.

## What independent checks really mean

Three source files actually imported by the CPU process (`continual.py`, `evaluation.py`, `streams.py`) are hashed **directly from runtime-imported file bytes**, before and after the probe. An optionally caller-provided *expected* hash map must match before any optimizer call. Input train/eval tensors (including shapes, dtypes, labels and synthetic example-ID channels) are byte-hashed before and after execution. Both checks reject unexpected intra-run modifications.

**Limits, precisely:** A digest of actual imported file bytes is stronger than trusting a digest written in an arbitrary result JSON, and the identity-channel hook checks actual *synthetic* evaluation inputs. However, caller-supplied digest expectations are **not a signed/independent trust anchor**; loaded Python functions could be monkeypatched; dataset identity channels could be forged; a post-step hook attests that `optimizer.step()` completed sufficiently to invoke the hook, not that every intended mathematical update was scientifically valid. The test does not instrument CUDA kernels or prove complete machine/file-system provenance. No real protected sample membership is checked. The final return packet explicitly says `source_attested_by_independent_signer=false` and `science_runner_integration=false`.

The small experiment intentionally uses engineering seeds **13 or 17** only and rejects proposed confirmation seeds **501–510**. The source emits no scientific accuracy/performance verdict; `actual_science_optimizer_steps=0` and `science_executable=false` remain true even on a 12-step CPU PASS. The input generation and RNG-state guards prevent accidental reuse of protected data and restore Python, NumPy and Torch CPU RNG state after the probe.

## Research code coverage

The new unit tests cover the 12-step actual Runner execution for both small models, 14 forward-event order/membership equality, source-byte pin success/failure, artificial stream-tensor mutation, injected optimizer failure and its explicitly preserved terminal record, invalid seeds/knobs and caller RNG restoration.

**Do not call CI PASS until the exact PR-head hosted run completes.** No CLI that loads arbitrary files or accepts scientific run flags is exposed.

## What is still missing for D2

1. A **separately reviewed, full-shape real execution control** over 10 tasks ×20 epochs/8000 steps with a trusted source/feature manifest and fully recorded optimizer/Mask state for every scientific attempt. **Not implemented or authorized here.**
2. A real SHA-attested **ordered feature/label membership** for all 1000 evaluation samples per task and the B256+256+256+232 grouping. The tiny probe observes synthetic B4+3 only, while PR #58 checks B256/B232 **position geometry** on fabricated indices. Neither asserts protected dataset membership.
3. Source and feature fingerprints bound to an independently checked Git revision, trusted immutable evidence root, host state, model/optimizer loading, step/resume/partial failure ledger and scientifically appropriate retry/exclusion policy.
4. Owner approval of evaluation policy and material legacy batch-composition effects; no evaluation-size or batch order selection based on protected results.
5. Outstanding **D3 final mean-based inference method/paired n**, **D1 arms/order/external baseline fidelity**, and **D4 total GPU-hour ceiling**, followed by a new explicit owner execution GO.

For review, run locally on CPU:

```bash
uv run --locked python -m pytest -q tests/unit/test_m13b_d2_live_cpu_runner_probe.py
uv run --locked python -m pytest -q \
  tests/unit/test_m13b_d2_eval_guard.py \
  tests/unit/test_m13b_d2_runner_output_contract_cpu.py \
  tests/unit/test_m13b_d2_live_cpu_runner_probe.py
```

**Disposition:** `D2_ACTUAL_TINY_CPU_PATH_WITNESSED` (only if hosted test succeeds); `D2_FULL_SCIENCE_RUNNER_UNTRUSTED`; `SCIENTIFIC_GPU_NO_GO`.
