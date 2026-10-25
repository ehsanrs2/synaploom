# CUDA preflight repair — offline qualification

Baseline: 2e6214019a5c0acb8fe533f17225a192fa84deab.
Branch: rnd/batch-composition-performance. 2026-10-25.

Root cause reproduced on torch 2.14.1+cu130 / RTX 4090 using the baseline
research module and only batch=4, width=8, hidden=16, nodes=4, recruited=2,
k=2, classes=3. See baseline_reproduction.txt. The benchmark created features
and labels on CUDA and reused them as TensorTask evaluation data.
iter_tensor_batches constructs CPU order/indices and performs index_select before
evaluate_task moves the resulting batch to the target device. CUDA input and CPU
indices fail in this installed runtime. No production change is justified here:
the benchmark violated the host-data lifecycle expected by the evaluation path.

The sole implementation change is research/sample_performance.py: detach/copy
features and labels to CPU for TensorTask construction. TrainingEngine still uses
the original device-resident tensors. RNG generation, feature values, model,
labels, optimizer, numerical checks, hash guards, warmups and timing boundaries
remain unchanged. The CLI's direct batch/sample forward probes are unchanged.
The older frozen research/performance.py is not changed or requalified by this fix.

Data-residency policy:
- Model and training-step inputs/labels reside on the requested device. Their
  allocation and any initial model placement happen before timing.
- TensorTask data reside on CPU. The one-time device-to-host setup copy is before
  both warmup and measurement. CPU evaluation indexing stays on CPU.
- evaluate_task transfers each indexed feature/label batch to the model device
  inside each evaluation call. On CUDA, these H2D transfers ARE included in the
  ordinary and instrumented TrainingEngine/evaluation wall measurements and trace.
  CPU .to(cpu) incurs no physical H2D transfer.
- Direct batch/sample forward cost probes use resident inputs: initial H2D setup
  is excluded. transfer_to_cpu separately measures D2H; idle_sync is a baseline.
- GPU-resident TensorTask evaluation is NOT supported as a separate benchmark case
  in this revision. Resident forward probes do not imply support for that case.

Timings include two warmups outside measurement and synchronization at CUDA wall
boundaries. Profiler durations include instrumentation/export overhead. CUDA stream
event elapsed time includes launch gaps; use traces for kernel/memcpy/sync spans.
CPU/CUDA wall comparisons must retain the residency distinction above. Do not
interpret tiny smoke times as representative performance. Measurement/result JSON
schemas remain unchanged; policy is documented here rather than changing keys.

Qualification:
- Full offline suite: 74 passed, one existing sparse-state warning (validation.txt).
- Parameterized CPU/CUDA tests exercise actual index_select device compatibility,
  CPU task residency, requested model device, evaluation state hashes, examples,
  identical repeated evaluation outputs and ordinary/instrumented final-state parity.
- CPU final state equals the frozen baseline CPU smoke hash. Returned measurement
  keys and shape schema match the frozen baseline. CUDA parity is within-device;
  no CPU/CUDA bitwise numerical parity claim is made.
- Fresh tiny CPU and CUDA CLI runs completed (one measured step/repeat plus two
  warmups; four examples). Both emitted traces and operators. smoke_summary.json
  records result-schema checks, final-state parity and raw-artifact locations/hashes.
- No representative profiling, scientific experiment, retraining, push or merge.

Only the research module, its integration test, and this new qualification package
are changed. Existing batch_composition_v1 and sample_routing_v1 packages, production
sources, frozen H1a protocols, CPU results and failed CUDA artifacts are untouched.
The old batch manifest fully verifies. The frozen sample manifest verifies all
preserved artifacts and flags exactly the two intentionally changed source/test
files; it is retained unchanged. data/ remains untracked and untouched.

GO for manual CPU/CUDA profiling with this host-evaluation policy. This qualifies
execution/preflight, not representative speed, scientific accuracy or optimization.
Use fresh directories; the CLI refuses existing output directories.

From repository root, manual representative-shape synthetic commands (NOT run):

```sh
uv run --locked --offline python experiments/scripts/rnd_sample_performance.py --device cpu --threads 1 --steps 5 --repeats 5 --output /tmp/new-host-eval-cpu-profile
uv run --locked --offline python experiments/scripts/rnd_sample_performance.py --device cuda --threads 1 --steps 5 --repeats 5 --output /tmp/new-host-eval-cuda-profile
```

Bounded regression command:

```sh
uv run --locked --offline pytest tests/integration/test_sample_performance.py
```
