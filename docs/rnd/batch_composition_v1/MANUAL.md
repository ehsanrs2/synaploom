# Manual execution package — do not run automatically

From repository root, on rnd/batch-composition-performance. Use committed uv.lock.
Every --output directory must be new; existing evidence is rejected.

## A. Trained fixed-state composition diagnostics

Provide an authorized exported **plain state_dict** checkpoint (not the training
checkpoint envelope), a JSON architecture config, and a torch tensor dataset.
No training command is supplied. Historical checkpoints were not saved; obtain
an explicitly authorized new checkpoint separately if needed. Do not reconstruct
or retrain a historical cohort for this package.

Dataset saved with torch.save: {"x": float tensor [N,D], "y": long tensor [N],
"sample_ids": unique list of strings}. At least eight rows per class, at least
two classes; all rows become anchors. IDs must refer to a frozen external source
manifest. The harness hashes tensors and records companions; provenance also
hashes the serialized dataset and checkpoint. No class masking is applied.

Routed config example (replace with actual checkpoint architecture):
```json
{"kind":"progressive_reserve","input_dim":512,"num_classes":100,
 "graph":{"num_nodes":16,"initial_recruited":4,"d_model":256,
 "d_hidden":512,"router_dim":64,"top_k":4,"message_rounds":0}}
```
All graph fields must match the original config, including q_max/relation_types
when nondefault. For all_recruited use kind all_recruited and the original
initial_recruited=16. Loaded recruitment buffers define the measured capacity.
For dense use kind dense, input_dim, d_model, d_hidden, num_classes with no graph.
Repeat separately for each available early/later state and control. This example
is a schema illustration, not an H1a checkpoint reconstruction.

```sh
uv sync --locked --offline
uv run --locked --offline python experiments/scripts/rnd_batch_composition.py --checkpoint /ABS/model-state.pt --config /ABS/model-config.json --dataset /ABS/anchors.pt --seed 7 --device cpu --output /ABS/new-trained-composition-run
```

Outputs: results.json, with artifact hashes, state hashes, condition summaries,
per-anchor raw logits, weights, companion IDs, permutation residuals and checks.
CPU is the default; CUDA requires explicit --device cuda and owner execution.
Task accuracy is condition accuracy over unique anchors, not historical continual
accuracy. Rows across conditions are correlated; no independent-sample inference.

## B. CPU/GPU performance profiling

These commands operate only on fresh tiny synthetic models, not on scientific
cohorts. Each executes two warmups plus five measured iterations per path. The
ordinary path and profiler path start identically and must finish hash-identical.
Compare wall_seconds only among ordinary paths. Instrumented durations include
trace setup and export overhead. Synchronization is included at wall boundaries.

```sh
uv run --locked --offline python experiments/scripts/rnd_performance.py --device cpu --threads 1 --steps 5 --output /ABS/new-profile-cpu-t1
uv run --locked --offline python experiments/scripts/rnd_performance.py --device cpu --threads 2 --steps 5 --output /ABS/new-profile-cpu-t2
uv run --locked --offline python experiments/scripts/rnd_performance.py --device cpu --threads 4 --steps 5 --output /ABS/new-profile-cpu-t4
uv run --locked --offline python experiments/scripts/rnd_performance.py --device cuda --threads 1 --steps 5 --output /ABS/new-profile-cuda-t1
```

For the default-thread comparison pass the default torch.get_num_threads() value
recorded in environment.json as --threads, in a separate process and new output.
Record OMP_NUM_THREADS/MKL_NUM_THREADS, CPU affinity, background load and GPU model.
Repeat ordinary timings at least five processes per setting before optimization
ranking by measured benefit. This fixture is instrumentation qualification only;
representative H1a dimensions/data require a separately frozen profiling revision.

Outputs: timing.json (ordinary and instrumented), operators.txt (CPU operator and
named region table), trace.json (open locally in Perfetto). Inspect aten::to,
_copy/copy_, aten::item/_local_scalar_dense, CUDA memcpy/device synchronization,
backward autograd operators and GPU kernel/launch lanes. Named regions isolate
router, executor, optimizer, numerical checks, importance, evaluation and hashing.
CPU traces cannot measure CUDA launch or synchronization costs. Process CPU / wall
is an approximate busy-core count, not per-thread utilization. Use trace thread
lanes and OS thread sampling for contention; do not equate low GPU utilization
with a proven bottleneck.
