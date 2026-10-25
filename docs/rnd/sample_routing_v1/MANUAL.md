# Manual commands and performance methodology

Run from repository root. Output directories must not exist. No command loads
scientific data or checkpoints. Representative-shape defaults are synthetic H1a-like
proxies (128x256, hidden 512, 16 nodes, 4 recruited, k=4, 100 classes), not a measured
historical workload. Freeze actual deployment dimensions and input provenance
before drawing representative conclusions. Use --recruited 16 for later capacity.

Bounded offline qualification:
```sh
uv run --locked --offline pytest
uv run --locked --offline python experiments/scripts/rnd_sample_qualification.py --output /tmp/new-sample-qualification
```

CPU manual thread sweep, each in a new process; repeat the sweep five times with
new output paths. Include the environment's default thread count as another value.
```sh
for t in 1 2 4 16; do
  uv run --locked --offline python experiments/scripts/rnd_sample_performance.py --device cpu --threads "$t" --steps 5 --repeats 5 --output "/tmp/new-h1a-cpu-t$t"
done
```

CUDA manual command (not executed during qualification):
```sh
uv run --locked --offline python experiments/scripts/rnd_sample_performance.py --device cuda --threads 1 --steps 5 --repeats 5 --output /tmp/new-h1a-cuda
```

Two warmups precede measurements. Each ordinary/instrumented TrainingEngine pair
starts at the same seed and must end with identical canonical state SHA256.
Numerical checks, gradient/parameter scans, importance, optimizer, evaluation and
hash guards remain enabled. Named regions isolate SparseRouter, GraphExecutor,
TrainingEngine components, checks, hash and evaluation. Trace exports contain CPU
and CUDA activities. Compare ordinary wall times; profiler wall includes setup,
wrappers and export. Their difference estimates instrumentation cost with noise.
Process CPU time and wall time are distinct. CPU affinity, torch/CUDA/device,
threads, interop and OMP/MKL environment are recorded; record background load and
power/clock settings alongside results.

Matched model batch/sample forwards additionally report repeated synchronized wall
costs; both use identical parameters and inputs, k and capacity. D2H and idle-sync
probes are separate baselines. CUDA events measure stream elapsed intervals,
INCLUDING launch gaps, not summed kernel execution. Use trace kernel durations for
kernel execution, CUDA memcpy lanes for transfer activity, and host synchronization
API spans for blocking overhead. These overlap: never subtract them into an
additive decomposition. Idle sync is not loaded sync cost. Inspect aten::item,
_local_scalar_dense, to/copy, device synchronization and host thread lanes.
Hash CPU serialization/SHA and transfers are inside the hash region. Evaluation
scalar transfers remain inside evaluation. No checks may be disabled for timing.

Oversubscription is a hypothesis: compare ordinary wall distributions at fixed
shapes and capacity across threads, inspect contention, and verify outputs,
ID-aligned weights, gradients, optimizer and state-transition parity before using
a cap. Thread count alone is not evidence. CUDA exact-hash parity may reject
nondeterministic runs; retain failures and investigate, do not weaken guards.
