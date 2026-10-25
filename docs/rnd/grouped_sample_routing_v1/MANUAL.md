# Execution and interpretation

Run from repository root with committed uv.lock, offline. Every output path must
be fresh; existing directories fail closed. Defaults are bounded qualification:
batches 1/4/8, width 8, hidden 16, six nodes, router_dim 8, three output classes,
recruited/k = 2/1, 4/2, 6/4. Seed 7 with identical shared parameters per cell.
Unit tests add seed 19, projected inputs, gradients and CPU/CUDA parity.

```sh
uv run --locked --offline pytest
uv run --locked --offline python experiments/scripts/rnd_grouped_performance.py --device cpu --output /tmp/new-grouped-bounded-cpu
uv run --locked --offline python experiments/scripts/rnd_grouped_performance.py --device cuda --output /tmp/new-grouped-bounded-cuda
```

Representative-shape synthetic commands, MANUAL ONLY (not executed):

```sh
uv run --locked --offline python experiments/scripts/rnd_grouped_performance.py --representative --device cpu --threads 1 --repeats 10 --output /tmp/new-grouped-representative-cpu
uv run --locked --offline python experiments/scripts/rnd_grouped_performance.py --representative --device cuda --threads 1 --repeats 10 --output /tmp/new-grouped-representative-cuda
```

Representative mode fixes width 256, hidden 512, router_dim 64, 100 classes,
16 nodes; batches 1/32/128; recruited/k 4/1, 4/4, 16/4. These are shape proxies,
not historical datasets. Freeze real workload/config separately before generalizing.
Repeat in at least five fresh processes, changing output paths, and randomize
external process order to assess thermal/background drift. Record power/clock
settings and concurrent GPU load. All modes disable TF32 for parity; compare
matching precision policies. Metadata records versions, affinity, threads,
interop, device and OMP/MKL settings. Try threads 1/2/4/default on CPU manually.

Each cell shares one initialized model and features across production batch,
old row-wise sample reference and new grouped sample routing. All have matched
per-example k, but production has different batch-level routing semantics and
usually a smaller expert union. Hooks separately count expert calls, unique
executed experts, and expert-row work; hooks are removed before timing. Parameter
counts include dormant circuits and unused relation parameters.

Two warmups precede three default ordinary measurements with synchronization at
CUDA boundaries. CPU wall and process CPU time are distinct. CUDA events run in
a separate forward (stream elapsed, including launch gaps). One separately
instrumented forward per operation exports CPU/CUDA traces and operator tables.
Instrumented wall includes export and is not used for speed ratios. Traces report
kernel count/summed duration, launch API count/host duration, synchronization APIs,
and copy APIs. These durations overlap and must not be added or subtracted into
an additive timing decomposition. Runtime counts include the explicit trailing
synchronization. Sorted top-k dispatch requires host metadata and dynamic-output
ops: grouped execution still synchronizes and is not CUDA-graph compatible.

Start CUDA profiling in its own fresh process. With this torch/CUPTI build, a
CPU-only profiler initialized earlier in the same process produced CUDA traces
without kernel/runtime lanes. The benchmark now rejects a missing forward-kernel
trace; preserve failed directories and rerun only with a new output directory.
Integration tests use fresh subprocesses rather than accepting incomplete traces.

All forward inputs/model parameters are device-resident: initial H2D placement is
outside timing. D2H expert dispatch inside the forward is INCLUDED. Unlike the
TrainingEngine/evaluation profiler, this runner does not time evaluation batches
or include host-data evaluation H2D transfers. Full canonical state hashes are
checked outside isolated forward timings, not removed from production paths.

Guard profiling separately executes the unchanged finite-logit check, gradient
scan, parameter scan, and model_state_sha256. Gradients are created by one synthetic
backward, with no optimizer or weight update. All guard/hash operations remain
intact; D2H copies and host scalar reads are included. Hash state before/after each
probe and route. No scientific training or numerical-check bypass is performed.

Raw trace archives in this package contain all final bounded CPU/CUDA traces and
operator tables. evidence_cpu.json/evidence_cuda.json summarize the same runs;
SHA256SUMS binds code and evidence. prior_profile_audit.json binds the owner's
existing representative artifacts without overwriting or re-running them.
