# Reproduction and owner commands v1

Run from /home/ai/project/My/synaploom using the committed uv.lock, offline.
Existing directories are rejected. Do not reuse or delete previous evidence paths.
All automatic qualification in this phase was tiny synthetic CPU/CUDA execution.

Offline parity tests and bounded probes:

```sh
uv run --locked --offline pytest
routing_run_root=$(mktemp -d /tmp/synaploom-routing-design-bounded.XXXXXX)
uv run --locked --offline python experiments/scripts/rnd_routing_design_probe.py --device cpu --output "$routing_run_root/cpu"
uv run --locked --offline python experiments/scripts/rnd_routing_design_probe.py --device cuda --output "$routing_run_root/cuda"
```

The probe defaults are B1/8, R4/16,K4, width8, hidden16, router_dim8, classes3,
capacity16, five repeats. Compare grouped and row baselines to separate singleton,
index_copy restore and precomputed-row probes. Parity tests additionally use seed19,
input projection, K1, dormant reserve and recruitment/consolidation between calls.
No optimizer or scientific training is executed. All forward costs use resident
features: initial H2D excluded, dispatch D2H included; state hashes stay outside
isolated-forward timings. Numerical/hash probes from the representative review
are separate from these forward ablations. CPU/CUDA tensor RNG/parameter setup
and TF32 settings must be matched; no precision-policy speed advantage is allowed.

Re-read existing owner artifacts without executing any model:

```sh
routing_review_root=$(mktemp -d /tmp/synaploom-routing-review.XXXXXX)
uv run --locked --offline python experiments/scripts/rnd_routing_design_review.py --cpu /tmp/synaploom-grouped-cpu-v1 --cuda /tmp/synaploom-grouped-cuda-v1 --output "$routing_review_root/review"
```

Representative baseline and ablation commands, OWNER MANUAL ONLY. These were not
executed in this design phase. Fresh CUDA process per command is essential for
CUPTI activity. Each output directory is new; errors remain preserved for review.

```sh
routing_owner_root=$(mktemp -d /tmp/synaploom-routing-owner.XXXXXX)
uv run --locked --offline python experiments/scripts/rnd_grouped_performance.py --representative --device cpu --threads 1 --repeats 10 --output "$routing_owner_root/baseline-cpu"
uv run --locked --offline python experiments/scripts/rnd_grouped_performance.py --representative --device cuda --threads 1 --repeats 10 --output "$routing_owner_root/baseline-cuda"
uv run --locked --offline python experiments/scripts/rnd_routing_design_probe.py --representative --device cpu --repeats 10 --output "$routing_owner_root/probes-cpu"
uv run --locked --offline python experiments/scripts/rnd_routing_design_probe.py --representative --device cuda --repeats 10 --output "$routing_owner_root/probes-cuda"
uv run --locked --offline python experiments/scripts/rnd_routing_design_review.py --cpu "$routing_owner_root/baseline-cpu" --cuda "$routing_owner_root/baseline-cuda" --output "$routing_owner_root/review"
```

The baseline representative sweep is B1/32/128 and R/K4/1,4/4,16/4. Probes focus
B1/128 and R4/16,K4. Both use dimensions256/512/64, classes100 and16 allocated
nodes. A later promotion investigation must also cover B32 probes, per GATES.md;
add that case in a separately reviewed probe revision before claiming promotion.
The current script is an investigation tool, not a complete promotion evaluator.

Repeat in at least five fresh process sets, creating a new root each time. Vary
external process order to expose clock/background effects; keep ordinary duration
arrays and traces, record GPU load/power/clocks, and retain all failures. Benchmark
runners report kernel/launch/copy/sync metrics separately, not additive time shares.
No missing CUDA kernel lanes may be accepted. Guards, hashing and finite-score
checks cannot be disabled. No session/training command is supplied: that interface
is a design, and its implementation requires the acceptance tests in GATES.md.

Package evidence: investigation_cpu.json/investigation_cuda.json are final bounded
summaries; identically named .tar.gz files contain their raw traces/tables and the
original results.json. probe_cpu/probe_cuda archives retain earlier development
runs. Inspect archives with tar -tzf, or extract into a fresh directory. Their
original bytes are bound by raw_artifact_hashes.json and the package SHA256SUMS.
