# Grouped sample routing — qualification and decision

2026-10-25; branch rnd/batch-composition-performance;
base 2f5ebe7526d34845f45a520b574227ac71e769ca.

GO for controlled integration planning and owner-run representative profiling.
NO-GO for production replacement, training effectiveness claims, or a universal
performance claim. No representative profiling was executed in this phase.

## Reviewed prior evidence

[MEASURED] Independently inspected the owner's existing host-evaluation v3 CPU/CUDA
results and CUDA trace-0 under /tmp. Median batch/sample times reproduce the task:
CPU 2.219862/23.540364 ms; CUDA .477111/44.276449 ms (92.801x sample/batch).
CUDA trace-0 contains 3,230 cudaStreamSynchronize, 3,245 cudaMemcpyAsync and two
cudaDeviceSynchronize calls. Artifact paths, complete hashes, shapes and environment
are in prior_profile_audit.json. These traces cover TrainingEngine/evaluation and
integrity instrumentation, not the separate isolated sample forward timing loop.
Their sync/copy counts must not be attributed to sample routing alone.

[DOCUMENTED] The old research reference loops over each row, projects/sorts/checks
scores separately, copies selected IDs to CPU and invokes k experts separately.
This multiplies small launches and host synchronization with batch size. Its
correctness-first design was never an optimized GPU executor.

## Implementation and parity

New research/grouped_sample_routing.py shares the same model object, encoder,
query, keys, circuits, head and recruited state with the old reference. It batches
queries/scores/softmax, stably selects per-example k, groups selected assignment
slots by expert and calls each expert only on its selected rows. A small dispatch
plan goes to CPU once. Unique slots are restored to row/top-k order before summing,
with no atomic scatter-add reduction. No unselected or dormant expert executes.
Per-batch union size can exceed k, so budget reporting includes both expert union
and effective expert-row work rather than pretending the production router has
the same batch execution budget. All circuits/relations remain allocated.

Exact computed ties choose ascending node ID, as in the old research reference.
Production topk tie semantics are untouched. Supported topology remains q_max=0,
message_rounds=0; construction and forwards explicitly reject other configurations.
The new finite-score check covers the full score matrix before expert dispatch;
the old reference checked each row before dispatching that row. Neither suppresses
nonfinite detection, and production numerical diagnostics are unchanged.

[MEASURED] Full offline suite: 92 passed, one existing sparse-state warning;
final_validation.txt. New tests cover two seeds, three recruited/k ratios, CPU
and CUDA, projected inputs, companions including extreme rows, singleton production
parity (IDs, logits, weights), permutations, repeated deterministic outputs,
parameter and input-gradient parity, exact ties, expert hooks, dormant/unselected
gradient isolation, unsupported inputs/topology, nonfinite rejection and state
preservation. Cross-device comparison loads identical CPU state into the CUDA
model. Frozen tolerance policy is CONTRACT.md; TF32 is disabled for qualification.

[MEASURED] Eighteen final bounded performance cells preserve exact state hashes,
selected IDs and predictions. Max old/new logit delta: CPU 1.1921e-7, CUDA 2.3842e-7;
max weight delta 1.4902e-8 on each device. Tolerances are not permission to change
selected IDs; near-boundary generalization remains unqualified. Each cell records
parameter count, feature hash, structural dimensions, selected expert work and
old/new cost distributions in evidence_cpu.json/evidence_cuda.json.

## Isolated performance evidence

All final measurements use batch 1/4/8, width 8, hidden 16, six nodes, router_dim 8,
three classes, recruited/k 2/1, 4/2, 6/4. TIMINGS.md contains all 18 cells.
Ordinary medians have only three repeats; they are qualification evidence, not
robust representative speedup estimates. Traces are separate instrumented forwards.

At B=8, recruited=4, k=2, CUDA ordinary old/new times are 2.2979/.7145 ms (3.22x).
Both routes perform 16 expert-row evaluations across four selected experts; old
executes 16 expert calls, grouped four. Trace kernel/launch counts fall 241 to 69;
summed kernel durations 363.144 to 111.394 us, launch API host durations 680.087
to 431.633 us. Synchronization APIs fall 23 to 10; copy APIs 45 to 16.
At B=8, recruited=6, k=4, old/new calls are 32/6 and kernel counts 319/81.
These counts and wall improvements support the combined per-row routing/dispatch
mechanism as a cause of overhead ON THESE fixtures. They do not isolate Python
interpreter cost from launch gaps, GEMM grouping or host copies individually, nor
prove how much of the owner's 92.8x representative gap this fixes. Attribution of
the original representative slowdown remains a hypothesis pending manual profiles.

The grouped singleton is slower in every measured ratio, on both CPU and CUDA:
sort/group/restore overhead is not amortized. At B=8/k=2 it is still about 2.13x
slower than the semantically different production batch forward (.3354 ms).
Do not hide these negative results or claim CUDA is always preferable for tiny ops.

## Independent integrity-cost measurements and proposals

The unchanged finite-logit check, gradient scan, parameter scan and canonical
state hashing are profiled independently after a synthetic backward, without an
optimizer. At B=8/r=4/k=2 median CPU costs are .0050/.2591/.3010/.2767 ms;
CUDA costs .0352/1.6995/1.9487/.6556 ms respectively. Each probe's trace and wall
samples are preserved. These are standalone finite synthetic probes: not additive
wall shares, not failure-path timing, and not representative bottleneck rankings.

Ranked future proposals (expected benefits remain unmeasured outside this scope):

| rank | proposal | potential benefit | scientific risk and required parity |
|---:|---|---|---|
| 1 | retain grouped selected-row execution | bounded B>1 benefit measured; larger batches unknown | preserve per-row IDs/weights/logits/input and parameter gradients, expert-row accounting, singleton/companion behavior and state transitions |
| 2 | tune CPU threads per fixed workload | potentially reduce tiny-op contention | low; output/gradient/optimizer/structural and nonfinite parity; fresh-process thread sweep |
| 3 | aggregate finite-status transfers while retaining tensor order | fewer CUDA host synchronizations in scans | high; check every tensor, preserve earliest failure name/context/details and total-norm failure, no deferred optimizer step |
| 4 | batch state transfer from an immutable snapshot | fewer D2H synchronizations in hashing | high; all tensors, exact frozen name/dtype/shape/byte encoding and SHA256, synchronization before host reads; no cached digest substitution |
| 5 | cache only static dispatch/shape metadata with explicit invalidation | reduce repeated metadata work | stale structure risk; bind to recruitment/structural version, never cache input-dependent selection or parameter bytes |

No guard, diagnostic or hash implementation is changed here. Norm/max numerical
reductions and canonical byte order remain protected parity gates, even if a
candidate transfers fewer scalars. Speed alone cannot authorize these changes.

## Risks, provenance and next decision

Dynamic grouping uses sorting, unique counts and a host dispatch plan: residual
sync, allocation and Python expert loops remain. B*k*d_model intermediate memory
can be material; no representative peak-memory qualification, CUDA-graph support,
mixed precision, pathological near-tie or general graph claim is made. Expert-union
cost depends on input routing diversity. Gradient equality is tolerance-based,
not bitwise across GPU kernels. No continual-learning, forgetting or accuracy
benefit follows from evaluation-only invariance.

A first combined test process emitted CPU-only traces for CUDA after CPU profiler
initialization. initial_validation_failure.txt preserves that failure. New tests
use fresh profiling subprocesses; missing CUDA forward kernels now fail closed.
Final bounded trace archives include every final CPU/CUDA trace and operator table;
raw_artifact_hashes.json and SHA256SUMS bind outputs/code. Earlier development
profiling directories remain in /tmp untouched. Existing representative CPU/CUDA
and failed CUDA directories were never modified. Production/base tracked files and
prior evidence packages are unchanged; data/ remains untracked and untouched.
No commit was requested for this phase, so the additive package remains uncommitted.

Historical verdicts remain EXP-H1A-002 INCONCLUSIVE, EXP-H1A-ORDER-001 ORDER_SENSITIVE,
and EXP-H1A-EVALBATCH-001 DIRECTION_FLIP_OBSERVED. No scientific data was loaded.
No production integration, scientific training, representative CUDA profiling,
push or merge occurred. Owner should execute the fresh-path manual commands in
MANUAL.md, compare distributions at representative shapes/diversity, review memory
and launch traces, then separately scope any controlled integration/training study.
