# Integration parity/performance gate v1

This is the acceptance contract for a later implementation. A green existing
research suite is not proof the proposed session/checkpoint/training adapters exist.
Current offline qualification covers the research baselines/probes only.

## Matrix and required evidence

| axis | required cases | current evidence | later implementation obligation |
|---|---|---|---|
| device | CPU, CUDA, identical CPU state loaded on CUDA | grouped v1 tests and new probe tests | fresh process per CUDA profiler; actual device required, skips do not qualify a release |
| batch | 1, 2, 8, 32, 128, final short evaluation batch | prior bounded/representative artifacts and tests span these | proposed session tested at every size; manual performance at 1/32/128 |
| capacity/k | R/K=4/1,4/4,16/4; dormant reserve; recruited=K | existing representative sweep; probe R4/R16,K4 and R4,K1 | all route modes, both seeds7/19, graph changes between transactions |
| routing | default batch, sample row, sample grouped, qualified singleton optimization | existing baselines and probes | default output/type/trace is legacy unchanged; invalid option combinations reject before dispatch |
| composition | singleton, distinct random/extreme companions, arbitrary row permutation | existing/new seeded tests | compare logits, ID-aligned weights, IDs, query and argmax; identical anchor features |
| backward | same loss/upstream gradient, shared/input/node gradients, recruited unselected/dormant, consolidated scaling | old/grouped/probe gradient tests; existing scaling tests | loss, gradients, selected-row work, finite/error behavior and no unintended updates |
| state | forward/eval/backward hashes, mode/RNG/input identity; recruitment/consolidation between calls | baseline/probe state hashes; existing diagnostic guards | wrapper mode/RNG/exception restoration and stale transaction rejection |
| checkpoint | plain strict state_dict and full envelope/optimizer/controller/RNG | production loader inspected; NOT new adapter-qualified | no key changes or copies, metadata fail-closed, default replay, explicit counterfactual and resume mismatch cases |
| topology | q_max=0 and rounds0; reject either nonzero | baseline constructor/runtime checks | session checks every call and after load/device move |
| ties/nonfinite | exact ties, tied cutoff, nonfinite score/logit/loss/gradient/parameter, norm overflow | existing ties/numerical tests and probe tie cases | stable node-ID sample tie rule; untouched production rule; same first-failure identity/context/phase |
| work/cost | expert union, expert calls, B*K row work, budgets, CPU wall, CUDA launches/kernel time, memory | all prior/new profile summaries and hashes | no extra expert/row; peak-memory and multi-process wall gates below |

## Numerical and semantic pass criteria

Use frozen float32 tolerances: forward/query/weights CPU atol=rtol=1e-6;
CUDA and CPU/CUDA 1e-5; input/parameter gradients 2e-5. Same parameters, initial
capacity and features, fixed seeds7/19, TF32 off. Class IDs and selected node IDs
must match exactly on qualified fixtures; compare weights by node ID. Report
maxima, score/logit margins and all failures, including near ties. Near-tie failure
is a limitation requiring a new decision, not a reason to relax selection equality.
Production singleton parity excludes exact ties with different legacy topk ordering.

Row/grouped sample routes must agree on logits/loss/gradient and B*K work within
tolerance. Batch routing is not a multi-example sample parity oracle. Repeated
outputs must be deterministic in a fixed supported environment. Test supported
strict-determinism kernels explicitly; fail closed if unsupported rather than
disabling policy. Hashes must be EXACT for read-only state; optimizer-step numerical
parity, if later authorized, requires separate parameter/buffer/optimizer/controller
transition comparisons with predeclared tolerances and identical event plans.

No expert outside the selected union may run; each grouped expert gets only the
rows selecting it. Expert parameter grad=None when never executed; unselected key
rows have zero gradients. Verify accounting by hooks outside timing. No optimizer
step is authorized by these backward diagnostics. Persistent RNG, modes and source
features must remain unchanged on normal and exceptional forwards.

Checkpoint and capability rejection must occur before any parameter/RNG/statistic/
optimizer mutation. Training integration remains a separate NO-GO even if all
inference gates pass. Tests must not rely on accidental TypeError for safety.

## Manual performance pass criteria (not passed by this design phase)

Freeze hardware, versions, threads, TF32/determinism, model/feature hashes, shape,
capacity and residency. Same input/state and per-example K per comparison. At least
five fresh processes per candidate/baseline, two warmups and >=10 ordinary repeats
per cell, interleaved/randomized external run order. Keep all raw durations; report
paired per-process median ratios plus p95 values and environment/load. A profiler
trace is separate and must contain CUDA kernel/runtime lanes. Keep hash checks
outside isolated-forward timing and separately report full guard/evaluation cost.

All cells must pass correctness and accounting first. Proposed promotion thresholds:
- Legacy/default delegation: no new routing ops or allocations; process-median
  wall regression <=5%, p95 regression <=10% against unchanged production.
- Grouped session B32/B128: process-median <=1.05x qualified grouped baseline and
  p95 <=1.10x; kernel launches no greater than baseline; no increased expert work.
- Singleton fallback: <=1.05x sample row-reference median and <=1.10x p95, and
  >=10% median improvement over grouped B1 in at least four of five paired
  processes on each supported device, with no per-cell regression >5%.
- Each additional dispatch optimization: >=5% median improvement on its target
  R16/K4 B32/B128 cells in at least four of five process pairs, <=5% regression in
  other cells, launches no greater, no extra experts/rows or persistent parameters.
- Peak allocated/reserved CUDA memory must be recorded; >=10% regression requires
  a separate design decision. No OOM or stale artifact reuse is acceptable.

These thresholds are predeclared for future promotion, not retroactive claims
about the one-process diagnostic probes. Row-reference is a semantic oracle,
not a performance target at large B. Never require grouped to equal the batch
router's latency by executing fewer per-row experts or changing weights.

Trace metrics distinguish ordinary CPU wall, process CPU, CUDA stream elapsed,
summed kernel durations, host launch/synchronization durations and copies. They
overlap; no additive decomposition or inferred synchronization cost by subtraction.
Timing includes resident-forward dispatch D2H, excludes initial H2D, and excludes
outside hash checks. Evaluation H2D-inclusive timings are a separate workload.
Finite checks and hashing must not be disabled to reach any threshold.
