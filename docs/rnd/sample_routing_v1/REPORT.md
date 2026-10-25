# Offline qualification and decision

2026-10-25. GO for later controlled integration design and manual profiling;
NO-GO for production replacement or scientific effectiveness conclusions.

[MEASURED] Existing suite independently passed 63 tests. Final suite: 72 passed,
one existing sparse-state warning (validation.txt). All prior SHA256SUMS entries
verify. Both qualified historical synthetic JSON packages regenerate exactly,
including every field across 1,728 rows; provenance_audit.json records all eight
cells and state hashes. Original package files and base production files remain
unchanged. These are initialized synthetic states, not trained checkpoints.
Development evidence remains preserved and is not promoted to qualified evidence.

[MEASURED] New evidence/results.json records 288 companion comparisons over four
seed/capacity cells, full model hashes, singleton deltas and five tiny CPU wall
samples per mechanism. New tests additionally cover extreme companions, arbitrary
permutation, independent dense control, repeated calls, fixed-set/variable-weight
negative control, exact tie handling, dormant eligibility, call hooks and backward
isolation. All use the same underlying routed parameters. Comparisons in each
cell are correlated; random-label accuracy is not a scientific outcome.

[DOCUMENTED] Parameter budgets are identical within each paired cell, including
unused relation parameters; evidence records totals. Both perform 24*2 expert-row
evaluations. Batch routing calls two experts on 24 rows each; reference calls two
experts separately for every row (48 calls). Across a batch sample routing may
use a larger union of experts, despite matched per-example k. Tiny measured wall
costs describe this reference only, not representative performance or speedups.

[QUALIFIED] New representative-shape profiling runner passes tiny CPU integration
and CLI smoke (profile_smoke). Unchanged engine checks remain active and ordinary
and instrumented final hashes agree. Defaults are shape proxies, not qualified
historical workload representativeness. No representative CPU or CUDA profiling,
GPU scientific execution, optimizer-based scientific training or accuracy study ran.

Ranked proposals, benefits unmeasured:
1. CPU thread sweep: potentially high benefit on small operators; low scientific
   risk; require output/gradient/state-transition parity across thread settings.
2. Cache host selected/structural metadata: medium potential if sync dominates;
   stale-state risk; exact IDs and structural-version invalidation required.
3. Combine numerical host reductions: potentially high CUDA benefit; high risk
   to failure order and diagnostics; preserve every finite/gradient/parameter check
   and nonfinite failure semantics.
4. Group selected sample rows by expert: potentially high prototype speed benefit;
   numerical/reduction and union-budget risk; singleton, companion, gradient and
   state parity plus kernel-shape tolerance qualification required.
5. Optimize state-hash transfers: workload-dependent benefit; high integrity risk;
   identical canonical bytes and SHA256, including exceptional guards, required.

Training-time effectiveness, forgetting, continual learning, near-tie stability,
GPU numerical behavior and general graphs are UNKNOWN. Evaluation invariance does
not establish learning benefit. Exact ties differ from production topk semantics
as declared in CONTRACT.md. This reference deliberately avoids changing production
statistics/training semantics and is not a TrainingEngine drop-in.

Next decision: owner runs manual profiles on explicitly frozen representative
shapes/workloads, reviews traces and cost distributions, then scopes a controlled
integration proposal with full output/loss/gradient/optimizer/structural transition
and nonfinite parity gates. Any scientific experiment needs a separate protocol.
Stop here after offline qualification; no push or merge.
