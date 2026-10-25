# Offline R&D qualification report

2026-10-21; branch rnd/batch-composition-performance; base a5453e2 (PR #20).
No production source, historical protocol or recorded historical result changed.
Untracked data/ retained. No commit, merge, GPU experiment or historical retraining.

## Mechanism analysis

[MEASURED FROM SOURCE] SparseRouter projects each encoded example and **means over
batch dimension**. Companion features therefore change the shared query. Scores
are recruited routing-key dot products divided by sqrt(router_dim); top-k picks
node IDs and softmax produces a shared weight vector. With recruited==top_k the
selected set is fixed but scores/weights need not be. Exception: top_k=1 gives
constant weight 1; identical projections or equal score shifts can also hide B.

GraphExecutor applies the selected circuits independently to every row, then
sums their representations with that shared weight vector. Selection changes
can also change selected-endpoint edges when message rounds are enabled. H1a
uses message_rounds=0, so graph message passing is not needed for this effect.
The shared weighted representation passes through the head to logits and argmax.
A/B/C are distinct: changing queries need not change sets, changing weights need
not flip predictions. DenseControlModel uses per-row LayerNorm and linear/body/
head operations; it has no batch aggregation. Floating arithmetic still permits
small batch-shape/order residuals, so invariance is tolerance-based.

SynapLoomModel defaults update_statistics=False. Continual evaluation iterates
unshuffled tensor slices and evaluates with all configured visible classes;
changing eval batch size changes query contexts and final short-batch membership.
Its existing multi-size state guards freeze weights, not the predictions.
H1a progressive_reserve uses scheduled recruitment and TrainingEngine; all_recruited
starts with all nodes and uses the same engine without a controller. Dense H1a
controls use budget-selected widths; this tiny dense fixture is only a mechanism
control and makes no parameter-matching claim. Historical H1a top_k=4 with early
recruited=4 is exactly the fixed-set, variable-weight case.

## Qualified synthetic measurements

Source: evidence/qualified_seed{7,19}/results.json. Each variant/seed has 24 unique
anchors and 9 conditions = 216 correlated comparisons. All state hashes match.
No historical .pt/.pth/.ckpt files were found under runs; PROJECT_STATE.md also
states checkpoints were not saved. These states are untrained random initializations
and constructed capacity buffers. Early/later is capacity, not training age.
Later reserve and all_recruited share learned weights and routing behavior by
construction; they are not independent replications.

| seed / state | selected-set changes | max weight L1 | max logit delta | flips / 216 | condition accuracy range |
|---|---:|---:|---:|---:|---:|
| 7 early reserve | 0% | .020069 | .002116 | 0 | .4583–.4583 |
| 19 early reserve | 0% | .007848 | .001193 | 0 | .3333–.3333 |
| 7 later reserve / all recruited | 63.43% | 2.00000009 | .476931 | 7 | .4167–.4583 |
| 19 later reserve / all recruited | 67.13% | 2.00000009 | .364589 | 15 | .3750–.4167 |
| 7 dense | 0% | 0 | 1.19e-7 | 0 | .3750–.3750 |
| 19 dense | 0% | 0 | 1.19e-7 | 0 | .2083–.2083 |

Weight L1 slightly above 2 is float rounding. Max within-batch permutation logit
residual is 1.79e-7; qualified rows pass permutation checks. Confidence deltas,
queries, Jaccard/intersection, raw logits, aligned weights and condition-level
accuracy are in the evidence. Aggregate flip counts are repeated-context counts,
not independent examples or a population flip estimate. Random-label accuracy
has no scientific accuracy meaning. Initial evidence/seed7 and seed19 outputs
are preserved development evidence, superseded because mixed sampling initially
did not guarantee another class. Only qualified_seed* supports the table.

## Performance findings and prepared protocol

[DOCUMENTED] TrainingEngine checks logits/aux/router/context/loss, scans gradients
twice, updates importance, scales protected gradients, steps the optimizer, scans
parameters and updates utilization. Numerical diagnostics use scalar host reads
and tensor reductions. GraphExecutor transfers selected IDs to CPU; enabled
message rounds additionally iterate edges with .item(). Hashing copies every
state tensor to CPU, encodes bytes, and SHA256s on CPU. Evaluation repeatedly
forwards, checks finite values and transfers scalar loss/correct counts.
These are plausible costs, not measured dominant bottlenecks.

[QUALIFIED] Tiny CPU profiler integration test runs the actual engine, backward,
optimizer, guards and evaluation; ordinary and instrumented paths end in the
same full state hash. Named regions and operator/Chrome trace export work.
[UNKNOWN] Representative wall-time shares, GPU kernel/launch/synchronization
costs, transfer costs and oversubscription. No final profiling command was run.
MANUAL.md specifies separate ordinary/profiler runs, thread sweeps, trace analysis
and limits of process CPU/wall utilization. environment.json records local defaults.
Default thread count alone cannot prove oversubscription.

| priority / candidate | expected benefit (unmeasured) | scientific risk | complexity | determinism compatibility |
|---|---|---|---|---|
| 1 CPU thread cap sweep | potentially high for tiny ops | low | low | same algorithm; verify tolerances |
| 2 cache host structural metadata per structural version | medium if sync dominates | medium: stale structure | medium | explicit invalidation and exact IDs |
| 3 batch numerical reductions / fewer host reads | potentially high on CUDA | high: fail-fast order/details | high | preserve every check and failure semantics |
| 4 vectorize expert execution | potentially high | high: reduction/order changes | high | gradient and output parity needed |
| 5 optimize state-hash transfers | workload-dependent | high: encoding/guard drift | medium | exact canonical bytes/hash required |

Do not remove guards or checks. A separate parity gate must cover fixed-input
logits, selected IDs, ID-aligned weights, loss, gradients, optimizer state and
post-step state transitions, including recruitment/consolidation/edges and
nonfinite failure behavior. Require ordinary wall improvement on representative
workloads and unchanged frozen scientific metrics before proposing promotion.

## Recommendation and remaining uncertainty

GO for a separately scoped sample-level routing prototype with a frozen contract
and companion/permutation invariance tests. NO-GO for production replacement or
scientific-benefit claims: batch routing is current model behavior, and sample
routing changes semantics, execution/accounting and possibly active budgets.
First implement an isolated per-example top-k/weight reference and validate its
execution/accounting against a singleton oracle; re-register any scientific run.

GO for manual performance diagnostics and low-risk thread-count experiments with
a parity gate. NO-GO for runtime changes or removing checks before representative
CPU/GPU profiles and parity evidence. Trained-state prevalence, real task impact,
near-tie stability, edges/message rounds, GPU tolerance and representative speedups
remain unmeasured. Stop after offline qualification and hand off MANUAL.md.

## Validation and Git diff summary

`uv run --locked --offline pytest`: **63 passed**, one existing sparse-tensor
warning, 8.22 seconds (validation.txt). New tests cover deterministic grouping,
private RNG isolation, anchor content/ID identity, shared-sample accounting,
dense invariance, fixed-set/nonzero weights, zero aligned-weight differences,
permutation consistency, state mutation even on exceptions, duplicate-ID rejection,
strict checkpoint CLI loading, output overwrite rejection and profiler parity.
The checkpoint CLI test uses a newly created synthetic state_dict, not trained
scientific evidence. The qualified evidence audit passes all 1,728 anchor-condition
rows across eight variant/seed cells. CPU defaults: 16 intraop and 16 interop
threads, 24 CPUs in affinity. This motivates a sweep, not an oversubscription claim.

All tracked base files compare byte-unchanged to a5453e2; `git diff --check` passes.
Changes are additive/untracked: two research modules, two manual CLI scripts,
three test files, and this docs/rnd/batch_composition_v1 package (contract, report,
commands, environment, validation and four preserved evidence result files).
`git diff --stat` alone is empty because new files have deliberately not been
staged. data/ was already untracked and remains untouched. Package file hashes
are recorded in SHA256SUMS; the frozen contract SHA256 is
39c2e67b3f33c7340e122a40751302656b92892aff57ddc700ec8e49959ab37a.
