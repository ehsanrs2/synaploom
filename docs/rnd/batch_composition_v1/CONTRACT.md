# Exploratory batch-composition diagnostics v1

Frozen before fixture execution, 2026-10-21. Base a5453e20ad6a67ed6df8f9566bbc2529068142c4.
This is exploratory R&D, not a revision of any registered H1a protocol.
Historical INCONCLUSIVE, ORDER_SENSITIVE and DIRECTION_FLIP_OBSERVED results are frozen.

For each immutable input row ID, compare a singleton reference to batches containing
that exact row plus distinct companions (never the anchor), at sizes 2, 4 and 8.
Companion conditions: same class, other classes, and mixed classes. Use a private
seeded CPU generator; record every member ID and input/label SHA256. Reject
insufficient pools, duplicate IDs, nonfinite logits and any state_dict mutation.
Reverse each batch, find the anchor by ID, and compare against the original order.
Use eval mode and inference_mode, no optimizer or statistic updates. Hash before
and after every forward, including exceptional forwards. Restore module modes.

Report each anchor separately, and macro means over unique anchors per condition.
Companions are context, never additional accuracy observations. Singleton accuracy
and condition accuracy use identical unique anchors. No pooled repeated-sample
accuracy claim. All classes are visible; this is not a class-incremental score.
Report set Jaccard, intersection count, node-ID-aligned zero-filled weight L1,
query L-infinity, logits L-infinity, confidence (max softmax) signed delta, and
prediction flips. A = selected-set change; B = aligned weight change > 1e-6;
C = logit change > 1e-6 and separately argmax flip. Permutation uses atol/rtol
1e-6; report residuals without suppressing them. Near ties can change top-k.

Synthetic qualification: seeds 7 and 19, 24 rows, three classes, dimension 8,
6 nodes, top_k=2, progressive_reserve early=2 and later=6 recruited;
all_recruited=6; dense batch-independent control. Later reserve is constructed
by recruitment before measurement, not trained or a recovered trajectory.
Random labels/weights do not establish task quality. No historical trained state
is assumed available. Manual trained runs require explicit architecture config,
strict state_dict load, immutable tensor dataset and checkpoint hashes.

Performance protocol: separate ordinary wall timing from profiler timing; fresh
identically seeded synthetic engine per run, fixed SGD lr=.001 and no controller.
Keep numerical checks, gradient diagnostics and hash guards. Warm up 2 steps,
measure 5; synchronize CUDA at wall boundaries only. Record process CPU/wall,
thread settings, device, dimensions and torch version. Compare thread counts
1/2/4/default in separate processes manually. Profile actual TrainingEngine,
router/executor, forward/backward, optimizer, numerical checks, CPU transfers,
evaluation and state hashes. CPU and optional CUDA activities with shapes and
memory, Chrome trace plus operator and named-region tables. CUDA kernels and
launch/sync overhead require CUDA traces, not extrapolation from CPU timing.
No automatic GPU run; no performance changes before a separate output/gradient/
state-transition parity gate. Manual execution commands are not auto-run.
