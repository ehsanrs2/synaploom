# M1.3b stagewise budget and fairness audit — analytical v1

**Status: DESIGN / RESOURCE FEASIBILITY ANALYSIS ONLY. Not a frozen experiment; no training or GPU execution.** This follows merged [PR #35](https://github.com/ehsanrs2/synaploom/pull/35), in which the independently owner-supplied RTX 4090 train-only pilot succeeded, but all D1–D4 scientific decisions remained pending.

## Problem addressed

The fixed-sparse candidate was constructed against SynapLoom's **initial** R=4, K=4 per-batch linear-MAC proxy, while the actual progressive treatment recruits one additional Expert **after the first optimizer step of each subsequent task**. An externally valid resource fairness claim needs to account for all 10 tasks, full batches and the final short batch, not only a single B128 measurement.

The new standalone code `src/synaploom/research/m13b_stagewise_budget.py` computes:
- Frozen H1a2 10 × 10-class task stream with **5,000 training examples per task** (500/class), **20 epochs**, B128 giving **39 full batches of 128 and one final batch of 8 per epoch**.
- **800 optimizer steps/task; 8,000 steps per arm; 24,000 steps across the three proposed arms for one class order**. These are *proposed* workload counts, not executed training or cost estimates.
- The exact ideal **linear MAC estimator** already in `synaploom.research.budgets`, independently checked at initialization for R=4 and R=16. It counts encoder/query/one batch-level candidate score pass/K expert MLPs/head, excluding nonlinearities/softmax, backward, parameter updates and data movement.
- Per-task theoretical treatment trajectory R=4 for task 0; on task t=1..9 its **first minibatch has old R=3+t**, and following minibatches have new R=4+t. The last task ends at R=13. `all_recruited` retains R=16.
- `fixed_sparse_mlp` has fixed active masks and both the **ideal sparse linear-MAC count** and the **physically dense-mask body-matmul linear-MAC proxy**. This is *not* measured GPU FLOPs: actual torch `F.linear(weight*mask)` computes a full dense GEMM, and backward/kernel overhead is omitted.
- A physical parameter comparison against both original full-pool SynapLoom variants. In the frozen H1a2 configuration: SynapLoom **1,172,196**, Fixed Sparse **1,172,226** total trainable parameter elements, a difference of 30 (about 0.00256%). GPU pilot peak memory differences are recorded in `GPU_PILOT_EVIDENCE.md`, not inferred from this analysis.
- Exact per-task and aggregate numerical matching errors, plus `actual_gpu_compute_observed: null` to prevent a linear-MAC proxy from being mistaken for measured hardware work.

## Why this cannot authorize a claim of compute parity

Physical trainable parameter elements, nonzero masked weights, selected experts, forward linear MACs, backward/optimizer work, kernel launches, allocator memory, and wall time are **different resource axes**. Masking a large dense matrix is not computational sparsification on PyTorch CUDA kernels. Legacy routing is batch-level and the candidate's data/architecture differs from original RigL/NISPA. Even tiny theoretical MAC differences cannot establish fair training cost, equivalent model capacity, comparable task dynamics, or an advantage in accuracy/plasticity.

The aggregate theoretical MAC ratios and per-task trends are **diagnostic only**, computed deterministically and without loading a single sample of the frozen CIFAR data. They must not be substituted for independent hardware profiling or the initial vs progressive stage resource curves in a preregistration.

## Safe command

```bash
# Does not access training/eval data, CUDA, or run optimizer updates:
uv run --locked python experiments/scripts/m13b_stagewise_budget.py --audit

# Optional output JSON must not already exist; no overwrite:
uv run --locked python experiments/scripts/m13b_stagewise_budget.py --audit \
  --output /tmp/m13b-stagewise-audit-v1.json
```

The CLI offers **no** execution, seed, order, CUDA, features or science options. Result digest is a content checksum, not owner approval. The tests require exact task/batch counts, correct delayed recruitment, independent consistency with original estimator, deterministic SHA, all three arms, resource-tier separation and no global CPU RNG changes.

## Decision after review

This budget audit is one necessary input to the separate [OWNER_DECISION_GATE.md](OWNER_DECISION_GATE.md). **All D1–D4 remain PENDING**. Recommended protocol structure: O0 primary 30 trajectories, with O1 and O2 predeclared robustness adding 60, but do not interpret document PR merge as an approval or create confirmatory GPU jobs.

Before any science:
1. Independent reviewer verifies estimator omissions, masked dense hardware work and whether matched *physical parameters + ideal MACs* is a scientifically fair comparison tier, with measured GPU work disclosed separately.
2. Owner approves O0+O1/O2 scope, 501–510 seeds, exact masks, 3 candidate arms, the single external contrast/co-primary statistical rule and resource budget/cost.
3. Frozen hash-bound protocol, execution-ready runner, trusted data provenance and explicit user authorization are separately required.

Historical H1a verdicts `INCONCLUSIVE`, `ORDER_SENSITIVE`, and `DIRECTION_FLIP_OBSERVED` remain unchanged.
