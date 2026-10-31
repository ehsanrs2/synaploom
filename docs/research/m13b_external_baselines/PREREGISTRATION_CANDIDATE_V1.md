# M1.3b — Complete 90-slot preregistration candidate, v1

**Status: `DESIGN_REVIEW_ONLY_UNFROZEN`. NO scientific execution is authorized.** This is a concrete reviewable study *proposal*, not a frozen protocol, training runner, scientific result, or authorization from the owner. Built from merged [PR #36](https://github.com/ehsanrs2/synaploom/pull/36), main `ce0fe5270d406b5f09cadb7d050eaaff737977e8`. Issue [#29](https://github.com/ehsanrs2/synaploom/issues/29). Independently checked source identity on the future execution host, owner approval and independent methodological sign-off are **not yet present**.

## One honest, fully enumerated candidate

`src/synaploom/research/m13b_prereg_candidate.py` builds and validates the complete *candidate* matrix using the existing frozen H1a2 protocol and `m13b_stagewise_budget` resource audit. **No data loading or training occurs.**

| Dimension | Proposed value | Gate |
|---|---|---|
| Primary comparison | `O0`, 10 paired seeds, 3 arms = **30 trajectories** | proposed, pending D1–D4 |
| Order robustness | frozen `O1` and `O2`, 10 identical paired seeds and 3 arms each = **60 additional trajectories** | predeclare before new outcomes; descriptive |
| Seeds | `501..510` new paired confirmation candidates | no engineering training with these IDs |
| Arms | `progressive_reserve`, `all_recruited`, `fixed_sparse_mlp` | only these candidates; RigL/NISPA still NOT admitted |
| Study length | 10 tasks × 20 epochs × 40 B128 batches per task; **8,000 steps/trajectory** | exact B128×39 + B8×1 task geometry |
| Aggregate workload | **90 × 8,000 = 720,000 proposed optimizer steps** | NOT executed; GPU hours unknown |
| Input provenance | frozen CIFAR-100 ResNet18 512D feature SHA-256 `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47` | trusted file on full execution host not rechecked |
| Routing | legacy batch-level top-k under H1a, NOT sample route | production training not altered |
| Fixed sparse | mask seed `20261026` proposal and derived bit SHA-256 | not independently certified or accepted as a fair matched-work method |
| Scientific outcomes | **NONE** | do not interpret planned slots as actual results |

The balanced rotation of arm execution sequence within each fixed class-order × seed block reduces deterministic first-arm bias. It is not randomization proof, thermal control, CUDA kernel benchmarking or an actual fixed host run order. No per-arm execution concurrency or automatic scheduling is available.

## Proposed inferential contract — review, DO NOT use as preapproved inference

One *proposed* primary external O0 contrast: `progressive_reserve - fixed_sparse_mlp`, paired by 10 seed IDs. Proposed co-primary endpoints are (1) final average accuracy and (2) late-task within-task mean accuracy; both higher is better. Proposed 10,000-seed-pair percentile bootstrap replicates (bootstrap seed 1337) with two-sided 95% percentile CIs:
- `FAVORABLE` only when both lower CI bounds exceed zero;
- `UNFAVORABLE` only when both upper CI bounds are below zero;
- otherwise `INCONCLUSIVE`, **if all expected trajectories are numerically valid and available**.
- Missing/invalid/nonfinite seeds, outcomes or arm records are not silently omitted; formal interpretation must stop and preserve adverse/failed evidence.

Scientific reviewers must explicitly evaluate this two-co-primary inference rule's size/coverage with only 10 paired seeds, whether a strict joint success criterion adequately addresses multiplicity, and if bootstrap percentile intervals are appropriate. The `all_recruited` arm remains an important internal mechanistic comparator but is **descriptive** for this *external* primary family, not an additional undeclared confirmatory contrast. O1/O2 are **predeclared descriptive order sensitivity**, not 20 new independent confirmation seeds and not optional result-triggered reruns. Fresh-network late-task tests remain necessary for stronger H6 plasticity claims.

No statistics are run against actual scientific outcomes in this PR. `EXP-H1A-002=INCONCLUSIVE`, `EXP-H1A-ORDER-001=ORDER_SENSITIVE`, and `EXP-H1A-EVALBATCH-001=DIRECTION_FLIP_OBSERVED` remain unchanged.

## Resource matching does NOT imply equal actual execution

The current fixed sparse MLP matches SynapLoom's **physically allocated trainable parameter count** within approximately 0.0026% in the H1a2 setting. Whole-run ideal linear-MAC estimates for progressive and fixed sparse are close, but the current Fixed Sparse implementation actually uses **dense matrix multiplication with a fixed weight mask**; it does not execute true sparse CUDA kernels. The stagewise audit separately records the ideal masked linear MAC proxy and the dense-matmul proxy. Neither includes backward/optimizer, nonlinearity, normalization, softmax, data transfers, dispatch, kernel launch or real GPU wall time.

The previous single short manually run CUDA engineering pilot cannot be extrapolated to a scientifically valid full-run speedup. Claims of equal compute or winner-by-throughput are **NO-GO** until the effective training work is properly measured and reviewed. A reviewer may decide this represents a *parameter-capacity-matched* study only.

## Immutable review and safe commands

```bash
# Build the complete, non-executable plan and write a NEW file
uv run --locked python experiments/scripts/m13b_prereg_candidate.py \
  --review-plan --output /tmp/m13b-registration-CANDIDATE-v1.json

# Check an existing file against the entire known-source candidate,
# including an independently regenerated reference and all slot keys.
uv run --locked python experiments/scripts/m13b_prereg_candidate.py \
  --verify-plan /tmp/m13b-registration-CANDIDATE-v1.json
```

The first command refuses to overwrite an existing file. The second rejects both ordinary edits and edited/re-hashed records that no longer match the frozen-for-review proposal. The SHA-256 of the candidate is a checksum **not a signature, immutable science approval or proof of independently trusted code provenance**. The CLI has no `--execute`, `--approve`, `--freeze`, `--cuda`, `--seed`, `--arm` or `--features` capability.

## Owner/reviewer decisions required before science

**D1** Approve O0 primary plus mandatory predeclared O1/O2 descriptive robustness (90 total), or explicitly restrict to O0 only and limit claims. **D2** Approve the parameter-capacity-matched/ideal-MAC resource reporting tiers and actual GPU workload disclosure, or require stronger actual executed-work matching. **D3** Independently review the single external contrast, two co-primary 95% intervals, per-seed pairing, incomplete-run handling, multiplicity and order robustness. **D4** Approve planned seed IDs, all-arm run order, isolated output roots, negative/failure preservation and measured runtime resource budget. These D1–D4 decisions currently remain **`PENDING`**.

Only after a separately attributable owner decision and method review should another PR create an actual versioned, SHA-pinned **frozen science protocol** and then a narrowly gated, explicit owner-invoked scientific runner. No automatic GPU tasks, no reruns of historical data and no use of confirmatory seeds for preliminary accuracy-based tuning.

**Engineering status: review-ready candidate, verified offline by CI if successful. Scientific status: `NO_GO_CONFIRMATORY_GPU_EXECUTION`.**
