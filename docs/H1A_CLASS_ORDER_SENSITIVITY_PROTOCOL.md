# EXP-H1A-ORDER-001 — Class-Order Sensitivity of Controlled Reserve Recruitment

**Protocol date:** 2026-10-11  
**Status:** preregistered; runner implementation pending; not run  
**Parent result:** EXP-H1A-002 — COMPLETE_NUMERICALLY_VALID; INCONCLUSIVE  
**Question:** Is the paired effect of `progressive_reserve` versus `all_recruited` directionally consistent across different fixed class orders?

## 1. Scope and non-claims

This is a robustness study for the H1a mechanism-isolation comparison. It is not a replacement, extension, or re-analysis of the primary EXP-H1A-002 protocol. EXP-H1A-002 remains the primary result and its preregistered verdict remains **INCONCLUSIVE**, regardless of this study's outcome.

This study changes only class order and uses a fresh seed block. It does not test the automatic recruitment trigger (H1b), end-to-end feature learning, replay, consolidation, structural edges, or message passing.

Evaluation batch size remains fixed at 256 in this study. Batch-size sensitivity is a separate measurement-focused follow-up because batch-level routing makes predictions depend on evaluation-batch composition.

## 2. Frozen representation and model setup

Use exactly the EXP-H1A-002 feature and model settings:

- Dataset: CIFAR-100 with frozen torchvision ResNet-18 `IMAGENET1K_V1` features.
- Feature bundle SHA-256: `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47`.
- Feature dimension: 512; 10 experiences x 10 classes; 100-way output with unseen logits masked.
- Model: `d_model=128`, `d_hidden=256`, `num_nodes=16`, initially recruited=4, `top_k=4`.
- Graph edges/message passing: `q_max=0`, `message_rounds=0`; no replay, no consolidation, no pruning or recycling.
- Optimizer: SGD, learning rate=0.01, momentum=0.9, weight decay=0.0005; no scheduler.
- Training: 20 epochs/task, batch size=128.
- Evaluation: `eval_batch_size=256`, deterministic class/task evaluation order.
- Primary variants only: `progressive_reserve` and `all_recruited`.
- Paired comparison: `progressive_reserve - all_recruited`.

Do not tune hyperparameters, change model definitions, add variants, or select orders after inspecting results.

## 3. Fixed class orders

Every order is a permutation of the 100 CIFAR-100 class IDs, partitioned sequentially into 10 tasks of 10 classes. The literal orders below are normative; implementation code must not regenerate them at runtime.

### O0 — primary order

Compact-JSON-list SHA-256 fingerprint: `81d7f0637b8e61477c1ef80b7eef33270aae8d81dfcaffad6e58252fd004b633`.

Re-use the exact frozen `H1A_CLASS_ORDER` from `src/synaploom/protocols/h1a.py` (the same order used by EXP-H1A-002):

```python
(
    75, 59, 1, 39, 93, 87, 37, 51, 7, 34,
    21, 41, 19, 86, 27, 89, 44, 0, 20, 14,
    49, 91, 30, 69, 23, 70, 62, 9, 29, 76,
    6, 15, 71, 32, 96, 95, 24, 99, 50, 66,
    80, 88, 53, 98, 68, 10, 97, 18, 73, 61,
    4, 45, 40, 72, 74, 63, 31, 28, 5, 92,
    90, 25, 3, 38, 48, 79, 94, 35, 65, 58,
    13, 60, 47, 54, 57, 85, 82, 42, 83, 43,
    56, 8, 17, 2, 84, 78, 16, 36, 67, 64,
    46, 81, 52, 55, 77, 12, 22, 33, 26, 11,
)
```

### O1 — alternate order A

Compact-JSON-list SHA-256 fingerprint: `c0dc7d8b234ec06670526cbc8d917f0e7e14ed9bbaee640272eab95573f1f8ed`.

Generated once with Python standard-library `random.Random(20261006).sample(range(100), 100)`, then frozen here. Its task grouping is:

```text
[63, 10, 86, 36, 96, 65, 25, 71, 29, 67]
[80, 90, 69, 4, 13, 1, 30, 14, 98, 32]
[59, 73, 54, 47, 27, 0, 81, 20, 99, 17]
[75, 16, 21, 58, 7, 3, 12, 45, 87, 97]
[11, 70, 94, 24, 31, 82, 78, 22, 77, 2]
[89, 55, 74, 40, 43, 15, 41, 85, 6, 57]
[50, 26, 48, 53, 39, 61, 52, 46, 88, 18]
[84, 9, 28, 44, 5, 83, 38, 60, 66, 91]
[33, 64, 51, 72, 56, 49, 95, 23, 37, 19]
[35, 8, 68, 76, 62, 92, 93, 79, 42, 34]
```

### O2 — alternate order B

Compact-JSON-list SHA-256 fingerprint: `b610995c70c2269fb28e7492f1c1eb2870703fa134f967e185ca02dbd44e2e63`.

Generated once with Python standard-library `random.Random(20261007).sample(range(100), 100)`, then frozen here. Its task grouping is:

```text
[29, 45, 9, 37, 77, 79, 90, 72, 12, 35]
[16, 64, 71, 13, 86, 60, 0, 80, 24, 15]
[69, 21, 22, 14, 40, 25, 46, 61, 30, 83]
[43, 33, 6, 27, 7, 67, 11, 2, 1, 62]
[32, 92, 99, 65, 75, 54, 18, 3, 20, 55]
[52, 49, 47, 57, 91, 59, 81, 44, 74, 8]
[76, 34, 93, 94, 50, 5, 42, 23, 84, 87]
[39, 96, 63, 73, 19, 58, 85, 66, 95, 56]
[26, 82, 88, 97, 98, 17, 36, 28, 41, 70]
[10, 31, 38, 4, 89, 48, 51, 53, 78, 68]
```

The fingerprints above are SHA-256 over the UTF-8 encoding of each order as a compact JSON integer list (`json.dumps(order, separators=(",", ":"))`, with no trailing newline); they are protocol fingerprints, distinct from the feature-stream audit's class-order hash. The O1/O2 generation seeds are provenance only. The frozen literal orders and fingerprints are authoritative even if a future Python version changes its random-number implementation.

## 4. Seeds and run matrix

Use exactly these fresh paired seeds:

```text
301, 302, 303, 304, 305, 306, 307, 308, 309, 310
```

Use the same seed IDs across all three orders and both variants. Seeds are disjoint from the invalid primary run (0–9), the numerical calibration (101–103), and EXP-H1A-002 (201–210).

Expected matrix:

```text
3 class orders x 10 paired seeds x 2 variants = 60 runs
```

Pair treatment and control within every (order, seed) block. Do not omit or replace seeds based on accuracy. Any detected NaN/Inf or artifact-integrity failure invalidates the entire EXP-H1A-ORDER-001 study; preserve the failure artifact and do not report partial aggregates as results.

## 5. Outcomes and reporting

The two co-primary outcomes stay identical to EXP-H1A-002:

1. final average accuracy;
2. late-task within-task mean accuracy.

For each order and each co-primary outcome, report treatment/control values by seed, paired deltas in percentage points, mean paired delta, and a paired bootstrap 95% percentile CI with 10,000 resamples. Use bootstrap RNG seeds 1337, 1338, and 1339 for O0, O1, and O2 respectively.

Also report a class-order-marginal pooled effect, weighting O0/O1/O2 equally. Because the same seed IDs recur across order blocks, bootstrap by **seed cluster**: for each of 10,000 replicates, sample 10 seed IDs with replacement and carry every selected seed's deltas from all three orders into that replicate. Use bootstrap RNG seed 7331. Do not treat the 30 order-by-seed blocks as independent.

Report descriptive per-order and pooled summaries for average forgetting and backward transfer, but do not use them for the registered classification.

## 6. Registered sensitivity classification

For each co-primary outcome, all effect signs refer to `progressive_reserve - all_recruited`; positive means higher accuracy for progressive reserve.

- **ORDER_ROBUST_FAVORABLE:** both pooled co-primary 95% CIs are strictly above zero, and all three order-specific mean deltas are positive for each co-primary outcome.
- **ORDER_ROBUST_UNFAVORABLE:** both pooled co-primary 95% CIs are strictly below zero, and all three order-specific mean deltas are negative for each co-primary outcome.
- **ORDER_SENSITIVE:** for at least one co-primary outcome, the order-specific mean deltas include at least one positive and at least one negative value.
- **INCONCLUSIVE:** none of the preceding rules is satisfied.

This classification summarizes class-order robustness; it does not retroactively upgrade, downgrade, or replace the primary EXP-H1A-002 verdict. `ORDER_SENSITIVE` is a descriptive directional-heterogeneity flag based on per-order point estimates, not a claim that the order-by-treatment interaction is statistically significant. Order-specific confidence intervals are reported transparently but are not separate primary-hypothesis tests.

## 7. Validity and provenance checks

The runner must validate before training:

- feature SHA-256 equals the frozen SHA listed above;
- each class order has exactly 100 unique IDs in `0..99`;
- the three class orders are distinct;
- variants are exactly the two registered variants;
- seed set is exactly 301–310;
- run count is exactly 60;
- `eval_batch_size=256` and all other frozen settings match this document.

Each per-run artifact must record experiment ID, order ID, literal class order, seed, variant, resolved protocol, feature SHA/audit, environment, Git commit, and numerical diagnostics. The aggregate must contain the per-order paired comparisons, seed-cluster pooled comparisons, registered sensitivity classification, and a complete run manifest.

## 8. Planned command

The dedicated runner will be added in a subsequent implementation PR. Its required interface is frozen as:

```bash
uv run --locked python experiments/scripts/m1_3_h1a_order_sensitivity.py \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt \
  --output runs/m1_3_h1a_order_sensitivity \
  --device cuda
```

A `--dry-run` mode must emit the protocol, order fingerprints, seed list, and `expected_runs=60` without loading feature data or training. Runtime overrides for class orders, seeds, variants, batch sizes, or optimizer settings are not permitted.

## 9. What follows this study

This study isolates class-order sensitivity. Evaluation-batch-size sensitivity remains a separate preregistration. Because routing is batch-context dependent and the late-task co-primary depends on epoch-level evaluations, the next batch-size study should evaluate identical trained model states at multiple evaluation batch sizes rather than conflate evaluation batching with a new training condition.

The scientific sequence is: merge this preregistration, implement and test the fixed runner, run all 60 combinations, audit the artifacts, then update `PROJECT_STATE.md` and `docs/EXPERIMENT_LOG.md` with the outcome. EXP-H1A-002 remains the frozen primary result.
