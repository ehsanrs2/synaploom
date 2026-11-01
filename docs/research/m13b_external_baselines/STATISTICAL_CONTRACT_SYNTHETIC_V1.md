# M1.3b — Statistical decision contract qualification (synthetic only)

**Status: ENGINEERING TEST / SCIENCE EXECUTION NO-GO.** This is a reviewable implementation of the *proposed* inference logic, not a frozen scientific decision, actual result, preregistration authorization or scientific outcome pipeline. Continues merged [PR #37](https://github.com/ehsanrs2/synaploom/pull/37) and [Issue #29](https://github.com/ehsanrs2/synaploom/issues/29). All D1–D4 remain **PENDING**.

## Why this PR

The 90-slot candidate registration is precise, but before any real CIFAR-100 confirmation, a scientific analysis must reject:
- incomplete/missing seed–arm–order cells, duplicate slot keys, wrong O0/O1/O2 fingerprints, wrong comparison arms and unexpected seeds;
- failed/nonfinite scores and post-hoc seed dropping;
- post-hoc relabeling of the all-recruited control, order-specific favorable selective reporting, or treating 30 order-by-seed cells as independent 30 seeds;
- unsupported claims that one good endpoint proves both final average accuracy and late-task within-task accuracy increased.

The independent methodology review is NOT considered complete merely because the code passes synthetic unit tests.

## Proposed scientific estimand — not yet approved

Under the proposed O0-only external inferential family, the ten paired seed differences are:
`delta_seed,metric = metric(progressive_reserve, O0, seed) - metric(fixed_sparse_mlp, O0, seed)`.

Both `final_average_accuracy` and `late_task_within_task_mean_accuracy` are measured as fractions in [0,1] and higher is better. The candidate bootstrap uses **10,000 draws** with **the same sampled seed index list across both endpoints per draw** (seed 1337), matching the prior historical percentile-index convention. Two-sided marginal 95% percentile CIs are recorded. Proposed classifications:

| Both endpoint intervals | Toy classification |
|---|---|
| Both lower bounds strictly > 0 | `SYNTHETIC_FAVORABLE` |
| Both upper bounds strictly < 0 | `SYNTHETIC_UNFAVORABLE` |
| Mixed, overlapping zero, or opposed signs | `SYNTHETIC_INCONCLUSIVE` |

The proposed global positive claim is an **intersection-union (both-needed)** form, conceptually different from declaring each endpoint individually significant; formal familywise size/coverage still warrants independent review. Under only **n=10 paired seeds**, the percentile bootstrap may have non-ideal coverage and limited power; do not claim it is statistically calibrated from these synthetic tests. A zero-containing CI is not evidence of equivalence. A favorable outcome for a parameter-capacity-matched comparison also does not automatically demonstrate a compute-matched or H6 plasticity advantage.

The all-recruited arm and O1/O2 remain **descriptive** for this proposed *external* primary contrast. The equal-order pooled descriptive interval is built by averaging each **seed's three order effects first**, then resampling the same ten seed clusters (not 30 independent order records). Reporting must explicitly expose order sign reversals and cannot discard O1/O2 because of unfavorable results. A real method-specific H6 conclusion would require a late-task fresh-network comparator not included here.

## Exactly what is and is not implemented

`src/synaploom/research/m13b_statistical_dryrun.py` and `experiments/scripts/m13b_statistical_dryrun.py` validate only **90 invented synthetic summary records** whose `fixture_kind` is `SYNTHETIC_NONSCIENCE_METRIC_FIXTURE_V1`; all have `actual_optimizer_steps=0`, `status=SYNTHETIC_NOT_TRAINED`, and `scientific_claim=NONE`. They are never model-generated training observations.

The validator checks the full proposed 90-slot candidate hash, every slot key, frozen order SHA, seed, arm, role, four metrics, finite numeric values, valid accuracy range, uniqueness and exactly zero actual optimizer steps. The analyzer refuses real results or scientific claims. It **does not parse arbitrary outcome files or possess a GPU execution command**. A later actual science runner needs independent feature/source/mask provenance checks, terminal run-state audits, per-seed complete evidence, and separate approval.

Fixtures demonstrate four logic branches:
- `positive`: O0 both endpoints positive (while O1 exhibits a negative order effect), demonstrates proposed joint favorable classification;
- `negative`: O0 both endpoints negative, demonstrates unfavorable;
- `null`: O0 both effects zero, demonstrates inconclusive;
- `opposed`: O0 one endpoint positive and the other negative, demonstrates inconclusive despite one good CI.

This synthetic pattern is intentionally fabricated and **must never be cited as a property of SynapLoom**. All produced labels begin with `SYNTHETIC_`.

## Safe commands (no results, data, or GPU)

```bash
uv run --locked python experiments/scripts/m13b_statistical_dryrun.py \
  --synthetic-self-test --scenario positive

uv run --locked python experiments/scripts/m13b_statistical_dryrun.py \
  --synthetic-self-test --scenario opposed \
  --output /tmp/m13b-inference-SYNTHETIC-review-v1.json
```

The optional output file is created with exclusive mode `x`, preventing silent overwrite. There is **no** `--results`, `--execute`, `--approve`, `--freeze`, `--cuda` or training-data argument. The emitted SHA-256 only detects modification of the fabricated report and is not independent scientific provenance.

## Before full confirmatory execution

1. An independent methodology reviewer must approve the *actual* inferential contract, including n=10 power/coverage, paired design, joint co-primary decision, missing/failed-run treatment and secondary/order multiplicity.
2. The owner must explicitly resolve D1–D4: O0 primary with or without mandatory O1/O2 robustness, the three arms and 501–510 seeds, honest resource fairness tiers and 720,000-step budget/cost boundaries.
3. Build a separately versioned, hash-bound **frozen protocol** with actual source/feature/order/mask provenance and a real science runner that refuses execution without explicit authorization; never infer owner GO from this docs/code PR merging.
4. Run only manual approved and finite/numerically auditable full runs. Preserve all failed and negative evidence; keep historical `EXP-H1A-002=INCONCLUSIVE`, `ORDER_SENSITIVE` and `DIRECTION_FLIP_OBSERVED`.

**GO for testing synthetic inference controls; NO-GO for interpreting any real scientific result or running the 30/90 benchmark.**
