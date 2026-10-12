# EXP-H1A-EVALBATCH-001 — Evaluation Batch-Size Sensitivity of H1a Outcome Measurement

**Protocol date:** 2026-10-12  
**Status:** preregistered; runner/tests pending; not run  
**Parent result:** EXP-H1A-002 — COMPLETE_NUMERICALLY_VALID; INCONCLUSIVE  
**Related sensitivity result:** EXP-H1A-ORDER-001 — COMPLETE_NUMERICALLY_VALID; ORDER_SENSITIVE

## 1. Question and scope

**Question:** How much do evaluation batch sizes change the two H1a co-primary accuracy metrics and the paired treatment-control contrast when the model weights at each evaluation checkpoint are held fixed?

This is a measurement-sensitivity study. It does not retest H1a independently, test automatic recruitment (H1b), compare new model mechanisms, or permit the registered verdict of EXP-H1A-002 to be changed. The class-order sensitivity result remains a separate finding.

### Artifact limitation and design consequence

The available EXP-H1A-002 and EXP-H1A-ORDER-001 output artifacts contain evaluation summaries and run provenance but do not contain model-state checkpoints. It is therefore not possible to claim that this follow-up retrospectively re-evaluates the exact historical weights. This protocol instead uses a new paired seed block and evaluates all batch sizes at every designated checkpoint of each single training trajectory. **Each seed/variant is trained once; training is never repeated separately for different evaluation batch sizes.** Within each checkpoint sweep, all batch-size evaluations must use exactly the same in-memory model state.

## 2. Frozen data and model configuration

- Dataset: CIFAR-100 using frozen torchvision ResNet-18 IMAGENET1K_V1 features, 50,000 training and 10,000 evaluation examples, feature dimension 512.
- Feature bundle SHA-256: \`c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47\`.
- Class order: O0 / the exact frozen H1A_CLASS_ORDER used by EXP-H1A-002. Compact-JSON-list fingerprint: \`81d7f0637b8e61477c1ef80b7eef33270aae8d81dfcaffad6e58252fd004b633\`.
- Task stream: 10 experiences × 10 classes; 100-way output with logits restricted to classes seen by the current stage.
- Variants: exactly \`progressive_reserve\` and \`all_recruited\`; paired contrast is progressive reserve minus all recruited.
- Model: feature dimension 512, d_model=128, d_hidden=256, 16 nodes, initially recruited=4 for progressive reserve, top-k=4, q_max=0 and message_rounds=0.
- Training: 20 epochs/task, train batch size=128, SGD learning rate=0.01, momentum=0.9, weight decay=0.0005, no scheduler, no replay, no consolidation, no pruning/recycling, and no message passing.
- The training minibatch stream, optimizer, learning schedule, class order, and parameter updates remain unchanged. Evaluation must not affect training decisions or update parameters/buffers.

### Seeds and matrix

Use exactly these fresh paired seeds, shared by both variants:

\`\`\`text
401, 402, 403, 404, 405, 406, 407, 408, 409, 410
\`\`\`

These seeds are disjoint from the previous primary and sensitivity seed blocks (201–210 and 301–310). Do not replace or omit seeds based on their metrics.

Expected training trajectories:

\`\`\`text
10 paired seeds × 2 variants = 20 trajectories
20 trajectories × 6 evaluation batch sizes = 120 run-size summaries
\`\`\`

This is a newly run cohort designed to isolate batch-size effects within identical model states; it is not a claim of posthoc evaluation of the historical EXP-H1A-002 checkpoints.

## 3. Fixed evaluation batch sizes and checkpoint semantics

Evaluate at exactly these sizes, in this order:

\`\`\`text
32, 64, 128, 256, 512, 1024
\`\`\`

Batch size 256 is the reference condition matching the original protocol. No runtime overrides are permitted. For every evaluation, use the same task examples in the same fixed order, no shuffling, and do not drop the final partial batch. Since each CIFAR-100 task has 1,000 evaluation examples, batch size 1024 places the task in one evaluation batch.

At each training checkpoint:

1. After every epoch, evaluate the current task for the late-task within-task learning curve at all six batch sizes.
2. At the end of each task/stage, evaluate every task allowed by the original protocol (all seen tasks) at all six batch sizes, producing the corresponding accuracy and loss matrices.
3. Do not perform an optimizer step, structural update, random sampling, or any other model mutation between evaluations at different batch sizes.
4. Record a deterministic model-state SHA-256 immediately before and after each complete checkpoint sweep. Compute it over sorted state_dict entries; for every entry, hash the UTF-8 name, dtype, shape, and contiguous CPU tensor bytes in a frozen canonical encoding. The before/after hashes must match; record the state hash with the stage/epoch and result rows.
5. Any state-hash mismatch invalidates the whole study. Evaluation mode flags alone are not model-state changes; the hash is over state_dict tensors.

The implementation may extend the continual evaluation harness or add a dedicated runner, but the training trajectory must run once per seed/variant and the six evaluations must share the same model state. Separate training runs configured with different evaluation batch sizes do not satisfy this protocol.

## 4. Outcomes and registered analysis

### Co-primary metrics

For each evaluation batch size \(b\), calculate these using the same definitions as EXP-H1A-002:

1. final average accuracy;
2. late-task within-task mean accuracy.

All treatment-control deltas and batch-size drifts are reported in percentage points (pp).

For metric \(m\), seed \(s\), and batch size \(b\), define the paired seed-level treatment effect as:

\[
\delta_{s,b,m} = M_{s,\mathrm{progressive\_reserve},b,m} -
M_{s,\mathrm{all\_recruited},b,m}
\]

The mean treatment effect at batch size \(b\) is the mean of these ten paired seed deltas, \(\Delta_{b,m}\). The reference drift is:

\[
\Gamma_{b,m} = \Delta_{b,m} - \Delta_{256,m}
\]

The reference drift at 256 is exactly zero by definition. For each metric and size, report the two variant means, the paired treatment effect with its 95% paired-bootstrap percentile CI, and the drift from 256 with its paired-bootstrap 95% CI. Also report per-seed deltas so the pairing can be audited.

Use 10,000 seed-cluster bootstrap resamples and RNG seed 7331. Each resampled seed carries both variants and all six batch sizes together; reuse the same sampled seed blocks for all metrics and sizes to preserve the paired comparisons. No individual run-size row may be treated as statistically independent.

### Secondary metrics

Report average forgetting and backward transfer by batch size as descriptive secondary metrics. They do not affect the registered directional classification or the original H1a verdict. Report per-variant descriptive changes from batch size 256 where useful.

### Descriptive direction classification

This follow-up uses a descriptive point-estimate label, not a new confirmatory hypothesis test:

- **DIRECTION_FLIP_OBSERVED:** for either co-primary metric, at least one non-reference batch size has a treatment-effect point estimate whose sign differs from the point-estimate sign at 256. Define sign(negative)=-1, sign(zero)=0, sign(positive)=+1.
- **NO_DIRECTION_FLIP_OBSERVED:** both co-primary treatment-effect point estimates keep the same signs as their respective 256-reference estimates at all other registered batch sizes.

This label reports directional measurement sensitivity only. It is not a test of statistical significance, and it does not establish a treatment-by-batch-size interaction. Confidence intervals should be shown transparently; do not make a familywise confirmatory claim from individual batch-size contrasts.

## 5. Validity, provenance, and failure rules

Before training, the runner must validate:

- the feature-bundle SHA-256 is the frozen value above;
- the literal class order matches O0 and its fingerprint;
- variants are exactly the two registered primary variants;
- seeds are exactly 401–410;
- batch sizes are exactly 32, 64, 128, 256, 512, and 1024 in the registered order;
- resolved model, optimizer, task, training batch, and epoch settings match this protocol;
- the expected training-run count is 20 and the run-size summary count is 120.

Each run artifact must include the experiment ID, seed, variant, class order and fingerprint, resolved protocol, feature SHA/audit, state hashes for every checkpoint sweep, per-size accuracy/loss matrices and within-task curves, model/metric summaries, numerical diagnostics, Git commit, software environment, and a complete run manifest.

Any missing run, missing batch-size result, non-finite value, feature or protocol mismatch, or model-state hash mismatch invalidates the entire study. Preserve failure artifacts and do not interpret partial aggregates as scientific results. The output directory must be empty before a real execution.

## 6. Required runner interface

The runner is planned as:

\`\`\`bash
uv run --locked python experiments/scripts/m1_3_h1a_eval_batch_sensitivity.py \\
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt \\
  --output runs/m1_3_h1a_eval_batch_sensitivity \\
  --device cuda
\`\`\`

A \`--dry-run\` mode must print the frozen feature hash, O0 fingerprint, seed list, variants, six evaluation batch sizes, 20 expected training trajectories, and 120 expected run-size summaries without loading the feature artifact or starting training. The runner must reject overrides to any registered seed, order, variant, training setting, or evaluation batch size.

## 7. Interpretation guardrail and next action

This is a separate measurement-sensitivity study on a fresh cohort. It can show that metric values or treatment-effect estimates depend on evaluation batch composition, but it cannot rewrite the EXP-H1A-002 result or the EXP-H1A-ORDER-001 result. Keep the historical primary verdict **INCONCLUSIVE** and the class-order sensitivity label **ORDER_SENSITIVE** regardless of this follow-up's result.

Next sequence: implement the fixed multi-size evaluator and unit tests, run local tests and the dry-run, execute all 20 trajectories, audit the complete artifacts, and then record the observed directional label and numerical results in PROJECT_STATE.md and docs/EXPERIMENT_LOG.md.
