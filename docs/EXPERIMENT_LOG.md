# Experiment Log

This is the append-only experimental memory for SynapLoom.

## Rules

Every result used to make a project decision should record:

- date
- commit SHA
- experiment ID
- hypothesis tested
- dataset / stream definition
- seed(s)
- configuration
- hardware
- command
- baseline(s)
- metrics
- active/total parameters
- active nodes/edges
- measured FLOPs or wall time when claimed
- artifact/checkpoint location
- conclusion
- known caveats

Do not promote a single-seed exploratory result to PROJECT_STATE.

## Result table

| ID | Date | Commit | Hypothesis | Dataset | Seeds | Status | Main result |
|---|---|---|---|---|---:|---|---|
| EXP-H1A-001 | 2026-10-07 | 188b32ab663301e8486657bf8661d40d58514159 | H1a | CIFAR-100 frozen ResNet-18 features, 10x10 CIL | 10 | INVALID_NUMERICAL_INSTABILITY | 39/50 runs developed NaN/Inf; H1a NOT_EVALUATED |
| EXP-H1A-STABILITY-001 | 2026-10-07 | 7d8622fdb47bdb79fce109c64066b854711bafb2 | numerical calibration only | same frozen H1a feature stream | 3 calibration seeds | COMPLETE | lr=0.01 selected: 15/15 stable; lr=0.03 was 1/15 stable |
| EXP-H1A-002 | 2026-10-11 | 7b301224151882d677403e1ea7a91d4b26b1291c | H1a | CIFAR-100 frozen ResNet-18 features, 10x10 CIL | 10 paired seeds (201..210) | COMPLETE_NUMERICALLY_VALID; INCONCLUSIVE | final accuracy +0.069 pp [0.036, 0.103]; late-task accuracy -0.041 pp [-0.171, 0.087]; preregistered overall verdict INCONCLUSIVE |
| EXP-H1A-ORDER-001 | 2026-10-11 | pending | class-order robustness for H1a | CIFAR-100 frozen ResNet-18 features, 10x10 CIL, three fixed orders | 10 paired seeds (301..310) reused across order blocks | PREREGISTERED / IMPLEMENTATION PENDING / NOT RUN | 60 runs; primary pair only; pooled seed-cluster bootstrap and per-order deltas; eval batch size fixed at 256 |

## Template

### EXP-XXXX — Title

**Date:**  
**Commit:**  
**Hypothesis:**  
**Status:** exploratory / validation / final

**Dataset / stream**

Describe task order, class/domain splits, preprocessing, and data version.

**Configuration**

Record the exact config file or full relevant parameters.

**Hardware**

Record GPU/CPU and software versions when relevant.

**Command**

Record the exact invocation.

**Baselines**

List fair baselines and what is matched: total parameters, active parameters, FLOPs, or wall time.

**Metrics**

Include continual-learning metrics and SynapLoom structural metrics.

**Results**

Provide seed-level values plus aggregate mean/std when the run is decision-relevant.

**Conclusion**

State what the experiment supports or falsifies.

**Caveats**

List confounds and follow-up tests.


### EXP-H1A-001 — Controlled reserve recruitment on frozen CIFAR-100 features

**Date:** 2026-10-07  
**Commit:** `188b32ab663301e8486657bf8661d40d58514159`  
**Hypothesis:** H1a — controlled reserve recruitment  
**Status:** **INVALID_NUMERICAL_INSTABILITY**  
**Scientific conclusion:** **H1a NOT_EVALUATED**

**Protocol**

The preregistered protocol remains unchanged in `docs/M1_3_PROTOCOL.md`.

**Execution integrity**

The execution itself completed mechanically:

- all 50 expected seed x variant JSON files were produced;
- preflight status was `ready`;
- feature bundle SHA-256 was
  `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47`;
- class-order SHA-256 was
  `4deb5f1b28e8f27095dbcdce2db355792a33d70b70e6a5080c787a56743189cf`;
- Python 3.11.15;
- PyTorch 2.14.1+cu130;
- Torchvision 0.29.1+cu130;
- CUDA 13.0;
- NVIDIA GeForce RTX 4090.

**Numerical validity**

Post-run inspection found non-finite training/evaluation values in **39 of 50 runs**:

| Variant | Finite runs | Runs with NaN/Inf |
|---|---:|---:|
| progressive_reserve | 2/10 | 8/10 |
| all_recruited | 0/10 | 10/10 |
| fixed_small | 7/10 | 3/10 |
| dense_active_proxy | 1/10 | 9/10 |
| dense_total | 1/10 | 9/10 |

The primary control `all_recruited` therefore had **zero numerically valid seeds**.

The old runner continued after NaN/Inf and generated finite-looking accuracy summaries from invalid states. Those aggregate statistics must not be interpreted as an H1a result.

**Result**

```text
EXP-H1A-001 = INVALID_NUMERICAL_INSTABILITY
H1a scientific conclusion = NOT_EVALUATED
```

Observed aggregate deltas are retained only as historical artifacts; they are not evidence for or against H1a.

**Decision**

Do not tune H1a based on accuracy from this run.

Next:

1. add fail-fast numerical diagnostics;
2. run `EXP-H1A-STABILITY-001` on independent seeds;
3. select a learning rate using numerical validity only;
4. preregister a new scientific run `EXP-H1A-002` if calibration succeeds.

**Caveats**

The instability occurred in dense and routed variants, so EXP-H1A-001 does not isolate the cause to SynapLoom recruitment. The shared optimization protocol is the first variable to calibrate.

---

### EXP-H1A-STABILITY-001 — Shared optimizer numerical calibration

**Date:** 2026-10-07  
**Commit:** `7d8622fdb47bdb79fce109c64066b854711bafb2`  
**Hypothesis:** none; numerical calibration only  
**Status:** COMPLETE

**Protocol**

See `docs/H1A_STABILITY_PROTOCOL.md`.

Frozen calibration:

- seeds: 101, 102, 103;
- variants: all five H1a variants;
- candidate learning rates, descending: 0.03, 0.01, 0.003;
- all other H1a settings unchanged.

**Selection rule**

Select the highest learning rate that completes all:

```text
3 seeds x 5 variants = 15 runs
```

with no detected NaN/Inf.

Accuracy, forgetting, transfer and routing metrics are explicitly ignored for selection.

If no candidate passes, select no learning rate and open a separate stability experiment before adding mechanisms such as gradient clipping.

**Command**

```bash
uv run --locked python experiments/scripts/m1_3_h1a_stability.py \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt \
  --output runs/m1_3_h1a_stability \
  --device cuda
```

**Results**

- LR 0.03: **1/15 stable**, 14/15 numerical failures.
- LR 0.01: **15/15 stable**, 0/15 failures.
- LR 0.003: not executed because the preregistered stop rule selected the first fully stable candidate.
- selected learning rate: **0.01**
- feature SHA-256: `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47`
- execution environment: Python 3.11.15, PyTorch 2.14.1+cu130, CUDA 13.0, RTX 4090
- maximum gradient L2 norm observed among stable LR=0.01 runs: approximately **49.99**
- maximum absolute parameter value observed among stable LR=0.01 runs: approximately **1.01445**

Accuracy, forgetting, transfer and routing metrics from this calibration were **not used** for optimizer selection.

**Conclusion**

`lr=0.01` satisfies the preregistered numerical-validity rule and is the shared optimizer setting for the next fresh scientific H1a test.

H1a remains `NOT_EVALUATED` until EXP-H1A-002.

---

### EXP-H1A-002 — Controlled reserve recruitment scientific rerun

**Date:** 2026-10-11  
**Commit:** `7b301224151882d677403e1ea7a91d4b26b1291c`  
**Hypothesis:** H1a — controlled reserve recruitment  
**Status:** **COMPLETE_NUMERICALLY_VALID; PREREGISTERED VERDICT INCONCLUSIVE**

**Frozen protocol**

- CIFAR-100 frozen ImageNet ResNet-18 features, 50,000 train / 10,000 eval examples, feature dimension 512.
- Feature bundle SHA-256: `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47`.
- Fixed class order, 10 experiences x 10 classes, 100-way classifier with unseen logits masked.
- Variants: `progressive_reserve`, `all_recruited`, `fixed_small`, `dense_active_proxy`, `dense_total`.
- 10 paired seeds: **201..210**; 50 total runs.
- Learning rate **0.01**, selected by EXP-H1A-STABILITY-001 using numerical validity only; SGD momentum=0.9, weight decay=5e-4.
- 20 epochs/task, batch size 128, eval batch size 256, N=16, initial recruited=4, top-k=4.
- No replay, no consolidation, q_max=0, message_rounds=0.
- Primary comparison: `progressive_reserve - all_recruited`.
- Co-primary outcomes: final average accuracy and late-task within-task mean accuracy.
- The original preregistered overall rule is unchanged: FAVORABLE only if both co-primary 95% CIs are strictly above zero; UNFAVORABLE only if both are strictly below zero; otherwise INCONCLUSIVE; INVALID on numerical/artifact failure.

**Command**

```bash
uv run --locked python experiments/scripts/m1_3_h1a_002.py \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt \
  --output runs/m1_3_h1a_002 \
  --device cuda \
  2>&1 | tee runs/m1_3_h1a_002/console.log
```

**Execution and artifact integrity**

- All **50/50** expected run files were produced and reported complete.
- All 50 files contained the expected experiment ID, variant, and seed metadata; no duplicate variant/seed pairs were found.
- No non-finite values were found in the serialized per-run JSON results.
- Feature hash matched the frozen protocol.
- Environment: Python 3.11.15, PyTorch 2.14.1+cu130, Torchvision 0.29.1+cu130, CUDA 13.0, NVIDIA GeForce RTX 4090.
- Commit recorded in the aggregate: `7b301224151882d677403e1ea7a91d4b26b1291c`.
- Full per-run JSON files, `aggregate.json`, and `console.log` remain in the local `runs/m1_3_h1a_002/` artifact directory; `runs/` is git-ignored.

**Per-variant descriptive results**

Accuracy means are percentages; values after ± are standard deviations in percentage points.

| Variant | Final average accuracy (%) | Late-task within-task mean accuracy (%) | Average forgetting (%) |
|---|---:|---:|---:|
| progressive_reserve | 9.524 ± 0.254 | 86.496 ± 0.280 | 88.467 ± 0.227 |
| all_recruited | 9.455 ± 0.271 | 86.537 ± 0.319 | 88.586 ± 0.271 |
| fixed_small | 9.465 ± 0.252 | 86.463 ± 0.308 | 88.547 ± 0.222 |
| dense_active_proxy | 9.392 ± 0.287 | 86.580 ± 0.287 | 88.382 ± 0.358 |
| dense_total | 9.315 ± 0.345 | 86.619 ± 0.238 | 88.310 ± 0.421 |

The high forgetting values occur under the frozen no-replay/no-consolidation protocol and are descriptive outcomes, not one of the two co-primary criteria.

**Preregistered paired comparison**

Differences and confidence intervals below are in percentage points (pp), for `progressive_reserve - all_recruited`.

| Co-primary outcome | Mean paired delta (pp) | Paired-bootstrap 95% CI (pp) | Outcome |
|---|---:|---:|---|
| Final average accuracy | +0.069 | [+0.036, +0.103] | Favorable |
| Late-task within-task mean accuracy | -0.041 | [-0.171, +0.087] | Inconclusive |

**Paired deltas by seed**

All values are percentage points for `progressive_reserve - all_recruited`.

| Seed | Final average accuracy delta (pp) | Late-task within-task mean accuracy delta (pp) |
|---:|---:|---:|
| 201 | +0.100 | +0.025 |
| 202 | +0.070 | -0.090 |
| 203 | +0.010 | -0.090 |
| 204 | +0.090 | -0.285 |
| 205 | +0.170 | -0.1075 |
| 206 | +0.120 | +0.0125 |
| 207 | -0.010 | -0.440 |
| 208 | +0.090 | +0.2425 |
| 209 | +0.000 | +0.025 |
| 210 | +0.050 | +0.2975 |

**Preregistered conclusion**

```text
Execution/artifact validity = PASS
Final-average-accuracy co-primary = FAVORABLE
Late-task within-task-accuracy co-primary = INCONCLUSIVE
EXP-H1A-002 overall verdict = INCONCLUSIVE
H1a benefit = NOT ESTABLISHED
```

The final-average-accuracy advantage is statistically above zero under the preregistered paired-bootstrap interval, but it is only **0.069 percentage points**. The late-task within-task outcome is slightly negative on average and its interval crosses zero. Therefore, the experiment does **not** establish that progressive reserve recruitment improves both final retention and late-task learning.

**Caveats and next action**

- This is one fixed class order and one fixed evaluation batch size (256); routing is batch-context dependent.
- Features are frozen; this is a mechanism-isolation test, not an end-to-end encoder result.
- No replay or consolidation is enabled, so the experiment does not test those remedies for forgetting.
- Do not treat the small favorable final-accuracy result as a broad architectural win.
- Next: preregister a separate sensitivity/robustness study for class order and/or evaluation batch size. Keep this primary result frozen and report any follow-up as a separate experiment.


### EXP-H1A-ORDER-001 — Class-order sensitivity of controlled reserve recruitment

**Protocol date:** 2026-10-11  
**Commit:** pending implementation/execution  
**Hypothesis:** robustness of the H1a treatment effect to class order  
**Status:** **PREREGISTERED; IMPLEMENTATION PENDING; NOT RUN**  
**Parent result:** EXP-H1A-002 remains the frozen primary experiment with verdict **INCONCLUSIVE**.

**Question**

Does the paired effect of `progressive_reserve - all_recruited` remain directionally consistent across multiple fixed CIFAR-100 class orders?

**Frozen setup**

- Same frozen ResNet-18 feature bundle and SHA-256 as EXP-H1A-002: `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47`.
- Same model, optimizer and continual-learning protocol as EXP-H1A-002: LR=0.01, momentum=0.9, weight decay=0.0005, 20 epochs/task, batch size=128, evaluation batch size=256, N=16, initially recruited=4, top-k=4, q_max=0 and no message passing, replay or consolidation.
- Primary variants only: `progressive_reserve` and `all_recruited`.
- Three fixed orders: O0 (the original `H1A_CLASS_ORDER`), O1 (one literal permutation generated from stdlib seed 20261006), and O2 (one literal permutation generated from stdlib seed 20261007). Exact order groupings are frozen in `docs/H1A_CLASS_ORDER_SENSITIVITY_PROTOCOL.md`.
- Fresh seeds: **301..310**, shared across the three orders and two variants.
- Expected run count: **3 orders x 10 seeds x 2 variants = 60**.

**Registered analysis**

- Co-primary metrics: final average accuracy and late-task within-task mean accuracy.
- For each order, report paired seed deltas and paired-bootstrap 95% CIs (10,000 resamples; RNG seeds 1337, 1338, 1339 for O0, O1, O2).
- Pooled order-marginal estimates weight each order equally. For pooled CIs, bootstrap seed clusters (10,000 resamples; RNG seed 7331), carrying all three order outcomes for every resampled seed; do not treat the 30 order-by-seed blocks as independent.
- Registered labels: `ORDER_ROBUST_FAVORABLE`, `ORDER_ROBUST_UNFAVORABLE`, `ORDER_SENSITIVE`, or `INCONCLUSIVE`, using the exact rules in the protocol.
- Any numerical or artifact-integrity failure invalidates the full sensitivity study; do not interpret partial aggregates.
- This study cannot retroactively change the EXP-H1A-002 verdict.

**Implementation status and command**

The dedicated runner and protocol tests will be added in a separate implementation PR. The frozen command interface is:

```bash
uv run --locked python experiments/scripts/m1_3_h1a_order_sensitivity.py \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt \
  --output runs/m1_3_h1a_order_sensitivity \
  --device cuda
```

**Deferred follow-up**

Evaluation-batch-size sensitivity is not part of this study. It will need its own protocol that evaluates identical trained model states at multiple batch sizes, since batch-level routing makes predictions depend on the evaluation-batch composition.
