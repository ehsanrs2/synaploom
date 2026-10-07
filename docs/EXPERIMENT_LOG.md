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
| EXP-H1A-STABILITY-001 | 2026-10-07 | pending | numerical calibration only | same frozen H1a feature stream | 3 calibration seeds | preregistered / not run | choose highest LR with 15/15 finite runs; accuracy ignored |

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
**Commit:** pending merge/execution commit  
**Hypothesis:** none; numerical calibration only  
**Status:** preregistered / not run

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

Not run yet.
