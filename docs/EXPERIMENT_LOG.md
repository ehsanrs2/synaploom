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
| EXP-H1A-001 | 2026-10-05 | pending | H1a | CIFAR-100 frozen ResNet-18 features, 10x10 CIL | 10 | preregistered / not run | protocol frozen; no result yet |

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

**Date:** 2026-10-05  
**Commit:** pending merge/execution commit  
**Hypothesis:** H1a — controlled reserve recruitment  
**Status:** preregistered / not run

**Protocol**

See `docs/M1_3_PROTOCOL.md`. The protocol, class order, seeds, model sizes, optimizer, metrics and paired comparison are frozen before inspecting primary scientific results.

**Primary comparison**

`progressive_reserve - all_recruited` on identical seeds.

**Co-primary metrics**

- final average accuracy
- late-task within-task mean accuracy

**Secondary metrics**

- forgetting
- backward transfer
- diagonal task accuracy
- structural/routing histories
- parameter/MAC/wall-time/VRAM views

**Command**

`python experiments/scripts/m1_3_h1a_benchmark.py --features data/features/cifar100_resnet18_imagenet1k_v1.pt --device cuda`

**Results**

Not run yet.

**Conclusion**

None. Engineering/CI smoke outputs are not scientific results.
