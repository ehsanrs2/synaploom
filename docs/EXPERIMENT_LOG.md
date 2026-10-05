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
| — | — | — | — | — | — | No experiments yet | — |

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
