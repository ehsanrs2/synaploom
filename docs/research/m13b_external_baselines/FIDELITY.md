# M1.3b external-method fidelity and feasibility audit

**Status: DRAFT — NOT A CLAIM OF METHOD REPRODUCTION.** Scientific admission requires a separately qualified implementation and independent methodological review. The public literature helps define candidate methods but does not certify their use in the SynapLoom frozen-feature harness.

## Primary sources and discovered reference implementations

| Method | Original publication | Public code anchor | What is verified vs still unknown |
|---|---|---|---|
| RigL | Evci et al., ICML 2020, https://proceedings.mlr.press/v119/evci20a.html | https://github.com/google-research/rigl | Published method/source location verified. Reference project has TensorFlow-oriented training examples; a direct drop-in into this PyTorch continual runner is **not** demonstrated. Pin commit/license/dependency versions and compare code to pseudocode before claiming reproduction. |
| NISPA | Gurbuz & Dovrolis, ICML 2022, https://proceedings.mlr.press/v162/gurbuz22a.html | https://github.com/BurakGurbuz97/NISPA | Published method/source location verified. Method defines stable/plastic pathways with connection rewiring under fixed density; available external code has **not** been imported or integration-qualified. Verify commit, license and transferability to frozen 512-D features. |
| Fixed sparse | Conventional fixed-mask sparse MLP control | internal implementation **not yet selected** | Useful method-neutral comparison if same architecture, sparse mask and training budgets are predeclared. This is not `fixed_small`, which selects 4 whole experts from a 16-expert pool. |

External reference code is not automatically executed, downloaded, copied into the Repository or silently modified. Check license/attribution before any reuse.

## Gate F0 — Correctly characterize the algorithm

**Fixed Sparse MLP** must freeze the mask throughout the run, verify exact nonzero/eligible weights and mask density at every task, and test that masked-out weights and their optimizer state do not become active due to momentum/weight decay. Clearly state dense-mask versus true sparse arithmetic. Choose mask distribution *before* results.

**RigL-style** must include at least (a) a fixed target active-density budget, (b) magnitude-based connection removal, (c) gradient-magnitude-based regrowth using gradients for currently inactive connections, (d) explicitly frozen topology-update interval, end time and drop-fraction schedule, (e) newly grown weights initialized according to the selected original-method variant, and (f) optimizer/mask state handling. A mere random rewiring or expert recruitment schedule is **not RigL**. Verify gradient-probe cost and gate any difference from official method as an approximation, not a reproduced algorithm.

**NISPA-style** must implement a methodologically justified distinction between stable and plastic units/connections, its stability-selection rule and rewiring of plastic pathways to reuse stable knowledge, with fixed density and task-phase rules. The published approach has validation-dependent phases; **never** use the held-out CIFAR-100 evaluation set for training-time model selection. Propose a deterministic **train-only** validation split, document its interaction with other arms, and only admit the comparator if it can meet a fair and leakage-free data protocol. A frozen/recruited SynapLoom expert mask with a NISPA label is **not NISPA**.

## Gate F1 — Applicability to the exact benchmark

For each candidate, fill the following evidence record (blank means not qualified):

| Evidence field | Required qualification |
|---|---|
| paper/code | DOI/URL, commit SHA, author/license, implementation language, algorithm variant |
| feature protocol | consumes identical 512-D pinned features, 10×10 class-incremental order, seen-class logit mask, no inference task labels |
| training control | optimizer and LR strategy, masks, stability/rewiring schedule, number of updates and examples, use of validation |
| reserve & stability | declares what "dormant", "masked", "stable", "plastic" mean — these are *not equivalent* by nomenclature |
| numerical fidelity | finite gradients/parameters and protected numerical checks without bypass; failure propagation |
| sparse behavior | masked weights never update spuriously, dynamic method preserves requested density after rewiring, gradient grow matches rule |
| resource honesty | total allocated/trainable/active/executed counts and counted vs actual operations, wall, peak GPU memory |
| reproduction | original tests or known fixture vs implementation, code pin, adaptation deviations and reviewer approval |
| feasibility | available input feature bytes/device, dependencies, local pilot pass, no hidden data download |
| decision | `GO_METHOD`, `CONDITIONAL`, or `NO_GO_FIDELITY` with explicit evidence links |

## Gate F2 — Pilot before scientific commitment

No accuracy/forgetting-based tuning and **no confirmatory seeds** in the qualification pilot. Accept only:
- constructed model/mask and parameter ledger passes;
- one bounded synthetic step verifies masked-grad/optimizer, density and selected-path semantics;
- one short task-stream smoke verifies output shapes, seen-class logits, task boundaries, metrics serialization and deterministic data order;
- basic CPU CI passes; GPU tiny smoke is a separately approved manual activity;
- no unregistered Tensor/optimizer state after checkpoint roundtrip for methods that need resumability.

Failure labels: `NO_GO_FIDELITY`, `DATA_UNAVAILABLE`, `DEP_UNAVAILABLE`, `BUDGET_UNMATCHABLE`, `INVALID_NUMERICS`, `LEAKAGE_RISK`, or `PROVENANCE_INCOMPLETE`. Preserve proof of each failed gate; do not replace with a weaker method under the same label.

## Current per-method decision (as of this DESIGN DRAFT)

| Method | Status | Reason |
|---|---|---|
| Fixed Sparse MLP | **CONDITIONAL; first engineering priority** | Concrete architecture and mask/density/resources not frozen; implementability plausible but **unverified** |
| RigL-style | **CONDITIONAL; fidelity audit before implementation** | Key magnitude-prune / gradient-regrow rules are documented, but adaptation and code integration are not qualified |
| NISPA-style | **CONDITIONAL; validation/data split design is a blocker** | Need faithful stable/plastic rewiring and leak-free validation; 512-D frozen features differ from many original training settings |
| Replay | **SEPARATE EXPLORATORY RESOURCE CONDITION** | Uses data memory that no-replay primary arms lack; buffer rule and capacity must be pinned if studied |

## Decision required before PROMOTION

Do not "promote" a method name or scientific run until method-fidelity evidence, frozen protocol, fair budget, separate validation data if required, test results, and an explicit owner gate are available. The historical H1a outcome remains inconclusive and does not acquire significance through increased baseline count.

This audit is a feasibility policy, not a report that external method binaries or data have been executed.
