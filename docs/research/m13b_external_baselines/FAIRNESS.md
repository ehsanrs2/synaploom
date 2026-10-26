# M1.3b — Fairness and budget-matching matrix (DESIGN PROPOSAL)

**Status: DRAFT / NO EXPERIMENT APPROVAL.** Companion to `PROTOCOL_DRAFT.md`. Any numeric thresholds are proposed review gates rather than post-hoc modifications to H1a or confirmed attainable budgets.

## Why a single "same number of parameters" is insufficient

SynapLoom allocates all 16 experts even when only 4 are recruited, and initially executes 4 experts per *batch* under legacy routing. RigL rewires individual sparse weights; NISPA maintains sparse *stable and plastic pathways*. Matching physically allocated parameters does not imply equal nonzero weights, active experts, gradient evaluation, update FLOPs, peak GPU memory or inference latency. The original H1a primary control `all_recruited` remains the only exact matched-total-parameter, matched-top-k control within the same architecture. External sparse controls are different mechanisms and must be reported as such.

## Shared scenario — proposal, not registered execution

- **Representation:** CIFAR-100 frozen ImageNet ResNet-18 IMAGENET1K_V1, 512-D, artifact SHA-256 `c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47`. GitHub cannot establish local file availability. Audit it on the execution host before any experiment.
- **Streams:** 10 × 10 class-incremental tasks, 100-way masked seen-class output, no task ID at inference. O0 literal/fingerprint from `src/synaploom/protocols/h1a.py`. Given historical order sensitivity, O1/O2 (fingerprints in `docs/H1A_CLASS_ORDER_SENSITIVITY_PROTOCOL.md`) must be either **predeclared secondary robustness** or a stated limitation before final approval; no ad hoc selection after O0 results.
- **Default anchor:** H1a 002 configuration uses SGD lr=.01 (not the original failed .03), momentum=.9, weight decay=.0005, no scheduler, 20 epochs/task, train B128, eval B256, no replay. Retaining these is an *anchor*, not an automatic prescription for every external method where original fidelity requires distinct schedules; any difference requires a disclosed separate comparison tier and no claim of an exactly optimizer-matched experiment.
- **Proposed confirmation seeds:** 501–510 for matched candidate arms (new, not run); no final seed set is frozen until owner approval. Use pilot-only seed(s) disjoint from confirmation seeds. No accuracy-based candidate inclusion.
- **Batch-router limitation:** For SynapLoom, eval B256 and sample ordering are part of the fixed measurement. Comparing against batch-invariant dense/sparse models does not erase that limitation.

## Candidate matrix and eligibility

| Candidate | Role / question | Status before D0–D4 | Defining constraint | Primary resource caveat |
|---|---|---|---|---|
| `progressive_reserve` | H1a-like treatment | available historic runner; fresh run is **not** authorized | schedule one recruit after first optimization step for each new task | dormant full parameter allocation counts |
| `all_recruited` | internal anchor, equal N/top-k | available historic runner; fresh run not authorized | same 16 experts and K=4 from task 0 | 16 candidate scores vs growing candidates |
| `fixed_small` | internal capacity floor, not external fixed-sparse | historic existing | permanently 4 routable experts of N=16 | **must not** be renamed as an unstructured fixed-sparse baseline |
| `fixed_sparse_mlp` | mandatory first external comparison candidate | **CONDITIONAL**, architecture/target masks not selected | fixed, explicitly documented unstructured mask; no rewiring; masked gradients and optimizer state | dense masked kernels may give no wall-clock savings |
| `rigl_like_mlp` | whether gradient-guided reallocation helps | **CONDITIONAL / not admitted** | fixed density + magnitude pruning + dormant-gradient growth + update schedule + zero initialization | periodic dense-gradient and mask update overhead |
| `nispa_like_mlp` | sparse stable/plastic pathway retention | **CONDITIONAL / not admitted** | stable path protection and constrained plastic path rewiring | requires task-specific train-only validation for phase rules; cannot see test/eval data |
| `replay` | resource-augmented reference | separate optional family, not no-replay | replay buffer size/examples/provenance pinned | memory/storage/data advantage disclosed; never mix into no-replay primary family |

A baseline may be demoted to **EXPLORATORY_APPROXIMATION** if a defining mechanism cannot be faithfully implemented, with a different name and no original-paper reproduction claim; it must not replace a frozen registered arm.

## Mandatory resource ledger per (candidate, seed, stage)

1. Total **physically allocated** trainable elements (including dormant pools, sparse masked zeros and auxiliary gate parameters), plus all tensors/masks/state buffers and optimizer memory in bytes.
2. Total **nonzero/active** connection count or recruited-expert count, and exactly which elements are trainable.
3. Routed/selected experts (SynapLoom K), active sparse matrix density by layer, and **actual forward/backward work**, including task-boundary updates. Do not treat K=4 as comparable to density=.25 without measured FLOPs.
4. Two independently labeled FLOP/MAC views: mathematical ideal nonzero operation count and **actual implementation** dense-mask/sparse kernel work, including gradient probes, rewiring, masking, dispatch, controller and statistical guards; disclose excluded ops.
5. Train and eval elapsed wall-clock, peak GPU allocated/reserved memory, number of launches if profiled, and number of structural events/weight rewires, with processor/device/precision/thread provenance.
6. Same class-order/seed within paired blocks, actual examples presented and update steps; no task ID at inference; no test-data access for training or dynamic rewiring.

### Proposed matching policy (reviewable, not presumed feasible)

- **Total-allocation-matched tier:** count all potential trainable tensors and require `|P_candidate-P_reference|/P_reference <= 0.02` before presenting as matched. Parameter-mask storage and optimizer states are reported even when excluded from trainable count.
- **Executed-work-matched tier:** require independently counted ideal linear MACs per standard batch within ±10% plus separately measured actual train+rewiring work; the two measures must not be conflated. If the threshold is infeasible for a faithful method, declare `UNMATCHED_RESOURCE` and report resource-response curves rather than altering the method.
- **Wall-clock matching:** only a separately approved exploratory efficiency tier with equal device/time budget; never quietly shorten one arm's epochs or training examples.
- **Training policy matching:** common input labels, task presentation and preprocessing are mandatory. A method-specific optimizer/sparsity schedule is explicitly identified as such; same-SGD comparison and method-faithful comparison are **different** questions.
- If no faithful configuration meets the proposed resource gates, report that negative feasibility result, **do not** loosen limits retrospectively to declare fairness.

## Outcomes and multiplicity

For an owner-approved confirmatory fixed-sparse family, propose paired differences `progressive_reserve - fixed_sparse_mlp` for:
(1) final average accuracy and (2) late-task within-task mean accuracy, with 10,000 paired seed-cluster bootstrap draws and both intervals required strictly above zero for `FAVORABLE` (strictly below for `UNFAVORABLE`), otherwise `INCONCLUSIVE`. This is a **new external comparison**, not an upgrade to EXP-H1A-002. Each other external method remains *exploratory* until an independent multiplicity-controlled primary plan is frozen. Always report full forgetting/BWT matrices and curve distributions. High forgetting and learning-curve accuracy are **not** by themselves proof of loss of plasticity; do not claim H6 without a fresh-network late-task comparator.

## No-go conditions

Reject missing/changed data hash, unpinned class order, finite-check failure, selected-result seed omission, optimizer/mask mismatch, leakage of test data to growth/pruning decisions, hidden extra training examples, missing FLOP ledgers, unqualified NISPA phase validation, or mutated historic result files. Every failure remains an auditable result with all partial artifacts preserved and **no favorable result claim**.

**Next review:** choose exact fixed-sparse target architecture/mask, resource ledger definitions and pilot-only dataset partition before calling any external baseline qualified.
