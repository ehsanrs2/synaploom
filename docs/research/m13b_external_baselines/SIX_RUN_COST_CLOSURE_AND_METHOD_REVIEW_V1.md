# M1.3b — Six-run train-only GPU cost evidence and D2/D3 review handoff

**Evidence decision:** \`COST_CALIBRATION_CLOSED__OWNER_LOCAL_AUDIT_PASS\`.

**Scientific execution decision:** \`NO_GO_CONFIRMATORY_SCIENCE\`.

This document records a narrow **engineering-cost observation**, not scientific learning results, comparative GPU-compute equality, an independently witnessed host measurement, or a frozen protocol. It follows merged [PR #40](https://github.com/ehsanrs2/synaploom/pull/40) and owner-submitted read-only six-run validation output in [Issue #29](https://github.com/ehsanrs2/synaploom/issues/29). The owner ran every command on an RTX 4090; this repository did **not** ingest the private local \`/tmp\` result artifacts. The following values were transcribed from the owner's validated terminal output, not independently obtained by GitHub CI.

## Fixed scope: six manually initiated, training-only engineering runs

Each of three arms used ten tasks, one epoch per task, 39 optimizer steps per task (**390 per run**). The frozen CIFAR-100 ResNet18 512-D feature **training** portion was split into 496 training and 4 train-derived diagnostic holdout examples per class. Proposed confirmatory seeds \`501..510\` and protected original evaluation examples were **not used**. Model/calibration execution code: pinned source \`4708615b04229a3bfb3beaf4051233b96b654dac\`; frozen feature SHA \`c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47\`.

- Round 1 manual arm order: **progressive → all-recruited → fixed-sparse**.
- Round 2 manual arm order: **fixed-sparse → all-recruited → progressive**.
- No unattended scheduler, automatic three-arm dispatch or 30/90 confirmatory training occurred.
- Owner's six-file integrity checker yielded \`SIX_RUN_EVIDENCE_VALIDATION: PASS\`, \`SAME_SPLIT_ENVIRONMENT: PASS\` after all six per-run SHA, source, feature, step, status, protected-eval, parameter and recruitment checks passed. The previously recorded mask identity for fixed sparse was the same in both runs: \`05c273126ac38b992eef60b98170f0d8af7b1ee81ee5af0594acb28187e7e461\`. Checksums indicate byte consistency with recorded artifacts, not an independent attestation of GPU measurement or file provenance.

### Owner-verified host measurements (seconds, MiB)

| Arm | Round 1 wall | Round 2 wall | Mean wall (n=2) | Round-2 change | Max allocated | Max reserved | Trainable parameter elements |
|---|---:|---:|---:|---:|---:|---:|---:|
| \`progressive_reserve\` | 12.2092 | 10.0895 | **11.1494** | −17.36% | **27.50** | **32.00** | 1,172,196 |
| \`all_recruited\` | 7.8323 | 9.5786 | **8.7055** | +22.29% | **28.26** | **34.00** | 1,172,196 |
| \`fixed_sparse_mlp\` | 2.1188 | 1.8600 | **1.9894** | −12.21% | **39.69** | **66.00** | 1,172,226 |

Descriptive ratios of means from the owner's output: \`progressive/all_recruited=1.281\`, \`progressive/fixed_sparse=5.604\`, \`all_recruited/fixed_sparse=4.376\`. Lower wall is faster on this **engineering protocol** only. Actual time ratios vary by run. Memory values are peaks from PyTorch's CUDA caching allocator, **not whole-device VRAM** and not evidence that all model runtime allocations are equal. The ~30-element parameter difference between progressive and fixed sparse is ~0.0026% and does **not** imply equal GPU work.

**Interpretation limits:** All three have n=2 independent manual run **invocations**, not a statistical sample of independent hardware settings. The arm order was reversed but not independently randomized; \`all_recruited\` was second in both rounds. Cold starts, power/clock changes, CUDA scheduling, host contention, initialization and stage-dependent routing are not independently controlled. The cost timer \`train_seconds_by_task\` also includes optional train-derived current-task diagnostic evaluation; it is **not** just kernel/optimizer time. The actual confirmatory job is **10 tasks × 20 epochs × 40 minibatches/task = 8,000 steps per trajectory** and has much larger original held-out benchmark evaluation work and altered training dynamics. Do **not** multiply a 390-step wall time by \`8000/390 × 90\` to assert a full-budget projection.

## D2: reviewable resource-fairness conclusion

**Supported:** Compare the three admitted architectures at near-equal **physically allocated trainable parameter elements**, report ideal linear-MAC accounting separately, and disclose **measured engineering time and PyTorch CUDA allocator peaks** as cost axes. Per-arm measured times differ materially in the six-run engineering workload.

**Not supported:** Claim \`compute-matched\`, equal actual forward/backward kernel operations, equal throughput/energy, a fair throughput-based scientific winner, or a real CUDA sparse-kernel baseline. The current fixed-sparse MLP computes with **dense masked \`matmul\`**, not genuine sparse matmul. Fixed-sparse model mechanism/fidelity relative to an independently named external research method still requires specialist review, as do changing optimizer states and resource costs over 20 epochs. The owner endorsed *separate resource-accounting tiers*, not any claim of compute equality.

**Recommended explicit reviewer disposition:** \`PARAMETER_CAPACITY_MATCHED_COMPARISON_ACCEPTABLE_FOR_CONDITIONAL_DESIGN_REVIEW\`, \`ACTUAL_GPU_COMPUTE_MATCH_NOT_ESTABLISHED\`. The words *conditional design review* matter: this is **not** an independent D2 signoff. If the scientific estimand requires identical actual computational work, redesign the comparator and re-preregister **before** examining outcomes; otherwise explicitly restrict the hypothesis to an equal-capacity comparison and report hardware time descriptively.

## D3: substantive statistical review questions — unresolved

The candidate external O0 contrast \`progressive_reserve − fixed_sparse_mlp\` is paired by ten seeds \`501..510\` across two co-primary endpoints, \`final_average_accuracy\` and \`late_task_within_task_mean_accuracy\`. The proposed algorithm resamples ten seed **pairs** together, 10,000 draws, uses **marginal 95% percentile confidence intervals**, and declares \`FAVORABLE\` only when **both** lower bounds are positive. It symmetrically labels both upper bounds negative as \`UNFAVORABLE\`. O1 and O2 are mandatory predeclared **descriptive** order-robustness analyses, sharing the same ten seed IDs, not 20 additional independent seeds.

A reviewer should resolve, **before freezing the protocol**:
1. **Hypothesis and direction:** Is the scientific primary claim a conjunction of two positive effects? A both-needed *intersection–union* testing construction does **not automatically require the same multiplicity adjustment as an any-endpoint claim*, provided each component test's size is controlled. The implementation's *percentile-bootstrap interval at n=10* may not have nominal coverage, so this logical argument **does not validate its numerical test**. Define the separate \`UNFAVORABLE\` directional label and what its error control is supposed to mean.
2. **Small-n calibration:** Independently evaluate coverage/type-I behavior with plausible paired-difference distributions (skew, outlier, heteroskedastic or multimodal seed effects), simulation uncertainty and correlated endpoints. Compare sensitivity to a prespecified paired t CI, sign-flip/randomization procedure **under its explicit symmetry/exchangeability assumptions**, or another defensible method. Test robustness to a single influential seed without post-hoc deletion. The previously written tiny toy Gaussian demonstration does **not** meet this requirement.
3. **Completeness and estimator:** All ten paired seed outcomes must have valid, finite, provenance-verified values on each endpoint. Define precise fail-closed invalid-versus-inconclusive and retry handling; do not silently drop failed/unfavorable runs or modify the endpoint/seed list after seeing results.
4. **Scope and reporting:** Both endpoint raw paired differences and per-seed results must be exposed; intervals containing zero are *inconclusive*, not evidence of equal performance. \`all_recruited\` and O1/O2 cannot be selectively elevated to post-hoc confirmatory contrasts. Shared-seed order observations must be analyzed with seed-cluster dependence in descriptive summaries.
5. **Mechanism:** A successful external capacity-matched comparison does not by itself establish an H6/biological plasticity claim. Fresh-network comparator and stricter method-fidelity controls would be needed.

**D3 remains \`PENDING_INDEPENDENT_STATISTICAL_SIGNOFF\`.** No new inference over actual learning outcomes took place here.

## Scientific runner still blocked

Before any 30/90 full training trajectories, require: (a) independent D2 comparator/resource-method review, (b) D3 full inferential contract/size/coverage review, (c) explicit owner budget of GPU-hours, walltime and restart attempts based on defensible real workload evidence, (d) final source/feature/class-order/mask/hash-pinned and reviewed protocol, (e) tested manual science runner and a distinct explicit owner execution authorization. Existing \`90 × 8000 = 720000\` is **planned optimizer work, not a completed workload or runtime estimate**.

Historical \`EXP-H1A-002=INCONCLUSIVE\`, \`ORDER_SENSITIVE\`, \`DIRECTION_FLIP_OBSERVED\` remain unchanged. No additional GPU work is proposed by this document. **Next task is independent D2/D3 review, not endless cost-only retries.**
