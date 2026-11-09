# SynapLoom M1.3b — D1/D2/D3/D4 integrated methodological evidence review (V1)

**Review date:** 2026-11-08  
**Decision:** \`INTEGRATED_RESEARCH_REVIEW_COMPLETE__DESIGN_DECISIONS_OPEN\`  
**Execution:** \`NO_GPU\`, \`NO_PROTECTED_OUTCOMES\`, \`NO_PROTOCOL_FREEZE\`, \`NO_SCIENTIFIC_GO\`

## Executive finding and provenance boundaries

This is a *research and decision-support* packet, **not** a new candidate runner, a retrospective analysis of scientific accuracy, or an owner's approval of a final design. It integrates (1) source code and historical review documents at verified GitHub \`main=148c35aa2feaa62ac7125ca88996f7a96044a2ff\`, (2) owner-supplied SHA-verified D4 *engineering* measurements, and (3) independently retrievable statistical and continual-learning literature. None of those sources gives the variance or distribution of the *actual unobserved primary paired-seed accuracy differences*, so a powered seed count is **not yet identifiable**.

**Fixed owner research direction**: main question is positive **population mean** difference \`progressive_reserve - fixed_sparse_mlp\`, on *primary* frozen class order \`O0\`, for **both** \`final_average_accuracy\` and \`late_task_within_task_mean_accuracy\`. The owner chose **+0.02 accuracy fraction (2 percentage points) on EACH metric as a scientifically meaningful ALTERNATIVE for power planning only**. The positive null stays \`H0,m: E[D_m] <= 0\`, not \`E[D_m] <= .02\`. A positive *global* claim requires both valid component tests rejecting. Neither a majority-of-seeds sign estimand nor an automatically formal negative-direction claim is allowed. Mandatory independent human signoff was waived for this **internal, budget-limited project**; correctness and owner-specific GPU permission were **not** waived.

### Evidence classification (not all evidence has the same status)

| Evidence | Source and method | What it establishes | What it does NOT establish |
|---|---|---|---|
| D4 Progressive engineering cost | Owner local SHA/provenance verifier PASS; canonical SHA \`01d908f8a8395689d2df1470759da2c1ef8cca1a64b128d42ef3f8770531b410\` | 8,000 steps, 130.52830013199127 s, training-derived 1,000-example eval shape; original feature/source pinned | Protected benchmark score; full-science runtime guarantee; variability across repeated runs |
| D4 Fixed Sparse engineering cost | Owner local verifier PASS; SHA \`d4f7c98ea08d556395f0be01a01861a5c53bb8831468beddf4e6e90c942105c2\` | 8,000 steps, 35.31436062100693 s, same *shape* cost protocol | True sparse CUDA kernel or compute parity; scientific accuracy |
| GitHub implementation | Exact \`main\` above; PRs #49–#53 merged | D2 strict *standalone* endpoint guard, D3 synthetic test research, D4 manual engineering cost CLI exist | Final manual confirmatory runner or frozen hypothesis/seed/test |
| External methodological sources | Peer-reviewed paper links below | Available proof-based methods and dangers of order/seed dependence | Real SynapLoom seed variance, task-order-generalized efficacy, or approval of a particular implementation |

**Important provenance distinction:** the assistant did not fetch the owner's local evidence JSONs directly. The owner **reported successfully running** the Python verifier, which checked both canonical observation hashes, arm identities, 8,000 steps, protected-evaluation non-use, source commit, pinned feature digest, and ten timing-array entries. Call this \`OWNER_LOCAL_SHA_AND_PROVENANCE_VERIFIED\`, **not** independent third-party GPU attestation.

## D1 — study scope, claims and comparator selection

Current repository candidate: **10 paired seed IDs (501–510) × 3 arms × 3 class orders = 90 complete trajectories**. \`O0\` is the only intended inferential order; \`O1/O2\` were predeclared *descriptive* robustness orders, not 20 additional independent O0 seed draws; \`all_recruited\` is an internal descriptive/mechanistic reference. Original candidate files \`m13b_prereg_candidate.py\` and \`PREREGISTRATION_CANDIDATE_V1.md\` are **not frozen and retain an obsolete percentile-bootstrap proposal**: merging newer review PRs did not silently replace that contract.

**Evidence**: historical repository observations include \`EXP-H1A-ORDER-001=ORDER_SENSITIVE\` and \`EXP-H1A-EVALBATCH-001=DIRECTION_FLIP_OBSERVED\`; these are context, not outcomes of M1.3b. Separately, Li & Hiratani (ICML 2025) demonstrate the methodological importance of task order in continual learning. Their paper does **not** quantify SynapLoom's order effect, nor justify choosing favorable orders post hoc.

| Prospective scope choice | Trajectories with n paired seeds | Scientific interpretation | Tradeoff / required action |
|---|---:|---|---|
| **A. O0 / 2 arms**, primary architectural contrast only | \`2n\` | Conditional difference on the *specific frozen O0* and current training/eval rules | Lowest cost; no separate recruitment-timing causality, no measured O1/O2 robustness; explicit prospective amendment of old 90-slot proposal |
| **B. O0 / 3 arms + O1/O2 / 3 arms** | \`9n\` | Original proposed broader descriptive design; O0 inferential only | High cost; full-shape \`all_recruited\` runtime not measured; requires precommitted order-complete reporting |
| **C. O0 primary + fixed secondary matrix** | \`2n + K\` with **predeclared K and seed/order/arm slots** | O0 inference plus strictly descriptive mechanistic/order context | Research compromise, but K must be specified **before outcomes** and executed irrespective of O0 sign; do not select O1/O2 only after favorable results |

**Research recommendation for owner review:** preserve O0 × 2 arms as the smallest *scientifically honest primary question*; retain O1/O2 and all-recruited either as a **fixed, outcome-independent descriptive budgeted block** or explicitly omit them and prominently limit generality. Do **not** conflate near-equal **physical parameter allocation** with matched real GPU computation. The in-house comparator is a *fixed-masked residual MLP implemented using dense masked linear operations*, not RigL/NISPA reproduction and not a matched-kernel architecture. Its distinct architecture prevents causal attribution of an effect specifically to recruitment timing.

**Not approved**: any shrinking of the old candidate, replacing a comparator, or stage-triggering secondary runs after observing confirmatory accuracy.

## D2 — measurement fidelity, batch routing, and a trustworthy final runner

### Repository findings

1. \`src/synaploom/core/router.py\` computes a batch-level routing query from the **mean of encoded samples**. For Progressive, a sample's prediction may change when batch companions change. Fixed Sparse does not have this same routed behavior. The current contrast is **defined partly by evaluation grouping**. Altering batch size or last-batch composition after viewing protected accuracy would change the target and risk biased selection.
2. \`src/synaploom/training/evaluation.py\` calls ordered, \`shuffle=False\` minibatches and **includes the final partial batch**. \`h1a2.py\` fixes train B128, eval B256, 10×20 epochs, SGD LR=.01, momentum=.9, WD=.0005 and seen-class evaluation. Primary group composition and feature order should be pinned and hashed *before protected evaluation*.
3. \`src/synaploom/training/continual_metrics.py\` currently uses \`torch.nanmean\` for final accuracy, which tolerates intentional future-task missing entries but could also silently omit **required** evaluated cells. The new standalone \`src/synaploom/research/m13b_d2_eval_guard.py\` validates a complete 10×10 accuracy lower triangle, explicit null upper triangle, ten 20-epoch learning curves, finite and in-range values, ordered epoch/step records and agreement with co-primary summaries.
4. **Guard is not wired into a real science runner.** It does not itself attest data/seed/model/mask provenance or approved attempts/retries. Its increasing-global-step check is not a substitute for enforcing every expected optimizer step and per-arm per-seed inventory.
5. \`fixed_sparse_baseline.py\` has a fixed-mask optimizer guard; the cost probe checked the final mask, but this is **not** a substitute for qualification of the exact final training runner, masked optimizer state across task boundaries, expected mask fingerprint and all-trajectory provenance.
6. D4 shape-only cost probe deliberately evaluates 100/class **training samples already seen by the model**; its scores are **unusable** as scientific or D3 variance data. The trusted loader deserializes the *whole pinned* bundle but only returns TRAIN tensors; original protected evaluation data were not used as probe batches.

### Pre-outcome D2 research and qualification protocol (NO protected evaluation outcomes)

| Gate | Low-cost verification | Acceptance / stop condition |
|---|---|---|
| D2.1 Reference grouping | Independently derive exact 1,000-per-task evaluation B256 membership sequence (256+256+256+232); bind original order, no shuffle, no drop-last, seen-class restriction and O0 | Group list/digest reproducible; cannot change after seeing scores |
| D2.2 Batch-companion sensitivity | CPU synthetic or **train-only** fixed-state Progressive and Fixed Sparse anchors across prespecified companion sets/sizes, including B256 and partial B232; log anchor/group IDs, route/label-flip rates, logits deltas and model fingerprints | Report rates and worst cases descriptively **without** tuning architecture or selecting B from protected outcomes; assess whether measurement limitation is acceptably disclosed |
| D2.3 Comparator state | CPU one/two task mask/optimizer/momentum tests and final post-task mask digest checks on the exact proposed runner path | No masked weight or optimizer state leakage; exact mask rule fixed prospectively |
| D2.4 Result completeness | Wire standalone guard into pre-publication result validation, enforce exact \`global_steps=8000\`, 10×20 curves, all expected 10×10 cells, reference B256, and full two-arm paired seed slot registry | Reject any missing, nonfinite, duplicated or inconsistent record; **no selective exclusions** |
| D2.5 Run integrity | Check frozen source SHA, feature hash, ordered class fingerprints, seeds, masks, runtime/device, immutable evidence root and bounded retries | Fail closed and preserve all failures; require a new *manual* owner science authorization |

**Disposition:** \`D2_RESEARCH_GAPS_PINPOINTED\`; no protected batch-sensitivity diagnostics, no final runner and no method promotion. A CPU **pre-outcome** integration PR is a reasonable next engineering task *after* the scope/test decisions are reviewed, not this research-only packet.

## D3 — paired population MEAN inference: research synthesis

### Correct global logic

For endpoint m in {final average, late-task within-task mean}, define \`D_{i,m} = progressive_{i,m} - fixed_sparse_{i,m}\`. The global *positive-direction* null is \`H0_global: (mu_final<=0) OR (mu_late<=0)\`, and the alternative requires **both >0**. This is an **intersection–union test (IUT)**: each *valid* one-sided test can be at alpha=0.05 without splitting alpha for a **single global BOTH-needed claim**. Endpoint correlation affects *joint power*, not this type-I argument. This is established in Hamasaki, Evans & Asakura, *J Biopharm Stat* (2018), a methodological reference from a different application domain; it is not a clinical regulatory requirement on SynapLoom.

**+2pp chosen by owner is a scientifically important *alternative* for power, not the null.** If the owner instead later wants proof that mean improvement **exceeds +2pp**, this would be a different hypothesis \`H0:mu<=.02\`, and the n/power calculations need revision. For the current hypothesis, a result can be statistically positive yet smaller than the intended scientifically important effect; report the actual estimated gain and confidence interval without conflating significance and scientific importance.

### Candidate methods and their honest evidence level

| Method | Type-I assurance for mean | Potential power / requirements | Current disposition |
|---|---|---|---|
| Original 10,000-resample percentile paired bootstrap | **Cannot claim uniform alpha=.05** under arbitrary bounded/skewed mean-zero seed differences. Internal exact null counterexample had >=34.87% false favorable | Invalid as universal confirmatory engine at n10; descriptive uncertainty use must be labeled approximate | **REJECT as current confirmatory method** |
| One-sided paired Student-t | Exact t calibration under IID normal differences, not distribution-free for arbitrary bounded skewed differences; internal n10/n55 exact skewed-null stress shows failure | Often efficient **if** justified model assumptions and sensitivity analyses hold; observed tiny confirmatory sample cannot be used to select test post hoc | **Assumption-dependent alternative only** |
| PR51 fixed \`product_i(1+D_i)>=20\` bounded-mean e-test | Mathematically valid <=.05 under independent bounded [-1,1] paired differences and common nonpositive mean | Poor power for small true gains; at all deltas +.02 requires n>=152 even for rejection | **VALID CONDITIONAL CANDIDATE, NOT SELECTED** |
| Variance-adaptive fixed-time empirical-Bernstein or bounded-mean betting | Nonasymptotic mean inference can have an error guarantee **when a correct theorem/implementation and assumptions are verified** | Better adapts to small variance; unknown whether it is feasible at +2pp in the actual seed distribution. Needs independent CPU implementation/theorem, adversarial null tests, power scenarios and full-run cost | **HIGHEST-PRIORITY METHOD TO INVESTIGATE NEXT; NOT APPROVED** |
| Paired sign/sign-flip test | Sign and median are different estimands; sign-flip requires symmetry/exchangeability or a justified sharp randomization null | Not a general paired mean-null test for skewed effects | **NOT A DROP-IN REPLACEMENT** |

**External methodological basis:** Waudby-Smith & Ramdas (JRSS B 2024) develop variance-adaptive confidence intervals and betting processes for bounded means, both fixed-time and sequential; Howard et al. (Annals of Statistics 2021) covers nonasymptotic confidence sequences. They justify *researching and independently qualifying* these methods, not silently selecting a package's default algorithm. The code must implement the exact paper's hypotheses/conventions and be tested against small-n adversarial cases **before outcomes**.

### Seed population and fixed n

A deterministic set of consecutive seed integers is not automatically a random sample from a well-defined scientific population. A preregistered **seed-generation design** should define the generator/key space, selection mechanism, deterministic mapping to both model arms, host and software deterministic controls, and what is assumed independent across runs. Pair arms on the *same seed* within O0 but avoid training/outcome-dependent selection. Distinguish: randomness over algorithm initialization/order/optimizer vs sampling of datasets or class orders; holding one frozen dataset/order fixed does not generalize to either of those populations.

For an illustration only, the prior **known-SD Gaussian normal planning proxy** at mean +.02 and hypothetically SD=.05 yields **n≈54 paired seeds sufficient for >=90% marginal power** under that model, which via a union bound suffices for >=80% joint power if both endpoints have that same marginal model. This is **NOT a valid distribution-free n** and **not** the small-n Student-t power after estimating SD. In contrast, PR52's fixed-bet invented two-point SD=.05 example gave **n=210** for that same sufficient joint-power criterion. Different methods/assumption classes produce different n; the actual paired-effect variability is unknown. No n is approved.

**D3 research acceptance prerequisites before final method/n:** (a) exact theorem-to-code mapping; (b) predeclared one-sided 0.05 null/decision rule and no undeclared formal negative result; (c) constructed zero-mean nulls spanning symmetric, p=.8/.9/.95 skewed, rare extreme-loss and mixed outlier cases, including perfectly correlated co-primary endpoints; (d) fixed-paired generator and seed-ID registry; (e) +.02 mean alternative joint-power envelopes across deliberately **assumed** SD/endpoints dependence; (f) cost/n tradeoff; (g) all missing/failure policies and complete results; (h) no choosing the winner among tests after actual outcomes. Numerical toy null experiments are useful regression checks, but proof-based guarantee rests on the assumptions and exact implementation.

## D4 — empirical engineering cost and prospective capacity

Current local two-arm engine-cost *point observations*, verified by the owner's read-only integrity script:

| Arm | One full-shape engineer-only run | PyTorch allocator peak allocated | Peak reserved |
|---|---:|---:|---:|
| Progressive Reserve | 130.528300 s | 28.12 MiB | 34 MiB |
| Fixed Sparse MLP | 35.314361 s | 39.76 MiB | 66 MiB |
| **One paired-seed arm set (sum)** | **165.842661 s** | **Not additive** | **Not additive** |

Progressive/Fixed observed time ratio ≈ **3.696**. Peak PyTorch allocated/reserved **is not whole-GPU memory usage**. One run per arm **cannot** establish a runtime distribution, P95 cost, actual original protected-evaluation runtime or guaranteed wall-clock bound.

**Illustrative single-GPU arithmetic ONLY** for a hypothetical future owner-approved **O0 two-arm** scope, both arms at these identical observed engineering time rates (no retries, parallelization, reruns, installation, protected-eval runtime difference, or host drift):

| n paired seeds | 2n full trajectories | Direct extrapolation | Multiply by illustrative 1.5× contingency |
|---:|---:|---:|---:|
| 10 | 20 | 0.461 GPUh | 0.691 GPUh |
| 20 | 40 | 0.921 GPUh | 1.382 GPUh |
| 40 | 80 | 1.843 GPUh | 2.764 GPUh |
| 54 | 108 | 2.488 GPUh | 3.731 GPUh |
| 152 | 304 | 7.002 GPUh | 10.503 GPUh |
| 210 | 420 | 9.674 GPUh | 14.511 GPUh |

The factor 1.5 is **NOT** an upper confidence limit or hard guarantee. At the hypothetical 1.5× contingency, 3/6/12 available single-GPU-hours would fit at most ~43/86/173 paired seeds **arithmetically** for two arms O0; this is not permission to run, a realistic failure/retry budget or an adequate power claim. The original 3-arm ×3-order design cannot be priced from these two arms because \`all_recruited\` **has no comparable full-shape cost observation**.

**D4 research decision:** the two-arm **local integrity subgate** passes, but the full scientific budget gate does not: owner needs to set an explicit total GPU-hours/wall-time ceiling, acceptable owner-manual retry policy and scope; full-science execution timing remains uncertain. A capped manual science runner must be qualified separately and can never be invoked by merging this document.

## Cross-gate status and next bounded work package

| Gate | What is already established | Still blocking | Recommendation for next decision |
|---|---|---|---|
| D1 study scope | O0 mean 2-endpoint primary target; existing 90-slot draft | 2 vs 3 arms, O0 vs O0+O1/O2, secondary slots and claim generality not accepted | Owner review **A/B/C** scope table above; no silent reduction |
| D2 measurement | Existing B256 eval source, synthetic batch-composition witness, standalone completeness guard and comparator mask guard | Frozen grouping and train-only companion test; full-run guard/provenance/optimizer integration | CPU-only nonprotected engineering qualification |
| D3 mean inference | +2pp meaningful alternative; mean estimand, IUT rule; invalid bootstrap and t-distribution caveats, conditional bounded fixed bet | variance-adaptive procedure qualification; iid seed-generating mechanism, useful joint power, n and formal direction | Research and independently CPU-check a valid adaptive bounded mean candidate, compare against model-based alternative under stated assumptions |
| D4 resources | Two owner-local SHA-verified full-shape engineering costs | selected budget, retry/variability margins, owner execution approval, 3rd-arm full cost if original matrix | Owner defines hours cap after viewing scenario, before any science |

**Recommended sequencing**:
1. Finish this method/scope review and have owner inspect recommendations. **No PR merge constitutes approval of scope or statistical method.**
2. If owner wants further engineering before choosing scope, execute D2.1–D2.4 **CPU-only** qualification with synthetic/nonprotected data and an independent fail-closed runner dry-run (no GPU). Do not use proposed science seeds for engineering.
3. Compare **one** theorem-specified variance-adaptive bounded-mean fixed-time method against the fixed bet and a *separately labeled assumption-dependent* paired-t candidate on CPU/analytical nulls and +2pp alternatives, including joint endpoint dependence. Do not promote without exact proof/code review.
4. Owner specifies a **finite total GPU-hour cap** and A/B/C scope; then a *new* protocol can be designed for explicit approval, with source/feature/seed/order/mask pins and manual confirmatory execution as a separate gate.
5. If cost or the assumptions make adequate mean-confirmatory power infeasible, **predeclare an exploratory-only study** rather than relaxing test validity post hoc.

**No new confirmatory data, GPU time, parameter tuning, final statistical method, final seed count, protocol freeze, or science execution is authorized by this review.**

## External source register (evidence, not certification)

1. Waudby-Smith, I. & Ramdas, A. (2024). *Estimating means of bounded random variables by betting*, JRSS B 86(1), 1–27. DOI https://doi.org/10.1093/jrsssb/qkad009 . Methodology for variance-adaptive bounded mean CIs and betting processes; not code certification for SynapLoom.
2. Howard, S.R., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021). *Time-uniform, nonparametric, nonasymptotic confidence sequences*, Annals of Statistics 49(2):1055–1080. DOI https://doi.org/10.1214/20-AOS1991 . Related fixed-/time-uniform calibration background; no claim such a method has been implemented here.
3. Hamasaki, T., Evans, S.R. & Asakura, K. (2018). *Design, data monitoring, and analysis of clinical trials with co-primary endpoints: a review*, J Biopharm Stat 28(1):28–51. https://pmc.ncbi.nlm.nih.gov/articles/PMC6135538/ . Intersection–union logic; conceptual transfer only.
4. Li, Z. & Hiratani, N. (2025). *Optimal Task Order for Continual Learning of Multiple Tasks*, ICML/PMLR 267:34578–34603. https://proceedings.mlr.press/v267/li25z.html . Order sensitivity motivates prespecified orders; not evidence about SynapLoom effect sizes.
5. *Measuring deep learning performance: an empirical study of performance distributions across architectures and tasks* (2026), Scientific Reports. https://doi.org/10.1038/s41598-026-49656-z . Shows repeated-seed variation in other settings, not a substitute for SynapLoom paired-seed SD or IID evidence.

**Owner decision register:** \`D1_D4_RESEARCH_DECISION_REGISTER_V1.json\` alongside this report. All unapproved items are explicitly \`PENDING_OWNER_DECISION\` or \`RESEARCH_PENDING\`—not implicit approvals.
