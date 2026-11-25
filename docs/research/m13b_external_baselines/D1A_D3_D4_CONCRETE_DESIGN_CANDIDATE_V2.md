# M1.3b D1-A — concrete final-design candidate for owner approval (V2)

**Status:** PROPOSED_NOT_FROZEN / D3_D4_OWNER_NUMERIC_APPROVAL_PENDING / SCIENCE_GPU_NO_GO.

Base: verified main 9e707167fa76bb630efa1c6a2c5fc9fe2ea21bb4 after merged [PR #63](https://github.com/ehsanrs2/synaploom/pull/63); [postmerge CI](https://github.com/ehsanrs2/synaploom/actions/runs/43809757842) SUCCESS. Owner approved D1 option A for design and agreed to the strategy of closing D3/D4, completing an auditable O0 comparison, and prioritizing M2 research only after interpreting results. This agreement does NOT ratify the newly proposed numeric choices below or authorize scientific or engineering GPU execution. The historical M1.3b 90-slot candidate remains intact and must not be used as the new protocol.

## 1. Proposed concrete design

- One primary frozen class order O0; progressive_reserve versus the in-house fixed_sparse_mlp. No all_recruited, O1/O2, replay, or named external RigL/NISPA claims in this study. This is an architectural contrast, not causal isolation of recruitment timing or matched physical GPU compute.
- **Proposed fixed n=360 independently sampled paired seed draws**; 720 complete trajectories, 5,760,000 optimizer steps (8,000 per trajectory), 10 tasks x 20 epochs. Fixed n is not a maximum for sequential optional stopping. No interim efficacy analysis, adaptive n, or method selection after outcome access.
- Retain frozen CIFAR-100 512-dimensional ResNet-18 features, existing H1a2 optimizer/architecture settings and seen-class evaluation; train B128 (tail 8), eval B256 (tail 232). Preserve the existing fixed-sparse mask design seed 20261026 (not the engineering calibration mask seed 17); pin the actual audited mask digest before execution. Actual protected sample order and membership must be independently verified, not merely the batch positions.
- Co-primary endpoints final_average_accuracy and late_task_within_task_mean_accuracy; paired differences are Progressive minus Fixed. Each positive null is population mean <= 0; +.02 is the owner's planning alternative, not the null boundary or a guaranteed clinically/practically important result.
- Proposed joint power planning objective 80%, **conditional on explicitly invented alternative scenarios**, not a claim of measured or uniformly guaranteed power. No formal negative-direction test.

## 2. Proposed D3 inference, including a closed-form lower bound

For each endpoint, under the null E[D_i | past] <= 0 and D_i in [-1,1], use the pre-fixed unit stake test E = product_i(1+D_i), with rejection when E >= 20. Each factor has nonnegative value and conditional expectation <= 1 under the null; hence by iterated expectation and Markov's inequality, the one-sided size is <= .05. Independent identically distributed seed-level paired outcomes with a common null mean suffice. The global positive claim requires BOTH component tests to reject (intersection-union test). The test is valid under the stated assumptions even for skewed bounded distributions; it need not have useful power for every such alternative.

For an arbitrary candidate mean m in (-1,1], the corresponding nonnegative wealth is E(m) = product_i((1+D_i)/(1+m)). If the true mean is m and the independent/common-mean conditions hold, its expectation is at most 1. A one-sided 95% lower confidence bound for the mean is therefore

    L = exp((sum_i log1p(D_i) - log(20))/n) - 1

with L=-1 if any D_i=-1. Evaluate in log-space with log1p/expm1; fail closed for nonfinite/out-of-range inputs and ambiguous threshold arithmetic. This is a marginal lower confidence bound, not a simultaneous two-sided 95% confidence interval; no independent endpoint superiority claims are made. Point estimates and both marginal lower bounds must be reported along with all paired differences and exact test evidence. A separate simultaneous CI family is outside this proposed contract. E>=20 corresponds to L>=0; the final implementation must pin exact treatment of equality and independently verify numerical decisions (no favorable rounding). If an incomplete registry or violated assumption occurs, do not call the inferential test.

### Independent seed-generation proposal (not executed)

The estimand is over independently sampled training seeds under the fixed source/data/mask/hardware policy, not over random class orders, new datasets, or random masks. Before drawing any seeds, audit and exclude all historical/pilot/calibration seed IDs and reserve disjoint engineering IDs. Proposed admissible integer range 1,000,000 through 2^32-1-100,000 inclusive, subject to actual RNG and downstream offset audit; this leaves headroom for the existing stage*10007+epoch schedule and the NumPy uint32 seeding interface. Exclude any audited historical IDs in that range.

After owner approval and pre-draw protocol/source commitment, draw exactly 360 independent uniform values **with replacement** from the fixed admissible set using a vetted unbiased OS-random selection implementation (e.g. secrets.randbelow and exact index-to-admissible-set mapping). Commit the draw transaction, complete ordered registry, generator implementation and evidence externally before the first outcome. No choosing among generated lists, seed deletion, resampling duplicates, seed replacement after failure, or post-outcome amendments. Each draw has its own stable index, pair identifier, and two arm slots even if a seed value repeats; the repeated draw is retained and handled under the prespecified iid-seed-function model, not silently deduplicated. No seeds have been drawn by this PR.

Use the same draw seed for both arms' initialization and data order, fixed mask identity across seeds, isolated fresh processes, verified RNG resets, deterministic execution settings and stable device/environment. Alternate arm execution order by draw index (180 pairs per direction); no data-dependent ordering. Seed IDs and arm ordering do not prove independence: confirm deterministic seed-to-result mapping or justify independent residual execution randomness before accepting the population-mean inference. Hardware drift, cross-run shared state, or unexplained nondeterminism must block the confirmatory interpretation. Do not use protected outcomes to choose the seed mechanism.

## 3. Exact conditional power and why n=360 is only a proposal

Independently recomputed exact binomial tails on CPU using rational arithmetic. For invented two-point delta populations with mean .02, a high observation count k qualifies when (1+high)^k(1+low)^(360-k)>=20. Marginal power is its exact binomial upper tail, and the worst-dependence joint bound for equal marginal power p is max(0,2p-1).

| Invented IID alternative | Minimum high count | One-endpoint power | Joint Frechet lower |
|---|---:|---:|---:|
| Equal 0/+4pp (SD 2pp) | 77 | rounds to 100% | rounds to 100% |
| Equal -3/+7pp (SD 5pp) | 143 | 99.996352% | 99.992705% |
| Equal -8/+12pp (SD 10pp) | 168 | 90.622132% | 81.244264% |
| 96% +6pp / 4% -94pp | 354 | 0.998634% | 0% |

These are not observed SynapLoom delta distributions. The severe-tail scenario is a direct counterexample to a universal 80% power statement. Thus the proposed test has conditional finite-sample size validity but only scenario-specific power planning. The actual variance/tails are unmeasured. The owner must either accept that power risk for a fixed bounded-mean confirmatory candidate, or prospectively choose an exploratory-only design. Do not substitute t or bootstrap to force a smaller n. A nonrejection is not evidence that the effect is zero or negative, and does not refute the architecture. The 80% figure is a planning objective, not yet an owner-approved requirement or a verified guarantee.

## 4. Proposed D4 budget and strict incomplete-run policy

Two prior owner-reported locally SHA/provenance-verified full-shape **engineering** observations used train-derived diagnostic evaluation, not scientific protected evaluation: Progressive 130.52830013199127 s, Fixed 35.31436062100693 s, pair 165.8426607529982 s. At exactly these point rates:

- n360: 720 trajectories, 16.584266 GPU-hours.
- Illustrative 1.5x: 24.876399 GPU-hours.
- **Proposed 30 occupied single-GPU-hour cap**, leaving 5.123601 h beyond that illustrative scenario. Neither the 1.5x factor nor the cap guarantees completion. Full protected-eval science wall time is unmeasured.
- Proposed per-attempt elapsed caps: Progressive 300 s, Fixed 120 s. These are timeout guardrails, not estimates; their sum across all 360 pairs exceeds 30h, so the overall cap is independently mandatory. Require process termination / watchdog qualification, including shutdown margin and measured monotonic elapsed accounting. Do not start an attempt unless its declared maximum time plus termination margin fits the remaining cap. Count failed attempts, idle reservation/setup and cleanup inside the occupied GPU budget. No overlapping runs or budget reset via restart. Freeze the shutdown margin and exact accounting implementation before execution.
- Proposed zero retries. Unexpected failure, timeout, integrity violation, or interrupted execution stops the study; preserve every terminal and not-started slot, all evidence, and consumed budget. Never drop a failed seed or infer from surviving pairs as confirmatory. If fewer than 360 complete valid pairs, report incomplete/invalid, no confirmatory verdict. Any future repair/restart requires a separately approved new prospective protocol, including treatment of previously exposed results and new independent cohort; no automatic rerun.
- No partial efficacy summaries available to the design/decision process during execution. Only cost, integrity and failure status may be monitored; protected outputs sealed until registered registry closes. No change to design after outcomes are accessed.

The 30h cap is a proposed owner decision, not an instruction to spend time or launch a command. A hard cap may prevent completing n. Scientific GPU, including an engineering repetition if separately desired, remains explicitly unauthorized.

## 5. D2 and final freeze admission checklist

Before asking for a separate manual science GO, implement and independently qualify one actual fail-closed science runner for this approved prospective design, with synthetic-only dry-runs until separately authorized:
1. Pin exact final source tree, runtime/imports/lock and device/determinism environment; trusted external feature digest before safe load, actual per-task ordered training/eval sample identities and labels, O0 digest, B128/B256 tail policy and seen-class logits rule. Pin mask and resource ledger. Inspect actual comparator fairness; report physical allocation, ideal MACs and dense-masked execution separately.
2. Owner-approved numeric D3/D4 choices, formal seed population, fixed n, test and confidence-bound decision conventions, draw/commit transaction, complete 2n slot registry, arm order and failure policy. No real seed generation yet.
3. Exact per-trajectory 8k optimizer steps, all expected epoch/evaluation records, finite outputs, fixed mask/optimizer-state safeguards; complete paired endpoint and result contract.
4. Independent pre-run and postrun evidence custody with authenticated source/manifest/attempt roots (local hashes alone not a witness); tested budget/watchdog and failure/incompletion handling; no protected-outcome access by design/tuning.
5. Independent review and explicit owner final protocol approval, then a separate owner manual execution authorization. Merging this PR is neither.

## 6. Interpretation and subsequent architecture work

If the registry and assumptions pass and BOTH endpoints reject, report limited positive mean evidence under O0, with measured magnitude and caveats (not proof of at least +2pp, H1b, forgetting mitigation, other class orders, or superiority to RigL/NISPA). If the valid complete study does not reject both, report inconclusive with effect estimates and uncertainty; no unsupported negative claim. If invalid/incomplete, report that separately. If positive evidence is absent or the observed effect is too small for the research goal, use it as an engineering decision input to revise the structural mechanism and plan separate prospective M2 need-trigger/consolidation ablations, not as a statistical proof of no benefit. Do not add multimodal complexity merely because a milestone was completed.

**Proposed owner approval bundle:** n=360, bounded-mean fixed-stake test with the specified assumptions, 80% conditional scenario power target, 30h occupied GPU ceiling, proposed attempt caps and zero retries. Alternatively owner may choose a smaller, explicitly exploratory design before outcomes. No execution authorization in either case. Numeric choice and final protocol remain unapproved.

**Disposition:** D1_A_DESIGN_APPROVED / D3_D4_CONCRETE_CANDIDATE_READY_FOR_OWNER_DECISION / D2_REAL_RUNNER_AND_ATTESTATION_PENDING / SCIENCE_GPU_NO_GO.
