# SynapLoom M1.3b — D1/D3/D4 owner decision bridge after merged D2 (V1)

**Prepared:** 2026-11-11. **Status:** `DECISION_SUPPORT_ONLY__NOT_FINAL_PROTOCOL`. **Source:** verified `main=efe30e41aea53addb27bf2e4eff54ce099f018f7` after merged [PR #61](https://github.com/ehsanrs2/synaploom/pull/61); its exact-head [CI #42980405534](https://github.com/ehsanrs2/synaploom/actions/runs/42980405534) passed. **No training, protected outcomes, GPU run, method selection, n selection, protocol freeze or new owner execution authorization.**

This paper is a **decision synthesis of already reviewed evidence**, not a fresh benchmark or an additional toy statistical method. See `D1_D4_INTEGRATED_RESEARCH_EVIDENCE_REVIEW_V1.md`, `D3_OWNER_TWO_PP_POWER_GPU_BUDGET_PLANNING_V1.md`, `D3_FIXEDTIME_HEDGED_BETTING_CPU_REVIEW_V1.md`, `D3_MAURER_PONTIL_FIXEDTIME_CPU_REVIEW_V1.md`, `D3_PRPL_EB_THEOREM2_CPU_REVIEW_V1.md`, and `D2_DETACHED_PIN_TRUST_CONTRACT_V1.md`.

## Executive judgement

1. **D1: smallest defensible primary contrast is O0 / 2 arms** (`progressive_reserve` minus in-house `fixed_sparse_mlp`). Recommend it **conditionally for owner decision** when a limited-budget claim, strictly conditional on fixed class order O0, is acceptable. It is **not** a full external RigL/NISPA comparison or a recruitment-timing causal ablation. The prior 90-slot, 3-arm × 3-order candidate must be **prospectively amended**, not silently discarded; there is no owner approval to reduce the matrix at this point.
2. **D3: no confirmatory n or method is supportable from existing real seed-distribution evidence.** The selected estimand is paired-seed **population mean**, and a meaningful alternative of **+0.02 accuracy fraction on EACH co-primary endpoint**. The primary formal null remains `mu <= 0` for each; a global favorable result needs **both** valid one-sided tests at marginal alpha=.05 (IUT). No formal negative-direction claim has been selected. The original n=10 percentile bootstrap and naive small-n Student t are **not** default-valid for arbitrary bounded, skewed mean differences. Four bounded-mean methods have been explored; none has been selected and no observed M1.3b variance is available.
3. **D4: observed engineering-only two-arm O0 pair cost is 165.842661 seconds** at exactly one owner-verified cost run per arm (Progressive 130.528300, Fixed Sparse 35.314361). This is **not** an estimate of protected/full-science runtime distribution or an assured cap. The **third arm has no comparable full-shape cost**; hence the original 9n-trajectory candidate cannot honestly be priced. No total GPU-hour budget has been approved.
4. **D2: CPU software-path exercises are complete as research evidence through PR #61**, but source/data trust roots, exact real protected sample membership and independent scientific-run custody remain unimplemented. A merged green CI does not certify the scientific runner.

**Practical recommendation:** decide **D1 scope and D4 budget first**, while registering D3's desired confidence and assumption tier. If no prospective n within budget can deliver an honestly defended joint-power target, label the study **exploratory-only before any protected outcomes**. Do not force n=10, choose a test from favorable protected results, or lower scientific standards because compute is available.

## D1 — mutually exclusive scope decisions

| Scope | Complete trajectories for n paired seeds | Valid headline claim | Limitation / outstanding input |
|---|---:|---|---|
| **A — O0 two-arm** (conditional recommended minimal primary study) | `2n` | Mean performance contrast for **this specific** O0 / B256 grouping and in-house fixed-mask comparator | No evidence of O1/O2 generalization or isolated recruitment-timing causality; requires an explicit owner-approved amendment of historical 90-slot candidate |
| **B — O0+O1+O2 three-arm** (original proposal) | `9n` | O0 primary inference plus *predeclared descriptive* order and third-arm context | Extra `all_recruited` cost unknown; order evidence cannot simply be pooled into 3n independent O0 seeds |
| **C — O0 primary plus predeclared secondary slots** | `2n+K` | O0 primary inference and bounded, **descriptive** extra comparisons | `K`, exact orders/arms/seeds and reporting must be specified **before outcomes**, independent of whether primary results look good |

**Comparator fidelity:** `fixed_sparse_mlp` uses dense masked linear operations; near-equal physical capacity and estimated ideal nonzero MACs are **not equivalent hardware costs**. Neither RigL nor NISPA is admitted. Do not publish a broad sparse SOTA or causally isolated recruitment mechanism claim from A alone. `O0` is selected for comparability, **not** because it is a representative random task order. Earlier results indicate order sensitivity and an evaluation-batch direction flip.

## D3 — decision on the TYPE of inference before deciding n

Define for every prospectively registered seed `s` and endpoint `m`:

`D(s,m)=accuracy(progressive_reserve,O0,s,m)-accuracy(fixed_sparse_mlp,O0,s,m)`, with `D∈[-1,+1]`.

The owner-selected estimand is `E_seed[D(s,m)]`, **not** `P(D>0)` and not an average over task orders. Formal positive global claim is `E[D_final]>0 AND E[D_late]>0`, both endpoints positive at their **predeclared** procedures. Scientific meaning of a +2 pp design alternative is separate from an inferential threshold at zero. Independence / common conditional means and an interpretable seed-generating population require an explicit prospective justification; merely calling seed IDs `501..510` independent is insufficient.

### Evidence-based method-screening gate (no winner yet)

| Candidate | What past review actually supports | High-priority limitation before choosing |
|---|---|---|
| **Fixed `lambda=1` e-bet** | Simple finite-sample bounded-IID-mean size argument and exact synthetic power; one-endpoint power **90.516%** at **n=210** in invented symmetric `{-0.03,+0.07}` (mean+.02, SD .05) case; same constructed endpoint coupling gave BOTH-needed joint **~81.03–90.52%** | Under invented rare severe loss `96% +.06 / 4% -.94`, n210 fixed-stake power is only **2.988%**; actual seed tails completely unknown |
| Fixed equal-weight hedged e-mixture | Valid pre-fixed stake mixture, sometimes better under extreme tails | Synthetic n210 / SD .05 marginal power **36.508%**, far worse than full-stake fixed `lambda=1` |
| Maurer–Pontil fixed-n empirical Bernstein | A reviewed theorem-based bounded-mean fixed-time bound | In invented SD .05 case marginal rejection **0% at n210** and **16.347% at n1000**; not a generic 'more adaptive = more powerful' solution |
| Predictable plug-in empirical Bernstein | Theorem-2/Eq-16 mapped predictable bounded-mean construction | Invented SD .05 / n210 simulation **0/4000 marginal rejections** versus fixed-stake ~90.6%; at n1000 SD .05 ~99.85%, but rare catastrophic-tail behavior can reverse the preference |
| Paired Student t under additional model assumptions | Potentially more power under justified near-normal/appropriate noise assumptions | Exact small-n skewed mean-null counterexamples show it is **not automatically universal size-.05 valid**; sign flips test a different symmetry assumption, not mean zero alone |

**Do not promote a method from a table of invented scenarios to a universal recommendation.** The M1.3b primary paired-seed variability, endpoint dependence and tail behavior are unobserved; cost measurements cannot estimate them. A candidate final method needs a fixed theorem/assumptions contract, correct implementation checks, full pre-outcome seed population, endpoint power target and a realistic alternative family. Prefer **one constrained decision gate** over additional unbounded synthetic-method exploration.

**Pointwise impossibility check, carefully scoped:** under the full bounded IID mean-null, the hypothetical null `P(D=+.02)=50/51, P(D=-1)=1/51` makes every n observations equal +.02 with probability `(50/51)^n`. Therefore any **nonrandomized 5%-size fixed-n test that rejects that exact all-+.02 sample** needs `n>=152`. This is **not** a universal 80% power requirement for SynapLoom, not a proven minimum n for all positive alternatives, not an observed result and not a rationale to run 152 seeds automatically.

**Joint power:** endpoint correlation is unknown. If marginal power under a stated hypothetical model is at least 90% on **each** endpoint, the worst-dependence union-bound suffices for at least 80% probability of rejecting both. The numerical scenario is not a guarantee about SynapLoom. Any final power plan must separately state whether *80% joint power* is actually the owner's desired target; it has NOT yet been ratified.

**Stop rule:** if confidence, population, tail assumptions, honest joint power and budget do not align, predeclare **exploratory/descriptive** scope and do **not** advertise confirmatory alpha=.05 success.

## D4 — evidence-labeled compute planning

Owner-local integrity-checked *engineering* point costs (10 tasks ×20 epochs, 8,000 training steps, **train-derived diagnostic evaluation**, not original protected performance evaluation):

- `progressive_reserve`: **130.528300 s**; hash `01d908f8a8395689d2df1470759da2c1ef8cca1a64b128d42ef3f8770531b410`.
- `fixed_sparse_mlp`: **35.314361 s**; hash `d4f7c98ea08d556395f0be01a01861a5c53bb8831468beddf4e6e90c942105c2`.
- Pair sum **165.842661 s**; observed Progressive/Fixed wall ratio ~**3.696**. This ratio is not a future performance guarantee, nor a proof of equal GPU computation.

**Simple serial O0/two-arm extrapolation** at precisely those two past engineering rates; **not actual science costs or a statistical runtime upper bound**, and not approved runs:

| Candidate n (scenario only) | O0 trajectories `2n` | Arithmetic GPU-hours | With illustrative 1.5× (NOT a bound) |
|---:|---:|---:|---:|
| 10 | 20 | 0.461 | 0.691 |
| 40 | 80 | 1.843 | 2.764 |
| 54 | 108 | 2.488 | 3.731 |
| 152 | 304 | 7.002 | 10.503 |
| 210 | 420 | 9.674 | 14.511 |
| 358 | 716 | 16.492 | 24.738 |
| 1000 | 2000 | 46.067 | 69.101 |

**Crucial:** the 1.5 multiplier is not validated by repeated full-science pilot runs. Nothing above includes measured full protected evaluation, variation of later seeds/driver/GPU load, failed attempts, retry budgets or a full-shape `all_recruited` run. **B and C cannot be costed completely from the existing evidence.** The owner must set a **hard total GPU-hour cap, wall-time/interrupt policy and worst-case attempt allocation** *before* any science. If budget is smaller than the desired conservative power design, report infeasibility without quietly changing the statistical claim.

## Decision ladder and required owner signoffs

These are **questions to decide**, not answers silently made by merging this PR:

| Order | Owner decision | Current defensible provisional preference | Freeze/GO state |
|---|---|---|---|
| 1 | **D1 claim & scope**: A two-arm O0, B original three-arm × three-order, or C predeclared bounded extras | **Prefer A conditionally** for limited-budget, precisely O0-specific inference; B only if scientific need and paid budget justify broader descriptive matrix | **PENDING** |
| 2 | **D3 statistical guarantee**: robust bounded-mean control vs assumption-dependent mean model vs declared exploratory outcome; desired joint power | Preserve the **mean of matched seed differences** and both-needed IUT; select method **before outcomes** using an explicit risk/assumption and seed-generation contract; do NOT revive n10 bootstrap | **PENDING** |
| 3 | **D4 budget**: hard GPU-hour cap, per-run timeout, max allowed attempts/retries and accepted runtime contingency | Price A with measured point rates **clearly only as a scenario**, do not price B until third-arm cost measured separately | **PENDING** |
| 4 | **D3 registered n and failure policy**: inferential power with both endpoints under justified assumptions; all seeds and retries | Choose only after (1)–(3); if no feasible justified n, downgrade **prospectively** to exploratory, with transparent uncertainty | **PENDING** |
| 5 | **D2 real-science trust**: pre-outcome signed/attested source/feature/seed/mask/eval grouping + actual runner/attempt ledger | Review independent attestation and protected-data firewall; pass CPU-safe qualifications, distinct from proposed D2 synthetic tests | **PENDING** |
| 6 | **Explicit final separate owner GO** | Manual invocation only after all earlier gates; never auto-execute from a merge | **NO_GO** |

**No claim of RigL/NISPA fidelity, no GPU science permission, no protocol freeze and no statistical test/n are adopted here.** Previous candidate `m13b_prereg_candidate.py` is an unfrozen historical proposal and must be amended explicitly if the owner selects A/C. Do not rewrite historical H1a findings.

**Disposition:** `D1_D3_D4_RECOMMENDATION_PACKET_READY_FOR_OWNER_REVIEW` / `D3_SEED_POPULATION_METHOD_N_UNRESOLVED` / `D4_FINAL_SCIENCE_BUDGET_UNRESOLVED` / `D2_REAL_SCIENCE_TRUST_PENDING` / `SCIENCE_GPU_NO_GO`.
