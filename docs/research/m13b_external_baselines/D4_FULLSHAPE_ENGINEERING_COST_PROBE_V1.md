# M1.3b / D4 — Full-shape cost-only GPU calibration: prepared, NOT authorized

**Decision:** \`D4_FULLSHAPE_COST_PROBE_IMPLEMENTATION_READY_FOR_CPU_REVIEW\`. **Scientific experiment and even the engineering GPU probe have NOT been owner-authorized.** This PR contains code and CPU safety/shape tests only, without executing one CUDA optimizer step.

## Why this is necessary

Earlier M1.3b engineering cost studies measured **390 training optimizer steps** per model, with ten tasks × one epoch and only **40 train-derived diagnostic evaluation examples per task**. Actual proposed science has **10 tasks × 20 epochs × 5000 training examples per task** (40 batches of B128, last 8), resulting in **8000 steps per complete model trajectory**. In the original evaluator, current-task evaluations happen **once per epoch** and full seen-task evaluations happen after each task.

For a 1000-example-per-task evaluation geometry, this corresponds to:

| Dimension | Full-shape engineering calibration |
|---|---:|
| Tasks / epochs | 10 / 20 |
| Optimizer steps | 8000 |
| Current-task epoch evaluation calls | 200 |
| Final-stage seen-task evaluation calls | 1+2+…+10 = 55 |
| Total task-evaluation calls | 255 |
| Examples per eval call | 1000 |
| Evaluated example occurrences (not unique examples) | 255,000 |
| Eval batches at B256 | 1020 |
| Final eval batch | 232 |

Thus simply multiplying a 390-step timing by 8000/390 would omit a large and stage-dependent evaluation workload. Existing H1a2 B256 routing is **batch mean dependent**; keep grouping and last partial batch semantics identical while comparing wall cost.

## Data and claim boundary — critical

The **only permitted source of tensor values** for this engineering cost probe is the originally frozen CIFAR-100 ResNet18 *training* feature partition from the trusted pinned feature bundle. The existing trusted loader verifies the entire feature-bundle SHA256 and model metadata **before** deserialization. Because the file contains both train and original evaluation tensors, **loading the bundle necessarily deserializes the whole file**; however the loader returns only training tensors and **the new probe never constructs evaluation batches from the original protected evaluation partition**. No actual protected-evaluation accuracy or model comparison is emitted.

To match the 1000-example evaluation geometry without reading the original evaluation tensors, select **100 samples per class from the same 500 original training examples per class** as diagnostic input. This intentionally creates *train/eval overlap*. Consequently **all diagnostic accuracy and forgetting values are meaningless and must never be used for scientific inference or to tune the model**; they remain internal to the continual training harness and are not written to the cost-only evidence. The probe is an engineering runtime approximation with real **batch/task/epoch/evaluation shape**, not a replica of the original held-out benchmark's statistical behavior.

- Engineering model seed **13**, diagnostic subset selection seed **41**, fixed-mask seed **17**; proposed science seeds **501–510** must not be used in this calibration.
- One selected arm per manual invocation: \`progressive_reserve\`, \`fixed_sparse_mlp\`, or \`all_recruited\`. Do not automatically chain arms, skip failed arms, or cherry-pick cost observations.
- The legacy batch-mean router and current training/evaluator path are reused unchanged.
- Resource observations, if the owner separately approves manual execution, are only **per-arm elapsed wall time, training/evaluation phase durations, CUDA allocated/reserved peaks, tensor/source/model/mask fingerprints and completed optimizer steps**. No accuracy, confusion matrix, forgetting, scientific p-value, training history or protected evaluation scores are exported.
- Parameter near-matching does not imply equal measured GPU work; the fixed-mask MLP still uses dense \`F.linear\`.

## Code, host protection and CPU audit

New source: \`src/synaploom/research/m13b_d4_fullshape_cost.py\` — emits a no-run plan, creates deterministic training-only stream with exact shapes, and has an owner-gated engineering worker using the existing continual harness.

New CLI: \`experiments/scripts/m13b_d4_fullshape_cost.py\` — **defaults to no-run** even without any flags. Safe inspection:

    uv run --locked python experiments/scripts/m13b_d4_fullshape_cost.py --describe-only

Before any optional later manual GPU execution the CLI requires all of:
- a **separate explicit owner decision approving that engineering-only GPU invocation** and a chosen time cap, not just merging a PR;
- explicit \`--manual-fullshape-cost-probe\`, **one** \`--arm\`, \`--owner-approval-acknowledged\`, \`--trusted-feature-source\` and \`--allow-in-sample-diagnostic-evaluation\`;
- owner-verified training artifact origin and pinned file digest \`c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47\`;
- a SHA-pinned \`--expected-head\` matching the actually checked-out clean Git commit and a fresh \`--output-dir\` (no resume/overwrite);
- mandatory \`--max-wall-seconds\` in the bounded range **60..7200** for each arm. The parent starts a separate worker with \`subprocess.run(timeout=...)\`, kills it if the budget is exceeded and saves an explicit failure record rather than partial positive evidence. This wall limit is on the worker process, not a scheduler/enforced energy quota.

The purpose of these controls is **operator error resistance, not cryptographic authentication of owner identity**. An invocation flag is not evidence that approval was given outside the tool. Never conflate engineering cost approval with authorization to train real confirmatory seeds or consume protected evaluation data.

The evidence root is created only after cheap source/input checks and reserved before feature data loading. On success it contains \`intent.json\`, \`cost-only-observation.json\` and a canonical SHA256 checksum; on a worker failure or timeout it contains \`failure.json\` and the initial intent, without a false-complete observation. The SHA identifies bytes but is not an authenticated external timestamp or signature. If the worker is killed, the process's exact final GPU usage might not be available; treat the probe as failed, not as a complete cost record.

Tests: \`tests/unit/test_m13b_d4_fullshape_cost.py\` — synthetic CPU-only 50k-row training inventory generated as an expanded view, validates 10×20 full geometry, 1000-per-task *in-training* diagnostic evaluation subset, per-class counts, deterministic sample identity and RNG isolation, nonfinite/bad-label refusal, default describe-only behavior, invalid output paths, wrong source SHA and direct worker entry denial **without dispatching CUDA**.

## Explicit next decision (not automatic on merging)

1. Owner **may** separately authorize one or more **bounded engineering-only GPU cost probes** after inspecting source/CI and choosing a real maximum total wall-clock/GPU-hours ceiling. Individual child process timeout must be <=7200 seconds and manual operator run order predeclared.
2. After actual host evidence exists, compare per-arm full-shape wall times and memory, record variability and conservatism/overheads, then choose a feasible experimental scope and n. A single cost observation is not a statistical estimate of full scientific runtime variance, and no guarantee of identical runtime on original protected evaluation data follows from matching shapes.
3. Only after D2 batch-sensitivity/evaluation-policy and fail-closed guard integration, D3 selected valid mean test/paired seed population/+2pp joint-power analysis or exploratory designation, and D4 frozen provenance/seed budget can a **separate manual confirmatory science authorization** be requested.

**No GPU run, training, source freeze, actual runtime figure, final number of seeds, hypothesis change or publication claim is authorized by this PR.**

## Reproduce CPU-only

    uv run --locked python -m pytest -q tests/unit/test_m13b_d4_fullshape_cost.py
    uv run --locked python experiments/scripts/m13b_d4_fullshape_cost.py --describe-only

The user-approved +2pp mean effect remains **for planning on both co-primary endpoints only** (merged PR52), and all toy error/power numbers remain hypothetical rather than measured SynapLoom results.
