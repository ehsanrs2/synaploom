# M1.3b D2 — CPU evaluation policy and fail-closed endpoint guard

**Status: REVIEW-ONLY CPU CANDIDATE. Scientific execution: NO-GO.**

This is the first bounded, no-GPU technical follow-up to the owner-adopted AI-advisory policy in merged PR #48. The owner accepts the AI reports as internal revision inputs, not as independent specialist signoff. D2 and D3 still require technical correction, and no GPU science run is authorized.

## Existing code, reused rather than duplicated

The existing router obtains one query per batch through the mean of the encoded samples and chooses top-k experts once for that entire batch. Thus output for a fixed input can depend on its batch companions. This behavior is *not* independently per-sample routing.

The evaluator in **src/synaploom/training/evaluation.py** uses the existing tensor batching iterator with shuffle disabled. The iterator preserves original order and includes an incomplete final batch; it does not pad or drop it. The H1a2 protocol proposes **training batch size 128, evaluation batch size 256, 10 tasks, 20 epochs/task**, seen-class restriction and current-task evaluation every epoch.

Existing research code **src/synaploom/research/batch_composition.py** and related unit tests already diagnose batch-composition effects on nonprotected fixed-state models with mutation guards. Existing H1a batch-size sensitivity protocol considers evaluation sizes 32, 64, 128, 256, 512 and 1024. Its historical Progressive-vs-All-Recruited comparison is NOT the prospective M1.3b Progressive-vs-Fixed-Sparse primary mean contrast.

Ordinary final-average accuracy uses **torch.nanmean**, useful for legitimate future-task NaNs but unsafe if a required evaluated lower-triangle cell is absent. Strict confirmatory completeness therefore needs a separate gate.

## New additive CPU-only functionality

Source: **src/synaploom/research/m13b_d2_eval_guard.py**.
Tests: **tests/unit/test_m13b_d2_eval_guard.py**.

1. A deterministic policy candidate describes the pre-existing reference evaluation behavior: B=256, ordered no-shuffle, trailing partial batches included, future tasks not evaluated, all seen classes allowed, 20 within-task evaluation epochs per task. It explicitly sets runner integration, science execution and owner GPU authorization to false. Content SHA records identity, not a signed final science protocol.
2. A fail-closed standalone validator checks a JSON-ready future continual-run observation: ten-by-ten lower-triangle finite accuracy fractions, explicit null future-task entries, ten 20-epoch within-task learning curves, strictly increasing global steps, finite losses and accurate co-primary summary recomputation. It rejects missing observations that legacy nanmean could conceal, nonfinite/out-of-range values, forbidden eval settings or summary mismatches. This guard is **not yet wired into a real science runner**.
3. Synthetic-only CPU tests demonstrate that adding a companion can switch the selected route for a fixed anchor, while a static fixed-masked MLP keeps its anchor logits invariant. A counting evaluator checks original-order batch evaluation and preservation of a 3-example trailing batch. All tests use artificial tensors, not the CIFAR feature pack, protected evaluations or GPU training.

## What this does and does not establish

The synthetic routing case proves only **the possibility of companion-dependent route selection**. It is not empirical evidence about how frequently it affects protected CIFAR accuracy, nor a statement about the direction of the mean comparative outcome. It cannot establish O0 primary ranking or a causal effect of recruitment. Fixed Sparse remains an in-house *dense-masked* MLP with near-equal physical allocation, not an actual compute-matched sparse CUDA kernel.

No training/evaluator production path changes in this PR. Do not claim the new validator already protects actual future science outputs: runner integration and final source/feature/seed attestations are separate unfinished work.

## Outstanding low-cost D2 gates

- Owner must ratify *pre-outcome* reference evaluation B=256, deterministic original ordering, no shuffle, no drop-last, task/epoch formulas, and an explicit policy for material batch-composition sensitivity. No choosing settings from observed protected accuracy outcomes.
- Use existing batch-composition diagnostics on **synthetic or explicitly nonprotected train-derived** observations for a relevant model or a registered surrogate, and record precise inputs, class labels, model-state fingerprints, companion groups and metrics. Its results remain exploratory.
- Validate final-run fixed-mask guard/optimizer state, fully qualified candidate configuration and actual completeness guard integration. The CPU-only prototype alone is not an enforceable science gate.
- Preserve resource ledgers separating physical vs active weights and ideal MAC vs real hardware time/memory. Full 20-epoch plus protected evaluation budget remains unmeasured.
- Record owner D2 acceptance only after technical evidence is complete. A standalone causal recruitment timing claim would separately require tightly controlled All Recruited and contextual-key imprinting ablation; it is NOT implied by the narrower primary external architecture contrast.

D3 still needs a valid mean-effect inference method, joint two-endpoint power, seed-size decision or explicitly exploratory scope and a fail-closed missing data policy. D4 still needs independently checked hashes, capped full-science resource budget, a qualified manual runner and a **new, explicit owner science-run authorization**.

## Local CPU-only commands

    uv run --locked python -m pytest -q tests/unit/test_m13b_d2_eval_guard.py
    uv run --locked python -m pytest -q tests/unit/test_batch_composition.py tests/unit/test_h1a_eval_batch_sensitivity_protocol.py

No approval is granted by running tests or merging this document.
