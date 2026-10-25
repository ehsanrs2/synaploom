# Opt-in routing session — draft implementation v1

**Status: REVIEW ONLY / NOT QUALIFIED.** Tracks [#21](https://github.com/ehsanrs2/synaploom/issues/21). This implementation is stacked on the research-evidence branch; it is **not** merged into `main`.

## Supported interface (research-only)

```python
from synaploom.research.routing_session import RoutingSession, RoutingSpec

legacy = RoutingSession(model)
legacy_output = legacy.forward(x)   # delegates model(x); original ModelOutput

sample = RoutingSession(model, RoutingSpec(
    granularity="sample",
    execution="grouped",  # alternatively "row_reference"
    tie_policy="node_id_ascending_v1",
))
with torch.no_grad():
    result = sample.forward(x, purpose="inference")
# result.logits [B,C]; result.trace.selected_node_ids [B,K]

# Diagnostic backward only: does not own an optimizer or update training state.
output = sample.forward(x, purpose="diagnostic_backward")
output.logits.square().sum().backward()
```

`RoutingSession` is a plain Python facade, **not an nn.Module**. It owns no parameters; saves/loads/checkpoints target `model`. The default batch path preserves the existing `ModelOutput`/`ForwardTrace` and constructor/config/forward behavior. The sample output uses a distinct versioned `SampleRoutingOutputV1` and `SampleRoutingTraceV1`; it must **not** be passed to `TrainingEngine` or legacy continual result accounting.

## Stage-1 restrictions and safety decisions

- Explicit allowed specs: `batch/legacy/legacy_topk`, `sample/row_reference/node_id_ascending_v1`, `sample/grouped/node_id_ascending_v1`. Invalid combinations reject at construction.
- Sample path rejects `q_max != 0`, `message_rounds != 0`, `update_statistics=True`, training-purpose calls, invalid shape/device/dtype, and non-finite routing scores/logits. Graph-state topology rechecked per call.
- Sample tie policy sorts computed scores descending and then node IDs ascending at **exact** ties. Batch delegates legacy `torch.topk`, including its own tie behavior. Do not use batch as sample oracle at exact ties.
- `purpose="inference"` disables graph construction for sample calls; `purpose="diagnostic_backward"` permits gradient collection but executes **no optimizer step**, structural controller, utilization statistic update or model mutation. The facade cannot prevent external caller code from using those gradients in an unauthorized optimizer step. No training integration is approved.
- Sample grouped execution dispatches only selected sample rows to the union of selected recruited experts. All installed model weights are shared. Report `effective_expert_rows=B*K`, actual expert calls and union size separately.
- No new checkpoint metadata/loader adapter is provided. Plain state_dict can be strictly loaded into the unchanged `SynapLoomModel`. **Full checkpoint-envelope semantic validation, RNG/optimizer/controller round-trip, safe replay/counterfactual metadata and stale transaction/concurrency controls are outstanding.**
- This draft does **not** implement automatic B1 fallback, slot restoration or precomputed-row optimizations. They remain separately gated research probes; avoid silent runtime optimization promotion.
- Supported float32 fixture tolerances: CPU forward 1e-6, CUDA/cross-device 1e-5, gradients 2e-5. Expected class/selected IDs are exact. Near-tie instability may still occur; do not widen tolerances to mask changed experts.
- The session checks structural version before/after forward but is **not** a concurrency lock or atomic transaction. Concurrent mutation is unsupported; state hashes/RNG/mode comparisons belong to diagnostic gate tests.

## Files and qualification status

- `src/synaploom/research/routing_session.py`: isolated opt-in prototype. Production source untouched.
- `tests/unit/test_routing_session.py`: proposed CPU/CUDA parity, companion/permutation invariance, gradient isolation, exact ties, checkpoint-state_dict compatibility, recruitment/consolidation, input/purpose failures and structural-version checks.
- Source and tests have been **authored remotely via GitHub**. They have **not yet been executed**. Previous 110 passing tests belong to the parent research branch only and are not evidence that this draft implementation works.
- Prerequisite: GitHub Actions must execute real test steps; historical runner failures with zero steps cannot qualify this change. Required review includes dedicated CPU/CUDA tests, full baseline suite, and provenance/metadata constraints from `docs/rnd/routing_integration_design_v1/GATES.md`.
- Do not merge/release until incomplete gates are explicitly accepted or implemented. Training integration, default replacement and scientific benefit claims remain NO-GO.

## Test commands for available CI/qualified environments

```bash
uv lock --check
uv sync --locked --extra vision
uv run --locked pytest -vv
uv run --locked pytest -vv tests/unit/test_routing_session.py
```

These commands are **proposed verification**, not recorded executions.

## Next gate

1. Review this stacked PR against parent draft evidence PR #22, especially all existing `GATES.md` obligations.
2. Run actual CI or bounded CPU/CUDA validation in a qualified environment.
3. Implement/review explicit checkpoint semantic provenance checks and deliberate diagnostic transaction ownership, if required for acceptance.
4. Measure default-delegation overhead and grouped-session overhead under the frozen five-process gates. Without passing representative gates, performance promotion remains NO-GO.
5. Decide whether to split raw artifacts out of PR #22 before either merge. Keep historical H1a verdicts unchanged.
