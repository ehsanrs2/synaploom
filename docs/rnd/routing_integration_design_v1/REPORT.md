# Controlled integration design — offline qualification report v1

2026-10-25. Branch rnd/batch-composition-performance.

GO for a separately implemented opt-in inference/diagnostic session, retaining the
legacy default. NO-GO for production replacement, training integration or automatic
optimization promotion. The proposed RoutingSession is a design, not an installed
API. Scientific training/effectiveness and historical verdicts remain untouched.

## Preservation and review

The previously qualified grouped research package was independently revalidated:
92 tests passed, SHA256SUMS verified, and existing tracked files matched
2f5ebe7526d34845f45a520b574227ac71e769ca. It was committed with scoped staging as
fec8aed6c826df1ebb869af496608501d384f186. No data/ paths were staged. The old report's
statement that its work was uncommitted describes its original qualification time;
it is preserved rather than rewritten. No prior evidence was overwritten.

Read-only review covered all 254 files in the owner's two representative grouped
CPU/CUDA directories: 126 traces re-parsed against stored metrics, 126 operator
tables retained and checked, two results JSONs, full file hashes. The review
completed before the user pause; on resume its terminal process state and complete
JSON were verified. It was not restarted or overwritten. representative_review/
review.json and rnd_routing_design_review.py provide reproducible review evidence.

PERFORMANCE.md records every one of 18 representative cells, empirical envelopes,
expert/row budgets, launches, guard/hash costs and negative singleton results.
The supplied CUDA B128/R4/K4 and B128/R16/K4 figures reproduce to reported precision.
Representative data are synthetic shape proxies, not scientific checkpoints/data.
No representative profiling was automatically executed in this design phase.

## Design and investigations

DESIGN.md specifies three valid combinations: legacy batch, stable row-wise sample,
and stable grouped sample. It separates legacy/sample trace types, parameter and
execution budgets, checkpoint semantic metadata, graph mutation boundaries, mode/
RNG/hash invariance, gradient isolation and numerical/tie policy. The production
SparseRouter/default ModelConfig/model.forward remain unchanged. Existing training,
utilization, StructuralObservation and accounting consumers are explicitly identified
as incompatible with [B,K] traces; Stage 1 rejects them before mutations. The later
training-statistics proposal is not an implemented or authorized training path.

Three isolated probes live in research/routing_design_probes.py:
- SingletonProbe delegates B1 to the exact stable sample reference, avoiding grouped
  dispatch, never to production torch.topk. Tests require exact singleton-reference
  tensors, including tie-selected IDs; other parity uses frozen tolerances.
- SlotRestoreProbe replaces inverse-permutation sorting with unique-slot index_copy.
- PrecomputedRowsProbe computes assignment row indices once, before per-expert splits.
  Integer-only dispatch probes require exact baseline forward logits/weights in tests.

All share the underlying parameters and preserve selected experts and effective row
work. Existing qualified files were not edited. Bounded probes used B1/8, R4/16,K4,
width8/hidden16/router8/classes3 only. Their raw traces, operator tables, ordinary
samples, CUDA event intervals, work counts and state hashes are archived. Earlier
two-probe development evidence is preserved separately from the final five-variant
investigations. Raw directories were moved without changing bytes to
/tmp/synaploom-routing-design-raw-20261025; portable copies are in the package's four
.tar.gz archives. raw_artifact_hashes.json binds every archived file.

The representative R16 overhead reflects growth from 4 to 16 executed experts despite
constant 512 expert-row work: launches 80 ->186. Bounded B8/R16 selected 13 experts;
precomputed rows reduced launches 130 ->118, median CUDA 1.2859 ->1.1454 ms. This is
a promising isolated result, not a representative speedup or promotion gate pass.
Singleton fallback lowers launch counts, but timings vary even versus equivalent
row-reference code. Slot restore has mixed CPU benefit. All negatives remain in
PERFORMANCE.md. Numerical guard/hash opportunities are evaluated separately there;
no fail-fast checks, canonical hashing, or integrity guarantees were weakened.

## Qualification and acceptance boundaries

Final offline suite: 110 passed, one existing sparse-state warning (final_validation.txt).
Eighteen new CPU/CUDA probe tests cover singleton/grouped parity, input/parameter
backward parity, companions/permutations, exact ties, selected-expert/row hooks,
dormant reserve, recruitment/consolidation between calls and full-state invariance.
Earlier 92 tests retain singleton production parity, cross-device parameters, general
research nonfinite/topology and profiling/output-protection coverage. Hashes remain
exact; CPU float32 forward tolerance 1e-6, CUDA/cross-device 1e-5, gradients 2e-5.

GATES.md defines the later session/metadata/test matrix and explicit multi-process
performance thresholds. Those proposed interface/checkpoint/training capability
tests have not passed by implication: the new interface does not exist yet. No
one-process probe can satisfy the future five-process promotion gate. This phase
qualifies the design, research ablations and reproducibility package only.

MANUAL.md supplies fresh-path bounded tests, read-only artifact review, and manual
representative baseline/ablation commands. Actual invocation syntax is backed by
the existing/new CLI parsers. The current ablation CLI focuses B1/128; its B32
extension is explicitly required before a later full promotion claim.

Historical results remain EXP-H1A-002 INCONCLUSIVE, EXP-H1A-ORDER-001 ORDER_SENSITIVE,
EXP-H1A-EVALBATCH-001 DIRECTION_FLIP_OBSERVED. No scientific dataset was read, the new probes used no optimizer, no
full scientific training ran, no production default changed, and no
push or merge occurred. Existing toy optimizer tests still run as part of pytest;
this is not scientific training. Stop after offline design qualification.
