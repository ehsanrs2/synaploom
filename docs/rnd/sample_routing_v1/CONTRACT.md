# Sample routing reference v1

Base a5453e20ad6a67ed6df8f9566bbc2529068142c4. Isolated offline R&D.
The assertions use float32 atol=rtol=1e-6, declared in tests before execution.
This allows modest accumulation/batch-shape residuals; it is CPU-qualified only.
GPU tolerances require separate qualification, including near-boundary cases.
Argmax equality is checked explicitly; no tolerance guarantees argmax at a tie.

Use the identical model object, feature rows and recruited state for both routes.
Only q_max=0 and message_rounds=0 are supported, checked at construction and each
forward. No statistics, recruitment or optimizer updates. No training integration.
Each row projects its own query, scores recruited keys, selects top-k and softmaxes
selected scores. Stable descending sort breaks exact ties by ascending node ID.
The unchanged production torch.topk has a different unspecified tie ordering:
singleton parity is qualified on non-tied fixtures, not guaranteed at exact ties.
Unselected/dormant circuits are never called. Selected routing-key rows alone get
applicable score gradients; shared encoder/query/head parameters remain shared.
No general graph compatibility claim. Relation parameters remain allocated but unused.

Seeds 7/19; capacity 2/6 of six nodes; k=2; dimensions 8/16/8; 24 synthetic inputs.
Compare singleton oracle, companions (empty, original, reversed/extreme), weights,
predictions, arbitrary row permutations, repeated forwards, and full state hashes.
Independent dense control uses separate parameters and is not parameter-matched.
Exact score ties test deterministic selection and gradient/execution isolation.
Historical scientific verdicts remain EXP-H1A-002 INCONCLUSIVE,
EXP-H1A-ORDER-001 ORDER_SENSITIVE, EXP-H1A-EVALBATCH-001 DIRECTION_FLIP_OBSERVED.
