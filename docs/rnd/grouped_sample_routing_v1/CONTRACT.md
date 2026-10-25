# Grouped sample routing v1 — frozen qualification contract

Base 2f5ebe7526d34845f45a520b574227ac71e769ca. Synthetic only; no protocol revision.
H1a q_max=0, message_rounds=0 only, revalidated each forward. Share the exact
underlying model parameters, recruited state, features, per-example k and dtype.
No statistics, parameter, structural, optimizer or production implementation changes.

Per-row queries, recruited-only scores, stable descending sort and per-row softmax.
Exact computed-score ties choose ascending node ID, matching the old research
reference. Historical production torch.topk tie behavior stays untouched; singleton
production parity is asserted only for non-tied fixtures. Rounded near ties may
change selection across devices or GEMM shapes. This is a compatibility boundary,
not a promise of arbitrary-input selection invariance in floating arithmetic.

Dispatch only selected rows to each selected expert, once per expert union. Restore
unique assignment slots before reduction in top-k order; no atomic scatter-add.
Dormant and unselected expert parameters receive no gradients; shared parameters
and selected routing-key rows receive applicable gradients. No optimizer integration.

Tolerances declared before qualification: float32 output/query/weight comparisons
CPU atol=rtol=1e-6; CUDA and CPU/CUDA atol=rtol=1e-5; gradients atol=rtol=2e-5.
These allow batched GEMM/LayerNorm accumulation differences, not semantic drift.
Require identical selected IDs on fixtures, argmax predictions, full state hashes,
and independent execution/row-count checks. Record actual maxima. TF32 is disabled
for qualification and recorded; representative performance policy must be explicit.
Near ties/argmax boundaries cannot be guaranteed by an allclose tolerance.

Bounded benchmark: batch sizes 1/4/8, dimension 8, hidden 16, capacity 6 with
recruited/k 2/1, 4/2, 6/4, seeds 7/19 parity tests. CPU and available CUDA.
Two warmups, three ordinary repeats, one separately instrumented forward per case.
Compare production batch, old reference and grouped reference using identical state.
Profile guards and hashing separately with synthetic gradients, preserve failures
and all numerical checks. No representative CUDA execution is authorized here.
