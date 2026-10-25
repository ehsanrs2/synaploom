# Measured performance envelope and priorities v1

Owner-executed representative synthetic runs were read, not repeated. All 254
files in the CPU/CUDA grouped-v1 benchmark directories were hashed; all 126 traces
were parsed and compared with their stored metrics, and all 126 operator tables
were reviewed/retained in representative_review/review.json. Each case has ten
ordinary repeats in one process, two warmups and a separate instrumented forward.
Hardware: RTX 4090; torch 2.14.1+cu130; CPU threads=1; TF32 off; dimensions
256/512/64, 100 classes, capacity16. These are shape proxies, not scientific data.
No multi-process confidence interval or universal latency guarantee is supported.

| device | B | R/K | production ms | row sample ms | grouped ms | row/grouped | grouped/prod |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 1 | 4/1 | 0.0964 | 0.1220 | 0.1364 | 0.89x | 1.41x |
| cpu | 1 | 4/4 | 0.1836 | 0.2428 | 0.2738 | 0.89x | 1.49x |
| cpu | 1 | 16/4 | 0.2214 | 0.2429 | 0.2844 | 0.85x | 1.28x |
| cpu | 32 | 4/1 | 0.2485 | 2.5572 | 0.4324 | 5.91x | 1.74x |
| cpu | 32 | 4/4 | 0.7631 | 5.6454 | 0.8510 | 6.63x | 1.12x |
| cpu | 32 | 16/4 | 0.7536 | 6.2248 | 2.2985 | 2.71x | 3.05x |
| cpu | 128 | 4/1 | 0.6932 | 9.7847 | 0.9422 | 10.38x | 1.36x |
| cpu | 128 | 4/4 | 2.2581 | 25.7697 | 2.3260 | 11.08x | 1.03x |
| cpu | 128 | 16/4 | 2.2029 | 24.7489 | 3.2765 | 7.55x | 1.49x |
| cuda | 1 | 4/1 | 0.9126 | 0.3972 | 0.4714 | 0.84x | 0.52x |
| cuda | 1 | 4/4 | 0.3878 | 0.5266 | 0.6664 | 0.79x | 1.72x |
| cuda | 1 | 16/4 | 0.3758 | 0.5040 | 0.6504 | 0.77x | 1.73x |
| cuda | 32 | 4/1 | 0.2529 | 6.8273 | 0.6343 | 10.76x | 2.51x |
| cuda | 32 | 4/4 | 0.3994 | 10.7984 | 0.7142 | 15.12x | 1.79x |
| cuda | 32 | 16/4 | 0.4039 | 10.9032 | 1.3494 | 8.08x | 3.34x |
| cuda | 128 | 4/1 | 0.2558 | 26.0827 | 0.7059 | 36.95x | 2.76x |
| cuda | 128 | 4/4 | 0.4013 | 42.1787 | 0.7371 | 57.22x | 1.84x |
| cuda | 128 | 16/4 | 0.4025 | 42.5024 | 1.4251 | 29.82x | 3.54x |

Observed envelope across those cells (min/max of cell medians, not population bounds):

| device | production ms | row sample ms | grouped ms |
|---|---:|---:|---:|
| cpu | 0.0964–2.2581 | 0.1220–25.7697 | 0.1364–3.2765 |
| cuda | 0.2529–0.9126 | 0.3972–42.5024 | 0.4714–1.4251 |

The B1/R4/K1 CUDA production median (.9126 ms) is an observed outlier relative
to larger-batch production cells; it remains in the envelope. No outliers
were removed. Grouped singleton regression versus row-reference is retained.

## R16/K4 dispatch investigation

At CUDA B128/K4, grouped R4 -> R16 costs .7371 -> 1.4251 ms. Expert-row
work stays 512; expert union/calls grows 4 -> 16; launches 80 -> 186. Production
executes four experts for either capacity. Grouped is 1.84x/3.54x production
and 57.22x/29.82x faster than the row-wise sample reference in these two cells.
The semantic union budget differs from production and cannot be optimized away
by dropping low-weight selected experts or running dense dormant capacity.

CUDA grouped trace CPU-op counts at B128 (R4/R16): linear 10/34,
native_layer_norm 4/16, gelu 4/16, index_select 5/17, div 5/17; sort 4/4 and
unique_consecutive 1/1. This implicates per-expert calls/gather/index arithmetic
and smaller expert batches, not a growth in inverse-permutation sort count.
These are observed op counts, not an additive causal wall-time decomposition.

## Bounded ablations executed in this design phase

Only B1/8, width8, hidden16, router_dim8, classes3, R4/16,K4 ran automatically.
Five ordinary repeats per cell; one process per device; traces separate.
SingletonProbe delegates to the exact stable sample reference for B1.
SlotRestoreProbe replaces the inverse sort with unique-slot index_copy.
PrecomputedRowsProbe divides all ordered assignment slots by K once, then
splits rows per expert; neither executes any additional expert or input row.

| device | B | R | grouped ms | singleton ms | slot-restore ms | precomputed-rows ms |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 1 | 4 | 0.1882 | 0.1533 | 0.2089 | 0.1784 |
| cpu | 1 | 16 | 0.1973 | 0.1573 | 0.1837 | 0.1835 |
| cpu | 8 | 4 | 0.1925 | 0.1968 | 0.1957 | 0.1906 |
| cpu | 8 | 16 | 0.4194 | 0.4523 | 0.4082 | 0.3821 |
| cuda | 1 | 4 | 0.7156 | 0.5749 | 0.7162 | 0.6589 |
| cuda | 1 | 16 | 0.6826 | 0.5365 | 0.6371 | 0.6871 |
| cuda | 8 | 4 | 0.7154 | 0.6724 | 0.6663 | 0.6497 |
| cuda | 8 | 16 | 1.2859 | 1.2624 | 1.1684 | 1.1454 |

At B8/R16/K4 the bounded fixture selects 13 experts, with 32 expert-row evaluations.
CUDA baseline launches 130; slot restore 128; row precomputation 118. The latter
saves 12 division launches and changes median 1.2859 ->1.1454 ms in this run;
it does not remove per-expert network/gather launches. CPU .4194 ->.3821 ms.
Singleton fallback reduces CUDA B1/R4 and R16 launches 68/66 -> 51/49, matching
the row-reference launch counts. Timings still vary: B1/R4 singleton .5749 ms
versus row-reference .4427 ms despite equivalent algorithms. Do not promote
either optimization on these single-process timings. Slot restoration has
mixed CPU results (B1/R4 slower .1882 ->.2089 ms); it is lower priority.
Initial two-probe development runs are preserved in probe_cpu.tar.gz/probe_cuda.tar.gz;
the five-variant investigation_*.json summaries and .tar.gz archives are the final ablation evidence.

## Numerical diagnostics and canonical hash — separate opportunity

Unchanged standalone probes in the existing representative runs give:

| device | R at B128/K4 | finite logits ms | gradient scan ms | parameter scan ms | state SHA ms |
|---|---:|---:|---:|---:|---:|
| cpu | 4 | 0.0217 | 2.1131 | 7.5494 | 7.7704 |
| cpu | 16 | 0.0222 | 7.9468 | 8.0586 | 7.8209 |
| cuda | 4 | 0.0340 | 2.0545 | 5.3529 | 10.8407 |
| cuda | 16 | 0.0343 | 6.9326 | 5.4835 | 10.7687 |

These scans/hashes have their own traces. They are not forward latency shares
and cannot be summed into an engine timing. Gradients were prepared by one
synthetic backward; no optimizer step or scientific training was performed.
Parameter/hash work scans full allocated capacity; gradient scans depend on
which expert parameters actually have gradients. Preserve both kinds of cost.

| priority | candidate | expected benefit | scientific risk / parity requirement |
|---:|---|---|---|
| 1 | B1 stable-reference fallback | bounded lower launches and wall vs grouped; representative unknown | preserve sample tie rule, gradients, work; never delegate to production topk; pass singleton manual gate |
| 2 | precompute assignment row indices once | bounded launch reduction; plausible R16 gain | exact integer index/work identity, no extra experts/rows; full backward and performance gate |
| 3 | CPU thread sweep and reduce avoidable session/export overhead | workload-dependent, unmeasured | preserve every check; compare identical precision/state; no automatic thread default change |
| 4 | aggregate diagnostic host transfers | potentially large CUDA scan savings | high: preserve first failing tensor order, phase/context/details, overflow and all norm/max diagnostics; no optimizer before checks |
| 5 | batch immutable state snapshot transfers | potentially reduce many D2H copies | high: canonical sorted name/dtype/shape/bytes unchanged; every tensor included; synchronize before hashing; exact SHA; no digest cache |
| 6 | unique-slot restoration | two fewer launches bounded, mixed wall benefit | no duplicates/unfilled slots, gradients/parity and manual nonregression gate; do not promote yet |

For diagnostic optimization, a fast all-finite status reduction may select the
existing ordered failure-detail path only if ALL current checks and first-error
semantics remain identical. Aggregate norm/max arithmetic can change rounding
and ties; include zero/None/empty gradients, NaN/Inf, overflow and equal-max
parameter-name cases. Do not amortize checks across optimizer steps.

For hash optimization, stage a coherent immutable snapshot and serialize with
the frozen model_state_sha256 encoding. Test dtype/shape/name/order/noncontiguous
and unsupported-layout failures, mutation during capture, and canonical hashes
across CPU/CUDA. Parameter updates invalidate any byte snapshot. These are design
proposals only: no numerical guard or hashing implementation is changed.
