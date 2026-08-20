# Rank-2 R V5 — Full-Pairwise Proposal-Level Global Address Decision

**Status:** `DIRECTIONALLY_SUPPORTED_HARDTAIL__OVERALL_HIT_MIXED__HARDTAIL_OPEN`

## Frozen scientific question

Does replacing V3-A's six-anchor approximation with the **entire same-view carrier relation structure** improve proposal identity/regret, while keeping R at proposal/correspondence level and keeping geometry set-valued?

This is **not** a geometry-selection or mechanics experiment. V5 produces no XYZ, performs no H collapse, and uses no F/D or teacher information in forward selection.

## Experimental invariants

Frozen across V3-A -> V5:

- exact eight e01 development-open families;
- 128 raster bytes and frozen Hybrid V11 checkpoint;
- frozen R6.3 observable states;
- exact V3-A top-4 proposal graph/evidence;
- descriptor-derived G unary;
- maximum-real-matching cardinality semantics K;
- proposal truth evaluator semantics;
- sealed21/external10 CLOSED;
- no training, retuning, threshold sweep, free XYZ, F, or D.

**Controlled change:** R only.

- V3-A: six high-confidence G anchors approximate global relational address.
- V5: every visible carrier pair contributes a same-view signed/directional/radial relation factor over the complete real top-k x top-k proposal support.

The semantic invariant is unchanged: **R is global relational address**, not a local signature.

## Pre-truth gates

All passed before evaluator truth opened:

- raster byte/SHA parity: 128/128;
- exact V3-A proposal evidence logical digest: 8/8;
- exact compressed proposal evidence file SHA: 8/8;
- independent Kuhn/Hungarian maximum-cardinality K parity: 64/64 views;
- injectivity + fixed-cardinality legality: 64/64;
- exact objective recomputation: max abs error 0.0;
- deterministic repeat: family 11032 byte-exact;
- all 64 family-view assignments frozen;
- three-store pre-truth persistence: GitHub + Drive + Library.

Pre-truth selection: 2356 visible rows, 2352 real assignments, 4 abstentions, 63/64 views with full real matching.

## Evaluator parity

Before interpreting V5, the evaluator reproduced the frozen V3-A result exactly:

- overall G hit: 490/2288 = 21.4161%;
- V3-A hit: 679/2288 = 29.6766%;
- overall mean regret: 3.64037 -> 2.56935 px;
- overall median regret: 1.72724 -> 1.40626 px;
- rank-3 hard-tail G hit: 18/79 = 22.7848%;
- V3-A hard-tail hit: 21/79 = 26.5823%;
- V3-A hard-tail carrier direction: 6 improve / 3 same / 6 worsen.

All historical parity checks passed.

## V5 result

### Overall applicable proposal rows (n=2288)

| Method | Oracle-best hit | Mean regret (assigned) | Median regret (assigned) | Abstain |
|---|---:|---:|---:|---:|
| G | 490/2288 = 21.4161% | 3.64037 px | 1.72724 px | 0 |
| V3-A anchor-R | 679/2288 = 29.6766% | 2.56935 px | 1.40626 px | 4 |
| **V5 full-pair R** | **676/2288 = 29.5455%** | **2.23448 px** | **1.33385 px** | 4 |

Relative to V3-A:

- oracle-best hit: **-3 rows / -0.1311 percentage points**;
- assigned mean regret: **13.03% lower**;
- assigned median regret: **5.15% lower**;
- directly comparable rows: **396 improve / 1548 same / 339 worsen**;
- 3 rows abstain in both; 2 rows change abstention coverage.

So V5 is **mixed overall**: spatial proposal regret improves materially, but exact oracle-best hit does not improve population-wide.

### Exact rank-3 hard-tail target (79 view-rows, 15 carriers)

| Method | Oracle-best hit | Mean regret | Median regret |
|---|---:|---:|---:|
| G | 18/79 = 22.7848% | 4.66508 px | 1.55431 px |
| V3-A anchor-R | 21/79 = 26.5823% | 3.22369 px | 1.52044 px |
| **V5 full-pair R** | **26/79 = 32.9114%** | **3.10965 px** | **1.40650 px** |

Relative to V3-A on the target tail:

- oracle-best hit: **+5 rows / +6.3291 percentage points**;
- mean regret: **3.54% lower**;
- median regret: **7.49% lower**;
- rows: **20 improve / 47 same / 12 worsen**;
- carrier-mean direction: **10/15 improve, 2/15 same, 3/15 worsen**;
- no hard-tail abstention.

A large heterogeneous failure remains: `14404 / carrier 32` was corrected by V3-A (mean regret 1.054 px) but V5 returns it to 19.454 px. Conversely `14404 / carrier 40` improves strongly (12.063 -> 1.099 px), so the family is not simply globally incompatible with full-pair R.

## Decision

Two claims must be separated.

1. **Does whole-structure full-pair R contain useful identity information beyond the six-anchor approximation? — SUPPORTED on the intended hard tail.**
   V5 improves hard-tail hit, mean regret, median regret, and 10/15 carrier means over V3-A under an evaluator that exactly replays historical V3-A.

2. **Is equal-weight full-pair G+R sufficient to close the hard tail? — NO.**
   Hard-tail oracle-best hit is still only 32.91%, three target carriers regress versus V3-A, and overall exact hit is slightly lower than V3-A.

Therefore V5 is **not** a hard-tail closure and does not authorize product qualification, geometry collapse, or claiming G+R sufficiency.

## Next scientific gate

Do not retune V5 after truth. Do not add F/D inside this result.

The next preregistered experiment should isolate one question before adding new evidence channels:

> Are the remaining/full-pair regressions caused by treating carrier pairs whose relative geometry genuinely changes under articulation as equally invariant global-address constraints?

This should first be answered by a **post-hoc diagnostic only** on the now-open development panel, followed by a new pre-truth experiment in which exactly one observable pair-compatibility/gating mechanism is the controlled change. G, proposal evidence, panel, evaluator and no-XYZ policy remain frozen.

If an observation-only R-compatible gate cannot recover the missing hard tail, then F or D becomes the next justified information channel rather than an untracked formulation shift.

## Authorities

- V5 contract blob SHA: `db98ddf3b885ad28598592f8e30350dab198b1a5`
- V5 solver SHA-256: `ea7ab996f881a08538c3cc049ca799e98783919255ed2c5214c01feada3161d7`
- pre-truth selection SHA-256: `277946d94a3685222fb0cef4ad3b34d4c4656657309cccdc65204e01a2386887`
- preflight SHA-256: `000edfe80d2e351ac71fda8db7bc24a5de6f1037a24858ec73c326bf61e51e5a`
- pre-truth freeze authority SHA-256: `4958392ac7da8ebd987f57339e2ed8526f50d0e6f49a380b7ab02b049e336bb6`
- pre-truth persistence SHA-256: `b3d947f3c8d5262a2ed656c964dca030d719747625422883082635f479412dbf`
- truth evaluator SHA-256: `250552f11458f024c8ba69401c8c2ca075b053414f2489184fe8bd19409554b8`
- raw truth evaluation SHA-256: `6c05bb0e0e92c586bedb3bc6460ef12ec104cfcbefac921c3039667e23504af7`
- result summary SHA-256: `301ee743bd9691f45b2cf2f2d01777cb4d514d4c5098489f71e7b1c14abb790b`

`sealed21 = CLOSED`; `external10 = CLOSED`.
