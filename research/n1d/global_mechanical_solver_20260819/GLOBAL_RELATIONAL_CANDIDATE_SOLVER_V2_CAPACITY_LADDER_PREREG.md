# RealSaS N1D — Global Relational Candidate Solver V2 — Nested Capacity Ladder Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_SOLVER_TEST__ONLY_CANDIDATE_DOMAIN_CAPACITY_MAY_CHANGE`  
**Parent:** `GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V1_PREREG.md`  
**V1 verdict:** `INCONCLUSIVE_CANDIDATE_COMPRESSION_FAIL`  
**Frozen geometry:** `BOUNDED_H_RESEARCH_CONTRACT_V1.md`  
**Frozen relation:** `R_REL_DIS`  
**Qualification:** unchanged; not a blind qualification.

## Purpose

V1 did not test the global relational solver because its observation-only M128 candidate domain narrowly failed the frozen preflight (`0.9695121951 < 0.970`). The full F16 source itself remains parity-validated and hard-tail sufficient.

V2 isolates a single variable:

> candidate-domain capacity.

No unary definition, ordering, pair relation, graph, normalization, objective weight, optimizer, endpoint gate or truth boundary may change.

## Panel / source parity

Use the same open-development e00 panel:

```text
09908, 11032, 12772, 13203, 14404, 14702, 14758, 15290
```

Before any capacity result is interpreted, replay full F16 and require exact per-family primary-2x parity with the canonical source:

```text
09908  1.0000000000
11032   .9491525424
12772  1.0000000000
13203   .9508196721
14404   .9682539683
14702  1.0000000000
14758   .9838709677
15290  1.0000000000
```

Reliable denominator must remain `492/512`. If parity fails, stop with `SOURCE_PARITY_FAIL`.

## Frozen candidate ranking / nested FPS sequence

For every carrier:

1. construct the exact frozen F16 H;
2. deduplicate XYZ after rounding each coordinate to `1e-5`;
3. compute the frozen observation-only unary reprojection score `U_raw` exactly as V1;
4. seed the domain with the same best 32 unique candidates ordered by `(U_raw,x,y,z)`;
5. from all remaining unique H candidates, run the same deterministic XYZ farthest-point sequence used by V1;
6. tie order remains: larger diversity distance, then lower `U_raw`, then lexicographic XYZ.

The FPS sequence is computed once. Therefore the capacity arms are **strictly nested prefixes** of one frozen candidate order:

```text
M128 ⊂ M192 ⊂ M256 ⊂ M384 ⊆ F16_unique_H
```

If unique H has fewer than M candidates, keep all.

No candidate may be added by truth proximity, target geometry, endpoint error or family identity.

## Capacity arms

Evaluate preflight only for:

```text
M192
M256
M384
```

M128 is carried forward only as the immutable V1 reference; it is not retuned or re-decided.

## Preflight gate — unchanged from V1

On mapping-reliable carriers, every capacity arm is evaluated independently against:

```text
pooled primary-2x containment >= 0.970
worst-family primary-2x containment >= 0.900
families primary-2x >=0.90 = 8/8
best-worst gap <= 10 percentage points
```

Strict-1x is reported.

## Capacity selection rule — frozen before results

Choose the **smallest** arm in the fixed order

```text
M192 -> M256 -> M384
```

that passes all four preflight gates.

- If M192 passes: select M192; M256/M384 may still be reported for monotonicity audit but may not replace M192 because their endpoint solver result looks better.
- If M192 fails and M256 passes: select M256.
- If M192/M256 fail and M384 passes: select M384.
- If all fail: verdict `V2_CANDIDATE_CAPACITY_PREFLIGHT_FAIL`; do not run/interpret solver.

Because domains are nested, containment must be non-decreasing per carrier as M increases. Any monotonicity violation is an implementation error and aborts interpretation.

## Solver — copied unchanged from V1

Only the selected smallest passing M is allowed into the solver.

### Graph

Same deterministic symmetrized four-nearest-neighbor P_A graph. Keep an `R_REL_DIS` edge only when >=2 common Pose-A-visible views have finite raw DIS evidence.

### Unary normalization

```text
U(i,h) = [U_raw(i,h)-median(U_raw)] / max(IQR(U_raw),1e-6)
```

### Pair relation

Frozen `R_REL_DIS`:

```text
candidate_relative_motion(v) =
    [project(h_i,v)-project(P_A_i,v)]
  - [project(h_j,v)-project(P_A_j,v)]

observed_relative_motion(v) = DIS_i(v)-DIS_j(v)

R_raw = median_v ||candidate_relative_motion-observed_relative_motion||_2
R = [R_raw-median(R_raw)] / max(IQR(R_raw),1e-6)
```

### Objective

```text
w_ij = 1/deg_i + 1/deg_j
E(x) = sum_i U(i,x_i) + sum_(i,j) w_ij R(i,j,x_i,x_j)
```

No lambda.

### Deterministic ICM

Exactly V1:

- initialize unary argmin;
- maximum 20 sweeps;
- odd sweep 0->63, even sweep 63->0;
- node candidate minimizes frozen local objective;
- tie lowest candidate index;
- stop on no-change sweep;
- no restart/randomization/annealing.

## Endpoint evaluation — unchanged

Primary evaluator-only set:

- mapping-reliable;
- truth-active (`||P_B*-P_A*|| > .005`);
- full F16 H truth-contained primary-2x;
- selected M truth-contained primary-2x.

Report `U_ONLY` and `G_REL_DIS` pooled/per-family:

```text
contain_1x
contain_2x
median normalized endpoint error
p90 normalized endpoint error
```

Hard tails: `11032`, `13203`, `15290`.

## Promotion gates — unchanged from V1

Absolute G_REL_DIS:

```text
pooled contain_2x >= 0.75
worst-family contain_2x >= 0.60
11032 contain_2x >= 0.60
13203 contain_2x >= 0.60
15290 contain_2x >= 0.60
pooled contain_1x >= 0.50
worst-family contain_1x >= 0.35
pooled median normalized endpoint error <= 1.00
best-worst contain_2x gap <= 0.30
```

Causal improvement vs U_ONLY:

```text
pooled contain_2x gain >= +0.08
```

and either:

```text
worst-family contain_2x gain >= +0.08
```

or:

```text
at least 2 of {11032,13203,15290} improve by >= +0.10
and none of those three degrades by >0.05
```

Secondary safety:

```text
all-reliable selected-M-contained pooled contain_2x
must not be lower than U_ONLY by >0.03
```

## Decisions

If no capacity passes preflight:

`V2_CANDIDATE_CAPACITY_PREFLIGHT_FAIL`

If capacity passes but solver fails promotion:

`V2_RELATION_SIGNAL_PRESENT_BUT_GLOBAL_SOLVER_FAIL`

If capacity passes and solver passes absolute + causal + safety gates:

`GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V2_PASS`

## Forbidden

- changing best-32 seed count;
- changing FPS metric/order/ties;
- choosing M after inspecting solver endpoint metrics;
- changing preflight threshold because V1 missed by one carrier;
- tuning pair/unary normalization or lambda;
- relation learning or relation replacement;
- standalone motion-head reopening;
- truth-active gating inside solver;
- free XYZ;
- Stage-B requalification;
- sealed21/external10 access;
- large/end-to-end training.
