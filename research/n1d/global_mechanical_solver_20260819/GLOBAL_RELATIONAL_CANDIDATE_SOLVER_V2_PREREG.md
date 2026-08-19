# RealSaS N1D — Global Relational Candidate Solver V2 Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_SOLVER_TEST__V1_IMMUTABLE_FAIL__ONLY_MODE_BUDGET_CHANGES`  
**Frozen geometry:** `BOUNDED_H_RESEARCH_CONTRACT_V1.md`  
**Frozen pair relation:** `R_REL_DIS`  
**Qualification:** unchanged; this is not a blind qualification.

## Motivation and V1 boundary

Global Relational Candidate Solver V1 is immutable with verdict:

`INCONCLUSIVE_CANDIDATE_COMPRESSION_FAIL`

Its M128 domain obtained pooled primary-2x containment `0.9695121951`, narrowly below the frozen `0.970` preflight gate. The solver was not run. V1 may not be retroactively changed.

V2 asks one causal follow-up only:

> If the exact same observation-only candidate compression is given a larger deterministic mode budget, does candidate-domain coverage clear preflight and thereby authorize the already-frozen relational solver test?

No result from M256 has been inspected before this preregistration.

## Single allowed change

Change only:

```text
M128 -> M256
```

Everything else is frozen exactly as V1.

## Panel / truth boundary

Open-development e00 families:

```text
09908, 11032, 12772, 13203, 14404, 14702, 14758, 15290
```

Use frozen checkpoint, observation-derived Problem-A carriers and frozen F16 H construction.

Truth is evaluator-only for source-parity check, M256 containment preflight and post-solver endpoint evaluation. Truth never enters candidate compression, unary scoring, graph, relation energies or optimization.

## M256 observation-only candidate domain

For every carrier independently:

1. construct frozen F16 H;
2. deduplicate XYZ after rounding each coordinate to `1e-5`;
3. compute the same V1 observation-only unary reprojection score:

```text
U_raw(i,h) = mean over Pose-A-usable views v of
             min_{q in frozen per-view top16 descriptor pixels}
             ||project(h,v)-q||_2
```

4. keep the best **32** unique candidates sorted by `(U_raw,x,y,z)`;
5. fill to **256** by the exact same deterministic XYZ farthest-point rule from remaining unique H candidates;
6. tie order remains: larger diversity distance, then lower U_raw, then lexicographic XYZ;
7. if H has fewer than 256 unique candidates, keep all.

No alternative M values are evaluated inside V2.

## Source-parity guard

Before M256 is interpreted, full F16 primary-2x per-family values and the mapping-reliable denominator must reproduce the canonical broad-8-family source exactly:

```text
reliable denominator 492/512
09908 1.0000000000
11032 0.9491525424
12772 1.0000000000
13203 0.9508196721
14404 0.9682539683
14702 1.0000000000
14758 0.9838709677
15290 1.0000000000
```

If parity fails, V2 result is invalid until execution/source parity is restored without changing experiment semantics.

## M256 preflight gate

Same V1 thresholds, on mapping-reliable carriers:

```text
pooled primary-2x containment >= 0.970
worst-family primary-2x containment >= 0.900
families primary-2x >=0.90 = 8/8
best-worst gap <= 10 percentage points
```

Report strict-1x.

If M256 fails:

`INCONCLUSIVE_CANDIDATE_COMPRESSION_FAIL_V2`

and do not run/interpret the solver. **No further M-budget tuning on this same 8-family panel is authorized after a V2 preflight fail.**

## Frozen graph

Identical to V1:

1. each P_A carrier connects to four nearest other carriers;
2. symmetrize;
3. keep R_REL_DIS edge when >=2 common Pose-A-visible views have finite raw DIS evidence.

No truth-active gating inside graph or solver.

## Frozen unary normalization

```text
U(i,h) = [U_raw(i,h)-median(U_raw)] / max(IQR(U_raw),1e-6)
```

## Frozen pair relation

```text
candidate_relative_motion(v) =
    [project(h_i,v)-project(P_A_i,v)]
  - [project(h_j,v)-project(P_A_j,v)]

observed_relative_motion(v) = DIS_i(v)-DIS_j(v)

R_raw = median_v ||candidate_relative_motion-observed_relative_motion||_2

R = [R_raw-median(R_raw)] / max(IQR(R_raw),1e-6)
```

No fusion, transport, delta3D, rigidity or learned relation.

## Frozen global objective

```text
w_ij = 1/deg_i + 1/deg_j

E(x) = sum_i U(i,x_i)
     + sum_(i,j) w_ij R(i,j,x_i,x_j)
```

No lambda.

## Frozen deterministic ICM

- initialization: unary argmin;
- maximum 20 sweeps;
- forward/reverse alternating node order;
- local update minimizes unary + frozen neighboring pair terms;
- tie: lowest candidate index;
- stop after a full sweep with zero changes;
- no restart, randomization, annealing, learned parameters or post-hoc iteration selection.

## Arms

- `U_ONLY`: M256 unary argmin before ICM.
- `G_REL_DIS`: final unchanged ICM configuration.

## Primary evaluator-only endpoint set

- mapping-reliable;
- truth-active (`||truth_P_B-truth_P_A|| > .005`);
- full F16 H truth-contained primary-2x;
- M256 truth-contained primary-2x.

Report pooled/per-family contain1, contain2, median normalized endpoint error and p90 normalized endpoint error. Hard tails: `11032`, `13203`, `15290`.

Secondary safety: all mapping-reliable M256-contained carriers.

## Absolute promotion gates — unchanged from V1

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

## Causal improvement gates vs U_ONLY — unchanged

Required:

```text
pooled contain_2x gain >= +0.08
```

and either:

```text
worst-family contain_2x gain >= +0.08
```

or:

```text
at least 2 of {11032,13203,15290} improve contain_2x by >= +0.10
and none of the three degrades by >0.05
```

Secondary safety:

```text
G_REL_DIS pooled contain_2x on all reliable M256-contained carriers
must not be lower than U_ONLY by >0.03
```

## Decision

If source parity passes, M256 preflight passes, and G_REL_DIS passes absolute + causal + safety gates:

`GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V2_PASS`

If M256 passes but solver fails:

`RELATION_SIGNAL_PRESENT_BUT_SOLVER_FAIL_V2`

If M256 fails:

`INCONCLUSIVE_CANDIDATE_COMPRESSION_FAIL_V2`

## Forbidden

- changing best-32 seed count;
- evaluating M192/M224/M384/etc. after seeing V2 results on this panel;
- changing unary or relation normalization;
- adding/tuning lambda;
- changing graph degree or edge rule;
- replacing R_REL_DIS;
- adding learned relation or motion head;
- truth-conditioned candidate compression;
- free XYZ;
- Stage-B requalification;
- sealed21/external10 access;
- large/end-to-end training.
