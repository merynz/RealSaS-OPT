# RealSaS N1D — Global Solver V2 Failure Localization V1 — Canonical Result

**Date:** 2026-08-19  
**Classification:** `FROZEN_LOCAL_OBJECTIVE_CONTEXT_FAIL`  
**Parent:** `RELATION_SIGNAL_PRESENT_BUT_SOLVER_FAIL_V2`  
**Prereg commit:** `ff1b0a76e4095b46c154ccb24782c0a3adbea8d5`

## Diagnostic A — M256 relation transfer: PASS

Frozen R_REL_DIS retains strong oracle-pair separability inside the complete M256 pair matrices.

```text
edges with oracle pair              1011
pooled median oracle percentile     .0573273
pooled fraction <= .25              .832839
worst-family median percentile      .150299
```

Frozen gates:

```text
pooled median <= .10       PASS
worst-family median <= .20 PASS
fraction <= .25 >= .70     PASS
```

Strata:

```text
active-active:   n=508  median .03753  frac<=.25 .89961
active-inactive: n=83   median .10271  frac<=.25 .71084
inactive-inactive:n=420 median .09386  frac<=.25 .77619
```

Per-family median oracle percentile / fraction<=.25:

```text
09908  .08961 / .788
11032  .01965 / .941
12772  .15030 / .694
13203  .08027 / .831
14404  .06430 / .798
14702  .07633 / .807
14758  .07924 / .816
15290  .01521 / .993
```

This rules out M256 relation-transfer failure as the primary explanation for V2 solver failure.

## Diagnostic B — oracle-neighbor local recoverability: FAIL

Neighbors are fixed to evaluator oracle candidate whenever available, otherwise frozen unary argmin. Each primary node then minimizes the **unchanged V2 local cost** once.

Aggregate:

```text
n                         261
pooled contain1           .521073
pooled contain2           .877395
worst-family contain1     .240000
worst-family contain2     .520000
best-worst contain2 gap   .480000
median normalized error   .982610
p90 normalized error      2.262800
absolute gate             FAIL
```

Per-family contain2:

```text
09908 .933
11032 .520
12772 .818
13203 .806
14404 .875
14702 .977
14758 1.000
15290 .920
```

The decisive witness is `11032`: R_REL_DIS oracle-pair percentile is exceptionally strong (`.01965` median), yet even with correct neighbor context the frozen local objective selects a target-near endpoint only `.52` of the time.

Therefore the relation is informative but the current unary + degree-weighted pair composition does not convert that information into sufficient local endpoint authority.

## Diagnostic C — oracle-initialized ICM stability: FAIL

Diagnostic oracle-hybrid initialization has, by construction on the primary set:

```text
contain1               .911877
contain2              1.000000
worst contain1         .760000
worst contain2        1.000000
median norm error      .322963
absolute gate          PASS
```

Running the **unchanged frozen ICM** from this nearly-correct configuration degrades it to:

```text
contain1               .478927
contain2               .827586
worst contain1         .080000
worst contain2         .480000
median norm error      1.078728
best-worst gap         .520000
absolute gate          FAIL
```

For comparison, unary-initialized V2 final is:

```text
contain1               .452107
contain2               .808429
worst contain1         .040000
worst contain2         .440000
median norm error      1.142215
```

So the bad endpoint quality is not explained only by unary initialization/local basin: even an evaluator-perfect M256 initialization is actively pulled toward the same poor family-specific basin by the frozen objective/ICM combination.

## Diagnostic D — frozen objective components

Summed over the eight family graphs:

```text
configuration              unary         pair          total
unary initial           -409.9149    -309.2307     -719.1456
unary-init final        -366.9485    -480.6305     -847.5790
oracle-hybrid initial   -167.5754    -345.2376     -512.8130
oracle-init final       -360.5757    -489.0047     -849.5804
oracle-neighbor local   -374.5444    -389.1727     -763.7170
```

The objective strongly improves as ICM leaves the truth-near oracle initialization. The oracle-initialized final objective is even slightly lower than the unary-initialized final objective, yet both have poor absolute endpoint quality. This supports **objective alignment/composition** as the dominant unresolved layer rather than a simple failure to optimize the frozen energy.

All family ICM runs converge deterministically. Oracle-init changes many nodes in the first sweep (typically 54–63/64) and converges in 6–12 sweeps, showing that the frozen objective actively rejects much of the truth-near initialization.

## Canonical classification

The preregistered hierarchy selects:

`FROZEN_LOCAL_OBJECTIVE_CONTEXT_FAIL`

because:

1. relation transfer PASS;
2. oracle-neighbor local recoverability FAIL.

Oracle-init instability independently reinforces the same conclusion.

## Mathematical interpretation / next hypothesis

R_REL_DIS constrains **relative** projected displacement:

```text
(d_i - d_j) - (DIS_i - DIS_j)
```

Under orthographic projection, adding the same 3D displacement `g` to both candidates cancels from the pair factor. Therefore local pair relations possess a common-mode / gauge degree of freedom. The current observation-only reprojection unary is the main absolute anchor, but hard-tail evidence shows that this anchor can prefer a globally coherent yet mechanically wrong common basin.

This gauge interpretation is a hypothesis, not yet a promoted architecture change. It should be tested by a separately preregistered truth-open gauge/common-mode audit before changing unary/pair weights or adding any new anchor factor.

## Boundaries

This result does **not** authorize:

- changing R_REL_DIS;
- tuning pair lambda/degree weights;
- changing graph;
- adding p_active to solver objective;
- adding restarts/new optimizer;
- changing M256;
- reopening per-carrier amplitude/vector heads.

Any modification requires a separate preregistration.

## Provenance

- diagnostic execution source SHA256: `e8b2fd05339ec0c596439ae4680f63d1ba0dc60d7fe89e451bbbfbab9d0f9932`
- full local diagnostic result SHA256: `6eb3242244c1a7c32766405ce09d91f8cb5aff0d59403e1e20f70b094a6bf46e`
