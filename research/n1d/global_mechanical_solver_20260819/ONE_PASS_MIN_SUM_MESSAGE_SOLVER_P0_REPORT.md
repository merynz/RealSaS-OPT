# RealSaS N1D — One-Pass Min-Sum Message Solver P0 — Canonical Result

**Date:** 2026-08-19  
**Verdict:** `ONE_PASS_MIN_SUM_MESSAGE_SOLVER_P0_FAIL`  
**Prereg commit:** `a1169be16dea0b706a179c59c979770b0ae06347`  
**Qualification:** truth-open development solver probe only.

## Parity

Exact frozen source/evaluator guards pass:

```text
M256 reliable denominator        492
M256 pooled primary2x            .9756097561
M256 worst-family primary2x      .9322033898
primary evaluator n              261
U_ONLY primary contain2          .6781609195
```

## Treatment

No hard neighbor candidate is selected before relation aggregation. For every directed edge:

```text
m_{j->i}(k) = min_l [U_j(l) + w_ij R_ij(k,l)]
B_i(k)      = U_i(k) + sum_j m_{j->i}(k)
```

with frozen V2 median/IQR-normalized unary/pair costs, graph and degree weights. Exactly one synchronous observation-only pass; no iteration, damping, tuning, new candidate or free XYZ.

## Primary result

```text
                    U_ONLY    HARD_U_LOCAL   MIN_SUM_P0
contain1             .37165       .45594       .47126
contain2             .67816       .79693       .77778
worst-family c2      .28000       .32000       .40000
median norm error   1.31793      1.11675      1.07155
p90 norm error      4.15209      3.04516      3.24548
```

Per-family contain2:

```text
family   U_ONLY   HARD_U_LOCAL   MIN_SUM_P0
09908     .8333       .9667         .9333
11032     .2800       .3200         .4000
12772     .8182       .9091         .8182
13203     .6111       .6389         .6389
14404     .5000       .6250         .5625
14702     .8182       .9773         .9773
14758     .8182      1.0000         .9697
15290     .7000       .8400         .8000
```

MIN_SUM_P0 gains vs U_ONLY:

```text
pooled contain2   +.0996168582
worst-family c2   +.1200000000
11032             +.1200000000
13203             +.0277777778
15290             +.1000000000
```

## Secondary safety

All mapping-reliable M256-contained carriers, n=480:

```text
                    U_ONLY    MIN_SUM_P0
contain1             .46667       .61042
contain2             .79583       .86667
worst-family c2      .58182       .67273
median norm error   1.04946       .86270
```

Safety passes strongly.

## Frozen gates

```text
pooled causal gate   PASS
hard-tail causal     PASS
secondary safety     PASS
absolute quality     FAIL
```

Absolute failure remains because, among other conditions:

```text
worst-family contain2 .40 < .60
pooled contain1       .47126 < .50
worst-family contain1 .12 < .35
median norm error     1.07155 > 1.00
best-worst c2 gap     .57727 > .30
```

Canonical verdict:

`ONE_PASS_MIN_SUM_MESSAGE_SOLVER_P0_FAIL`

## Scientific interpretation

This is a materially different failure from the compressed incident-set ranker.

Retaining the complete M256×M256 pair compatibility matrices and min-marginalizing neighbor uncertainty produces a large causal improvement over U_ONLY, improves the hard family 11032 from `.28` to `.40`, improves secondary pooled contain2 by `+.07083`, and improves secondary worst-family contain2 from `.5818` to `.6727`.

Therefore **hard conditioning each relation on one U_ONLY neighbor state was demonstrably lossy**. Candidate-to-candidate relational structure contains useful information that scalar incident summaries discarded.

However one synchronous min-sum pass is not sufficient for absolute endpoint authority. The remaining failure is concentrated in hard-tail exactness/family spread rather than broad safety.

This result supports preserving full pair structure in the next architecture/solver formulation, but does not by itself authorize additional message iterations, weight changes or a new optimizer on this same panel.

## Boundaries

Not authorized from this result alone:

- additional BP/min-sum iterations;
- damping or temperatures;
- changing/removing central unary;
- pair/unary lambda tuning;
- rank/z blends;
- graph or R_REL_DIS changes;
- M256/F16 changes;
- free XYZ;
- sealed21/external10 access;
- large/end-to-end training.

## Provenance

- execution source SHA256: `d7bbb2a7da35a245eaa894454686e71598f5cfceed5c17dedcf765c8012ef0f1`
- full local result SHA256: `9eb269ba9433fcd4df382f94532edd27d7f8143894089bab40584285439e583c`
