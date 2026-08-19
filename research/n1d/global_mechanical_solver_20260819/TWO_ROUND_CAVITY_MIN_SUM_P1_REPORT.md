# RealSaS N1D — Two-Round Cavity Min-Sum P1 — Canonical Result

**Date:** 2026-08-19  
**Verdict:** `SECOND_ROUND_NOT_MATERIAL__FACTOR_OR_OBJECTIVE_INSUFFICIENCY_REMAINS`  
**Prereg commit:** `85267a10f0cc53451ee9101b1d21338366aa4260`  
**Qualification:** truth-open fixed-depth solver probe only.

## Validity / round-1 parity

All frozen source/evaluator guards pass before round-2 interpretation:

```text
M256 reliable denominator          492
M256 pooled primary2x              .9756097561
M256 worst-family primary2x        .9322033898
primary evaluator n                261
U_ONLY primary contain2            .6781609195
round-1 primary contain1           .4712643678
round-1 primary contain2           .7777777778
round-1 worst-family contain2      .4000000000
round-1 median normalized error    1.0715530716
round-1 secondary contain2         .8666666667
round-1 candidate-index parity     512/512 exact
```

Thus round 1 is endpoint-identical to canonical `ONE_PASS_MIN_SUM_MESSAGE_SOLVER_P0` on every node and P1 is valid.

## Frozen two-round cavity update

Round 1:

```text
m_{j->i}^{(1)}(k_i)
  = min_{k_j} [U_j(k_j) + w_ij R_ij(k_i,k_j)]
```

Round 2 excludes the destination echo:

```text
m_{j->i}^{(2)}(k_i)
  = min_{k_j} [
      U_j(k_j)
      + sum_{q in N(j), q != i} m_{q->j}^{(1)}(k_j)
      + w_ij R_ij(k_i,k_j)
    ]
```

Final belief:

```text
B_i^(2)(k_i) = U_i(k_i) + sum_j m_{j->i}^{(2)}(k_i)
```

All messages are normalized only by subtracting their own minimum. No damping, temperature, lambda, asynchronous schedule, third round, new factor, candidate or free XYZ is used.

## Primary endpoint result

```text
                    U_ONLY      ROUND1       ROUND2
contain1             .37165       .47126       .45977
contain2             .67816       .77778       .81609
worst-family c1      .00000       .12000       .12000
worst-family c2      .28000       .40000       .52000
median norm error   1.31793      1.07155      1.11291
p90 norm error      4.15209      3.24548      3.04516
best-worst c2 gap    .55333       .57727       .48000
```

Per-family contain2, round1 -> round2:

```text
09908   .93333 -> .90000   (-.03333)
11032   .40000 -> .52000   (+.12000)
12772   .81818 -> .72727   (-.09091)
13203   .63889 -> .66667   (+.02778)
14404   .56250 -> .75000   (+.18750)
14702   .97727 -> .97727   (+.00000)
14758   .96970 ->1.00000   (+.03030)
15290   .80000 -> .82000   (+.02000)
```

Round 2 versus U_ONLY:

```text
pooled contain2 gain   +.1379310345
worst-family c2 gain   +.2400000000
11032 gain             +.2400000000
13203 gain             +.0555555556
15290 gain             +.1200000000
```

## Secondary safety

All mapping-reliable M256-contained carriers, n=480:

```text
                    U_ONLY      ROUND1       ROUND2
contain1             .46667       .61042       .60417
contain2             .79583       .86667       .88750
worst-family c1      .16364       .38182       .40000
worst-family c2      .58182       .67273       .72727
median norm error   1.04946       .86270       .86221
p90 norm error      3.20938      2.21658      2.16307
```

Round 2 remains broadly safe and improves secondary contain2 over both U_ONLY and round 1.

## Frozen gates

```text
absolute quality       FAIL
causal vs U_ONLY       PASS
hard-tail causal       PASS
secondary safety       PASS
depth materiality      FAIL
amplifies-factor-bias  FALSE
```

Absolute gates still fail, including:

```text
worst-family contain2 .52 < .60
pooled contain1       .45977 < .50
worst-family contain1 .12 < .35
median norm error     1.11291 > 1.00
best-worst c2 gap     .48 > .30
```

The preregistered depth-materiality gate also fails because family 12772 degrades by `.09091`, exceeding the permitted `.05` per-family degradation, even though pooled and worst-family contain2 improve materially.

The preregistered decision is therefore:

`SECOND_ROUND_NOT_MATERIAL__FACTOR_OR_OBJECTIVE_INSUFFICIENCY_REMAINS`

## Scientific interpretation

A second cavity round is **not useless**. It raises primary pooled contain2 `.77778 -> .81609`, lifts the classic hard family 11032 `.40 -> .52`, and raises secondary worst-family contain2 `.67273 -> .72727`.

But added message depth is not uniformly reliable under the unchanged factors: 12772 regresses strongly, pooled contain1 decreases, and median normalized error becomes slightly worse than round 1. The full product-style absolute gate remains far from closed.

Therefore the evidence does not authorize a blind “keep iterating BP until it works” route. Full pair structure and neighbor uncertainty remain important, but the residual limitation is now more consistent with **factor/objective insufficiency or context-dependent factor calibration** than with simple lack of propagation depth.

The next safe question should localize why round 2 helps 11032/14404 while hurting 12772, using observation/evaluator separation, before changing weights, relation or graph.

## Boundaries

This result does not authorize:

- round 3/4/5 sweeps;
- damping or softmin temperature;
- pair/unary lambda tuning;
- changing/removing the central unary;
- graph changes;
- R_REL_DIS retuning;
- M256/F16 changes;
- free XYZ;
- learned message functions;
- sealed21/external10 access;
- large/end-to-end retraining.

## Provenance

- prereg commit: `85267a10f0cc53451ee9101b1d21338366aa4260`
- execution source SHA256: `37852c2151a14e1bbc62d704263c731cc0ef7ad7688d69e3c47301e255858c64`
- full local result SHA256: `7572d5adae05198b013c4c4f4da0cf35e5b4574657691205865e2d0f8227af9f`
