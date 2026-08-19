# RealSaS N1D — Two-Round Cavity Min-Sum P1 — Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_FIXED_DEPTH_MESSAGE_PROBE__NO_DEPTH_SWEEP`

## Parent result

Parent: `ONE_PASS_MIN_SUM_MESSAGE_SOLVER_P0_FAIL`, prereg commit `a1169be16dea0b706a179c59c979770b0ae06347`.

One synchronous full-pair min-sum pass preserves M256×M256 neighbor uncertainty and is materially causal/safe:

```text
primary contain2        .6781609 -> .7777778
worst-family contain2   .2800000 -> .4000000
11032                   .2800000 -> .4000000
15290                   .7000000 -> .8000000
secondary contain2      .7958333 -> .8666667
```

but absolute endpoint gates fail. This P1 asks one question only:

> Does exactly one additional **synchronous cavity min-sum round** add useful multi-hop context, or does propagation amplify the same factor/objective bias?

No endpoint result from round 2 may be inspected before this preregistration.

## Frozen quantities

Identical to P0/V2:

- F16 H and M256 candidate domains/order;
- U_raw definition and median/IQR robust-z normalization (`IQR floor=1e-6`);
- graph and edge eligibility;
- raw R_REL_DIS and per-edge median/IQR robust-z normalization;
- degree weight `w_ij = 1/deg_i + 1/deg_j`;
- no lambda, temperature, damping or learned parameter;
- evaluator populations and local-scale endpoint metrics;
- selected endpoint must be an existing M256 candidate.

No relation, graph, candidate or weight changes are allowed.

## Message schedule — exactly two synchronous cavity rounds

Initialize every directed message to zero:

```text
m_{j->i}^{(0)}(k_i) = 0
```

### Round 1

For every directed edge `j -> i`:

```text
m_{j->i}^{(1)}(k_i)
  = min_{k_j} [ U_j(k_j) + w_ij R_ij(k_i,k_j) ]
```

Subtract `min_{k_i}` from the whole message vector after computation. This constant offset cannot affect any argmin.

Round-1 belief:

```text
B_i^(1)(k_i) = U_i(k_i) + sum_{j in N(i)} m_{j->i}^{(1)}(k_i)
```

The round-1 selected endpoints **must be exactly endpoint-identical** to canonical `ONE_PASS_MIN_SUM_MESSAGE_SOLVER_P0` on all 512 nodes before round-2 endpoint metrics are interpreted.

### Round 2 — cavity update

For every directed edge `j -> i`:

```text
m_{j->i}^{(2)}(k_i)
  = min_{k_j} [
      U_j(k_j)
      + sum_{q in N(j), q != i} m_{q->j}^{(1)}(k_j)
      + w_ij R_ij(k_i,k_j)
    ]
```

Again subtract the minimum over `k_i` from each whole message vector.

Final P1 belief:

```text
B_i^(2)(k_i) = U_i(k_i) + sum_{j in N(i)} m_{j->i}^{(2)}(k_i)
```

Select `argmin_k B_i^(2)(k)`; exact ties use lowest candidate index.

There is no third round, asynchronous update, damping, message reuse that includes the destination echo, or alternative schedule.

## Source/parity guards

Before round-2 interpretation reproduce:

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
```

In addition, round-1 selected candidate indices must equal the canonical P0 implementation for every node in every family. Any mismatch invalidates P1 until parity is repaired without changing this preregistration.

## Round-2 product-style gates

Primary absolute gates remain V2/P0 gates:

```text
pooled contain2 >= .75
worst-family contain2 >= .60
11032/13203/15290 each contain2 >= .60
pooled contain1 >= .50
worst-family contain1 >= .35
pooled median normalized error <= 1.00
best-worst contain2 gap <= .30
```

Causal gate versus U_ONLY remains:

```text
pooled contain2 gain >= +.08
```

and either:

```text
worst-family contain2 gain >= +.08
```

or:

```text
at least 2 of {11032,13203,15290} gain >= +.10
and none degrades by >.05
```

Secondary safety:

```text
round2 secondary pooled contain2 >= U_ONLY secondary contain2 - .03
```

## Fixed depth-materiality gate vs round 1

Round 2 is considered to provide **material additional context** only if all hold:

```text
round2 primary pooled contain2 - round1 >= +.03
round2 worst-family contain2   - round1 >= +.08
no family primary contain2 degrades from round1 by > .05
round2 secondary contain2 >= round1 secondary contain2 - .03
```

No alternative definition may be substituted after viewing results.

## Decision tree

If parity fails:

`INVALID_ROUND1_PARITY_FAIL`

Else if absolute + causal + safety all pass:

`TWO_ROUND_CAVITY_MIN_SUM_P1_ABSOLUTE_PASS`

Else if depth-materiality passes:

`TWO_ROUND_CAVITY_MIN_SUM_DEPTH_SUPPORTED_BUT_ABSOLUTE_FAIL`

Else if round2 pooled primary contain2 is below round1 by more than `.03` **or** any hard-tail family degrades from round1 by more than `.10`:

`SECOND_ROUND_AMPLIFIES_FACTOR_BIAS`

Otherwise:

`SECOND_ROUND_NOT_MATERIAL__FACTOR_OR_OBJECTIVE_INSUFFICIENCY_REMAINS`

## Boundaries

Forbidden on this panel after P1:

- testing round 3/4/5 as a sweep;
- changing synchronous/cavity schedule;
- damping;
- temperature/softmin;
- lambda/unary/pair weight changes;
- removing central unary;
- graph-degree/topology changes;
- R_REL_DIS changes;
- M256/F16 changes;
- free XYZ;
- learned message function;
- sealed21/external10 access;
- large/end-to-end training.

If P1 supports depth, any further depth experiment requires a new preregistration and cannot be selected by a sweep on this same panel. If P1 does not support depth, deeper propagation on unchanged factors is not authorized by this result.
