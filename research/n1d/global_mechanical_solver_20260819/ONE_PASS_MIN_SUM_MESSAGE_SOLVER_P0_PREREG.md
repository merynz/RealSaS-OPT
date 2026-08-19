# RealSaS N1D — One-Pass Min-Sum Message Solver P0 — Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_FULL_PAIR_STRUCTURE_PROBE__NO_RETUNE`

## Parent evidence

Incident-Set Candidate Ranker P0 fails family-LOFO despite stable signs on several fitted aggregate features. The ranker compressed every incident relation after conditioning it on the neighbor's single frozen U_ONLY state.

Hard family 11032 is precisely a family where U_ONLY endpoint selection is weak. A wrong hard neighbor state can therefore corrupt incident evidence before any local aggregation occurs.

This P0 tests one hypothesis:

> The missing context is the retained candidate-to-candidate pair structure and neighbor uncertainty, not another scalar edge confidence or symmetric summary. One synchronous min-sum message from each neighbor should allow every central candidate to select its best compatible neighbor hypothesis without committing the neighbor to U_ONLY first.

## Frozen quantities

Identical to V2:

- F16 H;
- M256 candidate domains/order;
- U_raw and V2 median/IQR normalization;
- graph and edge eligibility;
- raw R_REL_DIS and per-edge median/IQR normalization;
- degree weights `w_ij=1/deg_i+1/deg_j`;
- no lambda;
- no new features, heads or candidates.

Only inference composition below changes. Relation and unary values are not retuned.

## Frozen normalized costs

```text
U_i(k) = robust_z(U_raw_i(k))
R_ij(k,l) = robust_z(R_REL_DIS_raw(i,j,k,l))
```

using the same median/IQR floor `1e-6` as V2.

## One-pass message definition

Initialize every neighbor only by its full frozen unary cost vector; do not choose a hard neighbor candidate.

For directed message `j -> i`, for every central candidate `k`:

```text
m_{j->i}(k) = min_l [ U_j(l) + w_ij * R_ij(k,l) ]
```

For numerical invariance subtract `min_k m_{j->i}(k)` from the whole message vector; this offset cannot change any argmin.

Central belief:

```text
B_i(k) = U_i(k) + sum_{j in N(i)} m_{j->i}(k)
```

Select:

```text
x_i = argmin_k B_i(k)
```

Ties choose lowest candidate index.

This is exactly one synchronous observation-only message pass. There is:

- no iterative BP;
- no damping;
- no message temperature;
- no restart;
- no learned weight;
- no oracle state;
- no truth-conditioned edge/candidate removal.

Every selected endpoint remains an existing M256 candidate.

## Diagnostic baseline

Also report, without promotion, the hard-U conditioned local V2 cost:

```text
H_i(k) = U_i(k) + sum_j w_ij R_ij(k, x_j^U)
```

where `x_j^U=argmin U_j`.

This isolates whether min-marginalizing neighbor uncertainty improves over hard unary neighbor commitment.

## Parity guards

Before interpretation reproduce:

```text
M256 reliable denominator        492
M256 pooled primary2x            .9756097561
M256 worst-family primary2x      .9322033898
primary evaluator n              261
U_ONLY pooled contain2           .6781609195
```

Any mismatch invalidates execution until parity is repaired without changing P0.

## Evaluation

Primary population and absolute gates are unchanged from V2:

```text
pooled contain2 >= .75
worst-family contain2 >= .60
11032/13203/15290 each >= .60
pooled contain1 >= .50
worst-family contain1 >= .35
pooled median normalized error <= 1.00
best-worst contain2 gap <= .30
```

Causal gate vs U_ONLY:

```text
pooled contain2 gain >= +.08
```

and either worst-family gain `>=+.08`, or at least two hard-tail families gain `>=+.10` with none degrading by more than `.05`.

Secondary safety: on all mapping-reliable M256-contained carriers, pooled contain2 may not fall more than `.03` below U_ONLY.

## Decision

If parity + absolute + causal + safety all pass:

`ONE_PASS_MIN_SUM_MESSAGE_SOLVER_P0_PASS`

Otherwise:

`ONE_PASS_MIN_SUM_MESSAGE_SOLVER_P0_FAIL`

## Boundaries

Forbidden after seeing P0:

- additional message iterations;
- damping;
- max-product/softmin temperatures;
- removing the central unary;
- changing unary/pair weights;
- rank/z blends;
- graph changes;
- new relation features;
- M256/F16 changes;
- free XYZ;
- sealed21/external10;
- large/end-to-end training.

If P0 fails, the result specifically falsifies one-pass min-marginalization of neighbor uncertainty under the frozen V2 factorization. Any richer message-passing or higher-order factor requires a separate preregistration.
