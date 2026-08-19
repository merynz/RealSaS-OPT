# RealSaS N1D — Global Relational Candidate Solver V3 — Rank-Calibrated Objective Preregistration

**Date:** 2026-08-19  
**Status:** `PREREGISTERED_TRUTH_OPEN_SOLVER_TEST__SINGLE_OBJECTIVE_CALIBRATION_CHANGE`

## Parent evidence

- M256 passes candidate-domain preflight.
- R_REL_DIS oracle-pair percentile transfers strongly to complete M256 pair matrices.
- Fixed median/IQR z-score objective causally improves over unary but fails absolute endpoint gates.
- Correct-neighbor pair-only oracle rank is much stronger than unary, but multi-edge aggregation is borderline and fixed unary interaction is heterogeneous.
- Simple common-mode anchor and confidence-frozen unary anchors are not supported.

V3 tests one hypothesis:

> The relation ordering is useful, but summing unbounded/heterogeneous z-scored energies allows edge/node scale and extreme-value effects to miscalibrate the global objective. A rank/percentile calibration should preserve every factor's ordering while making factors comparable and bounded.

No V3 endpoint result has been inspected before this preregistration.

## Frozen quantities

Identical to V2:

- checkpoint;
- F16 H construction;
- M256 observation-only compression;
- candidate ordering;
- graph and edge eligibility;
- raw U_raw definition;
- raw R_REL_DIS definition;
- degree weights `1/deg_i + 1/deg_j`;
- no lambda;
- deterministic ICM update order/ties/20-sweep limit;
- evaluator sets and endpoint gates.

## Single objective change — empirical percentile calibration

Replace the V2 median/IQR cost calibration with a deterministic empirical mid-rank percentile independently inside each candidate domain.

For any finite cost array `c` with `N` entries, define for every entry:

```text
P(c_k) = [# entries < c_k + 0.5 * # entries == c_k] / N
```

Exact floating equality is used for ties because each array is produced deterministically in one computation. No epsilon binning or learned calibration.

### Unary

For node `i`:

```text
U_rank(i,h) = empirical_percentile(U_raw(i,h))
```

This transform is monotonic, so `U_RANK_ONLY` argmin is exactly the same candidate as V2 U_ONLY.

### Pair

For edge `(i,j)`:

```text
R_rank(i,j,h_i,h_j) = empirical_percentile(R_REL_DIS_raw matrix)
```

This transform is monotonic and therefore preserves the per-edge candidate-pair ordering measured by the relation-separability audits.

## Frozen objective

```text
w_ij = 1/deg_i + 1/deg_j

E_rank(x) = sum_i U_rank(i,x_i)
          + sum_(i,j) w_ij R_rank(i,j,x_i,x_j)
```

No offset, temperature, lambda, clipping threshold or additional factor.

## Preflight / source parity

M256 preflight and full-F16 parity must reproduce V2 exactly before V3 solver evaluation:

```text
reliable denominator 492/512
M256 pooled primary2x .9756097561
M256 worst family      .9322033898
8/8 >=.90
best-worst gap         .0677966102
```

If not, execution is invalid until source parity is restored without changing V3 semantics.

## Arms

- `U_RANK_ONLY`: unary percentile argmin, expected endpoint-identical to V2 U_ONLY.
- `G_RANK_REL`: deterministic ICM under `E_rank`.

The baseline endpoint configuration must exactly equal V2 U_ONLY; otherwise V3 is invalid.

## Primary evaluator-only set

Unchanged V2 primary:

- mapping reliable;
- truth-active;
- full F16 contained2x;
- M256 contained2x.

Secondary safety: all mapping-reliable M256-contained carriers.

## Absolute promotion gates — unchanged

```text
pooled contain2 >= .75
worst-family contain2 >= .60
11032/13203/15290 each contain2 >= .60
pooled contain1 >= .50
worst-family contain1 >= .35
pooled median normalized error <= 1.00
best-worst contain2 gap <= .30
```

## Causal gates vs U_RANK_ONLY — unchanged

Required:

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

Safety:

```text
secondary pooled contain2 must not decrease by >.03
```

## Decision

If parity + preflight + absolute + causal + safety all pass:

`GLOBAL_RELATIONAL_CANDIDATE_SOLVER_V3_RANK_PASS`

If preflight/parity pass but solver gates fail:

`RANK_CALIBRATION_INSUFFICIENT_V3`

No post-hoc lambda, blend between z/rank, alternative percentile formula, temperature or cost clipping is allowed on this panel.

## Boundaries

Forbidden:

- changing M256;
- changing factor ordering/raw definitions;
- adding p_active or anchors;
- mixing rank and z-score costs;
- graph/degree changes;
- restarts/new optimizer;
- trying rank powers/logits/temperatures;
- accessing sealed21/external10;
- large/end-to-end training.
