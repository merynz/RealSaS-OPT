# Relation Algebra Court V1 — preregistration

Status: SEALED BEFORE OUTCOME
Authority: research-only.

## Purpose

Determine whether current pair-head parameterization structurally respects the algebra of different evidence families, independent of training quality.

## Algebra families

- `SYMMETRIC`: r(a,b)=r(b,a). Examples: deformation coherence, same-support similarity, boundary/separation similarity when defined pairwise.
- `COMPLEMENTARY_ORDER`: p(a≺b)+p(b≺a)=1 for qualified non-tie pairs.
- `DIRECTED`: r(a→b) need not equal r(b→a). Example: attachment dependency / causal influence.
- `RAY_GLOBAL`: cannot be reduced to a pair relation.

## Current-head witness

Instantiate current `GraphicsRelationHeadV3` with fixed random seeds and measure:
- swap error for STATIC outputs;
- complement residual for `front_probability`;
- swap error for projected boundary outputs;
- confirm directed ordered outputs can differ;
- keep ray-global head out of pair algebra.

The court does **not** claim random initialization predicts trained performance. It only asks whether the architecture guarantees the required algebra.

## Candidate exact constructions

1. `SYMMETRIC_LOGIT`: build pair features from `(a+b, |a-b|, a*b)` or symmetrize a shared ordered score. Must have machine-zero swap error.
2. `COMPLEMENTARY_LOGIT`: define signed score `s(a,b)=g(a,b)-g(b,a)` and `p(a≺b)=sigmoid(s)`. Must have machine-zero complement residual under swap.
3. `DIRECTED_LOGIT`: retain ordered features; no symmetry constraint.
4. `RAY_GLOBAL`: retain complete-stack context, not pair head.

## Gate

If current random-weight witness violates an algebra by >1e-6 while the candidate construction satisfies <=1e-7, mark that algebra **not guaranteed by current parameterization**. This is a structural apparatus finding, not a learned-performance result.

Whether enforcing the algebra improves full training remains Truth-dependent and is deferred to the causal truth experiments.
