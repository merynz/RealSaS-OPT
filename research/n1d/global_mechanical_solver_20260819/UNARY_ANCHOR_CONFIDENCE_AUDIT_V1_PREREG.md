# RealSaS N1D — Unary Anchor Confidence Audit V1 Preregistration

**Date:** 2026-08-19  
**Status:** `TRUTH_OPEN_CONFIDENCE_DIAGNOSTIC__NO_SOLVER_CHANGE`

## Motivation

V2 shows error propagation: R_REL_DIS is strong, local correct-context selection is much better than global ICM, and unary influence helps some families while harming others. Before attempting confidence-anchored propagation, test whether a deterministic observation-only score can identify a subset of U_ONLY endpoints safe enough to freeze.

## Frozen candidate / graph state

Use exact V2 M256, U_raw, normalized U, graph and R_REL_DIS. No solver modification is evaluated here.

For every episode/family, confidence ranking is computed across all 64 carriers using observation-only quantities. Truth/reliability is not used to choose the top quartile/half; truth is evaluator-only afterward.

## Confidence arms

### C_MARGIN — unary winner margin

```text
score = U_second_best - U_best
```

where U is the frozen normalized unary candidate vector. Higher is more confident. Tie by carrier index.

### C_ABS — absolute unary reprojection score

```text
score = - min(U_raw)
```

Higher means smaller raw reprojection residual. Tie by carrier index.

### C_AGREE — unary / relation-context agreement

Construct pair-only local cost using the frozen **U_ONLY neighbor configuration** (not oracle context). Let `p_i` be its argmin candidate and `u_i` the U_ONLY argmin.

Define:

```text
agreement_px = mean over Pose-A-usable views
               ||project(M_i[u_i]) - project(M_i[p_i])||
score = -agreement_px
```

Higher means stronger agreement. Tie by carrier index.

This is fully observation-only and uses no truth labels.

## Selection

Within each 64-carrier family independently, rank by each score and select:

- top 16 carriers = 25% anchor candidate set;
- top 32 carriers = 50% diagnostic set.

No threshold is fitted.

## Evaluator

Primary anchor-quality evaluator for each selected set uses selected carriers that are mapping-reliable and M256-contained2x. Report endpoint quality of their frozen U_ONLY candidate:

- contain1;
- contain2;
- normalized endpoint error;
- active/inactive strata;
- family counts.

Also report coverage: selected evaluable anchors / all mapping-reliable M256-contained carriers.

## Safe-anchor gate

An arm is `SAFE_ANCHOR_SUPPORTED` on its top-25% set iff:

```text
pooled contain2 >= .90
worst-family contain2 >= .80
pooled contain1 >= .65
pooled contain2 gain over all-reliable U_ONLY secondary baseline >= .10
```

The reference all-reliable U_ONLY secondary baseline is recomputed from the same V2 solution/domain; no stored rounded constant is used.

Arm-selection priority if multiple pass:

```text
C_MARGIN > C_ABS > C_AGREE
```

for simplicity and lowest relational dependency.

If no arm passes, decision is:

`NO_SAFE_UNARY_ANCHOR_CONFIDENCE_FOUND_V1`

and confidence-freezing is not authorized.

If one passes, decision names the first arm under the fixed priority and only that arm may be used in a later separately preregistered anchored-solver experiment.

## Boundaries

This audit does not authorize:

- freezing any node in the current solver;
- tuning top fraction;
- trying 10%, 20%, 30%, etc.;
- fitting a confidence classifier;
- changing unary/pair weights;
- changing graph/relation/M256;
- adding p_active;
- accessing sealed21/external10.
