# RealSaS N1D — Global Solver Common-Mode Gauge Audit V1 Preregistration

**Date:** 2026-08-19  
**Status:** `TRUTH_OPEN_DIAGNOSTIC_PREREG__NO_ARCHITECTURE_CHANGE`  
**Parent classification:** `FROZEN_LOCAL_OBJECTIVE_CONTEXT_FAIL`

## Hypothesis

Frozen R_REL_DIS uses only pairwise differences of projected displacement. Under orthographic projection, a common 3D displacement added to all nodes cancels from every pair factor. Therefore the relation has a common-mode/gauge null direction and requires some separate absolute anchoring mechanism.

This audit asks whether the V2 endpoint hard tail contains a material family-level common-mode error and whether simple absolute raster motion carries information about that mode.

No metric below authorizes free XYZ or an anchor factor; this is evaluator-only diagnosis.

## Frozen evaluated solution

Use the exact M256 V2 `G_REL_DIS` solution obtained by the frozen unary-initialized deterministic ICM. Recompute it with exact V2 source/caches if needed; no solver quantity changes.

Primary evaluator set is exactly V2 primary.

## Audit A — algebraic gauge invariance

For any graph edge and continuous endpoints `(P_i,P_j)`, define frozen raw relation as in R_REL_DIS. For deterministic diagnostic common shift `g`, verify numerically:

```text
R_raw(P_i + g, P_j + g) == R_raw(P_i, P_j)
```

using family fitted common shifts from Audit B and all eligible graph edges. Report maximum absolute raw-energy difference. Numerical invariance passes if max difference <= `1e-9` pixels after median aggregation.

This is a mathematical property diagnostic, not a candidate-generation operation.

## Audit B — evaluator-only best common 3D correction

For each family independently, on primary nodes:

```text
e_i = selected_P_B_i - truth_P_B_i

g*_family = - mean_i e_i
```

This is the least-squares common 3D translation that best corrects the selected endpoints.

Report before/after:

- contain1;
- contain2;
- median normalized error;
- SSE;
- common-mode SSE explained:

```text
explained = 1 - SSE_after / SSE_before
```

Report pooled endpoint metrics after applying each family evaluator-only `g*`.

Classify `COMMON_MODE_MATERIAL` iff:

```text
pooled SSE explained >= .25
and at least 2 of {11032,13203,15290} gain contain2 >= .10
and no hard-tail family loses contain2
```

This classification does not claim the shift is observable.

## Audit C — observation-native absolute common-motion estimator

Construct one family-level common displacement from raster evidence only:

1. for each view, collect frozen raw DIS vectors at all Pose-A-visible carriers with finite DIS;
2. take componentwise median DIS vector per view;
3. use the known orthographic view bases to solve a single 3D displacement `g_DIS` by ordinary least squares over all available view-median x/y equations;
4. compute the current G_REL_DIS solution's observation-only mean 3D displacement

```text
mean_i (selected_P_B_i - P_A_i)
```

across all 64 carriers;
5. define diagnostic correction

```text
g_obs = g_DIS - mean_i(selected_P_B_i - P_A_i)
```

and apply it continuously to the selected endpoints only for evaluator diagnosis.

No truth-active filtering, family-specific fit to truth, learned parameter or threshold enters `g_obs`.

Report:

- `g_DIS`;
- `g_obs`;
- cosine and norm-ratio between `g_obs` and evaluator-only `g*` when both norms are finite/nonzero;
- primary contain1/contain2 before/after;
- pooled and hard-tail gains.

Observation-native anchor evidence is `SUPPORTED` iff:

```text
pooled contain2 gain >= .05
and at least 2 hard-tail families do not regress
and mean cosine(g_obs,g*) over evaluable families >= .50
```

Otherwise it is `NOT_SUPPORTED_UNDER_SIMPLE_MEDIAN_DIS_ESTIMATOR`.

## Audit D — residual structure after best common correction

After evaluator-only `g*`, report per-family residual covariance eigenvalues and ratio:

```text
largest_eigenvalue / trace(covariance)
```

This is descriptive only. A large residual directional mode may motivate a later low-frequency field audit, but no such extension is authorized here.

## Boundaries

Forbidden in this audit:

- selecting candidates using truth-fitted `g*`;
- shifting production endpoints off H;
- adding free XYZ;
- changing R_REL_DIS;
- tuning unary/pair weights;
- changing graph/ICM/M256;
- introducing p_active gating;
- trying alternative robust estimators after seeing results;
- accessing sealed21/external10.

Any new absolute anchor must be separately formulated and preregistered after this audit.
