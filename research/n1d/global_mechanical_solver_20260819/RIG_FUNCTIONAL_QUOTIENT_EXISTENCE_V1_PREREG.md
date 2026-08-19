# RealSaS N1D — Rig-Functional Quotient Existence Audit V1 — Preregistration

**Date:** 2026-08-19  
**Status:** `TRUTH_OPEN_PRODUCT_EXISTENCE_AUDIT__NO_SOLVER_CHANGE`

## Question

The product target is **not** exact teacher geometry. It is a clean editable rig whose downstream mechanical/deformation behavior is functionally equivalent.

The audit asks whether the current geometric hard tail is materially relaxed by the rig-functional quotient:

```text
Does geometrically distinct Pose-B surface state k sometimes belong to the
same downstream rig-functional class as the teacher-bound surface state j?
```

If yes at material/broad rates, geometric singleton is not required for the tested product contract. If even a deliberately relaxed functional signature almost never identifies geometrically distinct equivalents, then the tested rig-functional quotient is effectively as fine as the geometric quotient and research must return to resolving geometry/mechanics.

This audit is frozen **before inspecting any dense-weight or multi-intervention functional distances**.

## Frozen open-development panel

Families:

```text
09908 11032 12772 13203 14404 14702 14758 15290
```

Current N1D episode `e00` is the parent M256/P1 geometry panel. Current family interventions `e00..e07` provide deformation-response truth. Historical dense teacher sidecars provide pose-invariant 512×joint skin-weight truth.

All data are truth-open development evidence. `sealed21` and `external10` remain CLOSED.

## Canonical carrier binding

Use the exact parent V2/P1 evaluator binding. For raster carrier i:

```text
j_i = argmin_s || P_A[i] - T_A[s] ||
maperr_i = ||P_A[i] - T_A[j_i]||
reliable_i iff maperr_i <= 2 * s_A[j_i]
target_i = T_B_e00[j_i]
```

where `T_A/T_B` are normalized current e00 surface points and `s_A/s_B` are the frozen local surface scales.

**Dense weight row and all intervention-response rows are bound by the same physical surface index `j_i`.** No remapping by weight/deformation truth is allowed.

## Validity guards

Before functional results are interpreted, all must pass:

1. historical dense teacher `surface_points_A` vs current e00 `surface_points_A` max absolute difference `<= 1e-7` after comparing their stored physical coordinates, for all 8 families;
2. current `surface_points_A` is episode-invariant across `e00..e07` within max absolute difference `<= 1e-7` for all 8 families;
3. historical dense `weights` has exactly 512 rows and finite values; row sums are within `1e-5` of one for every nonzero row;
4. canonical M256/P1 replay reproduces parent parity, including `primary_n=261`, round-1 candidate index parity `512/512`, round-1 contain2 `.7777777778`, round-2 contain2 `.8160919540`, round-2 worst-family contain2 `.5200000000`, `11032=.5200000000`, `12772=.7272727273`;
5. current e00 camera-normalized `T_A/T_B` reproduces the parent evaluator mapping/target geometry used by M256/P1.

Failure of any guard yields `INVALID_FUNCTIONAL_AUDIT_ALIGNMENT_OR_PARITY` and no scientific conclusion.

## Frozen rig-functional signature

For family f and physical surface index s, let

```text
w_f(s) = dense skin-weight vector over teacher joints
Delta_f,e(s) = normalize_camera(surface_points_B_e[s] - surface_points_A[s])
```

for current interventions `e00..e07`.

Family deformation scale:

```text
S_f = q95 over (e,s) of ||Delta_f,e(s)||
```

with epsilon `1e-9` only to avoid division by zero.

For two physical surface indices j,k:

```text
E_weight(j,k) = 0.5 * sum_joint |w(j)-w(k)|
E_deform(j,k) = sqrt(mean_e ||Delta_e(j)-Delta_e(k)||^2) / (S_f + 1e-9)
```

`E_weight` is total-variation distance between normalized dense ownership distributions. `E_deform` measures actual multi-intervention downstream displacement-response disagreement.

### STRICT functional equivalence

```text
E_weight <= .05
AND
E_deform <= .10
```

### RELAXED functional equivalence

```text
E_weight <= .10
AND
E_deform <= .20
```

These thresholds are frozen before outcome inspection. Argmax-owner agreement and per-intervention errors are descriptive only and cannot change equivalence membership.

## Geometric distinction

Use the parent e00 normalized Pose-B surface and frozen local scale at reference j:

```text
E_geom(j,k) = ||T_B_e00[k] - T_B_e00[j]|| / s_B[j]
```

A functionally equivalent index k is **geometrically distinct** iff:

```text
E_geom(j,k) > 2.0
```

The strict `>2x` boundary matches the parent primary geometry containment threshold.

## Test A — quotient breadth independent of solver selection

For every canonical primary carrier reference `j_i`, evaluate all 512 physical surface indices k.

Report for STRICT and RELAXED:

- class cardinality `#{k: k ~functional j}`;
- geometrically-distinct equivalent cardinality;
- fraction of primary carriers with at least one geometrically-distinct functional equivalent;
- per-family fractions;
- median/max normalized geometric distance among equivalent indices.

This directly tests whether the rig-functional quotient is meaningfully coarser than the geometric quotient in the physical surface universe.

## Test B — actual P1 hard-tail rescue

For each P1 round-1 and round-2 selected M256 candidate endpoint `M_i[x]`, map it to nearest current e00 Pose-B physical surface index:

```text
k_hat = argmin_k ||M_i[x] - T_B_e00[k]||
candidate_mapping_reliable iff nearest_distance <= 2 * s_B[k_hat]
```

No functional truth may influence this mapping.

A geometric failure (`parent normalized endpoint error >2`) is functionally rescued iff candidate mapping is reliable and `k_hat` is STRICT/RELAXED functionally equivalent to reference `j_i`.

Report:

- round1/round2 geometric contain2;
- strict/relaxed functional contain = geometric pass OR functional rescue;
- rescue fraction among geometric failures;
- per-family rescue fractions;
- candidate mapping coverage.

## Frozen decision tree

### `RIG_FUNCTIONAL_QUOTIENT_RELAXES_GEOMETRIC_SINGLETON_V1`

Only if **STRICT** evidence satisfies all:

```text
Test-A pooled fraction with >=1 geometrically-distinct equivalent >= .30
at least 6/8 families have that fraction >= .10
round2 strict functional contain - geometric contain2 >= .05
round2 strict rescue fraction among geometric failures >= .25
```

Interpretation: exact geometric singleton is not required for a material, family-broad portion of the tested rig-functional contract.

### `GEOMETRIC_SINGLETON_EFFECTIVELY_REQUIRED_V1`

Only if deliberately more permissive **RELAXED** evidence satisfies all:

```text
Test-A pooled fraction with >=1 geometrically-distinct equivalent <= .10
every family Test-A fraction <= .20
round2 relaxed rescue fraction among geometric failures <= .10
round2 relaxed functional contain - geometric contain2 <= .02
```

Interpretation is limited to this eight-family open panel and this frozen dense-weight + eight-intervention rig-functional signature: functional quotient does not materially relax geometric identity, so research should return to resolving the remaining geometric/mechanical ambiguity.

### otherwise

`FUNCTIONAL_QUOTIENT_INCONCLUSIVE_V1`

No threshold relaxation or post-hoc combination is allowed.

## Boundaries

This audit cannot:

- change F16/M256 candidates, R_REL_DIS, graph, unary factors or P1 messages;
- use functional truth to select a candidate;
- train a model/ranker;
- redefine the parent geometric metric;
- add F/D/M5 evidence to the solver;
- tune functional thresholds after seeing results;
- access sealed21/external10;
- authorize large/end-to-end training.

The result is an existence/product-contract audit, not a new solver result.
