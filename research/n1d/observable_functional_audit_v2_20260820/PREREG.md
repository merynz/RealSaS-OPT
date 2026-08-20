# RealSaS N1D — Observable Functional Audit V2 — Preregistration

**Date:** 2026-08-20  
**Experiment:** `N1D_OBSERVABLE_FUNCTIONAL_AUDIT_V2_20260820`  
**Status:** `PREREG_FROZEN__TRUTH_SEMANTICALLY_CLOSED`

## Question

Starting from the frozen production-like raster-only N1D Hybrid V11 front door, when its committed Pose-B geometry is evaluator-far but the independently observation-derived feasible set `H` still contains an evaluator-near alternative, does replacing only that carrier geometry with the nearest feasible `H` alternative materially change downstream GFDR-V2 mechanics **when Pose-B normals and visibility are regenerated from the same raster/model front door**?

This is a functional-geometry audit. It is not an exact-teacher-geometry objective.

## Claim boundary

This experiment may establish one of the following within the frozen GFDR-V2 mechanics contract:

1. many geometrically distinct feasible states are mechanically equivalent, so geometric singleton is not required on the tested panel;
2. mechanically distinct feasible alternatives exist and the teacher-near feasible state often improves truth-relative mechanical consequences;
3. mechanics change, but teacher-near utility is not dominant;
4. the tested functional quotient remains ambiguous or the hard-tail population is insufficient.

**No outcome of this experiment alone is authorized to claim information-theoretic impossibility or raster non-identifiability.** If mechanically distinct alternatives exist, the next scientific question is whether frozen raster-observable typed evidence separates their functional classes.

## Frozen authority

- source bundle SHA-256: `3bd133e805880ccca2e6a177778e3e931064eb0178da5a2ac5b02c07f1595278`
- source manifest GitHub commit: `330cccea2e8fc18c1815bc99dae27b1affba6c6b`
- detailed input inventory R6 SHA-256: `719d0573b14e218bb081e7e1858df4121f09200436909d3e665e3ac96c47252c`
- compact input authority GitHub commit: `b09e828e53ea4c76032a8518ff32692d8bf823a3`
- frozen checkpoint SHA-256: `0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18`
- frozen GFDR-V2 source SHA-256: `ca22e3fd42e9c812632eb372544e2319ecf720f6dede1f4a60c8da14ff0cb8fa`
- observable raster set digest: `efad989dd8a1929e2a9f49e1e94d7e16eda3aea75ca455a1faa052722e349e2c`
- evaluator sidecar set digest: `8a4107673e65858378c8f0f11cf74704d372b40c4e2f9f1b11a6ae8605c1b033`
- prereg-preflight: source and 136/136 input-byte replay PASS.

Population: e00 of families `09908,11032,12772,13203,14404,14702,14758,15290`. `sealed21` and `external10` remain CLOSED. No training, retune, free XYZ, solver tuning, or threshold sweep is authorized.

The old P1/M256 48-witness result is historical context only and is **not a dependency** of this experiment.

## Phase separation / truth firewall

### Phase 1 — observable inference

`observable_phase.py` may read only the frozen raster panel, calibrated camera contract, frozen N1D model/checkpoint and frozen observation-derived inference code. It must produce, for every family:

- committed `P_A,P_B,N_A,N_B,V_A,V_B`;
- frozen observation-derived candidate pools `H` and their reprojection/descriptor scores;
- route identity and checkpoint/raster hashes;
- an observable state SHA-256;
- `truth_access = NONE`;
- exact `V_B` decorator replay and `N_B max_abs <= 1e-7`.

All eight observable state files and their panel manifest must be cryptographically frozen and persisted before semantic sidecar access is authorized.

Opaque sidecar-byte hashing/copying for provenance is allowed before truth-open; deserialization/key/array/value inspection is forbidden.

### Phase 2 — evaluator-only truth-open

Only after Phase 1 persistence:

- map the 64 observable `P_A` carriers injectively to dense Pose-A evaluator surface using the frozen Hungarian rule;
- use exact sidecar geometry only for evaluator mapping, witness labels, counterfactual selection, and truth-relative scoring;
- never copy sidecar `N_B` or `V_B` into baseline or swapped mechanics inputs.

For every counterfactual swap, `decorate_pb_observable()` must regenerate `N_B,V_B` from the frozen normal head and raster visual hull.

## Witness population

For carrier `i`, using local-scale `k=4` exactly as frozen in `evaluation_phase.py`, define a witness iff all hold:

1. Pose-A mapping reliable: `map_err_A <= 2 * scale_A`;
2. evaluator motion active: `||P_B^truth - P_A^truth|| > 0.005`;
3. observation-derived candidate pool `H_i` contains an evaluator-nearest feasible candidate with error `<= 2 * scale_B`;
4. frozen observable baseline committed `P_B[i]` has evaluator error `> 2 * scale_B`.

Witness identities are an evaluator-derived result population and must be reported explicitly as `(family, carrier)` IDs. No historical `n=48` guard applies.

## Counterfactual

For each witness:

- `x_base`: complete frozen 64-carrier observable baseline state;
- `h_near(i)`: lowest-index `np.argmin` evaluator-nearest point **inside the already-frozen H_i**;
- `x_swap(i)`: identical to `x_base` except `P_B[i] = h_near(i)`;
- regenerate full swapped `N_B,V_B` observably;
- recompute frozen GFDR-V2.

Teacher truth does not create XYZ and does not alter any observation-derived candidate score or inference state.

## Primary mechanical equivalence

The raw coordinate-identity blocks `F_delta` and `R_rel_B` are not primary equivalence criteria because they restate that geometry changed. `F_delta_NRMS` remains descriptive only.

For the frozen local GFDR motif around each changed carrier, all non-abstained blocks must pass:

```text
F_response_NRMS             <= .15
F_kernel_RMS                <= .10
F_rigid_residual_NRMS       <= .20
D_surface_action_NRMS       <= .15
D_residual_NRMS             <= .20
D_gradient_NRMS             <= .20
R_motion_NRMS               <= .20
R_transfer_NRMS             <= .20
R_surface_affinity_RMS      <= .15
G_direction_disagree        <= .15
G_axis_line_norm_RMS        <= .20
G_support_RMS               <= .15
```

G direction/axis-line abstain-equivalent only when no carrier in the motif has union support `>= .25`. Axis-point difference is gauge-free along the axis and is compared only perpendicular to the axis, normalized by motif radius.

A witness is `mechanically equivalent` iff every applicable block passes.

## Truth-relative utility (secondary but decision-relevant)

For baseline and swap, compare the same non-tautological mechanical blocks against evaluator GFDR truth. Each finite block error is capped at 10 and the composite is their median.

A swap is `material_improve` iff both:

```text
relative composite improvement >= .20
absolute composite reduction    >= .05
```

A swap is `material_degrade` by the symmetric negative rule.

Truth-relative utility scores the already-frozen counterfactual; it never changes candidate selection beyond choosing evaluator-nearest `H_i` as defined above.

## Population sufficiency

The result is decision-capable only if:

```text
witness n >= 20
AND >= 4 families contain at least one witness
```

For the geometry-singleton-not-required decision, every family with `n>=5` witnesses must additionally have equivalence fraction `>= .60`.

## Frozen decision tree

1. If population sufficiency fails:  
   `INSUFFICIENT_HARDTAIL_POPULATION`

2. Else if pooled mechanical equivalence `>= .80` and family guard passes:  
   `GEOMETRIC_SINGLETON_NOT_REQUIRED_UNDER_REAL_OBSERVABLE_GFDR_V2`

3. Else if pooled equivalence `<= .50` and material truth-improvement fraction `>= .50`:  
   `MECHANICALLY_MATERIAL_ALTERNATIVES_EXIST__NEXT_FUNCTIONAL_CLASS_OBSERVABILITY_AUDIT`

4. Else if pooled equivalence `<= .50`:  
   `MECHANICS_CHANGE_BUT_TEACHER_NEAR_UTILITY_NOT_DOMINANT__RESEARCH_REQUIRED`

5. Otherwise:  
   `FUNCTIONAL_QUOTIENT_AMBIGUOUS__RESEARCH_REQUIRED`

No threshold, population definition, metric, source, or decision branch may change after semantic truth-open.

## Forbidden interpretations

Even decision 3 does **not** mean paired rasters are insufficient or that the problem is impossible. It means the current observable front door admits mechanically consequential alternate feasible geometry on the tested panel and motivates a separate functional-class observability/separability audit.

Decision 2 also does not prove all geometry ambiguity is harmless universally; it is scoped to the tested panel and frozen GFDR-V2 contract.
