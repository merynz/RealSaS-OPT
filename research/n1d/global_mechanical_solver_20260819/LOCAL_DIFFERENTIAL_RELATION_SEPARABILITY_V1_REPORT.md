# RealSaS N1D — Local Differential Relation Separability V1

**Date:** 2026-08-19  
**Verdict:** `PASS__PAIRWISE_LOCAL_RELATION_STRONGLY_SEPARABLE__R_REL_DIS_SELECTED`  
**Status:** `TRUTH_OPEN_RELATIONAL_DIAGNOSTIC__NOT_SOLVER__NOT_TRAINING__NOT_QUALIFICATION`  
**Prereg commit:** `2f94f42127e4efde0b0483e90e7887997c3332bf`

## Question

After three increasingly permissive single-carrier tests failed—direct scalar amplitude, candidate-conditioned amplitude, and full projected 2D motion vector—V1 tested whether the correct feasible H configuration becomes identifiable only through **carrier-to-carrier local differential/mechanical relations**.

The geometry substrate remained the frozen F16 H research contract. No relation was fitted to truth and no graph solver was tuned in this experiment.

## Graph / evaluator contract

- Observation-derived `P_A` carriers.
- Deterministic symmetrized 4-nearest-neighbor graph in `P_A` space.
- Primary diagnostic edge: both carriers mapping-reliable, both frozen F16 H sets truth-contained under the existing primary-2x criterion, and at least one endpoint truth-active. Truth-active is evaluator-only here; the inference analogue is the separately supported `p_active` evidence.
- Each H is reduced for the separability audit to a deterministic observation-only 32-mode spatial sample: XYZ deduplication, lexicographic seed, then max-min farthest-point selection.
- Evaluator-only oracle candidate `h_i*` is the H member nearest truth `P_B(i)`.
- Relation quality is the oracle-pair energy percentile among <=1024 deterministic sampled alternative candidate pairs. Lower is better.

## Tested relation energies

### R_DISP

```text
||(h_i-P_A_i) - (h_j-P_A_j)|| / ||P_A_i-P_A_j||
```

Local displacement coherence.

### R_RIGID

```text
| ||h_i-h_j|| - ||P_A_i-P_A_j|| | / ||P_A_i-P_A_j||
```

Local metric preservation.

### R_REL_DIS

For each common Pose-A-visible view:

```text
candidate_relative_motion =
  [proj(h_i)-proj(P_A_i)] - [proj(h_j)-proj(P_A_j)]

observed_relative_motion = DIS_i - DIS_j
```

Energy is median native-pixel L2 residual across common views.

### R_REL_TRANSPORT / R_REL_DELTA3D

Same relative-motion relation using frozen learned transport and projected per-view `delta_point_map_srcA` evidence.

### R_REL_FUSION

Within-edge robust median/IQR normalization of `R_RIGID + R_REL_DIS + R_REL_TRANSPORT + R_REL_DELTA3D`, candidate-pair-wise median; no fitted weights.

## Result — every relation arm passes

### R_REL_DIS — preregistered selected relation

The preregistered simplicity rule selects the simplest passing observation-relative single relation unless fusion improves the worst-family median percentile by at least `.05`. Fusion improves only about `.0125`, so **R_REL_DIS is selected**.

```text
edges                                 603
pooled median oracle percentile       .01465
pooled fraction <= .25                .93035
pooled fraction <= .10                .80597
worst-family median percentile        .05249
worst-family fraction <= .25          .84615

11032 median percentile               .01367
13203 median percentile               .03711
15290 median percentile               .00195

active-active edges                   520
active-active median                  .01172
active-active fraction <= .25         .94231

active-inactive edges                  83
active-inactive median                .06445
active-inactive fraction <= .25       .85542
```

Per-family R_REL_DIS median percentiles:

```text
09908  .05249
11032  .01367
12772  .02881
13203  .03711
14404  .02637
14702  .03418
14758  .03174
15290  .00195
```

All preregistered support gates pass by very large margins.

### R_REL_FUSION

```text
pooled median              .01270
fraction <= .25            .94196
fraction <= .10            .82587
worst-family median        .04004
worst-family frac <= .25   .87037
```

Also strongly PASS, but not selected by the preregistered simplicity rule.

### Geometry-only relations

Even geometry-only local displacement coherence is extremely strong:

```text
R_DISP pooled median        .01367
fraction <= .25             .94945
worst-family median         .06641
```

R_RIGID also passes (`pooled median .05371`, worst-family median `.13281`).

## Scientific conclusion

This experiment sharply resolves the repeated hard-tail pattern:

> **The correct candidate is often ambiguous when a carrier is considered independently, but the correct candidate pair becomes strongly identifiable once local mechanical/differential relations to neighboring carriers are considered.**

This explains why the previous single-carrier amplitude/vector formulations could have useful signal yet fail family-robust selection.

The supported architecture is therefore no longer:

```text
H_i + standalone log_amp_i + dir_i -> endpoint
```

The evidence now supports:

```text
p_active
+ set-valued frozen H_i
+ local pairwise/differential mechanical factors
+ global compiler/solver
-> final feasible configuration
```

Amplitude should emerge from the selected globally coherent feasible configuration rather than being treated as an independent free per-carrier authority.

This does **not** mean `p_active` disappears: activity/non-activity remains a separate supported factor and is especially important for active-inactive edges. It means conditional motion magnitude/direction are resolved primarily through candidate relations and the global configuration.

## Next authorized experiment

Freeze `R_REL_DIS` exactly as the primary pairwise relation. Then preregister a **separate discrete/global candidate solver**. The relation and solver must not be tuned together.

The next solver experiment must:

1. keep F16 H as XYZ authority;
2. use an observation-only bounded candidate-mode set for tractability;
3. preserve a unary observation score as baseline/anchor;
4. use frozen R_REL_DIS pairwise factors on the same `P_A` neighborhood graph;
5. compare unary-only vs graph relation without tuning pair weights post hoc;
6. evaluate hard-tail endpoint containment and unchanged downstream GFDR only after solver behavior is frozen.

## Qualification boundary

- Stage A remains PASS.
- Frozen Stage B remains FAIL/no-retune.
- This is truth-open development evidence, not a qualification PASS.
- sealed21/external10 remain closed.
- Large/end-to-end training remains forbidden.

## Reproducibility / recovery note

Canonical machine-readable result: `LOCAL_DIFFERENTIAL_RELATION_SEPARABILITY_V1_RESULT.json`.

Exact local aggregate SHA-256 recorded before the execution-session boundary:
`613c9fec4ac1428fb41e200005f6839a7ab457b4161e0e1feace5e118390aaef`

Exact local source SHA-256 recorded before the execution-session boundary:
`88a71b50c5f0c1d12240b19fb2895ff7bdc8119fa022970a433e71e0a977f7b6`

The exact local source bytes were not persisted before that execution session ended. This report and the canonical result preserve the frozen formulas, protocol, metrics and exact reported source hash; they do **not** claim byte identity to the lost local source file. A recovered reproducer must be treated as a new source artifact and must not reuse the old source hash.
