# N1D Observable Functional Audit V2 — Clean Replication Preregistration R6.1

**Experiment ID:** `N1D_OBSERVABLE_FUNCTIONAL_AUDIT_V2_20260820`  
**Source revision:** `R6.1`  
**Panel:** untouched `e01` replication  
**Status:** `PREREG_FROZEN__TRUTH_SEMANTICALLY_CLOSED`

## 1. Scientific question

Within the actual frozen raster-only N1D Hybrid V11 front door, when the committed Pose-B geometry is evaluator-far but the independently observation-derived feasible set `H_i` contains an evaluator-near alternative, does replacing only that carrier by the nearest already-existing feasible `H_i` candidate usually leave downstream GFDR-V2 mechanics materially unchanged, or does it induce mechanically material consequences when counterfactual Pose-B normals and visibility are regenerated from the same raster/model front door?

This is a functional-geometry quotient audit. It is **not** an exact-teacher-geometry objective and it does **not** test information-theoretic impossibility.

## 2. Frozen authority

- GitHub freeze authority commit: `7b9ee7a8f4068dbf8f873508fceb1bfc28dd786d`
- deterministic source bundle SHA-256: `b4ab69fbcc91d8c57ab5f36f151b37495e8d5366554f3189982406d0b60c282f`
- source index SHA-256: `dca7f6a6395db21ee69c72c296392b6eb7b019db330e9b5c88831ae3e0d13005`
- exact e01 input manifest SHA-256: `91acc078a9f6c41604b3f6f829c2cb239728d407f75872493a6712dd4ba7ca02`
- preflight SHA-256: `e418e5ee10365b2c9bdba990710244fcc33db47a2161435effdb01852aa372dc`
- frozen checkpoint SHA-256: `0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18`
- frozen GFDR-V2 source SHA-256: `ca22e3fd42e9c812632eb372544e2319ecf720f6dede1f4a60c8da14ff0cb8fa`
- frozen R6.1 evaluator SHA-256: `a986dba818ab63cbd6beac133c5e713698489ca037d5eeba71329e6b216828e3`

The source bundle, source index, exact e01 input manifest and preflight record passed local ↔ Drive ↔ Library byte-identical readback before this preregistration.

## 3. Population

Only:

`[09908,11032,12772,13203,14404,14702,14758,15290] × e01`

The input gate is exactly 136/136 producer-manifest-verified bytes:

- Pose-A rasters: 64/64
- e01 Pose-B rasters: 64/64
- e01 observation sidecars: 8/8 byte-hash verified

Before observable-state freeze, sidecars may be copied or hashed as opaque bytes only. Semantic deserialization, key inspection or array/value access is forbidden.

`sealed21 = CLOSED`, `external10 = CLOSED`. No training, retuning, free XYZ, solver tuning or threshold sweep is authorized. The historical P1/M256 48-witness population and the invalidated R5 e00 result are not dependencies and may not define this run's population or decisions.

## 4. Phase 1 — observable inference

`observable_phase.py` may read only the frozen raster panel, calibrated camera contract, frozen N1D model/checkpoint and frozen observation-derived inference code.

For every family it must freeze:

- `P_A,P_B,N_A,N_B,V_A,V_B`;
- complete per-carrier observation-derived feasible candidate pools `H_i`;
- candidate reprojection and descriptor scores;
- route identity;
- checkpoint and raster hashes;
- an observable-state SHA-256;
- `truth_access = NONE`.

Every family must also pass:

- `V_B` decorator replay bit-exact;
- `N_B` decorator replay max absolute error `<= 1e-7`;
- non-empty `H_i` for every retained carrier.

All eight observable states and the observable-state manifest must be persisted and byte-verified before semantic evaluator truth is opened.

## 5. Phase 2 — evaluator-only truth-open

Only after Phase 1 freeze may `observation_sidecar.npz` be semantically parsed.

Evaluator truth may be used only to:

1. map the 64 observable Pose-A carriers injectively to dense evaluator Pose-A surface identities;
2. label reliable/active/baseline-hard carriers;
3. determine whether an already-frozen `H_i` contains an evaluator-near feasible candidate;
4. select the evaluator-nearest candidate **inside that frozen `H_i`** as a counterfactual existence witness;
5. score baseline and counterfactual mechanics relative to evaluator truth.

Evaluator truth may not create XYZ, change candidate scores, supply `N_B`/`V_B`, train, retune, alter the solver, or enter forward mechanics.

Every counterfactual must regenerate `N_B,V_B` through `decorate_pb_observable()` from the frozen normal head and Pose-B raster visual hull.

## 6. Witness definition

For each observable carrier `i`, use local evaluator scale from the median distance to its `k=4` nearest evaluator-surface neighbours.

A primary witness requires all four:

1. Pose-A mapping reliable: `map_err_A <= 2*s_A`;
2. evaluator motion active: `||P_B_truth - P_A_truth|| > 0.005`;
3. H-recoverable: nearest frozen `H_i` candidate has Pose-B evaluator error `<= 2*s_B`;
4. baseline-hard: frozen observable baseline `P_B[i]` has evaluator error `> 2*s_B`.

Witness identities are derived only on this e01 run and must be reported explicitly as `(family, carrier)`. No historical witness count is a guard.

## 7. Counterfactual

For witness `i`:

- `x_base` is the complete frozen 64-carrier observable baseline state;
- `h_near(i)` is the lowest-index `np.argmin` evaluator-nearest candidate already present in frozen `H_i`;
- `x_swap(i)` equals `x_base` except `P_B[i] = h_near(i)`;
- regenerate the complete swapped `N_B,V_B` observably;
- recompute the frozen full GFDR-V2 state.

No other carrier or inference variable may change.

## 8. Primary mechanical equivalence

The comparison motif is the changed carrier plus forward 12-neighbours and reverse incident neighbours from the frozen relational graph.

Raw coordinate-identity blocks `F_delta` and `R_rel_B` are descriptive only and excluded from primary equivalence because they directly restate that geometry changed. `F_delta_NRMS` remains descriptive only.

State-vs-state normalized effect uses symmetric NRMS:

`RMS(a-b) / (max(RMS(a),RMS(b)) + 1e-8)`.

Frozen thresholds:

- `F_response_NRMS <= 0.15`
- `F_kernel_RMS <= 0.10`
- `F_rigid_residual_NRMS <= 0.20`
- `D_surface_action_NRMS <= 0.15`
- `D_residual_NRMS <= 0.20`
- `D_gradient_NRMS <= 0.20`
- `R_motion_NRMS <= 0.20`
- `R_transfer_NRMS <= 0.20`
- `R_surface_affinity_RMS <= 0.15`
- `G_direction_disagree <= 0.15`
- `G_axis_line_norm_RMS <= 0.20`
- `G_support_RMS <= 0.15`

G direction is sign-invariant. Axis-point comparison is gauge-free along the axis and normalized by local motif radius. G direction/axis-line blocks abstain only when no carrier in the motif has union support `>= 0.25`.

A witness is mechanically equivalent iff every applicable non-abstained primary block passes.

## 9. Truth-relative utility

Prediction-vs-truth errors use truth-normalized NRMS. Each finite block error is capped at 10 and the composite is the median.

A swap is `material_truth_improvement` iff both:

- relative composite improvement `>= 0.20`; and
- absolute composite decrease `>= 0.05`.

A swap is `material_truth_degradation` by the symmetric negative rule.

This utility score never changes the candidate set or the counterfactual selection rule.

## 10. Population sufficiency and frozen decision tree

Decision-capable population requires:

- total witnesses `n >= 20`; and
- at least 4 families contain at least one witness.

For the geometry-singleton-not-required decision, every family with at least 5 witnesses must additionally have equivalence fraction `>= 0.60`.

Decision order:

1. if population sufficiency fails: `INSUFFICIENT_HARDTAIL_POPULATION`;
2. else if pooled mechanical equivalence `>= 0.80` and family guard passes: `GEOMETRIC_SINGLETON_NOT_REQUIRED_UNDER_REAL_OBSERVABLE_GFDR_V2`;
3. else if pooled equivalence `<= 0.50` and material truth-improvement fraction `>= 0.50`: `MECHANICALLY_MATERIAL_ALTERNATIVES_EXIST__NEXT_FUNCTIONAL_CLASS_OBSERVABILITY_AUDIT`;
4. else if pooled equivalence `<= 0.50`: `MECHANICS_CHANGE_BUT_TEACHER_NEAR_UTILITY_NOT_DOMINANT__RESEARCH_REQUIRED`;
5. otherwise: `FUNCTIONAL_QUOTIENT_AMBIGUOUS__RESEARCH_REQUIRED`.

## 11. Invalidation guards

The run is invalid if any source/input/prereg hash differs; if sidecar semantics are opened before observable freeze; if baseline decorator replay fails; if teacher normals/visibility or teacher-created XYZ enters mechanics; if sealed21/external10 is opened; if any training/retune/threshold sweep occurs; if witness definitions or thresholds change after this prereg; or if a result is interpreted as information-theoretic impossibility without a separate functional-class observability experiment.

## 12. Interpretation boundary

No possible R6.1 outcome alone proves that paired rasters are insufficient or that the RealSaS problem is impossible. If mechanically consequential feasible alternatives exist, the next question is whether the **global observable relational/mechanical evidence** selects their functional class. In particular, this R6.1 audit does not evaluate the user's global relational-address `R` solver; that is a separate preregistered experiment after this replication closes.

## 13. Required output order

`observable states -> observable manifest/hash -> persist/readback -> truth open -> RESULT -> REPORT -> DECISION -> result bundle -> three-store verification -> state promotion`.

No result-dependent source, threshold, population or decision edit is permitted.
