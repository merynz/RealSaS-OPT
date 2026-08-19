# N1D Observable Functional Audit V2 — Preregistration R4

**Experiment ID:** `N1D_OBSERVABLE_FUNCTIONAL_AUDIT_V2_20260820`  
**Source revision:** `R4`  
**Source bundle SHA-256:** `48d66705d210f05953e20c6787722a4e3397c513ff83a19e651a3a5992c9d187`  
**Input manifest SHA-256:** `b5c851cdb6db3463b6abf26c215853f899b16e3154f9f1bbd36541dee9fca600`  
**Status at freeze:** source/input/preflight PASS; evaluation truth semantics unopened.

## 1. Scientific question

Within the actual frozen raster-only N1D Hybrid V11 inference front door, when baseline geometry is wrong relative to evaluator geometry but the observation-derived feasible set H contains a teacher-near alternative, does replacing only that carrier by the feasible alternative usually leave downstream GFDR-V2 mechanics materially unchanged, or does it induce mechanically material consequences?

This is a **functional-geometry quotient audit**, not an exact-teacher reconstruction test.

The experiment does **not** test or declare information-theoretic impossibility. If mechanically material alternatives remain, a subsequent experiment must test whether the mechanically relevant functional class itself is observable from the raster observations.

## 2. Population

Primary panel only:

`[9908, 11032, 12772, 13203, 14404, 14702, 14758, 15290] × e00`

- sealed21: CLOSED
- external10: CLOSED
- training/retune: FORBIDDEN
- no M5/weight/skinning inputs
- no old P1 48-hard-tail cache or reconstructed V2 solver dependency

The witness population is newly derived from this experiment and is not required to equal any historical count.

## 3. Frozen observable phase

Before evaluator sidecars may be semantically opened:

1. Run the exact frozen Hybrid V11 baseline from the paired e00 rasters.
2. Rebuild the route-matched feasible H candidate pools using the exact frozen V8/V5 descriptor-search + calibrated triangulation implementation.
3. Freeze `P_A, P_B, N_A, N_B, V_A, V_B` and the complete per-carrier H pools.
4. Verify baseline `V_B` is bit-exact under replay and `N_B` replay max absolute error is `<= 1e-7`.
5. Persist a manifest containing every observable family-state SHA-256.

The observable source contains a static truth firewall: evaluator sidecars, teacher targets, dense surface points/normals/visibility, camera truth center/extent are forbidden in this phase.

## 4. Truth-open evaluator phase

Only after the observable state manifest is frozen and persisted may the evaluator parse `observation_sidecar.npz`.

Evaluator truth is used only to:

- map the 64 observable carriers to evaluator surface identities;
- define reliable/active/baseline-hard witnesses;
- determine whether H contains a teacher-near feasible candidate;
- select the nearest feasible H candidate as a counterfactual existence witness;
- score baseline and counterfactual states.

Teacher truth must never provide counterfactual `N_B`, `V_B`, free XYZ, solver cost, or forward-mechanics evidence.

For every counterfactual candidate, `N_B` is regenerated from the frozen N1D normal head at the candidate projections and `V_B` from the deterministic Pose-B raster visual hull, through the same observable decorator used by the real inference front door.

## 5. Witness definition

For each observable carrier i:

- map to evaluator Pose-A surface by Hungarian assignment;
- local evaluator scale uses median k=4 nearest-neighbor distance;
- mapping reliable iff `map_error <= 2 * s_A`;
- mechanically active iff evaluator displacement `||P_B - P_A|| > 0.005`;
- H recoverable iff nearest feasible H candidate has evaluator Pose-B error `<= 2 * s_B`;
- baseline hard iff frozen baseline Pose-B error `> 2 * s_B`.

A primary witness satisfies all four conditions: reliable, active, H-recoverable, baseline-hard.

## 6. Counterfactual construction

For witness i:

- keep the full frozen baseline state;
- replace only `P_B[i]` by the evaluator-selected nearest candidate already present in H_i;
- regenerate the complete counterfactual `N_B` and `V_B` from raster/model observables;
- recompute the full frozen GFDR-V2 state.

No candidate is created by evaluator truth.

## 7. Primary mechanical equivalence metric

The comparison motif is carrier i plus forward 12-neighbors and reverse incident neighbors from the frozen relational graph.

Raw geometry identity fields `F_delta` and `R_rel_B` are **descriptive only** and are excluded from primary functional equivalence because they directly restate coordinate differences.

State-vs-state normalized effect uses symmetric NRMS:

`RMS(a-b) / (max(RMS(a), RMS(b)) + 1e-8)`.

Frozen block thresholds:

- F_response_NRMS <= 0.15
- F_kernel_RMS <= 0.10
- F_rigid_residual_NRMS <= 0.20
- D_surface_action_NRMS <= 0.15
- D_residual_NRMS <= 0.20
- D_gradient_NRMS <= 0.20
- R_motion_NRMS <= 0.20
- R_transfer_NRMS <= 0.20
- R_surface_affinity_RMS <= 0.15
- G_direction_disagree <= 0.15
- G_axis_line_norm_RMS <= 0.20
- G_support_RMS <= 0.15

G direction is sign-invariant. Axis-point comparison is gauge-free along the axis and normalized by local motif radius. Direction/axis-line blocks abstain when support is absent (`G_support < 0.25` throughout the motif).

A witness is mechanically equivalent iff every non-abstained primary block passes.

## 8. Truth-relative utility

Prediction-vs-truth errors use truth-normalized NRMS rather than the symmetric equivalence normalization.

A robust composite is the median of finite block errors, each capped at 10.

The teacher-near feasible swap is a `material_truth_improvement` iff both:

- relative composite improvement >= 0.20; and
- absolute composite decrease >= 0.05.

Material degradation is defined symmetrically at <= -0.20 relative change and >= 0.05 absolute increase.

This utility test is secondary to the mechanical-equivalence test. It asks whether a mechanically different feasible alternative is actually closer to the evaluator mechanical target.

## 9. Aggregate decision tree

Population sufficiency:

- total witnesses `n >= 20`; and
- witnesses occur in at least 4 families.

If not: `INSUFFICIENT_HARDTAIL_POPULATION`.

Otherwise define:

- `equivalence_fraction` = mechanically equivalent witnesses / all witnesses;
- `material_truth_improvement_fraction` = materially improved witnesses / all witnesses;
- family equivalence guard: for every family with at least 5 witnesses, family equivalence fraction >= 0.60.

Decision:

1. If `equivalence_fraction >= 0.80` and family guard passes:  
   `GEOMETRIC_SINGLETON_NOT_REQUIRED_UNDER_REAL_OBSERVABLE_GFDR_V2`

2. Else if `equivalence_fraction <= 0.50` and `material_truth_improvement_fraction >= 0.50`:  
   `MECHANICALLY_MATERIAL_ALTERNATIVES_EXIST__NEXT_FUNCTIONAL_CLASS_OBSERVABILITY_AUDIT`

3. Else if `equivalence_fraction <= 0.50`:  
   `MECHANICS_CHANGE_BUT_TEACHER_NEAR_UTILITY_NOT_DOMINANT__RESEARCH_REQUIRED`

4. Else:  
   `FUNCTIONAL_QUOTIENT_AMBIGUOUS__RESEARCH_REQUIRED`

No branch authorizes an impossibility claim.

## 10. Invalidation guards

The run is invalid if any of the following occurs:

- any source/input hash differs from the frozen manifests;
- any observable phase reads evaluator-side semantic content;
- sidecar semantic parsing occurs before the observable state manifest is frozen;
- baseline decorator replay fails;
- teacher normals/visibility or teacher-created XYZ enters counterfactual forward mechanics;
- sealed21/external10 is opened;
- training or retuning occurs;
- source/thresholds/decision tree are changed after this prereg freeze;
- a canonical result path is overwritten;
- results are interpreted as information-theoretic impossibility without a separate functional-class observability experiment.

## 11. Required output order

`observable states -> observable manifest + hashes -> persist/verify -> truth open -> RESULT.json -> REPORT.md -> DECISION.json -> three-store result verification -> state promotion`.
