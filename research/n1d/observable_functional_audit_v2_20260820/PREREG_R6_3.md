# N1D Observable Functional Audit V2 — R6.3 Staged Process-Isolated Clean Replication Preregistration

**Experiment ID:** `N1D_OBSERVABLE_FUNCTIONAL_AUDIT_V2_20260820`  
**Source revision:** `R6.3`  
**Panel:** untouched `e01` replication  
**Status:** `PREREG_FROZEN__TRUTH_SEMANTICALLY_CLOSED`

## 1. Inheritance and allowed change

R6.3 inherits the R6.2 scientific contract verbatim, which itself inherits R6.1. Parent preregistration: `PREREG_R6_2.md`, SHA-256 `eb473e5bdeec1d2c0cf67aa93d6921fa893838060d43a31b4fa773b957c2ee3d`.

The **only** R6.3 change is finer execution isolation inside observable Phase-1. R6.2 separated baseline and state into two OS processes, but family 15290/e01 reproducibly exposed a runtime pathology when the frozen baseline process itself executed multiple model contexts sequentially. R6.3 therefore stages the unchanged frozen computation as four explicit OS processes through `R6_3_STAGED_PROCESS_ISOLATED_OBSERVABLE_HARNESS.py`, SHA-256 `c5ea7b0d6cd64fc5555b53f5dc9d82b587fbda1c54d452c28b1ea6c40d2da91e`:

`V8 process -> exit -> gate process -> exit -> baseline-finalize/seed process -> exit -> observable-state process`.

No model, checkpoint, raster input, candidate construction, candidate score, geometry operation, GFDR-V2 operation, witness definition, metric, threshold, population rule, evaluator rule, or decision rule may change.

R6.3 is admitted only after byte-exact parity on both route classes: 09908/e01 reproduced baseline NPZ/JSON and full observable-state NPZ/JSON exactly versus R6.2; 12772/e01 and 13203/e01 reproduced final seed-route baseline NPZ/JSON exactly. Parity record SHA-256: `313d66ad51276395ac7ad26657de179e5060923a8fe13a50eeae7173d3774a0d`. Teacher/outcome access during parity was `NONE`; truth semantics remained `CLOSED`.

R6.3 freeze authority SHA-256: `f1ac1630f696254340bc42864ef3e4ee29073c52ca2121ce89b3b71869e43cf8` (GitHub commit `18ff77da34a28fe60fd086d481898ee701630912`).

## 2. Scientific question

Within the actual frozen raster-only N1D Hybrid V11 front door, when the committed Pose-B geometry is evaluator-far but the independently observation-derived feasible set `H_i` contains an evaluator-near alternative, does replacing only that carrier by the nearest **already-existing, already-frozen feasible `H_i` candidate** usually leave downstream GFDR-V2 mechanics materially unchanged, or does it induce mechanically material consequences when counterfactual Pose-B normals and visibility are regenerated from the same raster/model front door?

This is a **functional-geometry quotient audit**. It is not an exact-teacher-geometry objective. It does not test or prove information-theoretic impossibility.

## 3. Frozen authority

Inherited R6.1 authority:

- R6.1 freeze commit: `7b9ee7a8f4068dbf8f873508fceb1bfc28dd786d`
- source bundle SHA-256: `b4ab69fbcc91d8c57ab5f36f151b37495e8d5366554f3189982406d0b60c282f`
- source index SHA-256: `dca7f6a6395db21ee69c72c296392b6eb7b019db330e9b5c88831ae3e0d13005`
- exact e01 input manifest SHA-256: `91acc078a9f6c41604b3f6f829c2cb239728d407f75872493a6712dd4ba7ca02`
- R6.1 preflight SHA-256: `e418e5ee10365b2c9bdba990710244fcc33db47a2161435effdb01852aa372dc`
- frozen checkpoint SHA-256: `0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18`
- frozen GFDR-V2 SHA-256: `ca22e3fd42e9c812632eb372544e2319ecf720f6dede1f4a60c8da14ff0cb8fa`

R6.3 execution authority:

- staged harness SHA-256: `c5ea7b0d6cd64fc5555b53f5dc9d82b587fbda1c54d452c28b1ea6c40d2da91e`
- execution isolation note SHA-256: `0e401d12543e83310553b3b4109fbfd89792a33914b5717b2396d2fbb3aa5e01`
- execution parity record SHA-256: `313d66ad51276395ac7ad26657de179e5060923a8fe13a50eeae7173d3774a0d`
- R6.3 preflight SHA-256: `6aefef3966c7b63896ffa4167ab0324fbff86233bd02ada4a69cec85048a8402`
- all four R6.3 artifacts passed local ↔ Drive ↔ Library byte-identical readback before this preregistration.

## 4. Population — frozen before evaluator truth-open

Only:

`[09908,11032,12772,13203,14404,14702,14758,15290] × e01`

No carrier/family may be included or excluded using teacher error, functional outcome, GFDR outcome, or any result observed after this preregistration.

The input gate is exactly 136/136 producer-manifest-verified bytes:

- Pose-A rasters: 64/64 PASS
- e01 Pose-B rasters: 64/64 PASS
- e01 observation sidecars: 8/8 byte-hash PASS
- `sealed21=CLOSED`
- `external10=CLOSED`

All eight families must be regenerated under R6.3 for common staged-execution provenance. All pre-existing R6.1/R6.2 observable states are parity/diagnostic evidence only and MUST NOT be reused as canonical R6.3 panel outputs.

## 5. Truth firewall and Phase-1

Before the complete 8-family observable panel is frozen, persisted, and read back:

- observation sidecars are opaque bytes only;
- semantic deserialization, key inspection, array inspection, or value access is forbidden;
- `_GENERATOR_ONLY.json` is forbidden;
- exact `surface_points_*`, `surface_normals_*`, `surface_visibility_*`, `surface_xy_*`, skeleton, dense weights, intervention labels, or teacher geometry may not enter mechanics or candidate selection;
- no training, retune, free XYZ optimization, or threshold change is permitted.

For each family, Phase-1 MUST be exactly:

1. invoke R6.3 harness `--mode v8` in one OS process;
2. that process exits;
3. invoke the same frozen harness `--mode gate` in a fresh OS process;
4. that process exits;
5. invoke `--mode baseline-finalize` in a fresh OS process; it may call the unchanged frozen `seed_predict` if and only if the unchanged frozen gate switches;
6. that process exits;
7. invoke `--mode state` in a fresh OS process;
8. verify source-member hashes, checkpoint hash, all intermediate/final prediction hashes, current raster hashes, `truth_access=NONE`, and state-baseline SHA consistency;
9. verify frozen decorator replay gates (`V_B` exact and `N_B` replay tolerance from inherited R6.1 contract);
10. persist the observable state and metadata.

The canonical observable panel freezes, at minimum, the R6.1 state fields:

`P_A, P_B, N_A, N_B, V_A, V_B, XY_A, XY_B, H_xyz, H_reproj_px, H_desc_score, H_offsets`, route and source/raster/checkpoint hashes.

An `OBSERVABLE_PANEL_MANIFEST_R6_3.json` must enumerate all eight state files and SHA-256 values. It must be persisted to GitHub, Drive and Library and pass byte readback. **Only after that gate may evaluator truth semantics be opened.**

## 6. Phase-2 evaluator-only truth use

After Phase-1 panel freeze, exact sidecar fields may be opened only in evaluator code to:

- measure committed-state evaluator error;
- identify, within the already-frozen `H_i`, the teacher-nearest feasible candidate;
- define the frozen hard-tail witness population according to the inherited R6.1 witness definition;
- evaluate improvement and mechanics consequences.

Teacher data may not create a new candidate, move a candidate, alter a candidate score, alter the global/committed observable state, or supply `N_B`/`V_B`/mechanics inputs. Counterfactual `N_B` and `V_B` must be regenerated from the frozen raster/model front door exactly as in R6.1.

## 7. Frozen mechanics equivalence test

Primary equivalence uses the inherited symmetric state-vs-state normalized mechanics comparison. Tautological raw-coordinate blocks `F_delta` and `R_rel_B` are excluded from the primary equivalence decision; `F_delta_NRMS` is descriptive only.

A witness is mechanically equivalent only if **all** frozen limits pass:

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

No limit may be edited after execution begins.

## 8. Population sufficiency and frozen decision tree

Population sufficiency requires:

- witness count `n >= 20`, and
- witnesses from at least 4 families.

For the singleton-not-required conclusion, every family with at least 5 witnesses must have equivalence rate `>= 0.60`.

Frozen decision order:

1. if population sufficiency fails → `INSUFFICIENT_HARDTAIL_POPULATION`;
2. else if pooled mechanical equivalence `>= 0.80` and the family guard passes → `GEOMETRIC_SINGLETON_NOT_REQUIRED_UNDER_REAL_OBSERVABLE_GFDR_V2`;
3. else if pooled equivalence `<= 0.50` and materially truth-improving alternatives occur at rate `>= 0.50` → `MECHANICALLY_MATERIAL_ALTERNATIVES_EXIST__NEXT_FUNCTIONAL_CLASS_OBSERVABILITY_AUDIT`;
4. else if pooled equivalence `<= 0.50` → `MECHANICS_CHANGE_BUT_TEACHER_NEAR_UTILITY_NOT_DOMINANT__RESEARCH_REQUIRED`;
5. otherwise → ambiguous/research-required outcome under the inherited R6.1 decision semantics.

None of these outcomes, including a low equivalence rate, by itself proves raster-only non-identifiability or impossibility.

## 9. Separation from the later Rank-2 R experiment

R6.3 is the clean corrected functional-geometry quotient replication with staged execution isolation. It is **not** the later set-valued global-world `R` experiment.

The later Rank-2 experiment remains gated on R6.3 closure. Its first gate remains historical Rank-3 behavioral parity; only then may `R` be lifted from singleton carrier state to selection over the frozen observation-consistent candidate sets `H_i` in a full-64 global-world solve. Teacher geometry remains evaluator-only there as well.

## 10. Run authorization

Execution is authorized only after this exact preregistration itself is persisted to GitHub, Drive and Library and all three byte copies match. A final execution-authority record must bind this preregistration SHA to `R6_3_FREEZE_AUTHORITY.json` before the first canonical R6.3 family run.
