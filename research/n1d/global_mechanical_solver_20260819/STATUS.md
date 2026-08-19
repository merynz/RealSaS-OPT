# RealSaS N1D — New Global Mechanical Hypothesis-Set Solver Status

**Date:** 2026-08-19  
**Current canonical state:** POST-STAGE-B DEVELOPMENT / V11 FROZEN QUALIFICATION CLOSED  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Current canonical frontier — 2026-08-19

Hybrid V11 is the last frozen qualification treatment on this line.

- Stage A: **PASS** on 16/16 frozen episodes with exact standalone-runner parity.
- Stage B: **FAIL, immutable/no-retune**. Geometry/mechanics retained strong values, but primary F-activity Spearman was `0.433160 < 0.50`.
- Stage-B truth is now **OPEN DEVELOPMENT EVIDENCE** only. It must not be reused as a fresh blind qualification panel.
- sealed21 / external10 remain **CLOSED**.

Post-Stage-B evidence now isolates the remaining failure more narrowly:

1. frozen V11 baseline was re-evaluated against the exact canonical evaluator with **12/12 episode parity, max diff 0.0**;
2. a direction-fixed exact-amplitude ceiling closes F activity and preserves all primary downstream quality gates;
3. more importantly, a **truth-rank / frozen-magnitude-distribution / frozen-direction** arm preserves the complete V11 displacement-magnitude multiset to `2.78e-17` numerical error while moving F activity from `0.433160` to `1.000000`; F-kernel `0.838823`, D `0.068851`, R `0.043841`, G-direction `0.897785`, G-line `0.076640`, all primary quality gates PASS;
4. therefore a scalar motion **ordering/ranking ceiling** is demonstrated without replacing the V11 direction field or increasing the global magnitude distribution;
5. this is **not yet a candidate-basin composition proof**, because frozen V11 NPZs did not persist each carrier's feasible candidate set `H_i`;
6. route-local frozen current evidence is strong on `V8_BASE`, but naive free radial composition can slightly worsen G-line, supporting the intended architecture: amplitude/activity should score/select endpoints **inside** feasible geometry rather than free-scale XYZ.

### Current architecture target

Keep common-world dual-time geometry and factor motion state explicitly:

```text
P_A, P_B_geom, N_A, N_B, V_A, V_B, Z,
p_active, log_amp, dir, U_pred
+ deterministic typed U_obs
```

`p_active/log_amp/dir` are evidence, not independent XYZ authority. Final `P_B` remains constrained by multiview feasible geometry.

### Next required experiment

Reconstruct/export V11 candidate sets `H_i` on the now-open Stage-B panel, recompute frozen N1D `delta_point_map_srcA` amplitude/activity for **all routes**, and use that scalar evidence only to rank/select candidates inside `H_i`. Compare:

- frozen V11 baseline;
- unconstrained scalar composition;
- candidate-basin current-amplitude composition;
- direction-fixed oracle rank ceiling.

Only after candidate-basin composition is demonstrated should the learned held-out `p_active + log_amp` ranking head be trained. A later qualification requires a **new untouched preregistered panel**.

## Canonical post-Stage-B artifacts

- `POST_STAGEB_CURRENT_AMPLITUDE_DIAGNOSTIC.json`
- `REALSAS_IRIS_POST_STAGEB_TARGET_REPRESENTATION_LOSS_RESEARCH_V1_20260819.md.b64`
- `post_stageb_factorization_ceiling_v1.py`
- `POST_STAGEB_FACTORIZATION_CEILING_V1_RESULT_SUMMARY.json`
- `POST_STAGEB_FACTORIZATION_CEILING_V1.md`

## Historical status preserved below

The following section is retained as provenance for the pre-V11 research frontier. Statements such as “Stage-B truth CLOSED” describe that historical point and are superseded by the current canonical frontier above.

---

**Historical date:** 2026-08-19  
**Historical status:** ACTIVE RESEARCH / NOT SOLVED YET  
**Historical Stage-B truth:** CLOSED

## Why this line exists

Historical source recovery is no longer the primary path. Code-level architecture audit and repeated local diagnostics converge on the same failure: the descriptor/multiview hypothesis set often contains a good physical endpoint, while per-carrier unary selection collapses the set incorrectly.

The new line therefore preserves set-valued 3D hypotheses and tests observation-only **global mechanical coherence** before committing `P_B`. It is informed by the prior RealSaS code-level audit of DPM/V-DPM, MV-TAP, MVTracker, St4RTrack, and GGPT.

## Preserved evidence

- Historical descriptor-top4 hypothesis oracle: flow/zero ~0.4254, direction ~0.9283.
- Historical conclusion: per-carrier unary geometry selection is the wrong abstraction.
- Local 10763/e01 witness: candidate-set containment is strong and a global rank-1 mechanical selection can recover a strong G solution.
- Full 16-episode Stage-A frozen V1 qualification **FAILS**; therefore no claim of solution and no Stage-B opening.

## Full Stage-A V1 frozen result

- D tangent: `0.075206` PASS
- F activity Spearman: `0.076923` FAIL
- F kernel Spearman: `0.611932` PASS
- R differential: `0.070552` PASS
- G direction: `0.412111` FAIL
- G line: `0.124538` FAIL
- G carriers: `146` PASS
- G families: `4` PASS
- valid F episodes: `7` FAIL
- family G pass: `1/4` FAIL
- episode-index G pass: `1/4` FAIL

Verdict: `FAIL__MOTION_SUPPORT_FRONT_DOOR_AND_FINITE_AXIS_GENERALIZATION`.

## Historical scientific interpretation

The single-witness global rank-1 success does not generalize across interventions. The next work must separate at least:

1. candidate containment vs selector failure per intervention;
2. finite-motion support/activity from near-zero silence;
3. one-global-transform rank-1 assumptions from multi-part/local mechanical coherence;
4. typed observation-only scores (descriptor, reprojection, view support, conditioning, local deformation coherence) rather than one unary reprojection score.

At this historical point, Stage-B remained closed.

## Historical persistence

Compact research bundle SHA-256: `833cfeb104c795323c4ded90b0608cb1c5ed3df2998cd82a44bd7aa28d01b5df`.  
V1 source SHA-256: `efa858c59489688f9937a207bf78f03cfe9614b24d12f22369d13142e4e897c1`.  
Qualification result SHA-256: `a9fcc2ad4ececae998e17a79f732d133f8a8a08c6220e440d93abb4262d09b33`.  
Freeze ledger SHA-256: `069ed3132d806ac10e809991ac1002bf03ef7e86d8ca0dc271f39f69e9922bd7`.

Library: `/RealSaS_OPT/N1D_20260819_GLOBAL_MECHANICAL_SOLVER_RESEARCH/`  
Drive: `RealSaS_IRIS_N1D_GLOBAL_MECHANICAL_SOLVER_RESEARCH_20260819/`
