# RealSaS N1D — New Global Mechanical Hypothesis-Set Solver Status

**Date:** 2026-08-19  
**Current canonical state:** POST-STAGE-B DEVELOPMENT / CANDIDATE-BASIN COMPOSITION DEMONSTRATED / V11 FROZEN QUALIFICATION CLOSED  
**Frozen Stage-B authority:** `STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`

## Current canonical frontier — 2026-08-19

Hybrid V11 remains the last frozen qualification treatment on this line.

- Stage A: **PASS** on 16/16 frozen episodes with exact standalone-runner parity.
- Stage B: **FAIL, immutable/no-retune**. Geometry/mechanics retained strong values, but primary F-activity Spearman was `0.433160 < 0.50`.
- Stage-B truth is now **OPEN DEVELOPMENT EVIDENCE** only. It must not be reused as a fresh blind qualification panel.
- sealed21 / external10 remain **CLOSED**.

Post-Stage-B development evidence now isolates and causally tests the remaining activity factor:

1. frozen V11 baseline was re-evaluated against the exact canonical evaluator with **12/12 episode parity, max diff 0.0**;
2. a direction-fixed exact-amplitude ceiling closes F activity and preserves all primary downstream quality gates;
3. a **truth-rank / frozen-magnitude-distribution / frozen-direction** arm preserves the complete V11 displacement-magnitude multiset to `2.78e-17` numerical error while moving F activity from `0.433160` to `1.000000`; F-kernel `0.838823`, D `0.068851`, R `0.043841`, G-direction `0.897785`, G-line `0.076640`, all principal primary quality gates PASS;
4. the previously missing candidate-basin test is now demonstrated on the three **primary seed-route** Stage-B episodes (`10763/e04`, `11214/e04`, `14714/e04`): keep the frozen mechanical basin `q_i`, derive scalar weights only from frozen N1D `delta_point_map_srcA`, and compose `P_B = P_A + w_i q_i` without free XYZ authority;
5. primary seed-route F activity moves `0.1261 -> 0.6872`, `0.4332 -> 0.68485`, and `0.4000 -> 0.8333`; when combined with unchanged primary `V8_BASE` episodes, aggregate F activity moves `0.433160 -> 0.687180`;
6. treated primary aggregate principal metrics are F-kernel `0.70775`, D `0.10163`, R `0.04274`, G-direction `0.89947`, G-line `0.08024`: **6/6 principal quality metrics PASS**; family G robustness remains `3/4`, episode-index robustness improves `2/3 -> 3/3`;
7. the treatment remains candidate-anchored: median distance from current-amplitude basin weight to nearest feasible candidate-projection weight is approximately `0.00706`, `0.00701`, `0.00431` on the three primary seed episodes;
8. silence must be factored separately: raw-current p95 is approximately `1.384e-3` on primary `11214/e04` versus `3.068e-4` on near-zero `11214/e06` (~`4.51x` separation), while q95-normalized composition erases absolute scale and raises the near-zero predicted moved fraction `0.1719 -> 0.1875`;
9. therefore the supported architecture is **candidate/mechanical basin as XYZ authority + separate `p_active` silence gate + conditional `log_amp` ranking/magnitude evidence + direction evidence**. A free XYZ motion head is not supported by the current evidence.

This development result does **not** rewrite frozen Stage-B as PASS. The historical qualification also has a separate design defect: all three `12907` Stage-B episodes are truth-stratum `near_zero`, so its primary-only four-family G-coverage requirement was structurally unattainable.

### Current architecture target

Keep common-world dual-time geometry and factor motion state explicitly:

```text
P_A, P_B_geom, N_A, N_B, V_A, V_B, Z,
p_active, log_amp, dir, U_pred
+ deterministic typed U_obs
```

with authority split:

```text
candidate/mechanical basin -> physical XYZ authority
p_active                  -> absolute motion / silence gate
log_amp                   -> conditional magnitude / ranking evidence
dir                       -> direction evidence
```

`p_active/log_amp/dir` are evidence, not independent XYZ authority. Final `P_B` remains constrained by multiview feasible geometry.

### Next required experiment

The candidate-basin causal prerequisite is now satisfied for the primary seed-route development panel. The next step is a small **family/episode-held-out learned `p_active + log_amp` probe**, using explicit conditional ranking loss plus silence/activity supervision, while keeping endpoint selection constrained to the candidate/mechanical basin.

Do **not** authorize a new blind qualification from Stage-B. Any later qualification requires a **new untouched preregistered panel**.

## Canonical post-Stage-B artifacts

- `POST_STAGEB_CURRENT_AMPLITUDE_DIAGNOSTIC.json`
- `REALSAS_IRIS_POST_STAGEB_TARGET_REPRESENTATION_LOSS_RESEARCH_V1_20260819.md.b64`
- `post_stageb_factorization_ceiling_v1.py`
- `POST_STAGEB_FACTORIZATION_CEILING_V1_RESULT_SUMMARY.json`
- `POST_STAGEB_FACTORIZATION_CEILING_V1.md`
- `POST_STAGEB_CANDIDATE_ANCHORED_MECHANICAL_BASIN_V1.md`
- `POST_STAGEB_CANDIDATE_ANCHORED_MECHANICAL_BASIN_V1_RESULT.json`
- `post_stageb_candidate_anchored_mechanical_basin_v1.py` — compact canonical treatment specification recovered after the execution-session boundary; full replay depends on the frozen V11 runner/candidate reconstruction and canonical GFDR evaluator already preserved in this research line.

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
