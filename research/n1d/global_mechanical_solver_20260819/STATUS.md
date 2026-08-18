# RealSaS N1D — New Global Mechanical Hypothesis-Set Solver Status

**Date:** 2026-08-19
**Status:** ACTIVE RESEARCH / NOT SOLVED YET
**Stage-B truth:** CLOSED

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

## Current scientific interpretation

The single-witness global rank-1 success does not generalize across interventions. The next work must separate at least:

1. candidate containment vs selector failure per intervention;
2. finite-motion support/activity from near-zero silence;
3. one-global-transform rank-1 assumptions from multi-part/local mechanical coherence;
4. typed observation-only scores (descriptor, reprojection, view support, conditioning, local deformation coherence) rather than one unary reprojection score.

No Stage-B data should be opened while this line is under development.

## Persistence

Compact research bundle SHA-256: `833cfeb104c795323c4ded90b0608cb1c5ed3df2998cd82a44bd7aa28d01b5df`.
V1 source SHA-256: `efa858c59489688f9937a207bf78f03cfe9614b24d12f22369d13142e4e897c1`.
Qualification result SHA-256: `a9fcc2ad4ececae998e17a79f732d133f8a8a08c6220e440d93abb4262d09b33`.
Freeze ledger SHA-256: `069ed3132d806ac10e809991ac1002bf03ef7e86d8ca0dc271f39f69e9922bd7`.

Library: `/RealSaS_OPT/N1D_20260819_GLOBAL_MECHANICAL_SOLVER_RESEARCH/`
Drive: `RealSaS_IRIS_N1D_GLOBAL_MECHANICAL_SOLVER_RESEARCH_20260819/`
