# RealSaS-OPT — Current State

**Date:** 2026-08-18

## Active line

IRIS-SEES post-N1D continuation. The product target is not exact teacher world-vector reconstruction. IRIS-SEES measures observation-native primitives; deterministic geometry/mechanics and the compiler consume them.

## N1D canonical baseline

N1D itself remains an informative FAIL, not a product/model-2 handoff. Important preserved signal:
- cross-pose descriptor top1 ≈ 0.8904;
- Pose-B is genuinely used;
- motion/activity ranking is learned better than world-vector direction;
- near-zero silence and exact same-pose/swap invariants are preserved.

## Post-N1D localization

The main failure was progressively localized:

1. Simple transport/output calibration and longer training do not close the world-vector gap.
2. Problem-A provides a coherent raster-only geometric front door, but direct Problem-A scene-flow alone is not family-robust enough.
3. Hard current-vs-Problem-A selector families failed to generalize; early singleton collapse is a recurring dead-end.
4. The two-candidate set `{N1D current, Problem-A}` has a strong oracle upper bound, showing useful candidates exist, but is too sparse for compiler-grade rank-1 mechanics.
5. Raw DIS → 3D lifting is geometrically well-conditioned, but ordinary LS, LMedS and forward/backward cycle weighting all fail family robustness. The blocker is 2D persistent correspondence, not camera inversion.
6. DIS-centered descriptor search loses true endpoints because a minority of DIS errors are large: fixed ±8 px search contains truth only ~85.34% of visible view cases; descriptor top-4 retains truth ~84.79% conditional on being in-window.
7. Removing DIS as search authority and using **global B-foreground descriptor search + calibrated multiview geometry** produced the first strong post-N1D development PASS.

## Frozen development PASS

Treatment:
- global B-foreground descriptor coarse grid stride 4;
- top-8 coarse candidates;
- ±4 px / 2 px-step local refinement;
- final top-4 candidates per view;
- calibrated multiview rank-3 3D hypothesis solve;
- current N1D fallback on abstain;
- frozen N1D activity gate.

16-family open-development result:
- flow/zero: **0.7409 → 0.4411**;
- weighted direction cosine: **0.5839 → 0.8994**;
- family direction non-regress: **14/16**;
- non-abstain coverage: **99.90%**;
- false activation: **41 → 39**;
- prereg gates: **5/5 PASS**.

This is development authority only. It does **not** authorize sealed21, external10, product handoff or retraining.

## Active next gate

Frozen untouched-open-dev qualification on four previously untouched DEV families:

`10763, 11214, 12907, 14714`, episode `e00`.

The algorithm is frozen exactly as above. Teacher/sidecar may open only after all four predictions are complete. Required gates:
- aggregate flow/zero non-regress;
- aggregate weighted direction non-regress;
- carrier non-abstain ≥ 0.95;
- false activation no regress;
- direction non-regress in at least 3/4 families.

**sealed21 = CLOSED**  
**external10 = CLOSED**

## Data / workspace policy

- GitHub: canonical code, prereg, compact results, reports, experiment history.
- Drive: heavy corpus/cache/checkpoint/proof-pack depot.
- Library: transient active diagnostics and working artifacts.

Never silently redefine GFDR semantics to fit an implementation. Distinguish canonical structured objects from lower-level learned observation primitives.
