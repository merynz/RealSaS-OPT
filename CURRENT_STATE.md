# RealSaS-OPT — Current State

**Date:** 2026-08-31  
**Canonical continuation branch:** `main`  
**Status:** `DINO_S1_CLOSED_EARLY__FROZEN_APPARATUS_BLOCK_AND_FIX__REPROJECTION_V2_PREREG_SEALED__GATE0_DETERMINISTIC_ONLY__DEV32_CLOSED__NO_V2_OPTIMIZER_AUTHORIZED`

This file is continuation authority only on `main`.

## One-line state

`DTB-ND1 tolerance and consumer replay are closed -> DINO S/B were nearly flat under SharedLearnerV1 -> S1 was explicitly early-terminated -> frozen-apparatus audit found a material high-resolution/access confound -> IRIS V2 is now sealed as a reprojection-centered partial-canonical evidence system -> only deterministic Gate 0 coding/testing is authorized; no V2 optimizer step is authorized yet.`

## 1. Product/system contract — STABLE

North star:

`ONE NEUTRAL 8-VIEW CHARACTER SHEET -> EDITABLE, RIGGED, ANIMATABLE PUPPET`

External route remains:

```text
8 neutral views + known orthographic cameras
 -> IRIS observation-grounded forward depth/evidence
 -> analytic P = O + dF
 -> SurfaceBuilder / RiggingSurfaceIR
 -> Geppetto SkeletonProposalIR
 -> Compiler skeleton qualification
 -> Arachne SkinProposalIR
 -> Compiler skin qualification
 -> CanonicalPuppetGraph
 -> deformation/motion proof
 -> runtime
```

IRIS/Geppetto/Arachne emit evidence/proposals. Compiler owns canonical product state.

Mode G / Mode E topology invariant is binding in `canonical/PRODUCT_CONTRACT_V1.md`: analytic geometry and learned evidence keep the same authority topology; uncertainty changes evidence burden, not geometry ownership.

## 2. Geometry tolerance — CLOSED FOR CURRENT CONSUMER PROFILE

Historical frozen-D2 boundary:

`0.00225 <= epsilon_critical < 0.00250 RMS`

DTB-ND1 robust local-plane boundary:

`0.00250 <= epsilon_critical < 0.00275 RMS`

Matched ell=0 depth absolute-P95 is approximately `.00490-.00539`.

Authority: `canonical/DTB_ND1_ROBUST_LOCAL_PLANE_CLOSURE_20260829.md`.

## 3. DINO S1 — EARLY TERMINATED / SEALED

| Rung | FIT primary | TRAIN primary | Verdict |
|---|---:|---:|---|
| S @32768 | 10/54 | 17/56 | `COMPLETE_PRIMARY_FAIL` |
| B @32768 | 10/54 | 18/56 | `COMPLETE_PRIMARY_FAIL` |
| L | no primary eval | no primary eval | `TERMINATED_PARTIAL_NOT_EVALUATED` |
| g | not run | not run | `NOT_RUN` |

L history ends at step 192 and has no scientific interpretation. DEV32 remained closed.

Permitted S1 claim:

> Under SharedLearnerV1 and the sealed S1 apparatus, S->B scaling produced no material downstream accessibility improvement.

Forbidden: extrapolation to L/g, global backbone irrelevance, or S/B flatness under another apparatus.

Authority: `canonical/DINO_LADDER_S1_EARLY_TERMINATION_CLOSURE_20260831.md`.

## 4. Frozen apparatus audit — CLOSED: BLOCK_AND_FIX

Authority: `canonical/DINO_FROZEN_APPARATUS_CODE_AUDIT_CLOSURE_20260831.md`.

Main finding: the S1 measuring apparatus had a material shared-access limitation: last-layer 37x37 DINO interface, common width bottleneck, 16x16 cross-view context branch, sparse 25-sample native detail path and pointwise final query head. This is a plausible confound, not a proven sole cause.

Evaluator / DTB-ND1 replay passed the audit. Loss/sampling/corpus remain separate plausible limitations.

## 5. IRIS Reprojection-Centered V2 — PREREG SEALED

Authority: `canonical/IRIS_REPROJECTION_CENTERED_V2_PREREG_20260831.md`.

Machine-readable seal: `experiments/iris_reprojection_v2_20260831/IRIS_REPROJECTION_V2_PREREG_V1.json`.

Central change:

> Exact cameras create canonical world candidates and exact eight-view reprojections; learned features judge evidence agreement. Analytically identifiable geometry is not relearned.

V2-A default backbone is frozen DINOv2-S/14 for compute-efficient apparatus development. This is **not** a proof that S is sufficient. Larger DINO is forbidden until S exhibits a representation ceiling under a qualified V2 apparatus.

Internal canonical representation is **partial observed-surface evidence**, never a completed occupancy/SDF authority. Unknown occluded/unobserved regions remain typed UNKNOWN and are not rendered as invented depth.

V2-A is deliberately single-pass. No recurrent visibility refinement, learned hidden completion, sparse-convolution dependency or new S/B/L/g ladder is authorized.

## 6. Gate 0 — ACTIVE, DETERMINISTIC ONLY

Gate-0 source:

- `experiments/iris_reprojection_v2_20260831/gate0_geometry_v1.py`
- `experiments/iris_reprojection_v2_20260831/test_gate0_geometry_v1.py`
- `experiments/iris_reprojection_v2_20260831/run_gate0_synthetic_preflight_v1.py`

CI: `.github/workflows/iris_reprojection_v2_gate0.yml`.

Gate 0 has **zero optimizer steps** and tests:

- exact projection/backprojection;
- common-world-Z row invariance;
- visual-hull truth containment x search-reduction frontier;
- spacing x thin-structure resolvability;
- robust view-evidence permutation/outlier behavior;
- streamed candidate memory accounting;
- explicit UNKNOWN state preservation.

Coarse spacing is not post-hoc tuned. Training-free candidates are `.016`, `.008`, `.004`. Hull padding candidates are `0,1,2,4,8` native pixels. Selection requires both containment and useful search reduction.

Program rule:

`NO_ARCHITECTURAL_MECHANISM_WITHOUT_A_DEMONSTRATED_FAILURE_IT_ADDRESSES`.

## 7. Next gates

Only after Gate 0 closes:

1. freeze exact DINO-S intermediate taps, native-pyramid implementation, learned evidence representation, loss, runtime and source hashes;
2. Gate 0.5: one real family, single-pass ceiling;
3. if successful, Gate 1: sealed heterogeneous 8-family ceiling with thickness/orientation strata;
4. only then small causal ablations: DINO ON/native ON, DINO OFF/native ON, DINO ON/native OFF;
5. full training requires a new explicit authorization.

A native-only success is legitimate evidence. No larger backbone is automatically scheduled.

## 8. Geppetto/Arachne corpus work — IMAGE INTEGRITY PHASE CLOSED

Authority: `canonical/GEPPETTO_ARACHNE_NATIVE_IMAGE_INTEGRITY_CLOSURE_20260831.md`.

Completed native image/raster measurement:

- `2874 / 2874` objective-pass assets measured;
- `22992 / 22992` views exact support equality;
- IoU `1.0` for every measured view;
- hard authority failures `0`.

This proves image/raster support integrity only. Semantic single-riggable-character review remains required and fail-closed before final clean C0. Random/blind visual sanity checking remains mandatory before final clean-C0 trust.

Corpus H0 work remains separate from V2 apparatus causality and may not reinterpret S1 post hoc.

## 9. Training authority

**V2 learned optimizer steps: NOT AUTHORIZED.**  
**New S/B/L/g ladder: NOT AUTHORIZED.**  
**Gate-0 deterministic implementation/testing: AUTHORIZED.**  
**DEV32: CLOSED.**
