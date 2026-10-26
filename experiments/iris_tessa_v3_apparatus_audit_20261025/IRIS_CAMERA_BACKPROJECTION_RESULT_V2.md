# IRIS Camera Geometry / Backprojection Court — Result V2

**Date:** 2026-10-26  
**Preregistration:** `IRIS_CAMERA_BACKPROJECTION_PREREG_V2.md`  
**Workflow:** `iris-camera-backprojection-v2-stage`  
**Run:** `40315350158`  
**SHA:** `db366db2cfa54658072c40089bfe2c5fdfb04932`  
**Artifact digest:** `sha256:02ea7ce9b46c75f12302e2ad016734530f11c3d99495c2cfb411cc09ad298965`  
**Status:** `CURRENT_FACTORIZED_CAMERA_GEOMETRY_SUFFICIENT_TINY_V2`

## Result

All validity gates passed. Exact deterministic backprojection reconstruction max absolute error was `5.96e-08`; random-label accuracy was `0.0630`, consistent with 1/16 chance.

Held-out yaw/orthographic-scale canonical-locus recovery:

| arm | mean | min | trainable params |
|---|---:|---:|---:|
| CURRENT_FACTORIZED_ADD | 1.000000 | 1.000000 | 5,616 |
| JOINT_RAW_CAMERA_UV | 1.000000 | 1.000000 | 4,528 |
| DETERMINISTIC_BACKPROJECTED | 1.000000 | 1.000000 | 4,368 |
| TRUE_CANONICAL_POSITIVE_CONTROL | 1.000000 | 1.000000 | 4,368 |

Paired deterministic-minus-current accuracy gain was exactly zero on every seed.

## Decision

The preregistered current-sufficiency gate passed. Deterministic backprojection is therefore **not selected**, because it supplies no measurable generalization gain on the isolated camera-geometry task.

Carry forward the current factorized learned camera+UV source-token geometry as the V3.1 pre-Truth baseline. Keep the deterministic backprojection formula as an available diagnostic / compiler-side geometry transform, not a required learned-input feature.

This result closes the camera-feature-placement question at the bounded architecture-search level. It does not claim robustness to noisy/perspective/un-normalized real cameras.

## Authority

No ProductRevision, ModelRelease, Truth139, FIT8, LOFO or genericity authority changes.
