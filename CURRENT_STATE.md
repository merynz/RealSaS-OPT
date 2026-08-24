# RealSaS-OPT — Current State

**Date:** 2026-08-24  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_SUFFICIENT__CI202_RAW_XYZ_DIAGNOSTIC_COMPLETE__CI244_V3_FIXED_SCALE_FALSIFIED__P_V4_OBSERVABLE_SCALE_FROZEN__OPTIMIZER_ZERO__TRAINING_FORBIDDEN`

## Single continuation authority

Active implementation: `experiments/iris_single_pose_v2/`.

Current scientific gate:

- `P_FORMULATION_V4_OBSERVABLE_SCALE_CLOSURE_PREREG_20260824.md`
- `model_pv4.py`
- `p_formulation_v4_preflight.py`
- `p_formulation_v4_corpus_audit.py`
- same frozen 16-asset FIT panel: `P_FORMULATION_V3_PANEL_V1.json`

**TRAINING FORBIDDEN.** The latest workflow must release only an optimizer-zero P-V4 closure artifact. Do not reuse the CI202 learner prereg/checkpoint and do not open TUNE/CAL/DEV/EXTERNAL during formulation closure.

## Representation gate remains CLOSED/PASS

CI104 label remains `P_GEOMETRY_SUFFICIENT`: exact legal P is sufficient for same-locus correspondence on the frozen representation panel. The current work changes extractor parameterization, not P ontology.

P remains the canonical/object-frame position of the observed physical surface locus.

N remains observation-local orientation supervision/diagnostic only and is forbidden from correspondence admission/ranking and checkpoint selection.

## CI202 — historical raw-XYZ learner diagnostic

CI202/V4 completed cleanly under the old free 3-channel XYZ P head. Selected epoch16 / optimizer step512.

Key FIT_SELECT metrics:

- P p95 `0.344555` vs frozen `<=0.005`;
- Zc@8 `0.789931`;
- oracle Zf p95 `35.947` native px.

TUNE P p95 `0.413841` and Zc@8 `0.762019` showed that the primary failure was absolute extraction precision, not a large generalization gap.

The CI202 checkpoint is historical and incompatible with later P formulations.

## P-V3 formulation diagnosis

Post-CI202 audit showed:

- R/2=128 P-field discretization at the 256 mini is not an adequate explanation for P p95 ~0.34; a real native1024 oracle-style 128 field gave ~0.0017 p95;
- triangle+barycentric -> canonical P -> raster projection is correct to near numerical precision on audited real data;
- canonical geometry is centered/unit-scaled;
- the free XYZ head was unnecessarily asked to rediscover two coordinates already fixed by orthographic observation geometry.

P-V3 therefore changed to one learned camera-forward depth scalar plus analytic screen-plane reconstruction.

## CI237 V1 — preserved apparatus false reject

CI237 G1 emitted `P_V3_CORPUS_GEOMETRY_CLOSURE_FAIL`, but all 16 assets were rejected before geometry metrics opened because V1 incorrectly required the master NPZ field set to equal exactly `{vertices,faces}`.

Canonical classification:

`APPARATUS_FALSE_REJECT__MASTER_NPZ_SUPERSET_MISTAKEN_FOR_CONSUMPTION_LEAK`

The result root is preserved and must not be overwritten.

## CI244 V2 — valid P-V3 real-corpus result

CI244 corrected only the source-container firewall: master NPZ may be a superset, while the G1 consumer loads values only from `vertices` and `faces`. Hidden object-dtype tripwire regression passed.

Authority:

- code-bearing head `985997ba87faa92cc004ad9ab70efeacc400886f`;
- Actions run #244 / ID `32746353085`;
- artifact ID `9527242584`;
- ZIP SHA-256 `e04bc7c9638038688304b558cb816f8bccaf5e253bc2195977c0c2328f693667`.

Valid G1 result: `P_V3_CORPUS_GEOMETRY_CLOSURE_FAIL`, **fatal 2/16 only**.

Across all 16 assets:

- canonical bbox-center infinity norm max `2.98e-08`;
- largest bbox extent min/median/max `0.99999994 / 1.0 / 1.0`;
- max absolute canonical coordinate `0.5`;
- raster-P projection p95 across-asset max `6.72e-08`;
- projection fraction >1e-3 max `0.00024414`;
- exact-depth analytic reconstruction p95 across-asset max `4.00e-08`.

Thus canonical gauge, raster/triangle/barycentric P authority, yaw/basis convention and the analytic P decomposition are healthy.

### The only falsified V3 assumption

14/16 assets use `half_extent=0.54` and pass every V3 gate.

Two assets are internally self-consistent but use a different constant acquisition scale across all eight views:

- `asset_00aa1b666ba193851a498194`: `0.5570941257476807`;
- `asset_76313e4bd82b82fcd1659c70`: `0.6172158837318421`.

Their projection and exact-depth reconstruction remain near numerical zero. Therefore **hardcoded 0.54 is falsified; P geometry is not**.

Canonical interpretation:

`P_FORMULATION_V3_CANONICAL_INTERPRETATION_CI244_20260824.md`

## P-V4 — image-derived observable sheet scale

Do not feed `camera.json half_extent` into IRIS. Product IRIS remains image-only.

Canonical largest bbox extent is one. With ordered views:

- V0 horizontal occupancy exposes X extent;
- V2 horizontal occupancy exposes Y extent;
- image height exposes Z extent.

For RGBA alpha foreground:

```text
w0 = V0 bbox width / R
w2 = V2 bbox width / R
hz = max bbox height over V0..V7 / R
m = max(w0,w2,hz)
h_sheet = 1/(2*m)
```

P-V4 reconstructs:

```text
P = h_sheet*gx*right(yaw)
  - h_sheet*gy*up
  + depth*forward(yaw)
```

Only depth is learned. The depth head receives `(gx,gy,sin(yaw),cos(yaw),h_sheet)`. `h_sheet` is deterministic image-derived gauge, not a learned latent and not camera metadata.

Manual alpha checks on the two CI244 fatal assets produced scale errors versus diagnostic camera metadata of about `0.00186` and `0.00114`, motivating but not closing V4.

## CURRENT GATE — P-V4 optimizer-zero closure

Frozen prereg: `P_FORMULATION_V4_OBSERVABLE_SCALE_CLOSURE_PREREG_20260824.md`.

G0 requires synthetic image-only observable-scale closure at 256/512/1024 and representative scales near 0.54/0.557/0.617, finite full loss/backward and nonzero depth-head gradient.

G1 uses the **same frozen 16 FIT-only assets**. No substitution. Camera metadata is teacher-side diagnostic only. For each asset, both styles and resolutions 256/512/1024 must satisfy reconstructed 3D P p95 `<=0.005` when the screen-plane scale comes only from alpha and exact teacher depth is used solely as an optimizer-zero ceiling.

Allowed G1 labels:

- `P_V4_OBSERVABLE_SCALE_GEOMETRY_CLOSED`
- `P_V4_OBSERVABLE_SCALE_GEOMETRY_FAIL`

Only after G0+G1 PASS may a separate FIT-only P-V4 depth overfit prereg be frozen. No learner optimizer step is currently authorized.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
