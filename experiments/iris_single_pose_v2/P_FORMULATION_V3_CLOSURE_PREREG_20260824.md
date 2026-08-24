# IRIS Single-Pose V2 — P Formulation V3 Closure Preregistration

Status: `FROZEN_PRE_RESULT__OPTIMIZER_ZERO`
Date: 2026-08-24

## Why this gate exists

CI202 established that the V2 learner extracted useful signal but failed absolute P precision badly: FIT_SELECT P p95 = 0.344555 and TUNE_FINAL P p95 = 0.413841 under the raw free-XYZ P head. The representation-authority result `P_GEOMETRY_SUFFICIENT` remains valid because it asked whether exact observable P discriminates persistent physical loci; it did not prove that `image -> P` had a well-conditioned neural parameterization.

Post-CI202 audit found that the controlled camera already determines two of P's three coordinates analytically. Requiring a convolutional head to rediscover full absolute XYZ is therefore unnecessary and poorly conditioned.

## Frozen P ontology

The latent/observable object is unchanged: P is the canonical/object-frame position of the physical surface locus seen at an observation.

This prereg changes only the neural parameterization used to extract P.

Controlled camera contract:

- `realsas.level_orthographic_z_orbit.v1`
- orthographic half extent `h_e = 0.54`
- yaw `theta in {0,45,...,315}` degrees
- `right(theta) = (cos theta, -sin theta, 0)`
- `forward(theta) = (sin theta, cos theta, 0)`
- `up = (0,0,1)`
- image grid uses `align_corners=False`, top-left origin, image y down, NDC y up.

For normalized observation coordinate `g=(g_x,g_y)`:

```text
P · right   = h_e * g_x
P · up      = -h_e * g_y
P · forward = d
```

Therefore V3 reconstructs

```text
P(g,theta) = h_e*g_x*right(theta)
           - h_e*g_y*up
           + d(g,theta)*forward(theta)
```

where **only `d = P · forward` is learned**.

## Frozen neural change

At the existing f2 dense field:

- remove the free 3-channel raw-XYZ P head;
- predict one camera-forward depth scalar per f2 location;
- explicitly condition the depth head on normalized `(g_x,g_y)` and `(sin theta, cos theta)`;
- reconstruct the public 3-channel P field deterministically using the equation above;
- do not add a hard depth clip in this closure gate.

All other evidence roles remain unchanged:

- N = observation-local orientation auxiliary target only;
- U_geo = detached P-error risk;
- Z_coarse = global/high-recall persistence descriptor;
- Z_fine = local-only refinement descriptor;
- N remains forbidden from correspondence admission/ranking and checkpoint selection.

## Frozen P loss semantics

The learned P objective is now SmoothL1 on the scalar camera-forward depth `d`; the two analytic screen-plane coordinates are not averaged into the learned loss.

Evaluation authority remains the reconstructed **3D canonical P**. P success/failure continues to be measured with object-space Euclidean P error, including p95. No threshold is relaxed by this reformulation.

## G0 — executable synthetic formulation closure

Optimizer steps: **0**.

Required PASS conditions:

1. exact camera basis for yaw 0 and 45 degrees;
2. analytic P screen-plane reconstruction at input resolutions 256, 512, and 1024;
3. P-field resolutions remain R/2;
4. arbitrary interior bilinear samples preserve `P·right=h_e*g_x` and `P·up=-h_e*g_y` to <= 3e-6 absolute error;
5. full loss forward/backward finite;
6. P depth head receives finite non-zero gradient;
7. existing Zc/Zf matcher role separation remains PASS;
8. AMP boundary remains model-forward only; all supervision/matching loss numerics FP32.

## G1 — real-corpus geometry closure

Optimizer steps: **0**. RGB must not be required.

Panel is frozen in `P_FORMULATION_V3_PANEL_V1.json`: 16 OPEN FIT assets only (8 FIT_TRAIN + 8 FIT_SELECT sentinels). TUNE, CAL, DEV and EXTERNAL_HOLDOUT are forbidden.

For every panel asset, read only:

- `primary_geometry.npz` (`vertices`, `faces` only);
- 8 x `raster_authority.npz`;
- 8 x `camera.json`.

Required asset-level invariants:

- finite canonical vertices;
- canonical max absolute coordinate <= 0.55;
- bbox-center infinity norm <= 0.01;
- largest bbox extent in [0.98, 1.02];
- camera contract exactly `realsas.level_orthographic_z_orbit.v1`;
- half extent = 0.54 within 1e-6;
- expected right/up/forward basis within 1e-5;
- raster authority resolution = 1024;
- reconstructed raster P projection p95 grid error <= 1e-4;
- fraction of sampled raster points with projection error > 1e-3 <= 0.005;
- V3 analytic reconstruction using exact depth must reproduce the reconstructed canonical P with p95 Euclidean error <= 1e-5.

Any fatal asset means `P_V3_CORPUS_GEOMETRY_CLOSURE_FAIL`. No asset replacement is permitted after results open.

If all pass, label `P_V3_FORMULATION_GEOMETRY_CLOSED`.

## G2 — only after G0 + G1 PASS

Training is still forbidden until a separate FIT-only overfit/optimization prereg is frozen.

The next causal question is then:

> Can the one-scalar depth learner fit legal P to the existing absolute precision target without a generalization question being mixed in?

That future test must use FIT only. TUNE must not be reopened merely to debug this reformulation.

## Explicit non-claims

G0/G1 do not prove image extractability, generalization, Zc/Zf promotion, SurfaceBuilder sufficiency, Geppetto sufficiency, or final native-1024 product quality. They only establish that the P target, camera geometry, corpus gauge, and executable neural parameterization agree before another optimizer step is authorized.
