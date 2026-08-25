# IRIS Single-Pose V2 — P Formulation V4 Observable-Scale Closure Preregistration

Status: `FROZEN_PRE_RESULT__OPTIMIZER_ZERO`
Date: 2026-08-24

## Why V4 exists

CI244/V2 opened the corrected real-corpus P-V3 geometry measurement on the frozen 16-asset FIT sentinel panel.

The result was structurally clean except for one falsified assumption:

- canonical geometry gauge passed on all 16 assets;
- raster-P projection p95 was near numerical zero on all 16;
- exact-depth analytic P reconstruction using each asset's actual camera scale was near numerical zero on all 16;
- 14/16 assets used camera `half_extent=0.54`;
- `asset_00aa1b666ba193851a498194` used `0.5570941257476807` on all views;
- `asset_76313e4bd82b82fcd1659c70` used `0.6172158837318421` on all views.

Thus the P-V3 idea “two screen-plane coordinates are analytic, only depth is learned” survives, but the **hardcoded 0.54 acquisition scale does not**.

Camera metadata may not be added as a model input: product IRIS remains image-only.

## P ontology remains unchanged

`P` is still the canonical/object-frame position of the physical surface locus.

Canonical gauge remains:

- bbox centered at origin;
- largest x/y/z bbox extent = 1;
- ordered orthographic views with yaw 0,45,...315 and camera up = +Z.

## Observable sheet-scale gauge

For an ordered 8-view RGBA sheet, canonical largest bbox extent is exactly 1 by definition.

- yaw 0 horizontal occupancy exposes canonical X extent;
- yaw 90 horizontal occupancy exposes canonical Y extent;
- image height exposes canonical Z extent in every view.

At image resolution R, with foreground alpha threshold >=0.5, define inclusive foreground bbox spans in pixels:

```text
w0 = bbox width in V0
w2 = bbox width in V2
hz = max bbox height over V0..V7
m  = max(w0/R, w2/R, hz/R)
h_sheet = 1 / (2*m)
```

`h_sheet` is therefore a deterministic **image-derived canonical gauge**, not teacher camera metadata and not a learned latent.

No `camera.json half_extent` is consumed by the model at inference.

## P-V4 extractor parameterization

For normalized observation coordinate `g=(gx,gy)` and known yaw:

```text
P(g,theta) = h_sheet*gx*right(theta)
           - h_sheet*gy*up
           + d(g,theta)*forward(theta)
```

where only `d=P·forward` is learned.

The depth head is conditioned on:

- local decoded feature;
- gx, gy;
- sin(yaw), cos(yaw);
- image-derived `h_sheet`.

No hard depth clip is introduced.

N/U/Zc/Zf authority roles remain unchanged.

## G0 — synthetic executable closure

Optimizer steps: **0**.

Required PASS:

1. alpha-only `h_sheet` estimator works at 256/512/1024;
2. at least three sheet scales are exercised, including values close to 0.54, 0.557 and 0.617;
3. no camera half-extent argument is accepted by the P-V4 model forward path;
4. reconstructed P satisfies screen-plane identities using the image-derived scale;
5. P field remains R/2;
6. full loss forward/backward finite;
7. P depth head receives finite non-zero gradient;
8. existing matcher role separation and AMP numeric boundaries remain intact.

## G1 — real-corpus observable-scale closure

Optimizer steps: **0**.

Panel remains exactly `P_FORMULATION_V3_PANEL_V1.json`:

- 8 frozen FIT_SELECT sentinels;
- 8 frozen FIT_TRAIN sentinels;
- 0 TUNE;
- 0 CAL/DEV/EXTERNAL.

No asset substitution is allowed.

Legal consumed information:

- `primary_geometry.npz`: values of `vertices` and `faces` only, for teacher-side measurement;
- `raster_authority.npz`, for teacher-side measurement;
- `camera.json`, **diagnostic/ground-truth audit only**, never model input;
- RGBA alpha from `cel_clean` and `ink_cel` at 1024 and 512, plus the exact staged 512->256 bilinear transform.

RGB color values are not used by this closure measurement; alpha only.

### Frozen camera/gauge gates

For every asset:

- canonical center infinity norm <=0.01;
- largest bbox extent in [0.98,1.02];
- max absolute canonical coordinate <=0.55;
- camera contract and yaw/right/up/forward conventions exact within existing tolerances;
- camera half_extent must be finite, positive, and constant across the eight views to <=1e-6; it is **not required to equal 0.54**;
- raster authority resolution =1024;
- teacher raster-P projection p95 <=1e-4 and sampled fraction >1e-3 <=0.005.

### Frozen observable-scale gates

For each style (`cel_clean`,`ink_cel`) and resolution (256,512,1024):

- alpha sheet-scale estimate must be finite and positive;
- using that image-derived `h_sheet` plus **exact teacher depth only for this optimizer-zero ceiling measurement**, reconstructed 3D canonical P must have p95 Euclidean error <= **0.005** on sampled raster loci.

This directly binds the observable scale estimator to the existing P precision target. No looser scale-specific success bar may replace the P-error gate after results open.

Camera half_extent error and cross-resolution/cross-style scale differences are reported diagnostically but do not replace the 3D P p95 gate.

### Allowed G1 labels

- `P_V4_OBSERVABLE_SCALE_GEOMETRY_CLOSED`
- `P_V4_OBSERVABLE_SCALE_GEOMETRY_FAIL`

## G2 — only after G0 + G1 PASS

Training remains forbidden until a separate FIT-only overfit/optimization prereg is frozen for P-V4.

The next question will be whether the one-scalar depth learner, with deterministic observable sheet gauge, can drive reconstructed P to the existing absolute precision target on FIT without mixing in generalization.

TUNE must not be reopened to debug this formulation.

## Non-claims

G0/G1 do not prove learned depth extractability, generalization, Zc/Zf promotion, SurfaceBuilder sufficiency, Geppetto sufficiency, or final product-domain robustness. They only establish that image-derived canonical scale + exact screen-plane geometry is a valid observable parameterization before another optimizer step.
