# IRIS Single-Pose V2 — P Formulation V5 Native-Scale-Once Closure Preregistration

Status: `FROZEN_PRE_RESULT__OPTIMIZER_ZERO`
Date: 2026-08-24

## Why V5 exists

CI269/V4 returned the frozen scientific label:

`P_V4_OBSERVABLE_SCALE_GEOMETRY_FAIL`.

The failure did **not** falsify canonical P geometry or image-derived scale at native product resolution.

On the same frozen 16-asset FIT sentinel panel, native1024 scale estimation passed the existing P p95 <=0.005 gate for all 16 assets x both styles. Across those 32 native cells, max P p95 was `0.0013928374974057078`.

V4 failed because it re-estimated sheet scale independently after spatial resize. On thin antialiased support, hard `alpha>=0.5` bbox support is not resolution-stable. `asset_00aa...` showed this strongly: native1024 h≈0.559 and P p95≈0.00139, while the exact 512 derivative yielded h≈0.723 and P p95≈0.12449 when scale was re-estimated after resize.

The 512 derivative itself was verified to be a correct corpus downsample, not render corruption.

No post-result alpha-threshold tuning is permitted.

## P ontology remains unchanged

`P` is canonical/object-frame position of the observed physical surface locus.

The geometric decomposition remains:

```text
P(g,theta) = h_sheet*gx*right(theta)
           - h_sheet*gy*up
           + d(g,theta)*forward(theta)
```

Only `d=P·forward` is learned.

## V5 execution boundary

The overall extractor remains image-only.

V5 changes **when** the deterministic observable gauge is measured:

```text
original ordered native1024 RGBA
    -> estimate h_sheet ONCE
    -> freeze/transport h_sheet
    -> resize image for learner as needed
    -> predict depth
    -> reconstruct analytic P with the transported h_sheet
```

Forbidden:

```text
resize -> re-estimate h_sheet from degraded alpha
```

Teacher `camera.json half_extent` is never a model/extractor input.

The scalar supplied to the learner is legal only if produced from the original native RGBA by the frozen image-derived gauge function. It is deterministic preprocessing evidence, not a learned latent.

## Frozen native observable gauge

The native estimator is unchanged from V4 and is not retuned post-result.

At native resolution R=1024 with alpha threshold >=0.5:

```text
w0 = V0 foreground bbox width
w2 = V2 foreground bbox width
hz = maximum foreground bbox height over V0..V7
m = max(w0/R, w2/R, hz/R)
h_sheet = 1/(2*m)
```

No alternative alpha threshold may be chosen from CI269 outcomes.

## G0 — executable pipeline closure

Optimizer steps: **0**.

Required PASS:

1. native scale estimator accepts only ordered 8-view 1024 RGBA and canonical yaw;
2. learner model accepts `(resized_images, yaw_deg, sheet_half_extent)`;
3. learner model has no camera-half-extent argument and no camera metadata dependency;
4. the transported scalar must be finite, positive, detached/non-learned, shape `[B]`;
5. the same native-derived scalar is used unchanged for learner inputs at 1024, 512 and 256;
6. model P remains R/2 and Zc remains R/8 at all three resolutions;
7. reconstructed P satisfies analytic screen-plane identities with the transported scalar;
8. full loss forward/backward is finite;
9. P depth head receives finite nonzero gradient;
10. optimizer steps remain zero.

## G1 — real-corpus native-scale-once closure

Optimizer steps: **0**.

Population remains exactly `P_FORMULATION_V3_PANEL_V1.json`:

- 8 FIT_SELECT sentinels;
- 8 FIT_TRAIN sentinels;
- no substitutions;
- no TUNE;
- no CAL/DEV/EXTERNAL.

Legal consumed information:

- native1024 RGBA alpha, solely to derive h_sheet once;
- 512 corpus derivative and exact 512->256 learner transform only to verify transport/preprocessing identity; h_sheet is **not recomputed** from them;
- `primary_geometry.npz` values `vertices` and `faces` only for teacher-side optimizer-zero measurement;
- `raster_authority.npz` for teacher-side measurement;
- `camera.json` only as diagnostic/ground-truth audit, never model/extractor input.

### Frozen geometry/camera gates

Keep CI269 gates unchanged:

- canonical center infinity norm <=0.01;
- largest bbox extent in [0.98,1.02];
- max absolute canonical coordinate <=0.55;
- camera contract/yaw/basis exact within existing tolerances;
- camera half_extent finite, positive, constant across views <=1e-6;
- raster authority resolution=1024;
- teacher raster-P projection p95 <=1e-4;
- fraction projection error >1e-3 <=0.005.

### Frozen native-scale-once P gate

For each asset and style:

1. derive `h_native` from native1024 RGBA alpha using the unchanged V4 estimator;
2. do **not** derive another h from 512 or 256;
3. transport the exact same `h_native` to the 1024/512/256 reconstruction conditions;
4. using exact teacher depth only for this optimizer-zero formulation ceiling, reconstructed canonical P p95 must be <= **0.005** at all three learner-resolution conditions.

The p95 threshold is unchanged. No new scale-specific success threshold replaces it.

Required transport invariant:

`h_used_R1024 == h_used_R512 == h_used_R256` exactly in stored float representation.

### Allowed G1 labels

- `P_V5_NATIVE_SCALE_ONCE_GEOMETRY_CLOSED`
- `P_V5_NATIVE_SCALE_ONCE_GEOMETRY_FAIL`

## G2 — only after G0 + G1 PASS

Training remains forbidden until a separate FIT-only depth/P overfit preregistration is frozen.

That later experiment may test whether learned depth reaches absolute P precision, but it may not reopen TUNE for formulation debugging.

## Non-claims

V5 G0/G1 do not establish learned depth extractability, generalization, correspondence promotion, SurfaceBuilder sufficiency, Geppetto sufficiency, or product-domain robustness. They only establish the corrected native-image canonicalization boundary before another optimizer step.
