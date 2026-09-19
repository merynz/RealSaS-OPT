# RealSaS — Rest Source Preservation Calibration V1 — 2026-09-19

**Status:** `FROZEN_BEFORE_SUBJECT_FREE_CALIBRATION_RESULT`  
**Scope:** Stage 32 rest/source-preservation admission policy only.  
**Forbidden inputs:** Knight source images, Knight rest renders, Mage images/renders/metrics, fitted-family percentile distributions, teacher/source mesh topology, human visual threshold tuning after result inspection.

## Question

What subject-independent silhouette-edge tolerance is sufficient to distinguish an exact rest replay from a one-reference-pixel geometric displacement under the frozen Stage-31 raster/appearance contract, while keeping appearance provenance and source color preservation exact?

This study does not prove Knight compliance, professional animation quality, unseen generalization, or motion quality.

## Frozen product invariants

The following are not calibrated from this sweep:

1. **Visible rest appearance must be DIRECT_SOURCE.**  
   `cross_view_source_geometry_fraction == 0`.
2. **Source color replay on source/render foreground overlap is exact.**  
   `overlap_rgba_mismatch_pixel_count == 0` and `overlap_rgba_max_abs_channel_error_u8 == 0`.
3. **Alpha/coverage admission inherits the already-frozen MESH G5 floors** from
   `canonical/QUALIFIED_MESH_PRODUCT_POLICY_V1_20260918.json`:
   - recall >= 0.97
   - precision >= 0.995
   - largest coherent hole fraction <= 0.005
   - interior uncovered fraction <= 0.005

These rules may later be tightened prospectively, but a Knight failure may not relax them.

## Exact synthetic family

No external image, mesh, checkpoint, run artifact, or subject data file is read by the calibration apparatus.

Procedural RGBA witnesses are generated at resolutions 64 and 128 using:
- centered rectangle;
- filled disk;
- concave L-shape;
- stepped asymmetric polygon-like raster shape.

Foreground RGB is a deterministic coordinate-dependent pattern so a spatial shift also changes appearance values. Background RGBA is zero.

Each witness is replicated across the exact eight-view measurement interface only to exercise production cardinality and aggregation semantics; no view-specific subject data exists.

## Frozen controls

### Positive control
- `IDENTITY`: rendered RGBA equals source RGBA byte-for-byte, foreground mask identical, all geometry-visible pixels marked DIRECT_SOURCE.

### Negative controls
- `SHIFT_X_POS_1`, `SHIFT_X_NEG_1`, `SHIFT_Y_POS_1`, `SHIFT_Y_NEG_1`: rendered raster translated exactly one reference pixel with wrapped pixels cleared.
- `COHERENT_HOLE`: a deterministic interior square foreground region is removed.
- `RGB_ONE_LSB`: exactly one foreground pixel has one RGB channel changed by one unsigned 8-bit level, alpha unchanged.
- `VISIBLE_CROSS_VIEW_DONOR`: pixels are visually identical but all geometry-visible pixels are marked cross-view rather than direct-source.

## Metrics

Use the shipping `RealSaS.RestPreservationMetricContract.v1` implementation:
- alpha recall / precision / IoU;
- largest coherent hole fraction;
- interior uncovered fraction;
- symmetric silhouette-edge mean / p95 / max distance in reference pixels;
- premultiplied RGB MAE / p95;
- alpha MAE;
- exact overlap RGBA mismatch count/fraction/max channel error;
- direct-source and cross-view geometry-visible fractions.

No LPIPS, SSIM, learned perceptual metric, or subject-specific feature extractor is used.

## Frozen silhouette selection rule

Let:
- `P` = maximum silhouette-edge p95 over every `IDENTITY` witness/view.
- `N` = minimum silhouette-edge p95 over every one-pixel translation witness/view.

The calibration is valid only if `P < N`.

If valid, the Stage-32 silhouette p95 ceiling is:

`T = (P + N) / 2`.

No rounding toward a subject result is permitted. The exact float emitted by the sweep is the policy candidate.

If `P >= N`, calibration status is `FAIL_NO_SEPARATING_SILHOUETTE_POLICY` and Stage 32 remains unbound.

## Frozen sensitivity requirements

In addition to `P < N`:

- every IDENTITY case must have zero overlap RGBA mismatch and zero visible cross-view donor fraction;
- every RGB_ONE_LSB case must produce at least one overlap RGBA mismatch and max absolute channel error >= 1;
- every VISIBLE_CROSS_VIEW_DONOR case must produce cross-view geometry fraction > 0;
- every COHERENT_HOLE case must produce a nonzero coherent-hole measurement;
- the calibration code may write only its optional JSON result path and may not open/read any external input data artifact.

Failure of any sensitivity check invalidates calibration; thresholds are not repaired by inspecting Knight.

## Policy materialization rule

This preregistration and the implementation commit must exist before the calibration result is inspected.

Only after a valid result may a separate immutable
`canonical/REST_SOURCE_PRESERVATION_POLICY_V1_20260919.json`
be materialized from:
- the inherited frozen G5 MESH floors;
- exact-zero direct-source/RGBA rules above;
- the exact selected `T` from the calibration result.

That policy still does not claim Knight PASS. Stage 32 may be bound only after the result and policy are sealed.
