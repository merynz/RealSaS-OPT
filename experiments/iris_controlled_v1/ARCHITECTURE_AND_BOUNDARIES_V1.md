# IRIS Controlled V1 — Architecture and Boundaries

## Mission

Under the strongest controlled conditions, recover a **persistent observable surface explanation** from one neutral pose observed in eight ordered views.

IRIS must answer: *what physical surface is observed, where is it across views, and how certain is that statement?* It must not answer: *which authored bone/owner/mechanical slot is this?*

## Deterministic frontend

- validates V0..V7 presence/order;
- preserves RGBA support;
- provides known controlled yaw as legal inference metadata;
- applies deterministic resolution conversion;
- defines legal geometry-constrained cross-view search structure.

No handcrafted joint/part segmentation or mechanical labels are introduced.

## Neural core

### SharedImageEncoder
One weight-sharing encoder processes all eight views and yields f2/f4/f8/f16 features.

### AxialWithinViewReasoning
At f16, row and column attention add within-image context before cross-view fusion.

### RowWiseMultiViewFusion
For controlled level orthographic Z-orbit cameras, physical loci preserve screen vertical coordinate. Cross-view attention therefore spans `(view, horizontal position)` within a row rather than arbitrary global 2-D attention. Known yaw is encoded with sin/cos features.

### Dense decoder and heads

- **P**: common-frame canonical surface position, direct learned core;
- **N**: geometric surface normal, direct learned current-best estimate;
- **U**: predictive risk/log-scale; geometry error is detached when training U so U cannot improve P by changing P;
- **Z_coarse**: high-recall persistence descriptor at coarse scale;
- **Z_fine**: local precision descriptor at fine scale.

There is intentionally no learned mechanical-node identity, owner, parent or skinning head.

## Deterministic evidence layer

- geometry-constrained candidate domain;
- coarse top-k retention, not early global singleton;
- reciprocal consistency;
- multi-view cycle consistency;
- local fine reranking only inside retained candidates;
- set-valued ambiguity when margin is insufficient;
- support derived from actual multi-view track support;
- robust common-frame geometry fusion;
- reprojection/geometric falsification;
- provenance retention.

A learned D3-style local matcher remains **reserve-only** if a reproducible hard tail survives the above system.

## Training truth

No hidden mechanical truth is forwarded. Exact legal authority is:

`raster pixel -> triangle + barycentric -> continuous physical surface locus P`.

Cross-view positives are defined by physical-locus equality/near-equality; negatives are distance-aware. Geometric N is derived from exact canonical vertices/faces. Visibility/support comes from actual raster observation, not a guessed semantic visibility label.

## Product-robustness roadmap (not Controlled V1 scope)

Controlled V1 freezes known exact cameras. After a controlled PASS, robustness should be expanded progressively:

1. jittered but known camera;
2. jittered hidden camera + camera estimator/uncertainty;
3. soft geometry corridors under camera uncertainty;
4. controlled artist/view-specific residuals `P_v = P* + Delta_v`;
5. real artist sheets.

This progression keeps the current core useful: hard geometry constraints become uncertainty-aware soft constraints rather than being discarded.
