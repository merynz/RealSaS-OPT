# Geppetto/Arachne V0.1 — Character / Render Purity Audit Prereg Amendment V1.1

**Date:** 2026-08-31  
**Status:** `FROZEN_AFTER_RUNTIME_PREFLIGHT__BEFORE_REAL_FIT_MEASUREMENTS`  
**Supersedes:** only the image-content decoding clause of `CHARACTER_RENDER_PURITY_AUDIT_PREREG_V1.md`.

## Why this amendment exists

A schema-compatible 2897-ID runtime preflight was executed before any real FIT character/render measurement was opened. The original V1 plan redundantly decoded all eight `cel_clean_512.png` files for every asset solely to compare image foreground with canonical raster foreground. That would duplicate image decoding that is already required by the next semantic character/render-integrity pass and makes Google Drive I/O the dominant cost.

No real FIT purity result, threshold, semantic label, Geppetto output, Arachne output, or model metric was inspected before this amendment.

## What remains unchanged

The complete structural Geppetto C0 population (`2897` assets; membership SHA-256 `cd3f5d14dbbfae0996cdade124af209da478cd36c2a0b3198a9a93fda5979147`) is still measured over all eight canonical views.

All V1 geometry/raster measurements remain frozen:

- mesh face/surface dominance;
- raster foreground occupancy and bounding box;
- frame margins / border touch;
- foreground centroid offset;
- 8-connected components and largest-component fraction;
- visible-triangle count and triangle pixel dominance;
- diagnostic visible skin support;
- exact controlled yaw sequence.

`cel_clean_512.png` must still exist for every measured view. Missing images remain a hard authority failure.

## Amended image-content rule

V1.1 does **not decode image pixels** during the objective raster/geometry pass. It records:

- `cel_clean_512_exists = true` for each passing view;
- `image_content_decoded = false` in authority metadata.

The former raster/image foreground-IoU measurement is removed from this objective pass before any real FIT result exists.

## Next image gate

The immediately following image gate must jointly evaluate:

1. semantic `single riggable character` purity;
2. blank/corrupt/material-broken render integrity;
3. obvious unrelated scene/background/prop content;
4. agreement of visible character content across the eight views.

This avoids decoding the same 23,176 controlled PNGs twice while preserving the scientific separation between objective geometry/raster quality and semantic/image-content quality.

No training is authorized by V1.1.
