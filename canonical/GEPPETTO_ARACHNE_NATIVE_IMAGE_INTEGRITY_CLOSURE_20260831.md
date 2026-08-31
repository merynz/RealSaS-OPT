# Geppetto/Arachne — Native Image Integrity Measurement Closure V1

**Date:** 2026-08-31  
**Status:** `PHASE_I_CLOSED__EXACT_IMAGE_RASTER_SUPPORT__SEMANTIC_GATE_NEXT__TRAINING_NOT_AUTHORIZED`

## Authority

Drive report: `NATIVE_IMAGE_INTEGRITY_REPORT_V1.md`  
Drive file id: `1h1XURTwT5ku3bLHjDMRWJrT5ZYazqWwo`

The measurement consumed the previously frozen objective-pass population from `OBJECTIVE_RENDER_C0_V1`.

## Result

- objective-pass expected: `2874`
- measured assets: `2874`
- hard authority failures: `0`
- native views measured: `22992`
- exact image/raster support equality: `22992 / 22992`
- IoU min/p01/p50/p99/max: `1.0 / 1.0 / 1.0 / 1.0 / 1.0`
- image-unsupported fraction min/max: `0.0 / 0.0`
- raster-missing fraction min/max: `0.0 / 0.0`

## Phase-I policy freeze

No empirical quality threshold is required or invented after the measurement.

For this corpus generation, Phase-I native integrity admission is exact and deterministic:

`IMAGE_INTEGRITY_PASS := all 8 views satisfy the frozen native-image/raster contract and exact_support_equal == true`.

Under the completed V1 measurement, all `2874` objective-pass Geppetto assets satisfy this condition. The nested Arachne population remains governed by its already frozen structural/objective membership; no target rewriting or transport is introduced here.

This result proves only native image/raster support identity and camera/file authority integrity. It does **not** prove that the depicted content is one clean riggable character.

## Semantic firewall

Phase II remains unopened as final authority:

- `PASS_SINGLE_RIGGABLE_CHARACTER`
- `FAIL_PROP_ENVIRONMENT_OR_UNRELATED_SCENE`
- `REVIEW_AMBIGUOUS`

`REVIEW_AMBIGUOUS` remains fail-closed for clean C0.

A random/blind visual sanity sample of automatic audit outputs is still required before final clean-C0 membership is trusted. This sanity sample is not allowed to silently alter the frozen Phase-I rule; any discovered apparatus error requires an explicit correction/version.

## Training authority

Geppetto/Arachne clean-C0 training remains **NOT AUTHORIZED** until semantic review and final membership closure are complete.

This corpus result is a separate H0 workstream and may not retroactively reinterpret the sealed DINO S1 experiment.
