# Geppetto/Arachne V0.1 — Character/Render Usability Policy Freeze V1

**Date:** 2026-08-31  
**Status:** `OBJECTIVE_RENDER_POLICY_FROZEN__IMAGE_AND_SEMANTIC_GATE_NEXT__OPTIMIZER_NOT_AUTHORIZED`

This policy consumes only the sealed input-quality result `CHARACTER_RENDER_PURITY_AUDIT_RESULT_V1_2.json` and the already frozen structural C0 membership. No Geppetto, Arachne, IRIS, Compiler prediction, optimizer output, TUNE/CAL/DEV, or EXTERNAL_HOLDOUT evidence is used.

## Authority

- purity audit status: `PASS_MEASUREMENT_COMPLETE__QUALITY_AND_SEMANTIC_FREEZE_NEXT`
- purity result SHA-256: `f71e2fb38b793b9110d4afe34701db1dfa171ff784b026fccf3ce394cfd18b0b`
- structural membership SHA-256: `cd3f5d14dbbfae0996cdade124af209da478cd36c2a0b3198a9a93fda5979147`
- measured Geppetto structural C0: `2897`
- measured nested Arachne structural C0: `2527`
- hard render/geometry authority failures in measurement: `0`

V1.2 measured native `raster_authority.npz` at 1024x1024 but intentionally did not decode PNG pixels. Therefore this policy freezes only objective geometry/raster usability; native image-content integrity remains a separate next gate.

## Observed tails

- mesh max-face area fraction p50 / p95 / p99 / max: `0.0031363 / 0.0213841 / 0.0688039 / 0.4985333`
- per-asset max visible-triangle pixel fraction p50 / p95 / p99 / max: `0.0163102 / 0.0973024 / 0.2070608 / 0.5007927`
- any-view frame touch: `0 / 2897`
- assets with minimum any-view occupancy `< 0.005`: `23 / 2897`

The smallest real extreme examples contain only tens to hundreds of authoritative raster pixels and are not a usable native-1024 character substrate. The 0.005 threshold is a **clean-C0 profile boundary**, not a claim that the underlying source asset is intrinsically invalid.

## Frozen objective hard exclusions

An otherwise structurally admitted asset is outside clean C0 if either condition is true:

1. `min_view_occupancy_fraction < 0.005`; or
2. `border_touch_view_count > 0`.

Policy name: `OBJECTIVE_RENDER_C0_V1`.

Current result under the sealed V1.2 measurement:

- objective hard exclusions: `23`
- nested Arachne objective hard exclusions: `18`
- objective-pass Geppetto candidates before image/semantic gate: `2874`
- objective-pass Arachne candidates before image/semantic gate: `2509`
- hard-exclusion asset-set SHA-256: `40f2bbcac14109c3857db2d34e4419c27e8e83bb24e814b059af13e59d6087ad`
- objective-pass asset-set SHA-256: `60096fdf837b9cf277d1e3493a956796faf173d1af8ba7f593009423d77fa78b`

No clipping, rerendering, rescaling, crop repair, or source mutation is permitted to turn a hard exclusion into clean C0.

## Frozen semantic-review priority flags — diagnostic only

The following conditions create review-priority flags but **must not exclude an asset by themselves**:

- `mesh.max_face_area_fraction >= 0.20` -> `MESH_FACE_DOMINANCE`
- `max_view.max_visible_triangle_pixel_fraction >= 0.30` -> `VISIBLE_TRIANGLE_DOMINANCE`
- `max_view.significant_component_count > 1` -> `MULTI_COMPONENT`
- `min_view.largest_component_fraction < 0.90` -> `FRAGMENTED_SUPPORT`
- `max_view.centroid_offset_fraction > 0.25` -> `OFF_CENTER`

The union currently flags `301 / 2897` assets (`197` nested Arachne); review-priority set SHA-256 is `4de5424421fb129540d1ed9a99f3c10979de72321766e8b4eaafe5c73a3e3460`.

These flags prioritize semantic inspection because valid stylized characters can legitimately contain large faces, multiple silhouette islands, accessories, wings, tails, disconnected costume pieces, or off-center geometry. They are not labels.

## Explicit non-exclusion channels

`visible_fully_unskinned_pixel_fraction` and related skin-visibility quantities remain diagnostic-only exactly as preregistered. They may not independently exclude a Geppetto asset or become an inference feature.

## Next gate

Before any Geppetto/Arachne optimizer output is opened:

1. execute the separately preregistered native-image integrity measurement on the `2874` objective-pass assets;
2. freeze image-integrity admission only after that input-only measurement is sealed;
3. execute the separately preregistered `single riggable character` semantic review on every remaining asset, not merely the 301 priority flags;
4. intersect structural, objective-render, image-integrity, and semantic decisions into final clean C0 membership;
5. only then build Geppetto zero-step apparatus.

**Training remains unauthorized.**
