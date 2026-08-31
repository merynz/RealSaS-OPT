# RealSaS — Clean-C0 Visual Anomaly Triage Prereg V1

**Date:** 2026-08-31  
**Status:** `FROZEN_BEFORE_VISUAL_TRIAGE_OUTPUTS__DIAGNOSTIC_ONLY__NO_AUTO_EXCLUSION__TRAINING_NOT_AUTHORIZED`

## Purpose

This gate schedules an image-only semantic sanity review of the objective-render-pass population. It is designed to surface cases such as a small character beside a dominant unrelated plane/card, a large quad occluding the character in one or more views, multiple unrelated dominant subjects, extremely small rendered characters, highly view-imbalanced scenes, and extremely thin visible structures.

This gate does **not** automatically exclude assets. It creates a deterministic priority sample plus a blind random control sample. Final semantic authority remains the frozen labels `PASS_SINGLE_RIGGABLE_CHARACTER`, `FAIL_PROP_ENVIRONMENT_OR_UNRELATED_SCENE`, and `REVIEW_AMBIGUOUS` from `IMAGE_SEMANTIC_CHARACTER_GATE_PREREG_V1.md`.

## Population interlock

Population is exactly `OBJECTIVE_RENDER_C0_V1`:

- expected Geppetto count: `2874`;
- source structural C0 count: `2897`;
- objective hard exclusions: min-view occupancy `<0.005` or any border-touch view;
- native image integrity measurement must already be complete with zero hard authority failures.

No DEV/heldout set may be opened.

## Image authority

Diagnostic metrics use only native `renders/V{v}/cel_clean.png`, exact RGBA 1024x1024, with the frozen foreground rule:

- if any alpha <255: foreground = `alpha >= 128`;
- otherwise foreground = complement of exact green RGB `(0,255,0)`.

Contact sheets use ordered `cel_clean_512.png`, V0..V7, 4x2, no crop or geometric transform, neutral gray padding/label area, and no metric overlays.

`raster_authority.npz` and existing render-audit metrics may be used only for **scheduling diagnostics**. They are forbidden to the reviewer while semantic labels are assigned.

## Frozen diagnostic metrics

Per native view, use 8-connected foreground components.

For each connected component record area, foreground-area fraction, bounding-box area fraction, bounding-box fill ratio, aspect ratio and centroid.

### Dominant rectangle / quad score

For each component:

`rectangle_score = bbox_area_fraction * fill_ratio * (1 + min(log2(max(aspect_ratio,1)),3)/3)`

Asset score is the maximum over its 8 views/components. This is a ranking diagnostic only; no threshold is an exclusion rule.

### Multiple dominant disconnected-subject score

If a second component exists:

`multi_score = second_component_foreground_fraction * centroid_separation_fraction`

where separation is normalized by the image diagonal. Asset score is the maximum over views.

### Tiny-subject diagnostic

Asset value is minimum foreground occupancy over 8 views. Lower ranks earlier.

### View-imbalance diagnostic

`imbalance = max_view_occupancy / max(min_view_occupancy, 1e-12)`.

Higher ranks earlier.

### Off-center / split-scene diagnostic

For each view measure foreground centroid distance from image center normalized by half the image diagonal. Asset value is the maximum over views. Higher ranks earlier.

### Extremely thin-support diagnostic

For each native foreground mask compute EDT, take `2*EDT` at 3x3 local maxima, and record p01 visible-support diameter in native pixels. Asset value is the minimum p01 over views. Lower ranks earlier. This is an observed silhouette proxy, not true 3D thickness.

### Existing raster triangle-dominance diagnostic

Use the already measured `max_visible_triangle_pixel_fraction` only as a scheduling signal. Higher ranks earlier. Reviewers must not see this metric before labeling.

## Frozen review sample

Seed: `20260831`.

From the exact 2874-member population select, before any review label is opened:

- top 8 dominant rectangle/quad score;
- top 8 multiple-disconnected-subject score;
- bottom 8 minimum occupancy;
- top 8 view-imbalance;
- top 8 off-center/split-scene;
- bottom 8 thin-support p01;
- top 8 existing raster triangle dominance;
- 16 deterministic blind-random controls from assets not already selected.

Deduplicate the union, then deterministically shuffle it with the same seed.

The scheduling manifest contains diagnostic reasons and metrics. The blind review manifest contains only slot, canonical asset id and contact-sheet path. The contact sheet itself contains only view labels outside image support.

## Review authority

The reviewer must assign exactly one:

- `PASS_SINGLE_RIGGABLE_CHARACTER`
- `FAIL_PROP_ENVIRONMENT_OR_UNRELATED_SCENE`
- `REVIEW_AMBIGUOUS`

The reviewer may not see source registry, skeleton/bone/skin statistics, diagnostic bucket, metric values, learned RealSaS outputs, or heldout identity while labeling.

Examples explicitly routed to semantic FAIL include:

- character plus a dominant unrelated plane/card/environment component;
- a large unrelated quad/plane substantially occluding or competing with the character;
- multiple unrelated dominant subjects;
- prop/object/vehicle/environment-only imagery.

Extremely small or thin but otherwise legitimate characters are **not automatically semantic FAIL**. They remain separate resolution/coverage diagnostics and may later route to `RESOLUTION_UNSUPPORTED -> UNKNOWN` if the geometry apparatus cannot support them.

## Independent sanity intent

The blind-random controls are mandatory. Priority-tail agreement alone cannot validate the automatic audit. The final review report must separately state false-pass candidates found in random controls and false-alarm rate among diagnostic priority cases.

## Outputs

- `CORPUS_VISUAL_ANOMALY_TRIAGE_V1.json` — full diagnostic/scheduling result;
- `CORPUS_VISUAL_BLIND_REVIEW_INDEX_V1.json` — reviewer-safe index;
- `CORPUS_VISUAL_REVIEW_LABELS_TEMPLATE_V1.csv`;
- `contact_sheets/*.png`;
- `CORPUS_VISUAL_REVIEW_CONTACT_SHEETS_V1.zip`.

All outputs must record source authority paths/hashes where available, seed, exact population count/set hash, and `training_authorized=false`.

## Claim boundary

Permitted before labels:

> The deterministic triage ranks objective-pass assets for blind image-only semantic review and includes independent random controls.

Forbidden before labels:

- any triage score is semantic truth;
- any diagnostic threshold is a clean-C0 exclusion rule;
- random-control cleanliness is assumed;
- training is authorized.
