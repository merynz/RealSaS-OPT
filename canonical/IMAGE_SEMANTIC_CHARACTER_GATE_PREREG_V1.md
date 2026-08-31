# Geppetto/Arachne V0.1 — Native Image Integrity + Single-Riggable-Character Gate Prereg V1

**Date:** 2026-08-31  
**Status:** `FROZEN_BEFORE_IMAGE_MEASUREMENT_OR_SEMANTIC_LABELS__OPTIMIZER_NOT_AUTHORIZED`

## Purpose

Close the remaining clean-C0 input-quality ambiguity after structural admission and `OBJECTIVE_RENDER_C0_V1` without using any learned RealSaS consumer output.

The gate has two ordered phases:

- **Phase I — native image/raster integrity measurement**: objective, deterministic, threshold-free measurement first;
- **Phase II — semantic single-riggable-character review**: image-only semantic decisions under a frozen label contract.

The final clean C0 membership cannot be emitted until Phase I has a separately frozen admission policy and Phase II labels are complete.

## Frozen population

Input population is exactly the objective-pass subset produced from:

- structural membership SHA-256 `cd3f5d14dbbfae0996cdade124af209da478cd36c2a0b3198a9a93fda5979147`;
- purity result SHA-256 `f71e2fb38b793b9110d4afe34701db1dfa171ff784b026fccf3ce394cfd18b0b`;
- objective policy `OBJECTIVE_RENDER_C0_V1`;
- expected objective-pass Geppetto count `2874`;
- expected objective-pass Arachne count `2509`;
- expected objective-pass set SHA-256 `60096fdf837b9cf277d1e3493a956796faf173d1af8ba7f593009423d77fa78b`.

TUNE/CAL/DEV/EXTERNAL_HOLDOUT remain closed.

## Phase I — native image integrity measurement

### Authority

For each objective-pass asset and each ordered view `V0..V7`:

- native image: `renders/V{v}/cel_clean.png`, required RGBA 1024x1024;
- canonical raster: `renders/V{v}/raster_authority.npz`, required 1024x1024;
- camera: `renders/V{v}/camera.json`, required yaw `45*v` degrees.

The derived `cel_clean_512.png` is **not** image-integrity authority. No resize or resampling is allowed in Phase I.

### Frozen image foreground extraction

- If any pixel has alpha `<255`, image foreground is `alpha >= 128`.
- Otherwise image foreground is the complement of exact RGB green `(0,255,0)`.

This restores the original measurement semantics at the correct native 1024 resolution without retroactively modifying V1.2.

### Per-view measurements

Measure and record, without an admission threshold:

1. image foreground pixel count;
2. raster foreground pixel count;
3. foreground intersection and union;
4. raster/image IoU;
5. image foreground unsupported by raster, normalized by image foreground count;
6. raster foreground missing from image, normalized by raster foreground count;
7. exact support equality boolean;
8. alpha-mode metadata (`ALPHA_THRESHOLD` or `OPAQUE_EXACT_GREEN`).

Malformed/missing native image, malformed raster authority, wrong resolution, duplicate/out-of-range raster pixel IDs, or camera-yaw drift is a **hard authority failure**.

A complete zero-hard-failure run is:

`PASS_IMAGE_INTEGRITY_MEASUREMENT_COMPLETE__POLICY_FREEZE_NEXT`

No image-quality threshold may be chosen before this complete result exists.

## Phase II — semantic single-riggable-character review

### What counts as PASS

The eight controlled views collectively depict one coherent riggable character subject. The protocol explicitly includes and must not bias against:

- humans/humanoids;
- animals and quadrupeds;
- robots/mechs when they are character-like articulated subjects;
- monsters, aliens, creatures, insects, fantasy beings;
- stylized, chibi, exaggerated, asymmetric, or partially amorphous characters;
- character-attached clothing, weapons, wings, tails, accessories, shells, or costume pieces.

### What counts as FAIL

Reject from clean C0 when the views are principally:

- prop-only or object-only;
- vehicle-only without a character-like articulated subject;
- furniture, building, architecture, or environment;
- a scene/set rather than one character;
- multiple unrelated dominant subjects;
- a character plus a dominant unrelated plane/card/environment component such that the training image is not a clean single-character observation;
- otherwise not reasonably interpretable as one riggable character from the eight controlled views.

### Labels

Every reviewed asset receives exactly one label:

- `PASS_SINGLE_RIGGABLE_CHARACTER`
- `FAIL_PROP_ENVIRONMENT_OR_UNRELATED_SCENE`
- `REVIEW_AMBIGUOUS`

`REVIEW_AMBIGUOUS` is fail-closed for clean C0 but remains eligible for future broader/pretrain pools. No asset is deleted from the master corpus.

### Review information firewall

The semantic reviewer may see only:

- canonical asset ID;
- ordered controlled images `V0..V7` (or a deterministic contact sheet built from them).

The reviewer may not see source-dataset category, skeleton count, bone names, skin statistics, objective review flags, DINO/IRIS/Geppetto/Arachne/Compiler outputs, or held-out split information while assigning the semantic label.

Objective flags may control **review scheduling priority only**; they may not alter the prompt or label rule.

### Deterministic contact-sheet contract

If contact sheets are used:

- input: `cel_clean_512.png` in exact order `V0..V7`;
- layout: 4 columns x 2 rows;
- each panel remains 512x512 with no crop or geometric transform;
- composite transparent pixels over fixed neutral gray `(127,127,127)`;
- add only a small view label `V0..V7` outside the image support;
- no enhancement, segmentation, captioning, saliency, or model prediction overlay.

## Classifier authority

This preregistration does **not** silently nominate an unfrozen VLM as scientific authority. An automated classifier may be used only after its exact model/version, weights/hash when locally hosted, prompt, decoding settings, view-order test, and ambiguity routing are separately sealed **before its outputs are opened**.

Human image-only review under the label contract is allowed. Any future automated triage disagreement must route to `REVIEW_AMBIGUOUS` or human adjudication; it may not auto-admit by majority after outputs are seen.

## Final membership rule

An asset enters final Geppetto clean C0 iff all are true:

`STRUCTURAL_C0 && OBJECTIVE_RENDER_C0_V1 && IMAGE_INTEGRITY_PASS && PASS_SINGLE_RIGGABLE_CHARACTER`.

Final Arachne clean C0 is the exact nested structural Arachne subset of that final Geppetto membership; no helper-mass transport or target rewrite is introduced.

Final output must record ordered IDs, set hashes, counts by exclusion reason, policy/result provenance, and `training_authorized=false`.

## Leakage firewall

Until final clean C0 is sealed:

- no Geppetto/Arachne optimizer step;
- no Geppetto/Arachne checkpoint selection;
- no IRIS or Compiler prediction used for admission;
- no held-out split opened;
- no master asset mutation/deletion/rerender.
