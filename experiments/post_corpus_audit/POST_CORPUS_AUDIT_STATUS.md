# RealSaS — Post-Corpus Audit Status

**Date:** 2026-08-23  
**Branch:** `g0-g1/single-pose-geometry`  
**State:** `CORPUS_RENDER_COMPLETE__POST_CORPUS_AUDIT_ACTIVE__TRAINING_BLOCKED`

This document is the continuation ledger for the post-corpus forensic ladder. It does not authorize training. `CURRENT_STATE.md` remains the top-level continuation authority.

## Frozen decision rule

`CORPUS FINISHED != START TRAINING`.

Before the first new optimizer step, close the fatal apparatus/data-plane checks and the Representation/Information/IRIS-parity gates. A negative result may not be promoted to an information-limit claim until apparatus, representation, target authority, and learner/solver explanations have been separated.

## Production corpus observed on Drive

Persistent corpus root:

`RealSaS_MASTER_CORPUS_1024_V3`

Builder:

`RealSaS_Master_Corpus_1024_RunAll_V4_3_LOCAL_FIRST_FULL_PRODUCTION_AUDITED.ipynb`

Build ID:

`REALSAS_MASTER_1024_V4_3_LOCAL_FIRST_FULL_PRODUCTION_20260822`

The heavy canonical render completed before the run was interrupted:

- canonical render selection: **3993**
- native render contract: **1024 x 1024 x 8 views**
- canonical render: **3993 / 3993 complete**
- consumer export was intentionally interrupted after render completion because the implementation duplicates/hashes existing observation files and repeatedly rewrites `records.jsonl`; it is not scientific evidence generation.
- observed partial IRIS consumer export before interruption: **11 records** only. This does **not** imply loss of master corpus evidence.

## Gate 0 — LIVE DRIVE CENSUS

### Ledger counts

- `MASTER_VARIANTS.jsonl`: **4008** rows
- unique candidate IDs: **4008**
- unique canonical asset IDs: **4008**
- `PROCESSED_VARIANTS.jsonl`: **4036** rows
- canonical selection: **3993** assets
- excluded from canonical render: **15 QA_ONLY** assets

No two rows in this run currently compete for the same `canonical_asset_id`; canonical selection therefore did not arbitrate among multiple same-family variants. Cross-registry/source duplicate physical assets remain a separate SHA/provenance audit.

### Selected capabilities

- IRIS: **3993 / 3993**
- Geppetto: **3890 / 3993**
- Arachne/full skin truth: **3414 / 3993**

### Selected split census

- FIT: **2981**
- TUNE: **318**
- CAL: **251**
- DEV: **274**
- EXTERNAL_HOLDOUT: **169**

Recomputation of the builder's deterministic hash split over all 3993 selected assets produced **0 split mismatches**.

### Source composition

- `objaverse_animated_originals`: **3765 (94.29%)**
- `quaternius_cc0`: **191 (4.78%)**
- `kaykit_cc0`: **37 (0.93%)**

This is a real domain-composition warning: 3993 physical entries must not be interpreted as a balanced 3993-family visual/domain sample. Source-stratified metrics are mandatory.

### Capability / decision composition in master ledger

- `ADMIT`: **3414**
- `ADMIT_RIG_NO_SKIN`: **484**
- `ADMIT_GEOMETRY_ONLY`: **110**

Technical rejects in the processed ledger:

- `REJECT_TECHNICAL`: **28**
- all 28 include `too_many_degenerate_faces`
- 2 also include `too_many_unweighted_vertices`

### Retrieval / extraction tail

`PENDING.jsonl` is an append-only history ledger; historical row count must not be treated as current unresolved count.

Latest-state unresolved population recovered from the ledger:

- total: **628**
- `PENDING_EXTRACTION`: **372**
- `PENDING_MATERIALIZATION`: **256**
- source: Objaverse **586**, KayKit **27**, Khronos QA **15**

Preflight had identified **10,653** commercial-clean linked unique families, so the 3993 production result is limited by materialization/extraction/gold/fallback throughput rather than by the legal candidate ceiling.

### Gold / fallback status

Ready and ingested paths include KayKit, Khronos glTF QA and two Quaternius packs.

Not recovered in this run:

- Blender Studio Storm: public `.blend` link not exposed by source page
- MPFB/MakeHuman: extension build failed
- Rig-XL processed fallback: run hit local disk `ENOSPC` before adding fallback full-truth families

These are corpus-completeness/domain-coverage issues, not proof that the current 3993 are invalid.

## Gate 0 — provenance spot-check

A live selected Objaverse variant contains typed admission/provenance with:

- source provider and immutable record identifier
- source revision
- source SHA-256 and byte size
- legal decision/evidence
- canonical asset ID
- technical geometry/rig/skin status
- raw-storage policy / preserved raw path when applicable

Full selected-set provenance existence, raw-path resolution and duplicate `source_sha256` groups are delegated to the batch auditor below.

## Gate 1 — LIVE DATA-PLANE SPOT CHECK

Representative full-truth Objaverse asset:

`asset_fffcea25a25b094fd8a2b06d`

Observed persistent structure:

- `RENDER_COMPLETE.json`
- `primary_geometry.npz`
- `renders/V0..V7`

V0 deep check:

- PNG: **1024 x 1024 RGBA**
- alpha foreground pixels: **135955**
- authority rows: **135955**
- alpha pixel set == authority pixel set: **exact**
- authority pixel indexes: **unique and in bounds**
- triangle IDs: **in range**
- barycentrics: **finite**, sums to one; minimum weight only numerical epsilon (~`-1.19e-7`)
- barycentric outside fraction at `1e-5`: **0**
- camera: `TOP_LEFT`, image-y-down, NDC-y-up, exactly one flip, yaw 0, `semantic_facing=UNKNOWN`
- P reconstructed from triangle+barycentric and projected back to pixels:
  - p50 ~ **7.9e-6 px**
  - p95 ~ **2.13e-5 px**
  - p99 ~ **2.93e-5 px**
  - p99.9 ~ **0.0337 px**
  - max ~ **3.10 px**
  - only **3** foreground points >0.5 px and **2** >1 px

The rare reprojection outliers are associated with extremely small/projectively ill-conditioned triangles in this sample. Max-only failure is therefore forbidden; the full audit must report quantiles and projected-triangle conditioning.

### Representative mesh warning

The same admitted asset contains **118 connected surface components**, despite otherwise clean finite geometry and no >2 edge multiplicity in the spot check. This is strong evidence that SurfaceBuilder sheet/component logic is not optional; spatial proximity alone cannot define a manifold/sheet identity.

### Provider-stratified publication spot check

Separate selected assets from Quaternius and KayKit also expose persistent `RENDER_COMPLETE.json`, `primary_geometry.npz`, and all `V0..V7` render folders. This is only a publication spot-check, not a substitute for the full binary audit.

## Gate 2 — known mesh-truth apparatus risk

The current Blender extraction path uses object data vertices transformed by `matrix_world` and deterministic polygon fan triangulation; it does not establish that the evaluated dependency-graph mesh is identical to extracted geometry.

Audit therefore remains open for:

- modifiers: Mirror/Subdivision/Solidify/etc.
- active shape keys / procedural geometry
- concave n-gon fan-triangulation mismatch
- custom/split normals and hard-edge authored normals

Do **not** rerender all 3993 by default. First quantify the affected subset and selectively re-extract/rerender only consequential assets.

## Gate 3 — appearance authority

Current `cel_clean` / `ink_cel` is a geometry-isolation control. The builder deliberately removes original texture/material appearance from the primary raster.

Therefore:

- it is legal as a controlled geometry observation arm;
- it must **not** silently become the sole natural/product observation authority for correspondence/identity conclusions;
- source appearance preservation and external texture dependency coverage remain open;
- an appearance-preserving sibling pass should be rendered only after coverage is quantified, and used as a causal A/B arm against the geometry-clean render.

## Batch auditor

`audit_master_corpus.py` performs a read-only Gate 0–3 audit against the persistent Drive tree and writes only compact reports under:

`reports/post_corpus_audit/`

Default behavior:

1. full ledger/census/split/provenance scan;
2. full 3993-asset shallow publication/camera/file-presence scan;
3. deterministic source/split/capability-stratified deep numeric audit of 128 assets;
4. no training, rerendering, consumer export, or corpus mutation.

Deep-all is possible but is deliberately not the default because reading all 31,944 raster-authority archives is unnecessary before a stratified sample establishes whether a systemic apparatus problem exists.

## Audit ladder status

| Gate | Status | Meaning |
|---|---|---|
| 0 Census/splits/provenance | **ACTIVE / mostly closed** | census and split exact; full provenance/SHA batch scan pending |
| 1 Geometry/data-plane | **ACTIVE** | strong live spot check; full shallow + stratified deep batch pending |
| 2 Mesh/surface truth | **ACTIVE** | extraction/triangulation/authored-normal risks identified; quantify subset |
| 3 Observation information | **ACTIVE** | geometry-control authority understood; raw appearance coverage pending |
| 4 Representation Authority Study | **BLOCKED by 0–3 fatal checks** | P / P+N / local frame / CSE / global R ladder |
| 5 SOI-2 / information ceiling | **BLOCKED by 4** | only after stronger legal representations |
| 6 IRIS architecture/data parity | **BLOCKED by 0–5** | 1024 path, epipolar, matcher, triangulation, U calibration |
| 7 SurfaceBuilder | queued | after geometry evidence is trusted |
| 8 Downstream sufficiency | queued | exact-vs-predicted Geppetto/Arachne consequence |
| 9 Ambiguity quotienting | queued | consequential product modes only |
| 10 Artist/product domain | queued | redraw/camera/appearance robustness |
| 11 Final owner decision | queued | PASS / bounded fix / true-limit prior completion |

## Immediate continuation

1. Run the Gate 0–3 batch auditor; inspect fatal/systemic findings first.
2. If no fatal apparatus problem exists, close Gate 0–3 with selective repairs only.
3. Freeze the Representation Authority Study before opening learner/training work.
4. No G1 optimizer step until the above gate authority says so.
