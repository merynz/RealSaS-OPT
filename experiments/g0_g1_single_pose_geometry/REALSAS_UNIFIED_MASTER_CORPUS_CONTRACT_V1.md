# RealSaS Unified Master Corpus Contract V1

Status: CANONICAL_DATA_ARCHITECTURE_FROZEN__IMPLEMENTATION_PENDING
Date: 2026-08-22
Branch: `g0-g1/single-pose-geometry`

## 1. Decision

RealSaS shall maintain one canonical, information-preserving **Master Corpus** rather than separate lossy corpora for IRIS, Geppetto and Arachne.

Initial source families:

1. Articulation-XL2.0
2. Rig-XL / UniRig

The master is source-rich. Model-specific datasets are deterministic, schema-checked exports from the master.

```text
UPSTREAM SOURCES
  Articulation-XL2.0        Rig-XL
           \                 /
            \               /
             v             v
          L0 SOURCE ARCHIVE
      immutable source records/blobs
                 |
                 v
          L1 MASTER TEACHER
  geometry + rig + skinning + source metadata
  + canonical cameras
  + native 1024x1024 x 8 renders
  + dense geometry render truth
                 |
       +---------+----------+
       |         |          |
       v         v          v
  IRIS export  Geppetto   Arachne
  geometry     skeleton   skinning
  firewall     teacher    teacher
```

Compiler evaluation artifacts may also be derived from L1, but Compiler is not treated as a learned model corpus consumer here.

## 2. Do not render every upstream record blindly

Upstream archives are source authority, not automatic admission authority.

### Articulation seed admission

The first canonical Articulation render set is the exact previously-audited RealSaS set of **3456** families/samples. Its historical split identity is preserved:

- 3072 historical train-side records;
- 384 historical validation records, remaining closed to model-selection use.

The 3456 identities must be reconstructed from existing RealSaS M1/M5 manifests and matched to upstream records by stable source identity/UUID. No substitute family may silently replace an unresolved identity.

### RigXL seed admission

The first canonical RigXL render set is the existing **RIGXL269** qualified depot/split, preserving its historical official split:

- 209 train;
- 29 dev;
- 21 sealed-qualification;
- 10 external holdout.

Sealed/external identities may be represented in L0 provenance, but model-specific exports must remain closed unless an explicit later gate opens them.

### Expansion

Later upstream records may be admitted only through a versioned admission audit. Expansion must never rewrite the identity of the seed sets.

## 3. Upstream source pins

### Articulation-XL2.0

Provider: Hugging Face dataset `Seed3D/Articulation-XL2.0`.

Known upstream fields include:

- `vertices`
- `faces`
- `normals`
- `joints`
- `bones`
- `root_index`
- `uuid`
- `pc_w_norm`
- `joint_names`
- sparse skinning triplets: `skinning_weights_value`, `skinning_weights_row`, `skinning_weights_col`, `skinning_weights_shape`

The source resolver must match all 3456 selected identities. It may use the diverse-pose and/or main Articulation NPZ archives as necessary. The notebook must not assume that every selected RealSaS identity is contained in one specific upstream NPZ until that is verified.

### RigXL

Provider: Hugging Face model repository `VAST-AI/UniRig`, source lineage pinned to the historical RealSaS/UniRig revision where possible (`bfb98220f01fe500398f57f517e67838a7c21b98`).

Primary processed source: `data/rigxl/processed.7z` plus `mapping.json` and official datalist/split metadata.

Known RigXL `raw_data.npz` fields include:

- `vertices`
- `vertex_normals`
- `faces`
- `face_normals`
- `joints`
- `skin`
- `parents`
- `names`
- `matrix_local`

All available source-native fields must be inventoried before normalization.

## 4. Information preservation rule

**No useful source field is discarded from L0/L1 merely because the current frontend does not consume it.**

For each source record the builder must:

1. enumerate every source-native key;
2. record key name, dtype, shape and content SHA-256;
3. copy or losslessly encode every field needed for geometry, skeleton, hierarchy, skinning, bind/rest transforms, source identity and reproducibility;
4. preserve unrecognized source-native fields in a passthrough artifact or fail the build if preservation cannot be guaranteed;
5. record every normalization/conversion as provenance rather than overwriting source authority.

Unknown-field silent loss is a build failure.

## 5. Entity-centric cross-source deduplication

Articulation and RigXL may contain overlapping Objaverse assets. The master therefore distinguishes:

- `canonical_asset_id`: RealSaS asset identity;
- `source_record_id`: one upstream annotation record;
- `source_annotations[]`: potentially multiple rig/skin authorities for one asset.

Cross-source duplicate candidates are detected using, in order:

1. exact upstream UUID/source identity when comparable;
2. normalized geometry fingerprint;
3. secondary geometric statistics/fingerprint.

A suspected duplicate is **linked, not silently merged**. Conflicting skeleton or skinning annotations remain separate teacher variants until an explicit adjudication rule selects one for a model export.

## 6. L0 Source Archive

L0 is immutable source authority and provenance.

For every admitted source record retain or address:

- provider/repository/revision;
- source archive path and archive SHA-256;
- member path / record index / UUID;
- complete source-native field inventory;
- source split/category/mapping metadata when available;
- source license metadata;
- source-native arrays or a byte-identical addressable source blob.

L0 performs no destructive normalization.

## 7. L1 Master Teacher normalized fields

L1 provides canonical normalized access while retaining links to L0.

### 7.1 Geometry

Required when available:

- vertices in source frame;
- vertices in canonical RealSaS frame;
- triangle faces;
- vertex normals;
- face normals when available/derivable;
- connected-component identity;
- bounding box / center / scale;
- canonical transform and inverse;
- dense or persistent surface sampling;
- source point-cloud-with-normal fields such as Articulation `pc_w_norm`.

Derived geometry must be marked `derived`, never masquerade as upstream truth.

### 7.2 Skeleton / rig

Required when available:

- joint positions;
- root identity;
- hierarchy / parent index;
- bone edge list;
- joint names;
- local/bind/rest transforms (`matrix_local` or source equivalent);
- source-specific additional rig metadata.

Representation conversions (e.g. Articulation bone edges -> canonical parent array) must retain the original representation and emit a conversion audit.

### 7.3 Skinning

Required when available:

- full per-vertex per-joint skinning weights or an exact sparse encoding;
- source sparse triplets where supplied;
- normalized/dense derivative for consumers;
- row-sum/finite/nonnegative diagnostics;
- influence count statistics;
- joint-index binding to the canonical skeleton representation.

Never discard exact sparse source values after producing a dense derivative.

### 7.4 Source-native passthrough

Any additional upstream array/field not yet assigned a canonical semantic role is preserved and listed under `source_native_inventory`. This prevents future rigging/skinning work from requiring a second raw-corpus archaeology pass.

## 8. Canonical single-pose 8-view render contract

Primary master render resolution:

```text
1024 x 1024 x 8 ordered views
```

Canonical view semantics:

```text
S, SE, E, NE, N, NW, W, SW
```

Default projection is a reproducible level orthographic orbit about canonical global Z, with screen-up +Z. Historical RealSaS projection conventions are used as lineage anchors; any change in camera framing/elevation must be versioned and measured rather than silently introduced.

Camera metadata per render must include enough information to exactly reproduce projection and unprojection.

## 9. Geometry-safe 2D/NPR appearance families

The 1024 observation domain is intentionally shifted toward 2D artwork without breaking exact geometry truth.

Allowed appearance transformations are rasterizer/NPR transformations that leave geometry, camera and coverage authority unchanged, including controlled variants of:

- flat/cel shading;
- quantized lighting bands;
- geometry-derived outlines;
- palette/material randomization;
- controlled shadow hardness;
- simplified specular response;
- geometry-anchored texture/pattern modulation;
- modest paper/paint-like tonal treatment that does not warp coordinates.

Generative/diffusion stylization is **not** part of the exact-GT core unless a future geometric-consistency gate proves pixel/surface alignment.

Every appearance variant of one view shares the same camera and surface truth.

## 10. Dense render truth

For every foreground pixel where rasterization defines a surface, the preferred teacher G-buffer is:

- foreground/coverage;
- depth;
- canonical XYZ;
- canonical normal;
- face/primitive ID;
- barycentric coordinates;
- optional component/material provenance when source-supported.

This enables exact cross-view same-surface correspondence without injecting skeleton or skinning semantics into IRIS.

If dense primitive/barycentric export is unavailable for a renderer backend, the fallback is a persistent surface sample set of at least 4096 points. The historical 512-sample set remains a lineage/parity subset, not the new information ceiling.

## 11. Model-specific export views

### 11.1 IRIS frontend export

Student input:

- only 8 raster observations and explicitly authorized observation channels.

Teacher/evaluation geometry:

- XYZ / normals;
- XY01;
- visibility/occlusion;
- dense surface provenance for correspondence;
- camera metadata as evaluator/build authority, not hidden student input unless explicitly preregistered.

Physically forbidden from the IRIS export:

- joints;
- parents/bones;
- joint names/roles;
- skinning weights;
- deformation/mechanics targets;
- compiler owner/control IDs.

### 11.2 Geppetto export

Input-side geometry authority may include canonical surface/point representations appropriate to the experiment.

Teacher targets may include:

- joints;
- root;
- hierarchy/parents;
- bone edges;
- bind/local transforms where relevant;
- source teacher variant/provenance.

Skinning weights are excluded unless a later joint-training contract explicitly authorizes them.

### 11.3 Arachne export

Input-side authority may include:

- canonical surface geometry;
- skeleton/hierarchy authority appropriate to the experiment.

Teacher targets include:

- exact skinning weights/influences;
- vertex/surface correspondence needed to map source weights to the canonical RiggingSurface;
- source teacher variant/provenance.

## 12. Split and seal policy

Master possession does not authorize model use.

Every asset carries split state per source and per RealSaS experiment. Export builders must require an explicit split policy and must fail closed on sealed/test records.

Historical closed sets are never reclassified as confirmatory development data.

## 13. Storage format

Large tensors are not embedded in JSON.

JSON manifests carry:

- semantic field name;
- artifact path;
- dtype;
- shape;
- byte/content SHA-256;
- authority (`source`, `normalized`, `derived`, `rendered`);
- provenance links.

Arrays are stored in NPZ/NPY or another explicitly versioned lossless numeric container. Raster observations use lossless PNG or another explicitly versioned lossless format for exact-GT core data.

## 14. Required audit gates

A family is exportable only if applicable gates pass:

- source identity resolved;
- source-native key inventory complete;
- no unknown-field loss;
- geometry finite/index-valid;
- normals finite/unit-or-audited;
- hierarchy valid/acyclic/root-valid;
- skinning finite/nonnegative and row normalization audited;
- canonical transform invertible;
- 8-view camera reproducible;
- dense G-buffer self-consistent;
- visible projected truth lands on foreground within rasterization tolerance;
- style variants preserve coverage/provenance geometry;
- artifact hashes complete;
- split/seal state preserved.

## 15. Initial build target

V1 build target is not “all public data”. It is the union of our already-qualified seeds:

```text
Articulation selected seed : 3456
RigXL qualified seed       : 269
---------------------------------
pre-dedup source records   : 3725
```

The final canonical asset count may be lower if cross-source duplicates are linked. All duplicate source annotations remain retained.

After V1 closes, a separate admission expansion may score the remaining upstream pools and add families without changing V1 identities.

## 16. Resolution derivatives

1024 is the master observation render.

Matched deterministic derivatives may include:

- 512 for resolution ablation;
- 256 when needed for legacy comparison;
- 128 only for historical/control lineage.

These derivatives are not substitute native masters.

## 17. Schemas

The following machine-readable schemas are normative companions to this contract:

- `schemas/REALSAS_MASTER_CORPUS_MANIFEST_SCHEMA_V1.json`
- `schemas/REALSAS_MASTER_ASSET_RECORD_SCHEMA_V1.json`
- `schemas/REALSAS_MODEL_EXPORT_MANIFEST_SCHEMA_V1.json`
- `schemas/REALSAS_SOURCE_ADMISSION_SCHEMA_V1.json`

## 18. Current scientific status

This contract changes data architecture, not experimental evidence.

- G1 remains frozen/unstarted.
- No 1024 frontend optimizer step is authorized by this document alone.
- Previous two-pose RigXL metrics remain historical architecture/data lineage only for the current single-pose frontend question.
- Master-corpus construction may proceed before model training because it preserves source truth and split/seal policy.
