# RealSaS Unified Master Corpus Contract V1

Status: `CANONICAL_DATA_ARCHITECTURE_FROZEN__BUILD_AUTHORIZED__MODEL_TRAINING_NOT_AUTHORIZED`
Date: 2026-08-22
Branch: `g0-g1/single-pose-geometry`

## 1. Decision

RealSaS maintains one canonical, information-preserving **Master Corpus** for the three learned modules:

- **IRIS** — observation-grounded geometry/correspondence;
- **Geppetto** — skeleton/hierarchy proposal;
- **Arachne** — skinning/weights.

The Compiler remains deterministic and may consume evaluation artifacts but is not a learned corpus consumer.

The master is not tied to one upstream dataset. It is the legally and technically admitted union of multiple source families.

```text
PUBLIC / OWNED SOURCES
      |
      v
L0 SOURCE AUTHORITY
raw asset or lossless source arrays
+ immutable provenance + license evidence
      |
      v
L1 MASTER TEACHER
canonical geometry
+ rig/hierarchy
+ skinning/bind/rest data
+ source-native passthrough
+ canonical 1024 x 8 observation generator
      |
      +----------------+----------------+
      |                |                |
      v                v                v
IRIS EXPORT       GEPPETTO EXPORT    ARACHNE EXPORT
geometry-only     skeleton teacher   skinning teacher
firewall          firewall           firewall
```

## 2. Admission is conjunctive and fail-closed

A source record enters the commercial/product-training master only if:

```text
TECHNICAL_ADMISSION == PASS
AND
LEGAL_ADMISSION == PASS
AND
SOURCE_PROVENANCE_RESOLVED == TRUE
```

Downloadability, a repository software license, a dataset-card tag, or prior academic use is not sufficient by itself.

This is an engineering compliance policy rather than legal advice. Ambiguous cases are quarantined until separately reviewed.

### 2.1 Legal auto-admission

Automatically admissible when the exact source asset is covered by a clearly commercial-permissive license and no conflicting source terms are found:

- CC0 / public-domain dedication;
- CC-BY, with attribution provenance retained.

### 2.2 Legal quarantine

Requires explicit later review:

- CC-BY-SA or other share-alike/custom terms;
- source/data license ambiguity;
- unresolved object-level license;
- unclear platform-specific restrictions;
- derived annotation datasets whose annotation-data license is not explicit.

### 2.3 Legal rejection

Commercial model-training admission is denied for:

- NonCommercial terms;
- explicit AI/ML training prohibition;
- research-only / academic-only terms;
- all-rights-reserved or unknown-rights records;
- assets requiring rights-holder permission that RealSaS does not possess.

No model export may contain `legal_decision != PASS`.

## 3. Canonical source registry

The normative market/source decision ledger is:

- `REALSAS_MASTER_CORPUS_SOURCE_AUDIT_V1.md`
- `REALSAS_MASTER_CORPUS_SOURCE_REGISTRY_V1.json`

The core V1 source classes are:

### Tier A — scale

1. **Objaverse / Objaverse-XL animated originals** — original artist mesh/armature/skinning truth; per-object license resolution.
2. **Articulation-XL2.0** — expert articulation annotation variant; underlying object license also resolved.
3. **Rig-XL / UniRig** — expert rig/skin annotation variant; Objaverse records resolved object-by-object; VRoid quarantined by default.

### Tier B — legal/artist gold

4. **MakeHuman / MPFB core** — CC0 synthetic/parametric humanoid gold.
5. **Blender Studio open character rigs** — explicit per-asset CC-BY production rigs.
6. **Quaternius CC0 rigged packs** — stylized artist-gold.
7. **KayKit CC0 character packs** — stylized artist-gold.

### Tier C — QA

8. **Khronos glTF Sample Assets** — selected compatible skinned assets for parser/rasterization regression.

Rejected/quarantined datasets remain in the source registry with the exact reason and are never silently downloaded into the product-training master.

## 4. Historical RealSaS sets are anchors, not ceilings

Historical known-good sets remain useful lineage/regression anchors:

- Articulation RealSaS historical set: **3456** records (3072 historical train-side + 384 historical validation);
- `RIGXL269`: **269** families (209 train + 29 dev + 21 sealed-qualification + 10 external holdout).

These counts do **not** define the new V1 corpus size. The new master admits all records from approved source families that pass the current technical/legal/provenance gates.

Historical closed identities remain closed for comparisons that rely on those historical splits.

## 5. Original asset is preferred L0 authority

Whenever a derived rigging dataset maps back to an original Objaverse/Objaverse-XL asset, the original object is the preferred L0 source authority.

The master separates:

- `canonical_asset_id` — one RealSaS physical-asset identity;
- `source_record_id` — one upstream source record;
- `source_annotations[]` — zero or more rig/skeleton/skin annotation variants for the same asset.

Thus an Objaverse original, an Articulation record and a RigXL record that refer to the same physical asset are **linked**, not duplicated as unrelated training families.

Conflicting annotations are retained separately. No annotation is silently declared ground truth merely because it came from one dataset.

## 6. Cross-source deduplication

Duplicate linkage uses, in priority order:

1. exact upstream UUID / fileIdentifier / original URL identity;
2. source SHA-256;
3. normalized geometry fingerprint;
4. secondary geometric/topological fingerprint.

Suspected duplicates that are not identity-proven remain separate but carry a duplicate-candidate relation.

All model splits operate on `canonical_asset_id`, preventing annotation-source leakage across train/validation/test.

## 7. L0 Source Authority

L0 is immutable provenance and preserves source truth without destructive normalization.

For every admitted record, retain or address:

- source registry ID;
- provider/repository/dataset and revision;
- original source URL/fileIdentifier/UUID;
- source archive/member/index identity;
- source SHA-256 and byte size when obtainable;
- source-native key inventory;
- source split/category metadata;
- dataset-level license metadata;
- object-level license metadata;
- license evidence URI + retrieval timestamp;
- attribution/author text when required;
- legal classifier version + decision + reason;
- original raw source blob or a reproducible address to it.

If a source-native field cannot be preserved or semantically represented, it must remain in a lossless passthrough artifact or the build fails with `UNKNOWN_FIELD_LOSS`.

## 8. L1 Master Teacher

L1 normalizes access while retaining links to all L0 authorities.

### 8.1 Geometry

Preserve when available:

- source-frame vertices;
- canonical-frame vertices;
- triangle faces;
- vertex normals;
- face normals;
- UVs and material-slot IDs;
- connected components;
- bounding box/center/scale;
- canonical transform + inverse;
- source point clouds such as `pc_w_norm`;
- topology fingerprints;
- persistent surface samples;
- shape-key/blend-shape inventory.

Derived fields are marked `derived`; they never masquerade as upstream truth.

### 8.2 Skeleton / rig

Preserve when available:

- joints / bone heads and tails;
- root identity;
- parent hierarchy;
- bone-edge representation;
- joint/bone names;
- deform-vs-control role evidence when source-supported;
- rest/bind/local transforms;
- armature-object transforms;
- source-specific rig metadata;
- control rig inventory and constraints when extractable;
- animation/action inventory.

Representation conversion (for example, bone edges -> parent array) retains the original form and emits a conversion audit.

### 8.3 Skinning

Preserve when available:

- exact source per-vertex/per-joint weights;
- sparse source triplets when supplied;
- dense normalized derivative for consumers;
- influence counts;
- row sums / finite / nonnegative diagnostics;
- mapping from vertex groups to canonical skeleton joints;
- inverse-bind/bind-transform information when available.

Exact sparse/source weights are never discarded after a dense derivative is produced.

### 8.4 Source-native passthrough

Every source key is inventoried. Additional arrays/metadata not yet assigned a RealSaS semantic role are retained under `source_native_inventory` or by immutable source-blob reference.

## 9. Source-specific normalization

### 9.1 Articulation-XL2.0

Known useful fields include:

```text
vertices
faces
normals
joints
bones
root_index
uuid
pc_w_norm
joint_names
skinning_weights_value
skinning_weights_row
skinning_weights_col
skinning_weights_shape
```

The builder resolves source/object identity from the dataset metadata and joins to object-level license authority before commercial admission.

### 9.2 RigXL

Known useful `raw_data.npz` fields include:

```text
vertices
vertex_normals
faces
face_normals
joints
skin
parents
names
matrix_local
```

The RigXL mapping is used to recover original Objaverse fileIdentifier/URL. VRoid records are not auto-admitted.

### 9.3 Original animated assets

For source files such as GLB/GLTF/FBX/BLEND, the headless extractor inventories and, when present, exports:

- all mesh objects and topology;
- armatures;
- deform bones and parent hierarchy;
- rest/local matrices;
- vertex-group weights;
- shape keys;
- action inventory;
- material/UV metadata.

Multi-armature and multi-mesh assets are preserved as source structure. A derived `primary_deform_rig` may be selected only by an explicit deterministic rule and does not erase secondary rigs.

## 10. Canonical single-pose observation contract

Primary render master:

```text
native 1024 x 1024
8 cyclic ordered views
ONE rest/neutral pose per rendered observation family
```

The render camera is a reproducible orthographic orbit about canonical global Z with screen-up +Z unless a source-specific canonical transform explicitly requires another versioned contract.

View slots are always cyclically ordered. Semantic labels `S, SE, E, NE, N, NW, W, SW` are used only when semantic facing is actually resolved. Otherwise the stored authority is `V0..V7` plus yaw angles and `semantic_facing=UNKNOWN`.

Camera metadata must be sufficient to reproduce projection/unprojection exactly.

## 11. 1024 geometry-safe 2D/NPR raster domain

The observation domain intentionally approaches 2D artwork while exact geometry truth remains unchanged.

Allowed deterministic appearance families include:

- neutral cel;
- quantized/cel-lit palette;
- geometry-derived **inner** contour/ink treatment;
- palette/material randomization;
- controlled shadow hardness;
- simplified specular response;
- geometry-anchored texture/pattern modulation;
- painterly-lite tonal modulation without geometric warping.

The exact-GT core does not use diffusion/generative image stylization.

A style transformation may change RGB/tonal appearance but must not change camera, surface provenance or geometry coverage. Outer silhouette expansion is not used as geometry foreground authority; outline effects are kept inside coverage or represented separately.

Same physical surface identity remains valid across both view and appearance family:

```text
(view_i, style_a, surface_p)
      == same teacher surface ==
(view_j, style_b, surface_p)
```

## 12. Dense correspondence/raster authority

Preferred per-foreground-pixel teacher authority:

- primitive/face ID;
- barycentric coordinates;
- depth or enough information to derive it exactly;
- geometry coverage;
- canonical vertex geometry and camera authority sufficient to derive XYZ and normals.

Dense XYZ/normals may be materialized as caches, but the master need not duplicate them permanently if they are deterministically reconstructable from immutable mesh + primitive/barycentric + camera with an equivalence audit.

This keeps dense truth information-rich without making redundant multi-terabyte storage mandatory.

Fallback when dense raster provenance cannot be emitted is a persistent surface sample set of at least 4096 points. Historical 512 samples remain lineage anchors only.

## 13. Storage hierarchy

Large numeric tensors are not embedded in JSON.

### L0/L1 numeric authority

Lossless NPY/NPZ or another explicitly versioned numeric container.

### Raster observations

Lossless PNG or another explicitly versioned lossless image format for exact-GT core data.

### Dense raster authority

Chunked/compressed numeric artifacts. Each field is described by:

- semantic field name;
- path;
- dtype;
- shape;
- SHA-256;
- authority (`source`, `normalized`, `derived`, `rendered`);
- provenance links.

All large-build stages are resumable and shard-addressed.

## 14. Model-specific exports

Master possession does not mean every field is visible to every model.

### 14.1 IRIS export

Student input:

- only eight raster observations and explicitly preregistered observation-derived channels.

Teacher/evaluator geometry may include:

- canonical XYZ / normals;
- XY01;
- visibility/occlusion;
- primitive/barycentric correspondence authority;
- camera metadata for build/evaluation.

Physically forbidden from the emitted IRIS export:

- joints;
- bone hierarchy;
- joint names/roles;
- skinning weights;
- deformation/mechanics targets;
- compiler owner/control IDs.

### 14.2 Geppetto export

Input-side geometry may include the canonical RiggingSurface/point representation authorized by its experiment.

Teacher fields may include:

- joints;
- root;
- hierarchy/parents;
- bone edges;
- rest/bind/local transforms where relevant;
- annotation-source provenance.

Skinning weights are physically excluded unless a later joint-training contract explicitly authorizes them.

### 14.3 Arachne export

Input-side authority may include:

- canonical surface geometry;
- skeleton/hierarchy authority authorized by the experiment.

Teacher fields include:

- exact skinning weights/influences;
- source vertex/surface correspondence;
- joint-index binding;
- annotation-source provenance.

## 15. Split and seal policy

Splits are canonical-asset based, not source-record based.

Required rules:

- all annotation variants for one physical asset share split;
- historical sealed/test identities remain sealed for experiments that inherit those splits;
- new master experiments receive a versioned deterministic family-disjoint split;
- legal/quarantined records are never included in any student export;
- source membership may not be used to leak duplicate identities across splits.

## 16. Required technical gates

A source record claiming a given teacher capability passes only when applicable checks succeed:

### Geometry

- finite vertices/normals;
- valid triangle indices;
- nondegenerate usable geometry;
- canonical transform finite/invertible;
- coordinate/frame audit recorded.

### Rig

- joint/bone values finite;
- root valid;
- hierarchy acyclic;
- parent indices valid;
- meaningful deform rig exists when rig truth is claimed;
- rest/bind transform representation internally consistent or explicitly qualified.

### Skinning

- vertex/weight row alignment valid;
- weights finite and nonnegative within tolerance;
- row sums audited;
- influence mapping to bones valid;
- no silent dropped influence columns.

### Render

- all eight cameras reproducible;
- raster coverage nonempty;
- primitive/barycentric authority self-consistent;
- visible teacher samples land on geometry coverage;
- style variants preserve coverage/provenance;
- artifact hashes complete.

A record may carry `geometry_pass=true` while `rig_pass=false`; its export eligibility is then capability-specific.

## 17. Required legal gates

Per admitted source record:

- exact original source identity resolved;
- dataset license recorded;
- object-level license recorded where applicable;
- conflicting terms absent or reviewed;
- no NC/research-only/AI-ban restriction;
- attribution evidence retained when required;
- retrieval timestamp and evidence reference stored;
- legal classifier decision == `PASS`.

License resolution happens **before expensive render production** whenever source metadata permits it.

## 18. Market-wide build target

V1 is no longer a fixed 3725-record experiment corpus.

The build target is:

```text
MASTER_V1 =
  all candidate records from the approved source registry
  that pass technical + legal + provenance admission
  after canonical-asset dedup/linkage
```

Historical Articulation3456 and RIGXL269 are regression anchors inside this larger process.

The candidate inventory may be much larger than the final admitted/rendered corpus. The builder writes counts for every rejection/quarantine reason so no silent selection occurs.

## 19. Colab execution contract

The canonical builder is Colab-oriented and resumable.

One `Run all` invocation must:

1. mount the designated Drive workspace;
2. fetch/update source metadata and source registry;
3. run legal resolution before large downloads where possible;
4. materialize source records in bounded shards;
5. extract/normalize source truth;
6. run technical gates;
7. deduplicate/link canonical assets;
8. render native 1024 x 8 geometry-safe NPR observations;
9. write dense/compact raster correspondence authority;
10. emit model-specific export indexes/firewalls;
11. hash all artifacts;
12. write admission/rejection/attribution manifests;
13. checkpoint after every shard and resume safely on the next Run-all session.

No local-PC heavy rendering is required.

## 20. Resolution derivatives

1024 is primary observation authority.

Deterministic controls may be derived:

- 512 for resolution ablation;
- 256 for legacy comparisons when useful;
- 128 historical/control only.

Normalized XY truth remains resolution-independent.

## 21. Machine-readable schemas

Normative companions:

- `schemas/REALSAS_MASTER_CORPUS_MANIFEST_SCHEMA_V1.json`
- `schemas/REALSAS_MASTER_ASSET_RECORD_SCHEMA_V1.json`
- `schemas/REALSAS_SOURCE_ADMISSION_SCHEMA_V1.json`
- `schemas/REALSAS_MODEL_EXPORT_MANIFEST_SCHEMA_V1.json`

The source registry is:

- `REALSAS_MASTER_CORPUS_SOURCE_REGISTRY_V1.json`

## 22. Current scientific status

This contract authorizes **corpus construction/audit only**.

It does not by itself authorize a new model-training run.

- G1 remains frozen/unstarted.
- No 1024 frontend optimizer step is authorized here.
- Previous two-pose RigXL evidence remains historical lineage, not evidence for the current single-pose frontend.
- Corpus construction may proceed because it preserves truth, legal provenance, and split/seal authority without changing model claims.
