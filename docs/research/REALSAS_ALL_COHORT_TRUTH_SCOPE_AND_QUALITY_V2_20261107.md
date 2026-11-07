# RealSaS: Global Truth Foundry all-cohort reconciliation V2

**2026-11-07. Read-only source/candidate provenance; not independent source truth or training permission.**

## New original ledger evidence recovered from private Google Drive

The original historical `RealSaS_MASTER_CORPUS_1024_V3/ledgers/` contains:

- `MASTER_VARIANTS.jsonl`: 4,008 exact canonical asset records, SHA-256 `475b12c6876a7ba91d8b1b32b6acac29536431134b44a96277a74c2493e86913`.
- `PROCESSED_VARIANTS.jsonl`: 4,036 records, SHA-256 `97c0b5bb27947689bc9eaacf8a46f1dd1c55b855b49fffaf59361801ba230b49`.

Reconciled **all 3,993** V1 Master selections against original candidate IDs, canonical asset IDs, provider identities and original splits: **0 mismatches**. The exact fifteen excluded `QA_ONLY` entries are all `khronos_gltf_qa`: 8 historical `ADMIT_RIG_NO_SKIN`, 7 `ADMIT_GEOMETRY_ONLY`. Neither decision is current training approval.

The processed ledger also has **28 unique independent Objaverse technical-reject candidate IDs**, **outside** the selected 4,008, with no canonical asset ID: all 28 record `too_many_degenerate_faces`, two also `too_many_unweighted_vertices`. They must stay in repair-review quarantine rather than silently vanish or become false accepted subjects.

## Conservatively measured universe

| Type | Rows | Training consequence |
|---|---:|---|
| Master3993 selected | 3,993 | Preserve original FIT/TUNE/DEV/CAL/EXTERNAL_HOLDOUT splits; no current head admission |
| Core370 | 370 | 139 `ESTABLISHED_BASE` plus 231 `EXPANSION` historical records; PR #327 authority not inherited |
| Exact original Khronos QA-only | 15 | Remain `QA_ONLY`; all 37 target routes masked |
| Original technical rejects without canonical ID | 28 | Preserve candidate_id and audit; no phantom 32 facets or 37 target routes |
| **Total known recorded candidate/source entities** | **4,406** | **NOT** certified unique characters |

**4,378 records have canonical subject IDs**; 28 rejects have candidate IDs only. All head training admission counts remain zero. The V2 private readback has 57,089 unchanged historical family evidence rows, 140,096 facet rows and 161,986 proposed target routes, all masked.

Cross-corpus duplicate closure remains **OPEN**. In particular, 191 Quaternius + 37 KayKit Master sources are `LEGACY_PATH_MATCH_UNREHASHED`, yet Core370 includes these publishers across glTF/FBX/GLB sources. Zero exact SHA matches between recovered 3,192 Master and 370 reported Core370 hashes cannot prove no cross-format duplicate.

## Concrete source-quality distribution / training-bias gate

| Master provider | Selected sources | SHA-recovered original receipts | Imported source numeric observations |
|---|---:|---:|---:|
| Objaverse original animated | 3,765 | 3,192 | **1,024** |
| Quaternius CC0 | 191 | 0 | **0** |
| KayKit CC0 | 37 | 0 | **0** |

**Every historical 1,024 full import comes from Objaverse.** This is a provider and morphology distribution risk, not evidence that the other 228 are invalid. We cannot infer their pose, mesh, rig, material, attachment, weight or rendered 2D graphics parity by extrapolating Objaverse imports. Core370 historical evidence has 196 glTF, 111 FBX and 63 GLB records, but historical court states are not independent current PR #327 training permissions.

Existing Master cleanup observations: 864 modular/open-mesh review flags, 350 texture relinks, 311 weight-sum anomalies, 302 unweighted-vertex reviews, 29 negative-determinant reviews and 22 source-recovery exceptions. These are review queues, **not automatic source rejection**.

## Head truth evidence routing: observation and historical evidence are not teacher labels

- IRIS Geometry source surface: 1,024 Master import hints, 86 Core370 historical PASS hints; full source geometry/camera parity not established by this union.
- IRIS Geometry camera: 341 Core370 historical PASS and 29 historical FAIL, not current.
- IRIS Semantic source face region and three-way drawable necessity: **no current accepted source truth in this ledger**. Material/object naming is a hint only.
- IRIS Graphics independently reviewed 2D stylized art / alpha / CAA: **zero accepted targets in this ledger**. Original 3D RGB may be diagnostic but not auto-designated game-art teacher.
- TESSA source topology: 1,024 imported hints, 86 historical Core geometry/topology hints; requires source face/mesh provenance and mechanical topology courts.
- AXIS skeleton graph: 1,024 import hints, 111 historical Core rig PASS hints; native IK, controller and RigProgram must be separately proven, not hallucinated from bone names.
- MIRA sparse weights: 1,024 import hints, 108 historical Core skin PASS plus 3 historical MASKED; bind/deformation source parity is still pending. FSQ tokenization can only be auxiliary until original sparse truth is certified.

## Implementation / safety

- `tools/corpus/reconcile_realsas_all_cohorts_v2.py`: SHA-pinned original 4,008/4,036 V1 ledger reconciliation, exact 3,993 source/split join, 15 genuine QA rows, 28 quarantined rejection candidates, all teacher masks closed.
- `tests/corpus/test_realsas_all_cohorts_v2.py`: synthetic negative cases for split tampering, candidate substitution, reject overlap and forged SHA.
- `canonical/REALSAS_GLOBAL_COHORT_REGISTRY_V2.json`: machine-readable count, source and head-safety registry.
- The private full V2 matrix is **not** committed to GitHub; producer output must remain in private Drive with exact SHA readback.

### Next mandatory courts

1. Revalidate original source rights, SHA and dedup/identity clusters across publishers and formats. Do not use only basename.
2. Stratify real scene imports and independent pose/camera/mesh/attachment QA across Objaverse, Quaternius, KayKit and Core370 format families before training.
3. Preserve original coordinates, native rest and source face/vertex/bone IDs; explicitly evaluate modifiers, nonmanifold/modular components and negative determinants.
4. Run source surface + semantic/graphics courts separately: native source vs approved 2D stylized projection, visibility, draw-order, edit-consequence.
5. Run native AXIS/MIRA RigProgram control and skin-deformation parity courts independently; never conflate derived IK prior with native authored control.
6. Only admit a head target if its source rights, independent family court, valid loss mask, split lineage and replayable SHA-pinned teacher receipt all PASS.

**No new renders, model weights, IK runs, source mutations, training or materialization permissions from these V2 ledger operations.**
