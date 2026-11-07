# Corpus4008: paper-grounded builder and model-supervision blueprint (2026-11-07)

**Status: independent research + route design, not family-truth PASS.** Parent draft: PR #391. No changes to Corpus370 PR #327, V18+, training, materialization or main.

## What we actually verified

**StdGEN public truth-preparation reference**: [hyz317/StdGEN/blender](https://github.com/hyz317/StdGEN/tree/main/blender) publishes source Blender rendering scripts. [StdGEN++](https://arxiv.org/html/2601.07660) (Sec III, IV-D, IV-E) describes Anime3D-EX: 10,811 selected VRoid-based anime models from ~14k candidates; full model / body+clothing / base-body synchronized multiview presentations; head-centric renderings; isolated iris, lashes/eyebrows, and face-skin texture layers; RGB, masks, semantic maps, normals and depth; staged NeRF/mesh supervision. StdGEN and StdGEN++ are different releases: **do not claim StdGEN++'s expanded exact builder implementation is public** without code-level verification. Humanoid A-pose and fixed three-part semantics are not universal source truth for our varied morphologies.

**SkinTokens/TokenRig reference**: [VAST-AI-Research/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) includes public inference/model/data modules. Research jointly represents skeleton hierarchy and per-vertex sparse skinning weights using FSQ-CVAE discrete skin tokens, sequence modeling and geometry/semantic rewards (bone-mesh containment, coverage, skin sparsity, smoothness). **A predicted rig is not the asset's native authored IK/FK/controller graph**. Published checkpoint training data rights (ArticulationXL 2.0 / VRoid Hub / ModelsResource) require original-license verification; MIT code does not license all underlying training assets.

## Mapping to BUSTED's six consumers

| Consumer | Source-bound truth builder | Potential supervision | Additional RealSaS gate |
|---|---|---|---|
| IRIS Geometry | Original mesh face and vertex addressing; invertible shared source frame; synchronized source cameras; 2D depth/normal/mask; guarded UDF/occupancy | surface/face correspondence, projection/silhouette, depth, normals, UDF; signed SDF only on independently valid solids | source geometry/source normals parity; nonwatertight/modular meshes not silently signed |
| IRIS Semantic | Independent source-face region/instance evidence, multiview source-hit palette; source scene semantics + morphology | semantic region/instance, 3D↔2D region, source component support | **necessary / alternative / undetermined drawable separation** based on source truth + capability contract, not material name or fixed body/clothes/hair labels |
| IRIS Graphics | Native source appearance and alpha/visibility/ray evidence **separate from** independently approved stylized 2D-game-art RGBA | edited/styled RGB/alpha, source-consistent coverage, occlusion/draw order, color and layer reveal | camera, silhouette, 2D art quality, identity across views, source CAA and edit-consequence courts |
| TESSA | Source topology, connected components, charts/seams and conditional motion-deformation correspondence | mesh/face topology, component/chart relationships, animatable surface | source topology parity; open/modular mesh semantics; no silhouette-only low-poly teacher |
| AXIS | Native skeleton hierarchy, rest/bind, joint curves; when present native IK/FK constraints, targets, poles, drivers, attachments and control interventions | skeleton/pose/motion; **typed RigProgram** with control, chain, constraint, attachment; conditional source-native control targets | imported bone forest not native controller program; inferred IK prior separate from native truth |
| MIRA | Original per-vertex/bone sparse influences; bind/skin/rest mapping; pose-sampled native mesh deformation and unaffected regions | skin weights, bone influence topology, deformation fidelity; optional FSQ token auxiliary | exact source IDs and source weight parity; token model cannot rewrite teacher weights |

**Important evidence taxonomy**: `RAW_SOURCE`, `IMPORTED_OBSERVATION`, `INDEPENDENT_SOURCE_TRUTH`, `REVIEWED_PRIOR`, `DERIVED_BEHAVIOR_VALIDATED`, `STYLIZED_2D_PRESENTATION_TARGET`, `REPRESENTATION_POLICY`. In particular, material names and mesh object groups are **hints**, not semantic ground truth. Full-control is a richer AXIS target, not a universal admission requirement for geometry/graphics/skin work.

## Global data and head routes

Current V1 provenance: 3,993 canonical master assets, 13 family slots each (51,909), six candidate head routes each (23,958), **zero new admitted targets**. Core370 historical 14-family matrix is independent and not presumed current #327 authority. Do not combine the 3,993 and 370 by filenames, undeclared aliases or nonmatching hashes. Import success and numerical roundtrip are NOT teacher authority.

The separate private read-only facet census checks 32 aspects per asset (127,776 candidate facet observations) from the existing immutable inventory and 1,024-source import/QA receipts. A proposed 37-target / 6-head mapping would create 147,741 potential asset-target rows: **every row remains masked/pending** until independent legal, source lineage, source-fidelity and target court receipts exist. These counts describe routing, not actual training supervision.

## Required builder payload per target

Each target packet must contain: `source_id`, `source_sha256`, `source_identity_cluster`, `split`, `head`, `target_type`, `source_face_vertex_bone_ids`, `camera_and_pose_id` (when applicable), `normalization_inverse`, `source_truth_refs`, `teacher_version_sha`, `court_receipts`, `quality_metrics`, `label_state` (PASS/FAIL/UNKNOWN/MASKED/N_A/QUARANTINED), `valid_target_mask`, `abstain_reason`. A missing target is not a negative class, and is never zero-filled.

## Priority and parallel engineering

1. Reconcile Wave4 already-imported receipts with V1, repairing test runner issues before claiming combined counts; preserve source SHA, split and original status.
2. Identity/rights, character-vs-prop, morphology, object/mesh/attachment graph, authoring pose, anatomical front/up and source/renderer camera courts, in parallel across independent asset shards.
3. Publish per-asset *source* supervision packets: Geometry/normal/topology, native rig/bind/skin/motion, source material/texture; quarantine anomalies and document provenance.
4. Run layer-semantic/visibility and stylized 2D Graphics teacher courts with independent source fidelity and aesthetics; control ambiguity via UNKNOWN/abstain.
5. Freeze versioned head target schemas, approved loss masks, cross-head overlaps and leakage-safe splits before materialization, training or a PASS claim.

**Non-goals**: rerun V18+/IK discovery as a substitute for source data cleaning; write artificial source truth; run generated graphics in place of approved 2D art; change active Corpus370 training or merge an unreviewed branch.
