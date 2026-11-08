# RealSaS Truth Foundry — Deep Source Audit / visual qualification V1

## User rule

> Know every source as if individually inspected. Distinguish the *breadth* of the recorded evidence from actual independent inspection. Nobody gets a review PASS just because a Blender importer reported a number or a text classifier guessed a class.

## Actual source records, not claims

- `MASTER3993_SOURCE_PHYSICAL_CENSUS_V1.csv`: 3,993 Master source records; 2,466 have imported physical summaries, 1,527 unobserved. Also 370 Core records in separate V1 4,363 truth matrix.
- `tools/corpus/truth_foundry/deep_source_audit_v1.py`: checks every SHA-bound **full compressed imported scene JSON** for all 2,466 Master observed sources, preserves node/mesh/armature/action counts and constraints/texture/attachment evidence. Outputs detailed per asset `jsonl`, 3,993-row coverage and action-by-action independent review tasks. All private to Drive. This is a **new independent readback step** beyond existing physical summary; no labels automatically admitted.
- Source name, material name, action name, bone name are *teacher-only metadata*, never available in student model input. No morphology or layer label may be derived from name alone.

## Mandatory description for each asset

| Group | Enumerated dimensions | Independent acceptance |
|---|---|---|
| Identity/rights | provider, original full SHA, archive/release, creator, usage rights, split and duplicate cluster | actual original license/version receipts, source-byte/cross-format courts |
| Scene | every imported object and hierarchy, collections, object types, parent graph, modifiers, constraints | original authoring scene vs importer parity |
| Geometry | mesh object count, original polygon+corner address, connected islands, open/boundary/nonmanifold, materials, LOD, topology | original bytes/parity; no island-as-semantic shortcuts |
| Anatomical pose | original rig rest/bind, *authored* pose, anatomical front/up, orientation source→canonical | source-bound canonical orientation image and bone/mesh pose controls |
| Character category | human, humanoid fantasy, quadruped mammal/other, reptile, dinosaur, bird, winged, serpentine, arthropod, aquatic, monster, robot/mech, tank, other ground vehicle, aircraft, watercraft, props/weapon, environment, plants, other/unknown | actual image/3D reviewer or validated independently checked categorizer; cross-view identity |
| Morphology attributes | limb/paw count, wing count, tail, shell, articulation axes, asymmetric appendages, hard/skeletal vs organic | source geometry/raster evidence with abstention and reviewer |
| Attachments | loose objects, bone-parented mesh, socket, independently controlled prop, rigid vs skinned object | causal edit/control test, source attachment ownership |
| Skin/controls | bind matrices, bone hierarchy, deform groups, sparse weights, nondeform controllers, IK/FK constraints, scripted drivers | source-authoring functional rig and pose-response, not imported count |
| Animation | clips, curves, frame ranges, animated bones, root vs skeletal motion, loop/contact/physical plausibility | replay with ground truth pose changes and contact controls |
| Textures/style | material slots, image files, external missing, UV, alpha, PBR, painted vs procedural, stylized candidate quality | source texture pixel and RGBA parity; 2D stylized identity/style review |
| Semantic | source-face ID exact ancestry, instance/role/part surface, seen/hidden/unknown mask, cross-view region consistent | independent source-face ontology+review receipt; not scene/mesh name |
| Graphics drawable | full/keep/hide/isolate, self-occlusion, edit reveal, slot necessity, independent control and painter ordering | real source 3D edit consequence + Compiler.Graphics qualification; NOT semantic class count |

## Review data schema

For every source, use a reviewed manifest `asset_id, original_sha256, category, morphology_tags, anatomical_front_up, pose_id, canonical_frame_transform, structural_evidence, texture_relink_status, native_ik_court, reference_renders_sha, stylized_render_sha, semantic_surface_receipts, layer_graph_receipt, evidence_origin, independent_reviewer, review_sha, qualification_state`. All fields must be explicit `PASS/FAIL/MASKED/UNKNOWN/N_A` with reasons, not false zeroes; typed names are from a versioned ontology.

## Realistic source rejection policy

Realistic characters may be valid Geometry/TESSA/AXIS/MIRA, even if IRIS Graphics cannot take native realistic RGB. 2D Graphics training requires independently approved native 2D art or source-bound stylized renders, including source identity preservation and silhouette + attachment + alpha parity. One source record can be excluded from Graphics and admitted for mechanical training only once separately reviewed.

## Machine and visual work split

1. **Full-import deep source evidence**: exact SHA/Bone/Mesh/Constraint/Texture source observations, no visual PASS.
2. **Geometry+canonical-pose contact-sheet generation**: original scene (after license/rights source verification), source-bound imported 3D front/back/side and ID/depth/normal masks, not guessed canonical anatomical front. Sample across morphologies; preserve unchanged source and hashes.
3. **Visual category and 2D style candidate review**: human/animal/vehicle/monster/prop, morphology, pose and texture relinking, reviewer receipt; suspected labels remain UNKNOWN until approved.
4. **Semantic Builder using StdGEN++ principles**: source-face labels, full/keep/hide/isolate and cross-view mask parity, source hidden-reveal, no semantics-from-material-name. Need actual original scene, review and renderer, not only previously extracted summary.
5. **Layer/Graphics courts**: causal region perturbation, transparency/draw order, view rotation, independent drawable necessity; no forced target number of layers.
6. **Head-specific PASS**: exact SHA+rights+lineage+geometry/pose/source-semantic/graphics/rig/skin independent family courts and actual consumer loss masks.

**All current review PASS values remain 0 in this new audit until an actual independent court writes evidence.** Generate full per-asset inventory now; then prioritize visual/causal quality gates rather than pretending the automated batch itself visually inspected 4,363 characters.