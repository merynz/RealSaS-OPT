# Corpus4008 V1 deep control, native-scene and stylized-presentation audit

Date 2026-11-07. Branch `corpus4008-deep-control-and-lookdev-audit-v1-20261107` from PR #379 `41e68159`.
Status: **pre-training imported observation + experimental lookdev; independent source and behavior courts NOT PASSED.**

## Owner separation

- **Source**: original bytes and full SHA; importer graph, per-bone transforms and source-side animation records. Imported evidence is not a proof of authored-source IK, because FBX/glTF exporters may bake or discard controls.
- **Designed prior**: motion/control affordance, semantic role and capabilities come from versioned product policy, not "deform=false" or bone name.
- **Truth**: native source-format or independently reproduced Blender/scene control behavior and observed pose-to-geometry consequences, verified by source/camera/identity courts.
- **Representation policy**: selection among independently proven control/mesh/layer alternatives; it cannot invent functional IK or semantic identity.
- **Presentation experiment**: shader/workbench-only, not source RGB authority. Geometry and pose stay identical, camera deliberately source-space unqualified.

## Control receipt V1 (new producer)

`tools/corpus/blender_corpus4008_control_profile_v1.py` returns an immutable `RealSaS.Corpus4008ImportedControlProfile.v1` JSON by original SHA + canonical asset/candidate identity. Captures:

1. All imported bone/armature counts, deform flag, nondeform candidates, parent graph, bone collections/groups, rest/pose matrices, lock/limit/IK stiffness and numeric custom-control properties.
2. Pose-bone and object constraints, type/influence/enable, IK chain count, target/subtarget, pole target/angle, owner/target spaces.
3. Driver channels, owner/data path, variable targets, transform and bone targets, expression hash only; do not execute driver source text.
4. Imported actions from layered channelbags or legacy F-curves, channel paths, keyframe counts and frame ranges; NLA strips/tracks and active object state.
5. Imported source cameras, world matrix and optics. **No anatomically qualified view**.
6. Explicit `UNKNOWN`: authored IK absence, functional IK, FK/IK switch, controller semantics, retargetability, source rest parity, source up/front, pose deformation and character-vs-prop.
7. Exact source SHA, Blender importer version, all training and semantic-family authority flags `False`. Abnormal sizes fail closed, not silently truncated.

**Gap to fill after V1:**
- Raw format-native controller extension/parity for FBX/glTF/BLEND and source-action dependencies, action-slot binding and authoring external rig programs. This should distinguish absent capability from import loss.
- Independent intervention court: perturb control targets, sample source action frames, measure constrained bone world matrices and resulting per-face/vertex deformation; verify heldout target response, attachment drift, contact stability, collision and draw-order.
- Explicit per-controller influence graph: `control -> driven_bone -> mesh/vertex -> attachment -> rendered_pixel`, with reproducible perturbations and negative examples.
- Determine IK vs FK from measured response, not name or imported IK presence; distinguish static rig hierarchy, runtime solver availability, source authored controller and our future compiler's controllability contract.
- Per-asset skeletal scale, axis/roll discontinuity, stretch, negative determinant, constraint cycles, graph disconnectedness and target resolution. Mark conditional and invalid with reasons.
- Independent control family court can PASS only when evidence bindings and source-to-canonical transformations are independently verified.
- Morphology balanced sampling before any readiness conclusion. Reference pilot is FBX/GLB/OBJ three sources, not 4,008 source courts.

## Presentation pilot V1

`tools/corpus/blender_corpus4008_lookdev_study_v1.py` renders the SAME original-byte source import/rest and SAME source-space orthographic diagnostic camera three times:

- `SOURCE_MATERIAL_DIAGNOSTIC`: CPU Cycles with basic diagnostic lighting. This is NOT native-source RGB parity, even if textures appear.
- `STYLIZED_FLAT_DIAGNOSTIC`: CPU Cycles emission-style flattened base color with linked textures if available, which may discard textures; NOT source-appearance truth.
- `STYLIZED_CEL_3BAND_DIAGNOSTIC`: CPU Cycles three-band cel lighting from normals, multiplication with source base-color graph where linked; NOT source-appearance truth.

All PNGs receive SHA256 hashes, and importer version/camera transform are included in the JSON. One source-space diagnostic view is never labeled FRONT/UP. **NO training data admitted.**

Recommended next experimental styles after a visual quality court: 3- and 5-band toon with source base-color preservation, depth/normal-sensitive ink silhouette, source alpha parity, cel shadow thresholds and explicit accessory/weapon semantic overlays. Compare across humanoid, fish, quadruped, winged, dinosaur, prop/surrounding scene, not solely one appealing character.

## Required per-asset visual dossier

Original scene/pose with source transform and character-space candidate frame side by side; neutral/rest and sampled authored action frames; material/native RGB candidate; toon/flat/outline exploratory views; alpha, normal, depth, face/owner ID, visibility and layer-off/reveal consequences. Record:
- Image SHA, source revision, original texture dependencies, renderer version/device/engine, lights, camera intrinsics and extrinsics, resolution, color management, native vs stylized label.
- Pose ID and which bones/vertices/attachments actually moved. Body bounds, prop bounds, crop occupancy and unexpected detached geometry.
- Source face and pixel coverage, alpha/texture fidelity, view-dependent occlusion/correct front determination.
- Human review flag and independent numeric court; unknown cannot silently promote to PASS.

Do not reject valid open meshes merely for being open. Do not generate watertight solid/SDF truth, semantic classification, controller roles or authored physics from a shader image.

## Workstreams / parallelism

A. **Source recovery** (PR379): finish 32 source shards, licensing/current-source verification, SHA and dedup lineage; protect 169 external holdouts.
B. **Scene and control anatomy** (this branch): real pinned Blender source-graph imports, exact controller/IK/driver/NLA/channel observations and intervention evidence.
C. **Graphics/semantic** (PR377 and separate experiments): original-material matched views, style-only comparison, StdGEN++-informed semantic source-face layer proposals, hand-checked evidence.
D. **Truth/readiness matrix** (PR327): per-subject per-target PASS/MASKED/FAIL/UNKNOWN independent courts, source→canonical→loss consumer binding, heldout split safety; no admission until closure.

Each workstream writes typed, SHA-locked receipts; **only the admission coordinator may join evidence into a model-target PASS**. Artifacts are source-owned; PRs remain isolated and unmerged pending independent review.

## First measured exercise

The hosted workflow `corpus4008-deep-control-and-lookdev-v1` gates private source inspection on the configured Drive secret and pins exact original FBX, GLB, OBJ SHA + Blender 4.5.14. It writes control JSON for three sources and three lookdev PNGs for KayKit under a private Drive quarantine path, never public artifacts. A separate validator checks source/asset joins, image hashes, false authority and internal counts.

**Workflow success alone does not claim full factual source rig functionality, camera correctness, source-RGB fidelity, meaningful toon style or training eligibility.**


## Real imported-source pilot closeout (2026-11-07)

Hosted private source run: https://github.com/merynz/RealSaS-OPT/actions/runs/42379725713 — **SUCCESS**, both synthetic authority-contract and real source/pixel audit jobs. Exact SHA-verified originals; public CI never holds source PNG artifacts.

Canonical private Drive original receipts: https://drive.google.com/drive/folders/17TCIU4z2EpMSvUSOCfWiOMYNIEZEuq6Q

| Imported evidence | Objaverse FBX | KayKit GLB | Quaternius OBJ |
|---|---:|---:|---:|
| Armatures | 1 | 1 | 0 |
| Pose bones | 34 | 41 | 0 |
| Imported IK constraints | 0 | 0 | 0 |
| Imported driver channels | 0 | 0 | 0 |
| Actions | 1 | 76 | 0 |
| Action F-curves | 349 | 29,252 | 0 |
| NLA strips | 0 | 76 | 0 |
| Imported cameras | 0 | 0 | 0 |

**Important negative distinction:** none of these three imports exposes an IK constraint, but authored source IK absence is **UNKNOWN** and functioning source IK is **UNKNOWN**. The first two imports expose no bone collections/custom control props and every imported bone has use_deform=true; this does not prove no real control rig existed in upstream authoring.

Known source appearance blockers already observed by PR379 source pilot: the Objaverse FBX import reports four missing external images; Quaternius OBJ import cannot locate `Zombie_Male.mtl`. Neither is a qualified native source RGB teacher.

Actual Master control coverage was materialized from SHA-pinned original selection: **3,993** original selected rows, **3** independent imported-control profiles, **3,990 NOT_YET_MEASURED**. The 169 external holdout identities and historical splits are preserved, and 15 QA-only Khronos entries remain outside the 3,993 selection JSON. This is a *coverage census*, not a family PASS census.

The first render composition had only 2.9% alpha occupancy and 1.47% flat-vs-cel RGB difference, so we classified it as a **lookdev quality failure** despite a successful PNG export. The corrected CPU Cycles source-material/flat/3-band cel study (same original SHA, rest, source-space diagnostic camera, mesh) has an independent Pillow-based numeric pixel court:
- Alpha occupancy ~9.8%; bounding-box width 52.2%, height 44.7%.
- Flat-vs-cel mean RGB difference ~5.40% of 255 (masked shared pixels).
- Flat/cel alpha mask Jaccard 1.000; source-vs-style Jaccard ~0.992.
- `numeric_pilot_gate_pass=true`. This is **not** human style approval, native source material parity, character-space camera admission, or training authorization.

Stylized graphics are exploratory `STYLIZED_...` records; do not overwrite source RGB or assume 2D-art appearance. Future contact sheets must cover more morphologies (including open components, non-human and attached props), held-out poses, readable body/accessory framing, source alpha and rendered evidence correspondence.

**Remaining P0:** independent native authored-rig source graph/intervention court; pose/IK/FK behavior; per-vertex deformation influence; source material/texture parity; anatomically qualified cameras; full multiview visual courts; per-target semantic/graphics/motion admission. No merges or model training from this branch.
