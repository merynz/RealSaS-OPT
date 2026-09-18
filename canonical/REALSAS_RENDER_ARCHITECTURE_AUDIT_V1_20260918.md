# RealSaS — Reference Rendering / Model–Compiler–Runtime Architecture Audit V1 — 2026-09-18

**Status:** `CLOSED__REFERENCE_BASELINE_DEFINED__ASSEMBLY_CERT_RETIRED__PEPPER_ROOT_CAUSE_OPEN`  
**Audited branch before transaction:** `72588079d90a0b8473af01b96a307cb1c750ea76`  
**Playback execution authority after this audit:** self-hosted `realsas` runner only  
**Colab role:** model training/inference only when GPU is materially required; no parallel playback/render authority.

## Executive finding

The current system mixed two different classes of requirements:

- established graphics-engineering requirements for correct 2D triangle deformation/raster/composition;
- RealSaS-specific product-quality and source-authority proofs.

The retired full-assembly certificate promoted a global boundary/manifold + continuous-embedding theorem into a render prerequisite. That was an architectural category error. A correct rasterizer must be able to draw legal triangle soup, disconnected components and canonical pinch vertices. Dynamic folds/conditioning may still be a real product problem, but they must be diagnosed separately.

The peppering root cause is therefore **still open**. This audit does not claim that visibility or geometry is cleared. It restores the correct ownership boundaries so the next self-hosted render can diagnose the real failure without a false topology blocker.

## 1. "Normal" production flow vs RealSaS

| Stage | Conventional 2D skeletal tools (Spine / Live2D / Rive pattern) | Learned 3D mesh pipeline (CharacterGen + downstream rigger pattern) | RealSaS current architecture | Audit meaning |
|---|---|---|---|---|
| Input | Artist layers/images + manually/automatically authored mesh | Single image -> generated/calibrated multi-view images | Exact multi-view artist observations + camera/evidence contracts | RealSaS input authority is materially richer and multi-view. |
| Mesh creation | Editor/user creates ArtMesh/mesh; triangulation becomes authored data | Model reconstructs SDF/implicit field and extracts explicit triangle mesh | Learned/geometric evidence -> qualified directional drawable mesh under exact observation authority | RealSaS mesh is not merely model output; Compiler qualification owns promotion. |
| Geometry representation before mesh | Usually none; mesh is direct authoring domain | CharacterGen: DINO/image conditioning -> triplane -> SDF -> DMTet mesh | Current pipeline includes observable surface/GSA plus directional render mesh; scientific substrate and drawable mesh remain distinct typed authorities | This split is RealSaS-specific and must stay explicit. |
| Skeleton | Artist-authored in classic tools | Separate downstream rigging (e.g. RigAnything/UniRig/SkinTokens family) | Model/evidence -> `QualifiedSkeletonIR` -> Compiler-bound lineage | Model prediction is evidence, not runtime truth. |
| Skin | Artist weights / editor weighting | Rigger predicts dense per-vertex weights | Model-qualified surface skin -> deterministic support transfer to exact drawable mesh -> `QualifiedMeshSkinIR` | RealSaS has an explicit model-to-Compiler-to-mesh binding seam. |
| Authority owner | DCC/editor file | Generated mesh/rig output file | Compiler canonical IR + hashes + qualification reports | Major architectural divergence. |
| Appearance | Texture/atlas attached to authored mesh | Generated/back-projected texture on generated 3D mesh | Exact source-raster appearance binding per directional mesh corner + donor/source provenance | RealSaS appearance is observation-authority aware. |
| Visibility | Attachment visibility, draw order, masks/clips | Conventional 3D visibility/raster after mesh generation | View-local active attachment + source provenance + explicit semantic order | RealSaS must ensure epistemic uncertainty does not become accidental per-pixel erasure. |
| Motion | Runtime deforms authored weighted mesh | Downstream rig + standard animation | Compiler-qualified motion state -> qualification-owned bake -> Runtime projection | Runtime does not own motion solving. |
| Runtime input | Editor-exported skeleton/mesh/weights/atlas | Mesh + rig asset | Sealed Compiler projection/package; models do not execute in runtime | Strong Compiler/runtime boundary. |
| Runtime job | Deform/raster/compose | Deform/raster/compose | Sample admitted posed geometry + source appearance + compose; no solver/model/completion | Should converge back to standard graphics behavior at this boundary. |
| Failure policy | Tool/editor validation; malformed assets rejected | Generation may repair/postprocess output | Fail-closed provenance/lineage; no unqualified completion | RealSaS-specific and valuable, but must not redefine raster coverage. |

## 2. External comparator findings

### Spine

Public Spine documentation describes slots as explicit attachment containers and draw order as an ordered slot list independent from bones. Spine's editor stores mesh triangulation so runtimes do not need to rediscover topology, and production runtimes may disable backface culling for 2D meshes.

**Lesson for RealSaS:** once Compiler output reaches runtime, topology and composition should be boring, explicit data. The runtime should not infer a new topology theorem.

Sources:
- https://esotericsoftware.com/spine-slots
- https://esotericsoftware.com/forum/d/24360-spine-json-mesh-attachment-attribute-clarifications
- https://esotericsoftware.com/spine-weights

### Live2D Cubism

Cubism exposes ArtMesh geometry, deformers, explicit draw order, clipping IDs and opacity as separate concepts. Culling is off by default specifically so turned-over/deformed 2D polygons remain drawable.

**Lesson for RealSaS:** deformation, draw order, clipping and culling are separate authored/runtime dimensions. Triangle winding is not itself a visibility oracle.

Sources:
- https://docs.live2d.com/en/cubism-editor-manual/concept-of-artmesh/
- https://docs.live2d.com/en/cubism-editor-manual/inspector-palette/
- https://docs.live2d.com/en/cubism-editor-manual/draworder/

### Rive

Rive's documented raster workflow is image -> mesh -> bones -> vertex weights -> animate bones.

**Lesson for RealSaS:** this is the standard downstream deformation pattern. RealSaS differs mainly in how mesh/rig/skin become trustworthy inputs before this point.

Source:
- https://rive.app/blog/new-features-released-mesh-deformation-and-psd-support

### CharacterGen — high-value mesh-generation comparator

CharacterGen is the closest inspected comparator for the learned **mesh-production** side. Its public project describes:

`single image -> image-conditioned multi-view generation/pose calibration -> transformer sparse-view reconstruction -> explicit 3D character mesh -> texture back-projection -> downstream rigging/animation`.

The released 3D code is more specific:

`DINO-conditioned images -> triplane features -> SDF decoder -> Marching Tetrahedra/DMTet -> explicit Mesh(v_pos, t_pos_idx) -> nvdiffrast rasterization`.

Its inference configuration uses a 256-resolution tetrahedral grid and exports an OBJ/UV texture. The code also contains mesh post-processing/outlier removal/UV unwrapping facilities and an empty-SDF fallback that modifies SDF signs to force a surface.

**Lesson for RealSaS:** CharacterGen validates the broad idea that learned evidence can terminate in an explicit triangle mesh before downstream rigging/rendering. But its public pipeline does not document a RealSaS-like authority firewall where generated geometry, skeleton and skin are separately qualified, lineage-bound, and only then promoted into a sealed runtime representation. Its generative repair/fallback behavior also must not be copied into RealSaS's no-hallucination product policy.

Sources:
- https://charactergen.github.io/
- https://github.com/zjp-shadow/CharacterGen
- inspected code: `3D_Stage/lrm/models/renderers/triplane_dmtet.py`, `3D_Stage/lrm/models/isosurface.py`, `3D_Stage/lrm/models/mesh.py`, `3D_Stage/configs/infer.yaml`

### RigAnything / UniRig / SkinTokens

These systems are useful comparators for learned rigging after an input mesh exists:

- RigAnything: mesh -> template-free autoregressive skeleton + skinning.
- UniRig: mesh -> autoregressive skeleton -> bone-point cross-attention skinning.
- SkinTokens/TokenRig: mesh -> unified sequence of skeleton + learned discrete skin tokens.

**Lesson for RealSaS:** their model outputs are close to our G/W evidence domains, but the inspected public interfaces expose complete rig outputs for standard downstream pipelines. RealSaS deliberately inserts Compiler qualification, exact surface/mesh bindings and runtime authority separation.

Sources:
- https://github.com/Isabella98Liu/RigAnything
- https://github.com/VAST-AI-Research/UniRig
- https://github.com/VAST-AI-Research/SkinTokens

## 3. RealSaS flow audit — actual ownership boundaries

Current FIT2 product lineage is structurally:

`multi-view observation -> observable/GSA surface S -> qualified skeleton G -> qualified skin W -> directional mesh M -> deterministic mesh-skin B -> source appearance/provenance A -> component assembly -> motion state -> qualification-owned motion bake -> Compiler Runtime-v4 projection -> sealed .rss -> native runtime`

Important implementation evidence:

- `fit2_current_authority_io.py` hard-pins exact current surface/skeleton/skin lineages and rejects drift.
- `materialize_fit2_p1_skin_binding_v1.py` explicitly forbids a fresh model query at mesh vertices; mesh skin is derived by deterministic surface-support convex transfer from Compiler-qualified FIT2 skin.
- `materialize_p1q_fit2_current_authority_v1.py` preserves qualified vertices/support/skin and permits only frozen exact face-subset repair.
- `current_v4_directional_runtime_v4.py` runs no solver/model/completion; it projects already-qualified product/bake data into view-local runtime attachments.
- Runtime-v4 validates typed source provenance, donor view identity, exact view/slot sets and completion policy before native execution.

This is not a cosmetic distinction. RealSaS is a **model-evidence + Compiler-authority + runtime-consumer** architecture.

## 4. Reference baseline vs current Runtime-v4 implementation

| Concern | Reference-correct behavior | Current RealSaS code | Audit |
|---|---|---|---|
| Pixel center | Explicit half-integer pixel center | `cover_pixel_center(... x+0.5,y+0.5)` | PASS |
| Shared-edge fill | Top-left / half-open convention | `is_top_left` + `edge_accept` | PASS |
| Both windings | Rasterize both unless culling explicit | edge sign normalized; no native cull in current path | PASS |
| Fixed topology | Faces stable through clip | Runtime-v4 asset triangles fixed; frames carry posed XYZ only | PASS |
| UV seams | Duplicate UV vertices allowed but geometry identical | exporter keys runtime vertex by `(source_index,u,v)` and replays posed XY from exact source index | PASS BY CONSTRUCTION |
| UV interpolation | Barycentric | native runtime interpolates UV with coverage barycentrics | PASS |
| Texture sampling | Explicit | clamped bilinear | PASS |
| Source alpha | Transparent texel may discard fragment | `src[3] <= 0` skips fragment | PASS / source-dependent |
| Draw order | Explicit semantic order | per-frame/per-view slot permutation | PASS |
| Equal-depth tie | Explicit deterministic policy | semantic order tie-break | PASS |
| Completion | Explicit only | Runtime-v4 `allow_completion=False`; native completion rejected | PASS |
| Global simple/manifold boundary | Not a raster prerequisite | old assembly cert required it indirectly through boundary-loop proof | **REMOVED FROM ADMISSION** |
| Continuous embedding theorem | Not a raster prerequisite | old assembly cert gated native render on it | **REMOVED FROM ADMISSION** |
| Dynamic fold/stretch/conditioning | Separate deformation/product-quality domain | historically mixed across proof artifacts; requires one self-hosted authoritative diagnostic lineage | OPEN |
| Per-face source provenance | Not common in standard DCC runtimes | explicit DIRECT/OTHER_VIEW/UNDER_RIGID/UNSEEN/COMPLETION | REALSAS-SPECIFIC |
| Model output authority | Usually editor/generated asset becomes input truth | model outputs must be Compiler-qualified and lineage-bound | REALSAS-SPECIFIC |
| Compiler/runtime separation | Often export/runtime split but no epistemic authority compiler | strict: no model/solver/completion in runtime projection/native path | REALSAS-SPECIFIC |

## 5. Critical RealSaS-specific seams to audit next

These are the places where "normal renderer correctness" is insufficient because RealSaS is genuinely different:

1. **Observation -> surface authority**  
   Which observed pixels/surface samples are admitted, and what is unknown?

2. **Surface -> drawable mesh**  
   Does the qualified directional mesh preserve the source raster domain needed for motion, not only rest-frame coverage?

3. **Surface skin -> mesh skin**  
   Does deterministic support transfer preserve deformation continuity across drawable neighbors? This is a Compiler concern, not a rasterizer concern.

4. **Qualified mesh -> appearance provenance**  
   Every face/corner must retain exact source/donor semantics without turning uncertainty into dynamic visibility holes.

5. **Motion state -> qualification-owned bake**  
   The bake must bind the exact mesh identity/topology being rendered. A motion PASS from another mesh lineage cannot be reused.

6. **Bake -> Runtime-v4 projection**  
   Runtime UV-seam duplication must preserve exact geometry; view-local activation must not remove owner-view source faces.

7. **Runtime visibility/composition**  
   ACTIVE attachment, provenance, depth/order and source alpha are separate decisions and should be inspectable independently.

8. **Compiler -> runtime equivalence**  
   The native consumer must render the exact projected posed vertices/faces/UV/provenance rather than derive a second mesh truth.

## 6. Peppering disposition after this audit

No root cause is claimed yet.

The next authoritative diagnostic must run **only on the self-hosted PC runner** and on one exact Runtime-v4 package/frame. It should separate:

- geometric triangle coverage;
- submitted/active face coverage;
- source-binding/provenance availability;
- depth/order acceptance;
- sampled source alpha;
- final alpha.

Only after the native renderer is reference-conformant should those masks be used to attribute pepper pixels to geometry/deformation vs visibility/provenance vs raster/alpha.

## 7. Repository transaction

This audit retires `playback_directional_assembly_cert_v1.py` from active Runtime-v4 render admission and changes admission semantics from "continuous full-assembly certificate PASS" to "reference render input contract PASS".

The following remain separate and intentionally **not** upgraded by this transaction:

- dynamic mesh/deformation quality;
- product visual PASS;
- founder visual PASS;
- unseen/FITK/generalization;
- contact lock, secondary motion, corrective deformation.
