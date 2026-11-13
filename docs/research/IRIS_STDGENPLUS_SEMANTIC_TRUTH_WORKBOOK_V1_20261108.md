# IRIS Semantic Truth Foundry — StdGEN++ Research Workbook V1

**Status:** implementation guide and source-specific truth policy PROPOSAL; no supervised semantic labels minted; no training/materialization authority.  
**Parent:** Truth Foundry PR #395; existing semantic candidate work PR #377, proposed six-head targets PR #392.  
**Scope:** IRIS Geometry / Semantic / Graphics interaction and Compiler.Graphics qualification, not a new model architecture decision.

## Keep these primary references open throughout IRIS work

1. **StdGEN++ (2026)** — https://arxiv.org/html/2601.07660 and https://arxiv.org/abs/2601.07660, *StdGEN++: A Comprehensive System for Semantic-Decomposed 3D Character Generation*, He et al. (IEEE TPAMI, 2026).
   - Section III: Anime3D-EX training evidence; subsection *Hierarchical Semantic Renderings*; full model / body with clothing / base body, multi-view, head crops, texture layers.
   - IV-A: canonicalized single-view → six synthetic RGB + normals (product input, **not all source-truth views**).
   - IV-B: dual Fullbody / Facial LoRA semantic+color+geometry triplane fields; a design reference, NOT justification to change IRIS model now.
   - IV-C: semantic-equivalent NeRF/SDF surface extraction and coarse-to-fine acceleration; **conditionally applicable only when our geometry/SDF validity is independently certified**.
   - IV-D: Stage 1 visible semantic maps, Stage 2 selected semantic subset/inside reveal render supervision, Stage 3 mesh/normal/depth and surface quality losses.
   - IV-E: facial textures as separately supervised eyelash/eyebrow, skin/eye white, iris; **texture decomposition ≠ generic articulated drawable identity**.
   - IV-F: layered refinement and collision preservation; source implementation and RealSaS-compatible counterpart must be checked before adoption.
2. **StdGEN original CVPR 2025** — https://stdgen.github.io/ and **public released original code** https://github.com/hyz317/StdGEN.
   - Public original code: `slrm/models/lrm_mesh.py`, `blender/`, and README's Anime3D++ rendering/semantic-map instructions.
   - Do not misrepresent the StdGEN 2025 public repository as the complete reproducible StdGEN++ 2026 training pipeline. Independently verify released ++ code and weights before claiming reproduction.
3. RealSaS PR #377 `models/iris/truth/semantic_source_groups_v1.py`: connected face domains + **privileged name hints only**, current no semantic training target.
4. RealSaS PR #383 `canonical/REALSAS_GRAPHICS_2D_ART_SUPERVISION_CONTRACT_V1.json`: source RGBA and reviewed stylized 2D image supervision separate.
5. PR #395 `tools/corpus/truth_foundry/target_matrix_v1.py`: full asset roster and 37 candidate routes, **training masks closed**.

## What StdGEN++ actually provides

The paper demonstrates geometry, color and per-semantic-part implicit fields. Its hierarchical training render states are source-grounded as (i) full character, (ii) body plus clothing, (iii) base body only. This is a **three-tier anime-human-specific source ontology**, not a general license to label every segmented Blender object as hair/body/clothing or every layer as an independent animator drawable.

For our source corpus, we generalize the **mechanism**, not those three categories:
- For each independently reviewed region or region set P, produce source-coherent render configurations `FULL`, `KEEP_SET(P)`, `ISOLATE(P)`, `HIDE_SET(P)`, and optional `EDIT_REVEAL(P,pose)`.
- Every view/pose/render receipt records the **exact original SHA, local face IDs and canonical mapping, camera, pose, visibility controls, RGBA, depth, normals, face-ID pixel map, and mask of known/unknown support**. Where a render or correspondence cannot be trusted, mark UNKNOWN; no fabricated label or alpha.
- One student input image is not one teacher observation: multiview source-truth render evidence may be necessary even if final inference uses a single front image.
- Only source geometry and independently reviewed regions can supervise 3D semantics. Object names, materials, bone names and weak ML masks are **annotation proposals**, never source truth by themselves.
- Source-native rest/neutral pose must be measured; canonicalized A-pose cannot be imposed on nonhumanoids.

## Three separate non-interchangeable questions

| Layer | Claim | Independent evidence | Route |
|---|---|---|---|
| SOURCE | A particular original source face belongs to a reviewed concept / instance / morphological component | Source SHA + source face/corner address + reviewer receipt + geometry parity | IRIS Semantic 3D region, instance, concept |
| OBSERVATION | From a particular view and pose, reviewed surface region P produces a particular mask, visibility and geometry/ray ordering | Source-bound renderer, camera, pose, face ID/depth/normal masks, full vs hide/keep pairs | IRIS Semantic 2D consistency; IRIS Graphics source visibility |
| CAPABILITY/POLICY | This region needs to become one or multiple independent drawable units when controls/pose/view change | Edit intervention, distinguishable consequences, compositing and stable independent control tests | Compiler.Graphics **qualification** with IRIS Graphics candidate evidence |

Semantic concepts are not bone constraints, skin weights, inverse kinematics, painter ordering or drawable count.

## Source semantic truth stages

**S0. Index/source identity:** master+core source record, full SHA / lineage / rights / holdout state; keep unresolved ambiguous identity quarantined.

**S1. Physical source face space:** source mesh object, material slot, face/triangle mapping (original polygon/corner addresses maintained), connected islands, attachments and modular open faces. Source importer re-triangulation must have parity receipt; a mesh object is not an anatomical part and one disconnected island may include multiple concepts.

**S2. Teacher *proposals*:** contiguous face-domain proposals from geometry/connectivity, material boundaries, bone influence similarities, texture features and object/material-name suggestions. All candidates inherit UNKNOWN concepts; names are privileged annotation hints excluded from student features. Special morphology (wings/tails/serpents/spiders) may need **region unions and splits**, not just mesh islands.

**S3. Independent semantic review:** one or more non-overlapping, source-addressed region instances with versioned concept/role from a scalable ontology; uncertain faces are UNKNOWN (not a negative); overlapping features must use separate typed annotation domains, not silently violate exclusive-face labeling.

**S4. Ground-truth intervention renders:** from the SAME original SHA, choose source-valid canonical pose and camera(s); render FULL, KEEP_SET, HIDE_SET, ISOLATE; preserve exact source face IDs/depth/normals and visible alpha for all configurations. Pairwise/subset reveal tests must show **real source-occluded faces**—never use inpainted guesses as source truth.

**S5. Semantic truth courts:** replay and compare source SHA, view/pose/camera matrix, source-face raster ID, semantic masks, all layers' union, visibility/draw order where observed, labels with review source. For invalid or unreviewed patch, abstain. Explicit reviewed negatives only if complete region exhaustivity is certified.

**S6. Head-specific teachers:** only after independent source-face semantic & view court, produce source semantics for IRIS Semantic and optional Graphics evidence; stylized presentation requires separate visual quality/art/source correspondence court. No AXIS control / MIRA skin / Compiler drawable PASS comes from semantic labels alone.

**S7. Training and consumer contract:** bind `semantic_support`, `semantic_instance`, `semantic_view_mask`, uncertainty/abstain masks, head losses, source lineage split and true nonleaking student input. Keep all `loss_mask_valid=false` until explicit independent admission; this document cannot grant it.

## Adapting the StdGEN++ SDF and NeRF extraction to RealSaS

StdGEN++ Eq. 3 and Eq. 12 construct semantic-equivalent SDFs from signed source fields and semantic probabilities. Our modular/open meshes often **do not have a valid global solid signed distance**. Thus:
- Reuse semantic-equivalent SDF only where sign topology has a source-qualified closed-solid court and no false interior/exterior is asserted.
- Default to **source-face-region, surface UDF, explicit layered visibility, ray compositing and face-ID raster supervision** for modular/open geometry.
- If a semantic boundary crosses a triangle, preserve a mapped surface split/soft supervision with explicit provenance, not false exact source-face identity.
- TESSA structuralization and AXIS/MIRA edit consequences remain independently qualified; semantic extraction cannot replace them.

## Acceptance fixture (before full Master)

A representative morphology-stratified pilot must include at least: a humanoid with layered hair/cloth, quadruped, winged creature, tail creature, articulated hard-surface/armor, and modular source with prop/attachment. Use actual licensed source only; not hypothetical screenshots. Each specimen needs:
- Source-front and 3D canonical orientation court, geometry rest/face IDs and appearance.
- At least one reviewed positive semantic region and an explicit UNKNOWN support region or complete negative-region review, tracked separately.
- A full vs remove/hide image pair and a newly revealed real source-surface example when the source allows one.
- Cross-view semantic consistency and no target/student name leakage.
- A downstream Compiler.Graphics **candidate** drawable decision with measured positive/negative edit consequence, explicitly not source semantic truth.

## Immediate product-data decisions

- **Corpus membership is per-source; training eligibility is per-target, per-head.** Stylized human/character candidates may serve all head families only with independent court coverage. Realistic source models may provide mechanical/geometry supervision without direct 2D-stylized Graphics training.
- **Golden Corpus** means a *small subset with actually complete, separately reviewed cross-head truth*, not an assumption that all 370 Core models already have it.
- StdGEN++'s 10,811 anime characters and 3-layer dataset are methodological examples, **not free additional commercially licensed RealSaS training samples**. Verify each original's underlying rights.
- Source-face semantics and graphical drawable necessity are separate typed evidence. The representation choice may be delayed to Compiler.Graphics after geometry, mechanical and intervention evidence are available.

## Open implementation gaps (track in PR)

1. Semantic source-court output schema with source face/polygon/corner lineage, per-region proof and typed UNKNOWN.
2. Source-bound keep/hide/full/isolated renderer and verified face/pixel visibility + new reveals for different morphology.
3. Multi-view semantic masked supervision composer and independent parity tests.
4. Cross-view/prompt/input semantic label leak prevention; original artist naming teacher-only.
5. IRIS Semantic head decoder/loss integration in real trainer only after courts are admitted; no architecture freeze inferred from the paper.
6. Compiler.Graphics interaction consequence and drawable policy/consumer gates.

**Research integrity:** Paper description ≠ verified public implementation. No new semantic samples, training PASS, stylized 2D art PASS or product-ready drawable count is claimed here.


## Clarification: StdGEN++ is a source-side **TRUTH BUILDER** reference as well as a student training reference (2026-11-08)

Do not describe StdGEN++ solely as "how to train from already separated labels." The paper's **Anime3D-EX data construction** explicitly produces hierarchical supervised render configurations: full character, body + clothing, base body; matching head-focused crops and pixel-aligned facial feature/texture layers. Its stage-2 learning uses semantic-subset rendering to reveal otherwise occluded source surfaces, then stage 3 adds image/mask/semantic/depth/normal losses.

**Our generic teacher pipeline:**
1. Recover immutable original SHA, qualified source mesh/face addresses, authored native materials/skin, rest-pose/camera and provenance; license and lineage remain separate courts.
2. Ingest PartVerse-XL `*_face2label.json` and `*_segmented.glb` as **3D part-instance proposal**, not anatomical or functional semantic truth. Verify full/part GLB face ID and geometry parity. If mapping changes through normalization/retriangulation, record explicit source triangle/corner barycentric address and unsupported faces as UNKNOWN.
3. Review/derive concept, role, instance and part-group hierarchy from source geometry, UV, material, rig influence, motion and cross-view evidence. A many-to-many part ↔ semantic relationship is allowed; UNKNOWN faces remain UNKNOWN. Do not force human body/clothing/hair categories on nonhumans.
4. Create source-bound candidate sets `P` for `FULL`, `KEEP_SET(P)`, `HIDE_SET(P)`, `ISOLATE(P)`, and intervention poses; each generated receipt contains source SHA, pose/rest, camera, chosen face/part IDs, RGB/RGBA, alpha, **global pixel face-ID + depth + normal + multi-hit visibility where available**, 3D support mask, genuine newly revealed source surfaces and exact generator version.
5. **Court the teacher itself**: full/part recomposition parity, 3D→2D ID consistency, hide/set reveal with original hidden face, no leakage from artist part labels to student input, source camera and rest fixed, no fictitious occluded completion. For open and modular meshes, never assert global sign/SDF without separate sign qualification.
6. Only then derive per-head losses: IRIS Semantic 3D instance/role/concept and multi-view pixel mask; IRIS Graphics source visibility, appearance and layered candidate relations. A **distinct source-bound 2D stylization court** is mandatory for student presentation training; native photoreal rendering is not 2D game-art ground truth.
7. Drawable count, painter order and independent editability are **Compiler.Graphics representation decisions** requiring intervention and counterfactual evidence from 3D pose, camera, control and source alpha. Neither PartVerse part count nor StdGEN++ human semantic count determines them.

**Tests before admission:** at least one per-source nontrivial `HIDE_SET` newly revealed original face; source ↔ part source-face parity PASS; pixel semantic/face-ID consistency PASS; camera/pose provenance PASS; explicit full/partial/unknown semantic region completeness; per-head training loss mask stays false until independent courts. This is a proposed RealSaS derivation policy; **no real PartVerse full/keep/hide audit is yet proven PASS by this document**.

Primary paper: https://arxiv.org/html/2601.07660 (Section III Anime3D-EX and Section IV-D, stages 1–3). PartVerse code: https://github.com/hkdsc/fullpart/blob/main/partverse/get_infos.py (`_face2label.json`, multi-view face ID and visibility render). 
