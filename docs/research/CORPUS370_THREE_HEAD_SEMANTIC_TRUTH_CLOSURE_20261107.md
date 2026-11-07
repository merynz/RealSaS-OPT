# RealSaS / Corpus370 — Three-head Semantic Truth Closure V1

**Date:** 2026-11-07  
**Target product:** **Controllable Puppet**, not merely editable or animated mesh.  
**Authority:** PRE-TRAINING ARCHITECTURE DECISION / TEACHER CANDIDATE CONTRACT.  
**Next hard objective:** Finish real Corpus370 family truth closure, NOT begin training with synthetic or speculative labels.

## Decision and precedence

1. IRIS shared Dense3D memory has three *logical supervised output families*: **Geometry, Semantic, Graphics**. These are not necessarily just three Python modules.
2. Semantic is the missing designed-prior bridge: **source-supported component instances, class/concept priors, functional affordances, structural relationship hypotheses**. It does not emit joints, skin weights, IK constraints, drawable authority, or motion.
3. TESSA structuralizes an **animation-ready mesh**, conditioned on IRIS scene/surface and qualified semantic/graphics evidence. Its topology must survive real mechanical deformation, source addressing and component boundaries; it does not author AXIS's rig.
4. AXIS generates a controllable typed RigProgram; MIRA reconstructs skin influences from its coupled codes. Mechanics, Graphics Compiler and runtime execute qualified programs. The Compiler may choose supported representations under capability policy but never invents missing perception.
5. Source truth, **designed teaching prior**, capability contract and representation policy remain typed and independently inspectable. An intentionally assigned prior is a legitimate training target, **not** a source-physics claim.
6. Keep the existing 370 source roster, 352 admitted observation / 18 mask FAIL split, source lineage and actual V5 family courts. Do not assert 352 admitted semantic subjects.

This supersedes the conclusion in `CORPUS370_STDGEN_STYLE_GENERIC_LAYER_COMPILATION_V1.md` that the semantic ontology should be omitted from learning. Anonymous source-locus and presentation evidence **remain useful**, but their indeterminate drawable equivalences are not the source of semantic category labels.

## What transferred from StdGEN

Read actual StdGEN source:
- `blender/blender_lrm_script.py` -> teacher `parse_material()` maps imported object/material conventions to hair/cloth/face/body; controlled full, no-hair and no-hair/no-cloth renders; semantic recolor, alpha/depth/normal and camera.
- `slrm/models/renderer/synthesizer_mesh.py` -> shared triplane queries produce geometry/SDF, RGB and four-channel semantic probabilities.
- `slrm/models/lrm_mesh.py` -> semantic probabilities intervene in SDF extraction via `levels`. Semantic information is a **downstream decision prior**, not just a display-only mask.

RealSaS does not copy the fixed human VRoid classes or a second articulation head. We extend the **teacher idea**: source-linked semantic concepts and controllability affordances, qualified separately from physical/source response.

## V1 ontology: concepts vs functional affordances

Canonical policy:
- `canonical/REALSAS_SEMANTIC_PRIOR_POLICY_V1.json` defines the *designed semantic prior*: examples HAIR_LIKE, CLOTH_LIKE, RIGID_PLATE_LIKE, HANDHELD_PROP_LIKE, TAIL_LIKE, WING_LIKE, ARTICULATED_APPENDAGE_LIKE, SURFACE_OVERLAY_LIKE, UNKNOWN.
- Each concept can imply *particular* positive or negative affordance labels: RIGID_PREFERRED, FLEXIBLE_DEFORMATION, SECONDARY_MOTION_ELIGIBLE, ATTACHMENT_ELIGIBLE, INDEPENDENT_CONTROL_ELIGIBLE, PRESENTATION_SEPARABLE.
- **Unspecified properties are UNKNOWN, not false.** Example: a tail can be flexible and independently controllable without IRIS predicting a tail joint count.
- These are **policy-labelled hypotheses**. Source-native material behavior or finite rig action may support or contradict a choice, but a source with no cloth simulation can still teach a cloth-like secondary-motion affordance if our policy says so.
- New morphologies can add concepts under a versioned policy after an explicit semantic coverage audit. No humanoid skeleton required.

## Semantic TruthBuilder: actual stages

| Stage | Teacher input | Output | Qualification |
|---|---|---|---|
| S0 Original-source parity | V5 exact source SHA + admitted rest observation + Stage50 rest vertices/faces/owner | Exact face/owner binding | Source checksum + rest parity, NO label authority |
| S1 Candidate grouping | Original face connectivity + source owner/material domains | Connected source-face groups | Coverage/non-overlap/source fidelity only; **not yet semantic components** |
| S2 Prior suggestion | Versioned teacher-only object/material keyword hints | Class **hints** or UNKNOWN/conflict | Weak hint; never auto-admit |
| S3 Semantic curation | Reviewed per-face/per-region concept and support evidence | Real per-source semantic class + instance support candidates | Independent review + exact geometry binding; no source names in student |
| S4 StdGEN-like layered render | Exact source face-group labels + camera/pose/alpha | Full / group-off / standalone / semantic masks + reveal/difference observations | Independent native reference parity, causal renderer/camera/reveal court |
| S5 Target conversion | Qualified 3D supports + 2D semantic rasters + view correspondence | Student `concept, instance, affordances` and qualified validity masks | Source-canonical identity, heldout view, leakage-safe lineage |
| S6 Cross-family admission | S5 + Geometry + Graphics + TESSA + response | Subject × target × head authority matrix | Family-specific PASS/MASKED/UNKNOWN; NEVER invented negatives |

**Implemented on this branch (code only; no real source run yet):**
- versioned designed semantic prior + teacher hint policies;
- `models/iris/truth/semantic_source_groups_v1.py`: source face / mesh-material domain connected grouping with candidate-only semantic hints and abstention;
- `tools/truth_foundry/blender_corpus370_semantic_source_groups_v1.py`: replay original import, compare Stage50 exact rest geometry and emit S1/S2 candidates;
- `models/iris/truth/semantic_prior_contract_v1.py`: reviewed, face-addressed S3 proposal and sparse functional-label masks; no family authority;
- `tools/corpus/build_corpus370_semantic_teacher_v1.py`: exact 370 V5 candidate/unknown census; no training authorization;
- synthetic unit/contract tests.

**Not implemented/claimed:** 370 real semantic labels, source-label review receipts, S4 semantic layered image rendering, S5 3D→view mask projection, a qualified semantic family court, new head/trainer, new TESSA geometry/deformation loss, checkpoint, or product proof.

## Target / loss ownership (proposal, not yet shipping)

| Target | Teacher truth | Model/head | Loss and mask | Downstream consumer |
|---|---|---|---|---|
| Surface UDF / source normals / signed solid where qualified | Existing exact source geometry | IRIS Geometry | UDF / normal / conditional SDF+occupancy | TESSA |
| Semantic concept | Curated source-face group concept | IRIS Semantic | Masked concept CE; UNKNOWN rows ignored | TESSA + Graphics/Compiler |
| Component support / permutation-independent instance | Curated connected or manual region mask, exact 3D source address | IRIS Semantic | Sparse masked segmentation + permutation-invariant matching | TESSA / AXIS conditioning |
| Functional affordance | Versioned designed prior derived from curated concept | IRIS Semantic | Independent masked BCE for each supported attribute | AXIS / Compiler / Mechanics |
| Structural relations | Semantic-region adjacency/anchor candidate and independently measured response | IRIS Semantic | Typed relation losses only when a qualified target exists | AXIS, Graphics compiler |
| RGBA / appearance boundary | Existing source/CAA and graphics measurements | IRIS Graphics | Masked RGB/alpha + boundary | Source-preserving Graphics runtime |
| Occlusion / visibility | Full ordered ray stack and source interventions | IRIS Graphics | Complete-stack visibility/ray losses | Graphics/CAA |
| Edit/reveal consequence | Real source conditional renders and held-out interventions | IRIS Graphics | Masked render/consequence losses | Graphics compiler |
| Animation-ready topology | Qualified surface topology + semantic support + deformation response | TESSA | Topology CE/coordinate + held-out deformation/mesh gates | AXIS/MIRA |
| Controls/constraints/SkinCodes | Source mechanical teacher, designed capability goal | AXIS / MIRA | RigProgram + SkinCode/skin objectives (separate) | Mechanics/runtime |

A designed semantic class may not, by itself, dictate that two source triangles are **separate final drawables**. Compiler presentation policy and real ray/CAA consequences still decide the representation.

## Immediate admission gates

- [ ] Reuse exact 352/18 source roster: zero change to original source bytes.
- [ ] Run S0/S1/S2 on morphology-stratified original assets, with unknown/conflict counts; no training promotion.
- [ ] Audit labels on human, quadruped, fish, dinosaur, winged and non-skeletal cases; measure macro coverage and false-positive source hints.
- [ ] Build explicit reviewed semantic support and 2D layered render for each selected source, then independent source truth court.
- [ ] Extend V5 authority matrix with separate Semantic concept, component support, functional prior, and render/reveal families.
- [ ] Finish Geometry and Graphics pending family courts without blocking independent valid targets.
- [ ] Version semantic loss/trainer materialization to consume ONLY qualified semantic masks and exact source-bound subject identity.
- [ ] Evaluate TESSA animation-ready topology under real native deform/replay before calling it ready.
- [ ] Only then qualify Corpus370 training split, IRIS three-head and TESSA training.

**Do not mistake new source group candidates for real semantic closure.** The semantic teacher—not another drawable-ownership hypothesis search—is now the P0 research work on the existing Corpus370 truth closure path.
