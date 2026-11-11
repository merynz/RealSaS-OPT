# RSS.vNext — Unified Character Compiler: architecture freeze V1

**Decision date:** 2026-11-09  
**Status:** **FROZEN DESIGN** — **not** an implemented, trained, unseen-qualified, or production-promoted model.  
**Normative machine contract:** \`canonical/RSS_VNEXT_UNIFIED_CHARACTER_ARCHITECTURE_FREEZE_V1_20261109.json\`.  
**Base source:** main \`9f9f53674777e961488262182260f039ea06ed77\`.  
**Product target:** source artwork -> coherent 3D mechanics -> editable, controllable 2D-presented puppet. The persistent puppet, not a video or framesheet, is the product.

This document freezes **architecture and responsibility**, not model weights, head count, sequence width, optimizer, data admission, scientific outcomes or final exported proof. Design freeze must not be interpreted as a green experiment. No existing model release, ProductRevision, running Corpus370/Truth Foundry workflow, source bytes, existing V3.2 freeze or user-facing runtime is changed here.

## 0. Executive decision

**One unified character model; four specialized learned experts; one persistent shared carrier; one compiler authority.**

1. **IRIS**: scene understanding through shared dense 3D evidence, **Geometry / Semantic / Graphics** logical supervised families.
2. **TESSA**: **animation-ready structural mesh expert**; jointly resolves component/chart topology, mechanically relevant partitions, seam/attachment structure, surface fidelity and deformation-sensitive geometry. It is not merely a detached autoregressive triangle transcription service.
3. **AXIS**: learned **RigProgram** expert, generating joint/rest frames, DOF, kinematic chains, authoring handles, constraints, effectors and socket/control relationships. Skeleton generation alone is explicitly insufficient.
4. **MIRA**: sparse influence and **SkinTokens** expert sharing causal rig state with AXIS, directly operating on the exact TESSA mesh and per-vertex structural carrier.
5. **Compiler/Mechanics/Graphics runtime**: qualify, bind and execute generic declared mechanisms and 2D compositing. Never become a hidden second perception network or an unlimited repair service.

**Models communicate by consuming and extending the same typed object, \`SharedCharacterCarrierIR.vNext\`.** They may run in an ordered implementation schedule, but they must not independently infer four incompatible identities for the same character. One combined logical model does **not** imply a single unbounded end-to-end training step or a single monolithic tensor.

Product inputs: two user-supplied front/back views are a targeted product mode. Existing eight-view source research/training apparatus remains valid and **is not** silently reinterpreted as two observations. Generated auxiliary views are predictions, never original observations.

## 1. Shared Character Carrier — the common language

\`SharedCharacterCarrierIR.vNext\` is an **evidence-preserving typed state graph**, backed by immutable artifact blobs and versioned expert deltas, not a small token vector into which source information is irreversibly compressed.

**Persistent domains** (each may be \`UNKNOWN\` or independently masked):

| Domain | Payload | Authority / invariant |
|---|---|---|
| Observation | Original image refs, source SHA, view/camera and art/alpha references | Immutable; teacher-only original mesh/rig/skin must not leak into inference |
| World frame | Source-to-canonical transforms, camera conventions, unit/scale, rest and local frames | Invertible mappings with exact revision references |
| Scene | Dense3D \`Z\`, queryable local \`H(x)\`, observations and geometric coordinates | IRIS-predicted, never categorical topology truth |
| Geometry | Surface/UDF, oriented normals, sign/occupancy **only where qualified**, geometry validity | Open/modular surface does not become invented globally signed solid |
| Semantic | Spatial region/instance support; concept and functional-affordance distributions; unknown masks | IRIS-predicted; class/concept prior is not a physical joint or drawable |
| Graphics | Appearance/RGBA, projected boundaries, view and ray/visibility evidence | IRIS-predicted or source-observed, with distinct authority |
| Surface identity | Stable surface patch IDs, vertex/face/chart IDs and source/image correspondence | Identity must survive remeshing by **explicit mapped correspondence**, not shared index assumption |
| Structure | Mesh candidate, adjacency, charts, seams, component relations, rigid/flexible/attachment candidates | TESSA-owned; structural relation is not the final rig parent |
| Mechanics | RigProgram, joints/frames/DOF/constraints/effectors/sockets and per-vertex skin codes/influences | AXIS/MIRA co-owned by disjoint typed fields, exact common mesh lineage |
| Provenance | Source, parent hashes, producer version, uncertainty, qualification, edit-consequence evidence | Append-only; \`UNKNOWN != false\`; no silent promotion |

**Identity discipline:** During supervised truth building, correspondence may reference *real original source faces + barycentric coordinates*, exact source object IDs and native transform frames. At inference from artwork, those original 3D faces **do not exist**. The model instead produces *predicted* persistent surface patch/region IDs and source-image/ray correspondence, explicitly typed as predicted. No fake original-source face IDs are allowed. Source-authoritative appearance and predicted completion must remain distinct.

**Graph-level relation kinds** are typed and independent: \`geometric_adjacency\`, \`qualified_seam\`, \`structural_independence\`, \`rigid_attachment_candidate\`, \`deformable_continuity\`, \`contact_candidate\`, \`projected_overlap\`, \`visual_instance_membership\`, \`controller_attachment\`, \`drawable_relation\`. Equality of XYZ, shared material, mesh object name, edge contact and co-visibility are *not* interchangeable evidence for these relations.

**No-loss policy:** no producer deletes an earlier observation/evidence blob when publishing a more compact latent. Compression in neural features is necessarily lossy; therefore the promise is **no information silently destroyed at typed interfaces**, not mathematical losslessness of embedding vectors. One source fact can legitimately condition multiple expert losses with different validity masks.

## 2. IRIS — shared scene, three logical learned families

IRIS owns a global/camera-aware Dense3D scene, a queryable local feature field, and three **logical** heads. The term "three heads" is an architecture of responsibilities, not a constraint that these are exactly three Python \`nn.Linear\` modules.

- **Geometry**: geometry field or validated unsigned surface distance, source-consistent normals, conditional signed field and occupancy, local geometric support, uncertainty/coverage. Do not force global sign on open or non-manifold assets.
- **Semantic**: persistent region/instance probabilities, conceptual affordances such as limb-like, rigid-like, cloth-like, appendage-like, prop-like, and *proposed* region correspondence across views. A concept may be \`UNKNOWN\`. Teacher object/material labels are weak hints and prohibited as privileged inference inputs.
- **Graphics**: texture/appearance RGBA, 2D and projected structural boundaries, edit/reveal support and view-dependent ray/visibility evidence. The source art remains appearance authority.

**Freeze boundary:** IRIS Semantic may propose **what region / instance / behavior family** a surface belongs to. It does not declare a weld, split final mesh charts, choose a bone hierarchy, emit skin weights, or mint a drawable merely because a class was predicted.

StdGEN++ transfers here: a joint geometry/appearance/semantic field, multi-configuration hierarchical rendering supervision, and semantic-aware surface extraction. Its fixed human body/clothing/hair ontology is **not** imported; varied morphologies require variable-cardinality region identities and typed affordances.

## 3. The differentiable surface bridge — shared, not a fifth model

An IRIS–TESSA bridge obtains candidate geometry and regional surface witnesses from the shared carrier. **Semantic-aware FlexiCubes is the canonical differentiable *scaffold* where sign is valid.**

- Sample geometry field and region probabilities in the exact common camera/world frame.
- Extract qualified semantic-aware provisional surfaces; maintain region ID and source/predicted surface-address mapping.
- Expose differentiable surface vertex and extraction parameters to geometry, view-render, deformation and multi-pose training losses where the operator is actually differentiable.
- For nonwatertight, open, thin, modular, multi-owner assets with unqualified global sign: use UDF/explicit oriented patch surface candidates and preserve owner separation. **Do not invent signed interior** merely to make FlexiCubes run.
- FlexiCubes is not the final mesh owner. It does not replace TESSA structural assembly, mechanical topology decisions, or compiler qualification. Do not silently enforce source watertightness via cross-owner weld.

This bridge is **not** an optional competitor to TESSA or a new separate trainable product head. It is a representation/gradient path shared by IRIS geometry and TESSA mesh refinement.

## 4. TESSA — animation-ready structural mesh expert

TESSA reads **the actual IRIS shared \`Z/H\` features**, qualified \`G\`, semantic region probabilities, Graphics boundaries, and the bridge's candidate surfaces; no 17-D or other compressed replacement of shared scene memory is allowed to discard the only copy of structural context.

It emits **\`StructuralMeshCandidateIR.vNext\`** with:
- vertices/faces, components/charts, explicit topological connectivity and stable face/vertex/region IDs;
- nonweld/seam/intersection/contact relations, attachment-**candidate** endpoints, independent rigid islands and deformable regions;
- rest-frame and camera binding, region-to-face support, surface addresses and remesh correspondences;
- expected deformation suitability and qualified/measured mechanical quality evidence;
- all original IRIS region/appearance support retained by reference, never flattened into a hard partition.

TESSA's final *learned* responsibility is **an animation-ready rest mesh**: a mesh with topology and region placement appropriate for an unknown but supported rig, including articulated bends, extreme deformation, rigid accessory islands and flexible/secondary-motion parts. This does **not** mean TESSA already knows joints or IK targets. AXIS still builds the rig; MIRA still binds skin.

**Attachment split:** TESSA predicts geometric/structural boundaries and likely attachment sites; AXIS chooses joint/control/socket parent, DOF and constraint realization; Compiler validates the physical binding. Thus attachment is *structurally learned when the structure is made*, while the later rig is not hallucinated ahead of its owner.

**Model design:** retain learned mesh structuralization and the existing autoregressive sequence expertise, but **do not enshrine all topology generation as a disconnected token-only decoder**. Structural graph and differentiable extracted-scaffold refinement are first-class TESSA outputs. Discrete topology construction remains learned; coordinates and surface extraction are differentiable where valid.

**Animation-aware supervision:** source-qualified rest/posed surfaces, attachment behavior, region deformation, normals, strain/Jacobian, fold-through, collision, seam separation and projected silhouette. A selected static teacher sequence is insufficient by itself. Source rig/skin/motion may be privileged training **targets** but not inference input.

**Gradient truth:** multi-pose loss may propagate through LBS, differentiable rendering and continuous extracted mesh positions/parameters. Discrete face-token choices, hard chart splits and control event selections do **not** become differentiable simply because a loss exists; train these with masked structural/sequence/relationship supervision and source-qualified candidate-quality targets. Do not claim full connectivity-gradient proof without a real relaxation/estimator. Mechanical qualification is separate from training loss.

## 5. AXIS + MIRA — one coupled rig/skin/control construction

Adopt SkinTokens/TokenRig's key insight: **joint skeleton + skinning must share a causal structural representation**, not be serially produced from unrelated reductions. Preserve our separate AXIS and MIRA parameter groups, expert names and release artifacts because their responsibilities differ.

**AXIS must extend beyond TokenRig.** SkinTokens publication guarantees a skeleton hierarchy + weight field, not native DCC controllers, complete inverse-kinematics program, target/pole authoring handles, IK/FK switches, attachment behavior, support contacts or secondary-control policy. A bare skeleton+weights therefore **cannot** satisfy BUSTED's full-control contract.

Freeze \`RigProgramIR.vNext\` as a typed, source/canonical-mesh-bound control program with events/objects for:
- \`JOINT\`, \`LOCAL_REST_FRAME\`, \`DOF\`, \`LIMIT\`, \`CHAIN\`, \`EFFECTOR\`, \`POLE_TARGET\`;
- \`CONTROL_HANDLE\`, \`CONTROL_TARGET\`, \`PARENT_SPACE\`, \`IK_FK_MODE\`, \`CONSTRAINT\`;
- \`SOCKET\`, \`ATTACHMENT\`, \`CONTACT_SUPPORT\`, \`SECONDARY_CONTROL\`;
- \`SKIN_CODE\` and stable interface refs to the exact TESSA mesh.

AXIS learns **which** controllable units exist, rig parentage, effective directions, range/limits, chain constraints, anchors, control-purpose and interplay. Generic learned *control affordance* is explicit supervision; it is not magically recovered from geometry via deterministic code.

MIRA shares causal state and the **exact same mesh/rig identities**. MIRA reconstructs sparse per-vertex weights via SkinTokens-style discrete coding and predicts deformation compatibility. Its consumer must receive full per-vertex and local-surface features plus TESSA region/structural context (not only 7 anonymous XYZ/normal features and a substituted mesh).

**Compiler owns generic execution**, not controller intent: FK, IK (FABRIK/CCD or suitable constrained solver), contact and foot planting, DOF/limits, multi-chain/root compensation, attachment transform, skin deformation and declared secondary dynamics. The compiler may synthesize standardized solver primitives from a **sufficiently supported** AXIS control graph, but may not infer arbitrary semantic intent, hallucinate missing controllers, silently reparent an unsupported chain, or claim full control from skeleton+weights alone.

**Kavya-style control** is the target *capability type*: editor-accessible limb/prop handles, target dragging, transform/IK constraints, secondary motion and attachment editing. This is a product capability contract, not a claim of verified internal Kavya implementation. The specific sword-tip-drag, full-body compensation and 45° rotated-view continuation remain named behavioral proof requirements.

## 6. Physics and 2D presentation retain first-class ownership

A 3D correctly moving puppet is not automatically a correct 2D puppet.

Graphics assembly reads **the same** posed mesh, skin deformation, socket program, IRIS source appearance, semantic region support, and source-ray/depth evidence. Compiler Graphics determines **necessary, alternative or indeterminate** drawable separation under actual edit and visibility consequences. Semantic segmentation alone does not mint a drawable.

Required presentation operations: layered texture/addressable surfaces, camera-aware per-hit occlusion, local dynamic draw order where supported, visibility and alpha/CAA source consistency, independent accessory edits and hidden-surface/reveal behavior with provenance. A global integer z-order is not a replacement for actual 3D depth or geometric intersections; sword/body interpenetration must be recognized as collision/shape/attachment issue rather than patched with a misleading render rank.

Unseen hidden 2D art not observed in the supplied images may be predicted only as **predicted completion**, not as authored truth. A failed capability must be flagged, not covered with cosmetic inpainting.

## 7. Inference and training coupling

**Inference order is causal, representation is shared:**

\`\`\`text
Observations (product two views; independently qualified camera contract)
 -> IRIS shared Dense3D Z/H + Geometry G + Semantic S + Graphics A
 -> semantic-aware differentiable surface witness / explicit patch witnesses
 -> TESSA StructuralMeshCandidate + surface correspondence + typed relation graph
 -> AXIS ↔ MIRA coupled RigProgram + sparse SkinCodes on exact TESSA carrier
 -> Compiler qualification + qualified M/Rig/Skin + Mechanics + Graphics
 -> persistent editable/controllable Puppet ProductRevision
\`\`\`

*There are four specialized neural expert roles, not necessarily four disconnected training jobs.*

- Each expert may first be trained using its admitted masked targets and stable upstream state.
- TESSA loss gradients can enter IRIS **selected** scene/semantic feature projections through differentiable paths when training explicitly permits; qualified geometry is preserved by geometry/source-identity regularization and frozen references rather than a blind always-on joint optimizer.
- AXIS/MIRA's admissible posed deformation and skin loss can train their coupled state and condition TESSA's coordinate/surface refinement with a qualified teacher path. Mesh connectivity and control program tokens remain discrete and need distinct structural losses.
- At inference, there is **one committed state revision** with exact upstream IDs. Any mesh remeshing, frame alignment change or region lineage change invalidates dependent rig, skin, attachment and graphics bindings or requires an explicit independently checked remap. No silent reused W on new M.
- An expert may publish new evidence but must not overwrite another expert's authority. Cross-head feedback updates **learned parameters during training** or publishes versioned refinement proposals; it is not an unbounded runtime cycle.

**Default training goal** is coherent final puppet and qualified edit consequence, not independent short-term proxy scores for each model. Free-running geometry and mechanics matter more than teacher-forced token metrics. This is a frozen *training objective family*, not a claim that any trained model already succeeds.

## 8. Compiler boundary and failure policy

Compiler qualifies and binds source-specific proposals through generic, versioned rules. Static geometry repairs can be bounded, deterministic and independently proved, but **they do not replace the expert's learned responsibility**. Publish the original pre-repair result, exact repair operation, post-repair result and preserved correspondence.

Forbidden:
- undocumented subject-specific Knight fixes in production;
- importing teacher skin into the prediction path;
- silently welding multi-owner components;
- repairing topology after rig/skin without exact invalidation and rebind;
- inventing joints, full IK intent, appearance or hidden art solely to make product PASS;
- claiming product readiness because static mesh passes while source motion is unqualified.

Fail-closed reason categories: \`STRUCTURE_AMBIGUOUS\`, \`SURFACE_SIGN_UNQUALIFIED\`, \`ANIMATION_READY_MESH_UNQUALIFIED\`, \`ATTACHMENT_BINDING_UNSUPPORTED\`, \`RIGPROGRAM_INCOMPLETE\`, \`SKIN_MESH_LINEAGE_DRIFT\`, \`CONTROL_CONSTRAINT_UNSATISFIABLE\`, \`GRAPHICS_VISIBILITY_UNQUALIFIED\`, \`REPAIR_BUDGET_EXCEEDED\`. These names are frozen architecture error categories; not yet runtime error enums.

**Existing canonical mesh v3.2 contract** remains watertight-required for the current product carrier; this does not convert every source render mesh into a watertight original. If hidden closure is required, it must have separate policy and proven source/presentation consequences; never weld across source owners or invent physical source truth. If full qualification is impossible, do not ship a false puppet.

## 9. Source-of-truth and development precedence

This design incorporates, but does not falsify:
- \`canonical/IRIS_TESSA_V32_PRETRUTH_ARCHITECTURE_FREEZE.json\`: source-valid geometry, direct shared Z+G, TESSA mechanical teacher and frozen initial training.
- IRIS three logical heads from PR #377; **semantic candidate truth is not admitted training truth**.
- TESSA V5 mechanical teacher from PR #306, which is a teacher/court design **not** a neural TESSA V5 checkpoint or validated motion-ready inference.
- AXIS V6 typed RigProgram and AXIS–MIRA V7 research from PRs #297/#352; these are implementation starting points, not general full-control proof.
- Existing \`models/iris/vnext/flexicubes_v1.py\` pinned extraction wrapper and \`models/iris/vnext/scene_core_v1.py\`.
- Existing \`models/mira/conditioning_v6.py\` binds exact mesh/rig lineage, but its 7-D vertex projection does not exhaust this new carrier contract.
- Knight survivor authority: TESSA T1B autoregressive teacher-free reconstruction and Static G3 repaired mesh PASS are FIT1/static evidence, **not** dynamic/FIT8/unseen proof.
- Knight source-frame correction and generic vs special-purpose repair accountability remain critical.

This freeze supersedes **future architecture choice discussions** about deleting TESSA or semantic supervision as a separate logical responsibility. It **does not supersede past experiment facts or existing production authority**. A later change in topology decoder microarchitecture, backbone width, training curriculum or runtime optimization is compatible only if all named contracts remain preserved. Changes to expert ownership, typed carrier semantics, compiler authority or full-control target require an explicit new versioned architecture change.

## 10. Immutable decision summary

1. **KEEP** IRIS, TESSA, AXIS, MIRA as four specialized expert responsibilities within **one** unified character model.
2. **KEEP** IRIS Geometry, Semantic and Graphics as three logical heads, with cross-view instance and affordance evidence, not a hard human-only ontology.
3. **KEEP / INTEGRATE** semantic-aware FlexiCubes between IRIS and TESSA as differentiable surface scaffold; **do not** replace TESSA with StdGEN-style raw extraction.
4. **EXPAND** TESSA from triangle transcription into animation-aware structural mesh understanding, learned component/attachment relations and motion-sensitive topology.
5. **COUPLE** AXIS and MIRA as TokenRig-inspired skeleton+skin **plus our explicit learned RigProgram/control extension**; compiler provides generic solvers, not unsupported intent.
6. **FREEZE** SharedCharacterCarrier as append-only typed evidence, shared structural IDs, coordinate frames, correspondence, uncertainty and exact product invalidation. No independent re-interpretation after interface compression.
7. **KEEP** Compiler bounded and source-preserving: qualify, assemble, render, execute; report repairs and reject unsupported outputs.
8. **FREEZE** outcome as a persistent, editable, 2D-presented **controllable puppet**, with genuine interaction behavior as final product contract.

### Research grounding

- StdGEN++ (2026): https://arxiv.org/html/2601.07660 — geometry/color/semantic field; layered supervision; semantic-aware FlexiCubes; refinement. Does not supply a generic learned full control rig.
- StdGEN (2025): https://stdgen.github.io/ — predecessor with semantic-decomposed 3D reconstruction.
- SkinTokens / TokenRig (2026): https://arxiv.org/abs/2602.04805 — unified skeleton + sparse learned SkinTokens; no published full IK authoring program guarantee.
- FlexiCubes (SIGGRAPH 2023): https://research.nvidia.com/labs/toronto-ai/flexicubes/ — differentiable geometry and animated-object mesh optimization.
- Local mechanical/architecture provenance: https://github.com/merynz/RealSaS-OPT/pull/294 , https://github.com/merynz/RealSaS-OPT/pull/377 , https://github.com/merynz/RealSaS-OPT/pull/306 , https://github.com/merynz/RealSaS-OPT/pull/352 .

**Authority boundary:** frozen **design** is a decided engineering architecture, not evidence of a trained puppet; no unseen, source rights, corpus or product capability status is changed by this document.
