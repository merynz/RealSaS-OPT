# RSS.vNext — Consumer-First Controllable Puppet Evidence Contract

**Date:** 2026-11-11
**Status:** Working runtime-first requirements specification / **NOT the final architecture freeze**
**Source baseline:** EsotericSoftware/spine-runtimes commit 19431ae32bd7419e74ba160a46080b878c73840c, branch 4.3 (TypeScript and C++ implementations inspected). The Spine Runtimes License is NOT permissive blanket authorization to copy/redistribute Spine code. This document records behavior and interface requirements; all RSS code must be independently authored.
**RSS consumer prototype:** compiler/realsas_compiler_core/controllable_puppet_runtime_v1.py
**Source/product contract:** 2 actual observed user FRONT/BACK RGBA images -> single shared 3D carrier -> one qualified persistent, interactive, 3D-mechanics puppet -> eight-camera 2D presentation. Additional constructed views are *predictions* or training-only source renders, never third/fourth real observations.

## 0. Reverse the dependency (do not design unteachable heads)

    User-visible runtime behavior and reproducible intervention
      -> exact runtime READ fields and executable invariant
      -> Compiler qualification plus deterministic derivation
      -> minimal LEARNED evidence missing from those derivations
      -> producing RSS.vNext expert and information-access path
      -> source-native / qualified-render / intervention supervision producer
      -> admissibility masks, losses, negative/abstain policies

Runtime capability is the goal, not a label. An intermediate class is useful only if it makes the downstream behavior more controllable, robust or source faithful. Preserve original pixels, alpha, camera, canonical coordinates, uncertainty and source provenance separately; never collapse them into a universal semantic score.

### Evidence authority taxonomy

- SOURCE_NATIVE: directly present in admitted original mesh, material, rig, weights, clip or artist-authored metadata, with exact IDs.
- SOURCE_RENDERED: measured from original admissible source (cameras, ray stacks, depth/normal, semantic role passes, UV/alpha, detach/occlusion counterfactual render).
- DERIVED_GEOMETRIC: exact calculation on a *candidate* or qualified mesh (screen projection, distance, visibility, intersection, interpolated anchor). This is mathematical truth **about that representation**, NOT proof the source geometry is correct.
- DERIVED_INTERVENTION: qualified response to source rig/skin/motion perturbations or declared generic solver programs, with teacher-vs-proposal distinction.
- REPRESENTATION_POLICY: allowed defaults and runtime tradeoffs (which controls, which dynamic region, max stiffness, click-depth policy); not a discovered source fact.
- LEARNED_HYPOTHESIS: probabilities/unknown masks until qualified. This may become a valid editable user-controlled proposal without claiming artist intention.
- PRODUCT_QUALIFIED: Compiler validation with source/lineage hashes, executable solver, source-support and edit consequence gates. Never emitted solely by a neural head.

No missing label is silently converted to background, negative, zero, non-attachment, hard surface, immobile region or true anatomical semantics. For each family use *family-specific qualification and supervision masks*; distinguish eligible source-positive, eligible source-negative, indeterminate, unobservable and policy-only examples.

## 1. Runtime-first capability → evidence crosswalk

| Consumer behavior (must be testable) | Minimal explicit runtime fields | Deterministic Compiler / renderer responsibility | Missing or learned evidence from shared RSS graph | Source supervision that can actually be produced | Current gate |
|---|---|---|---|---|---|
| Pixel click selects the same piece after pose and 45°/8-way view change | Persistent component/attachment/drawable ID, indexed face address, UV+alpha, posed camera, selectable control binding | Triangle ray/pixel hit + depth/alpha/clipping, barycentric coordinates, selected persistent ID | Structural component boundaries, reliable UV/appearance support, candidate selectable control domain | Native source component IDs; admitted per-view ray visibility and alpha; original source mesh/UV; source component ownership; user policy for selectability | Partial synthetic consumer; clipping-aware alpha pick and product atlas are open |
| Mouse drag controls a limb/effector, not regenerates frames | Control ID, reference frame, joint chain, DOF, target, depth/plane UI policy, skin matrix, constraints | Screen-to-world projection with chosen depth, deterministic multi-joint IK/FK/LBS, residual reporting | Chain, parent graph, candidate effector/root, joint coordinate frames, editable DOFs; confidence | Original source joints/weights + qualified pose suite; compiler-generated intervention PROGRAM targets labeled as policy and judged by source-relative posed consequences | Real three-joint synthetic IK/LBS works; coupled body/contact still open |
| Eight 2D directions from one 3D mechanical identity | Exactly eight camera projection contracts, canonical mesh/rig/skin identities, appearance authority by face/direction | Project same solved 3D mesh through view camera; 3D depth+composite, not eight independent sprite rigs | 3D surface/depth + semantic/component and appearance coverage; uncertain hidden content | Two real input views; other camera renders only as held-out supervised teacher; verified visibility/appearance masks | Eight synthetic camera projections work, no real art acceptance |
| Draw order, self-occlusion, clipping, translucency | Per-view camera, surface fragments, face/drawable ownership, alpha/depth write, clip intervals, slot policy | Exact z-buffer for 3D body; alpha compositing, clipping, stable tie/order for art semantics | Source-facing layered surface semantics and appearance support; candidate drawable membership; NOT global all-pair relation oracle | Original source ray stacks, alpha layers, material IDs, counterfactual component-removal renders if feasible | RuntimeV3 has typed contracts; complete CSS/atlas and source pixel tests open |
| Grip, weapon attachment, detach, rebind | Stable holder/held component IDs, surface addresses + barycentrics, local coordinate frames, target socket, attach mode, switch events | Compute compatible surface anchor and relative transforms; follow/copy on bound local frame; undo/switch on explicit event | Candidate distinct components, controllability/support region, contact/relative transport affordance; inferred grip remains UNKNOWN without evidence | Native skeleton/socket/control metadata where available; qualified relative motion in source clips; contact pairs measured in source pose interventions | Surface-anchor deterministic path exists; full generic discovery/proof open |
| Collision / contact | Qualified collision meshes or regions, pair filter, contact/penetration tolerances, allowed/forbidden behavior | Broad/narrow phase geometric collisions; optimize contact constraints only where declared; report residuals | Candidate component separation, joint/skin control that could avoid collision; optional learned contact-likelihood **not equal to intent** | Source geometry and collision measurements; source author collider metadata; motion source interventions; fixed-policy negative probes | Intersection predicates exist; complete physical collision solver not established |
| Dynamic supported region: cloth/hair/tail etc. | Exact vertex/face influence domain, pinning/anchor boundary, solver method and physical parameters, control and source IDs | Time-step an explicit dynamics solver, handle reset/contact/damping; never infer dynamics merely from geometry | Early generic presentation strata + structural component separability + mechanical flex region/anchor *hypotheses* | Source skin weights/animations for valid dynamic deformation; source material/cloth settings if present; physics probes tagged as synthetic **policy** | Source motion truth availability has gaps; no verified generic secondary solver |
| Full-body mouse-control and planted feet | Multiple effectors and priorities, support/contact constraints, reach limits, center-of-mass surrogate if declared, root mobility | Coupled constrained IK, contact preservation and failure feedback | Support/effector chain affordances, root/DOF graph, intended foot/ground points when support qualified | Source mechanics+motion and rendered contact; controlled body-IK intervention preference targets | Existing FABRIK is single chain; no certified general simultaneous goal solver |
| Appearance changes/skin swaps while moving | Stable drawable/slot/attachment variant, UV/material, face coverage, source provenance, compatible mesh/rig binding | Swap active art with exact geometry/skin identity and depth/alpha policy | K-way presentation roles, face ownership, hidden layer uncertainty and art grouping | Source material role ID, source alpha/depth, layered source render; open/unknown materials masked | RuntimeV3 attachment/slot states; not a qualified authored product |
| Animation clip mixed with live control/physics | Track state, animation event timing, precedence, override mask, frame delta, reset semantics | Deterministic scheduler and blending with explicit constraint ordering | Motion library retarget mapping and control affordance/DOF; not image-to-arbitrary-animation oracle | Native source clips with valid rig mappings and pose/intervention outcomes; absent clips are masked | Existing motion compiler and solvers partial; consumer timeline mixer still missing |
| Deterministic reload/export and editor controls | ProductRevision hash, canonical vertex/face IDs, rig+skin IDs, camera hashes, active controls, attachment state | Serialize and restore, replay identical control interaction, qualify edits, enforce ownership | Stable instance and component identity through structuralization | Source asset lineage; product invariance/roundtrip and exact canonical identities | Consumer synthetic IDs only; real product persistence/Studio bridge open |

**Spine's 2D runtime cannot prove any of the 3D/depth-specific rows by itself.** A one-dimensional draw-order permutation is not a universal substitute for 3D z-buffer visibility, especially for translucent fragments and self-occlusion. An eight-view visual bank without persistent mechanical identity is not a controllable puppet.

## 2. Neural information visibility and ownership — no private disconnected latent islands

| Existing RSS.vNext expert | Must be able to READ | Candidate outputs; DO NOT grant runtime authority | What the Compiler rather than neural expert should derive |
|---|---|---|---|
| Two-view perception / shared scene | FRONT/BACK true pixels and alpha, source camera; DINO token and original ray bank | Z triplanes, spatial query H(x), uncertainty, source-semantic candidate labels | Camera validation; pixel/ray address transforms |
| Early semantic + Graphics evidence | Original source pixels/alpha, camera, full H(x), raw region memory, estimated geometry and source-ray comparison | K-way general presentation roles; layers, region/component/face ownership probabilities, appearance/surface support, UNKNOWN masks | Depth/ray-stack visibility, occlusion consequences, alpha compositing, deterministic layer-policy constraints |
| Structural topology + dynamic surface | H(x), semantic roles, source ray evidence, FlexiCubes hypothesis surface, region relations, source-conditioned uncertainty | Indexed animation-ready mesh with stable candidate component/face/vertex IDs, surface normal, topology uncertainty and affine motion probes | Mesh admissibility, topology/hash/ownership validation, camera projections, collision geometry |
| Mechanical / RigProgram | Original H(x) **and** exact structural vertex H(x), structural memory, component/region evidence, optional FC motion witnesses | Joint forest, DOF/local frames, chain/effector/handle, constraints, candidate sockets/anchors, FSQ codes, unsupported-move uncertainty | Executable constraint filtering, deterministic FK/IK, local frame transport, reachable control target; report residuals |
| Skin spatial query | Exact structural vertex H/XYZ/normal, all predicted causal joint states and FSQ skin codes, component-boundary evidence | Index-preserving W[V,J], influence uncertainty and dynamic response candidate region | Validate weight coverage/simplex/index identity, LBS and deformation consequence tests |
| Differentiable appearance and dynamic-region readouts | Same H(x), 2 observed original RGBA sources, exact indexed structural geometry, control domain and skin-supported motion regions | Per-face UV/material/color/alpha authority and candidate secondary region pin boundaries | Source compositing, atlas patch packing, stable per-direction draw calls, explicit temporal simulation |

A semantic role must **not** claim it solved anatomy, grip, pressure, relative attachment ownership or dynamics. It establishes renderer-trainable presentation strata and supporting 2D/3D part evidence. The later mechanical expert can exploit that *same evidence* together with mesh topology and motion constraints.

A geometry surface guess with high neural confidence is not source depth authority. Original input RGBA survives end-to-end.

**Discrete gradients:** FlexiCubes/continuous geometry/skin loss may backpropagate into shared H. Discrete structural triangle tokens and causal control grammar require their own token teacher, motion preference or behavioral reward. A successful deterministic Compiler solver does NOT automatically differentiate a missing contact or control selection across discrete token choices.

## 3. Explicit family-specific Truth Foundry contract — candidate requirements, NOT final materialization tables

| Evidence family | Necessary source inputs | Candidate supervised targets | Eligibility (per sample) | Ambiguous/negative control |
|---|---|---|---|---|
| SOURCE_RGB_ALPHA_CAMERAS | true admitted source image pair, canonical calibrated FRONT/BACK | source pixels/alpha, exact camera and ray source | both source views, camera provenance, valid alpha/color | missing views do not become inferred input truth |
| SOURCE_PRESENTATION_LAYERS | native mesh/material IDs plus original source layered render (or verified material semantics) | K generic role IDs, per-pixel/face role coverage, counterfactual layer removal | only proven role mapping, face→material mapping and renderer court | unknown material/combined shells masked; no implied 'hair' label from color alone |
| SURFACE_GEOMETRY | source source mesh, coordinate frame, component topology and valid source views | 3D positions, SDF sign **only in qualified closed-domain**, normals, UDF/patches otherwise | source vertex topology/coordinate quality and layer eligibility | open source cannot be silently closed and called signed ground truth |
| TOPOLOGY_MOTION_READINESS | native mesh/skin/rig plus qualified pose/intervention suite | mesh face tokens, source-relative posed silhouette/stretch/seam/degenerate metrics, candidates ranked by behavioral consequence | exact source mesh/rig/pose relation, differentiable path or discrete ranking | quality relative to source, not penalize valid articulation |
| CONTROL_PROGRAM | native rig hierarchy/DOF/author controls, source animation and eligible intervention policy | joint/chain/frame tokens, candidate constraints, reach/effectors and behavior of resulting program | only fields present or calculable from qualified source or formally declared generic intervention policy | autogenerated handles labeled POLICY not historical artist/source truth |
| SKIN_WEIGHTS | native skin influences on exact source mesh and rig, remeshing correspondence if applicable | vertex/joint influence rows or dense sample, FSQ code reconstruction, posed error | exact vertex/rig lineage and normalized weights | invisible geometry ≠ zero influence; unbound components excluded |
| ATTACHMENT_CONTACT | source authored parent/socket/binding or source-relative motion with source geometry | holder/held positive relation, surface anchor, relative transform and switch behavior | confirmed original socket or relative-motion intervention evidence | proximity alone not positive; absence alone not negative |
| VISIBILITY_GRAPHICS_CAA | source material/alpha/atlas, rays/depth/normals/UV/camera, valid removals | face source-supported mask, visible ray-stack owner and appearance continuity under view/pose | qualified renderer, source camera and material/alpha/UV | hidden or unobserved pixels are UNKNOWN, not fake source texture |
| COLLISION_AND_SECONDARY | collider metadata/valid native clips/rig parameters; otherwise explicitly policy-generated probes | contact masks, temporal response, pin regions, stiffness/damping only when authored or policy-generated | real authored physics or controlled synthetic policy with method tags | no learned 'real stiffness' from static 2-view alpha |
| VIEW_CONSTRUCTION | original admissible source 3D mesh and renderer, train-time 6 held-out directions | RGB/alpha/normal/depth + visibility/uncertainty masks for novel cameras | true original source render provenance, canonicalized camera ring | model-generated views NEVER used as original source labels |

Source rights, dedup/lineage and real-character qualification are global hard gates **in addition** to family-specific masks. A truth table built from unverified inventory rows is not training admission. A synthetic solver-probe label is useful for a declared behavior task but must never be renamed source-native truth.

## 4. Runtime-first executable proof, current status

New consumer file:
- compiler/realsas_compiler_core/controllable_puppet_runtime_v1.py
- tests/compiler/test_controllable_puppet_runtime_v1.py

The first consumer intentionally uses *qualified bundle shape with synthetic fixture*:
- 8 explicit camera directions, one canonical 3D mesh/rig/skin identity, same face indices and vertex IDs, no eight unlinked puppet instances;
- mouse drag selects a UI depth-preserving ray-plane projection then executes REAL Compiler IK/FABRIK and LBS, reprojects the same posed geometry at 45 degrees;
- runtime picking returns persistent component and triangle source address using ray/depth/alpha policy, requires caller-supplied alpha provider and never invents opacity;
- attachment switching only works for pre-bound variants; no invented source grip;
- fail-closed mesh/rig/skin lineage and unsupported parent-transform inversion;
- **NOT** a GPU image renderer, actual sprite output, source-admitted product, coupled full-body physics, clipping-aware hit-test, real layered Graphics or source-qualified appearance.
Hosted CPU: https://github.com/merynz/RealSaS-OPT/actions/runs/42910926462 — 13 passed, synthetic consumer and existing deterministic IK/LBS tests.

## 5. Necessary staged real-character / behavioral gates before final RSS.vNext freeze

A. REFERENCE COMPLETE: inspect Spine TS/C++ scheduler, IK, slots, mesh deform, draw order, clipping, physics and alpha/picking; record version and code method. Product behavior differences must be explicitly annotated, not attributed to undocumented internal competitor implementation. Avoid copying Spine-licensed source code.

B. IDEAL MANUAL SOURCE CONSUMER: take ONE genuinely qualified original character with mesh/UV/rig/skin/source rendering, hand author a correct independent runtime typed bundle, prove user can click component, drag IK, turn 45°, preserve real RGB/alpha and appearance, modify attachment and verify consequence. This deliberately bypasses ML and is the first true consumer acceptance gate.

C. COMPILER FROM SOURCE EVIDENCE: generate the same runtime bundle from native source and mathematically derived evidence; ensure no learned semantics/author intent assumptions. Report which fields are impossible or ambiguous.

D. MODEL PRODUCER: make existing RSS.vNext experts produce the remaining candidate fields while preserving H(x), source-ray provenance, mesh identity, uncertain assignments and variable topology. End-to-end inference sees exactly two real views.

E. BEHAVIORAL LEARNING: train eligible family masks and source-qualified interventions; separately test motion-ready topology, skin, full-control program, animation, 8-way visibility and source appearance.

F. FULL PRODUCT: Studio persistent state, reset/replay/export, 3D collision, component-aware secondary motion, clip-aware selection, and real full-body multi-goal control; no silent approximate fallback.

**Freeze rule:** do not freeze a model head or require a truth target until there is a downstream runtime consumer, an executable Compiler binding, a stable typed address, and an actual qualified source builder or explicitly-labeled policy-generated intervention supervision. These are separately testable and separately versioned.
