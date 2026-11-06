# RealSaS / BUSTED — Mechanics post-V7 product gap map

**Date:** 2026-11-06  
**Scope:** Mechanically executable editable puppet after AXIS–MIRA V7; *not* model training.  
**Repository:** merynz/RealSaS-OPT  
**Branches inspected:** PR #352 \`research/axis-mira-v7-skintokens-control-370\` (V7), PR #353 \`research/mechanics-generic-closure-v1\` (Mechanics), PR #297 \`axis-v6-rig-program-product\` (shared base).  
**Status:** \`RESEARCH_AUDIT__NO_PRODUCT_READINESS_ASSERTION\`

## Architectural verdict

**Yes, the product is being coded**, including real compiler/runtime behavior and independent source tests. However, green isolated Mechanics regression tests do not imply that a V7-generated RigProgram can yet be compiled, edited and exported as a reliable generic puppet.

Three domains of truth must remain disjoint:

1. **Source evidence:** native mesh/rig/weights, joint transforms, attachments and responses independently observed on research-exposed source assets. Unknown source relations stay unknown.
2. **Puppet capability contract:** what a user is *entitled* to do with a compiled puppet (grab a tip, plant a foot, rotate a prop, control a tail, edit an attachment, blend clips and scrub). A passed rest pose is not evidence of these operations.
3. **Representation/runtime policy:** choosing internal axes, sparse solver DOFs, rank hysteresis, damping/frequency, caching, performance and advisory priorities. These do not mint new source joints, mechanics labels, grip frames or source constraints.

### Verified head and source boundaries

- **V7 PR #352** at \`1906cd75be332f7964410498b3d5cd924411b715\`: 16-layer/768-hidden/12-head causal AXIS using the **V6 RigProgram event grammar**, MIRA 64k FSQ skin codec and a learned-query attention candidate; separate read-only skin-influence-based control picking. The PR is draft, explicitly **no trained ModelRelease** and **no Corpus370 training admission**.
- **V6 PR #297** carries CanonicalRigIR.v2, CanonicalMechanicsIR.v1, V6 RigProgram, SkinIR.v2, CanonicalMotionIR.v2 and control-space execution. **V7 and Mechanics PRs are siblings built on V6**, not an integrated product branch.
- **Mechanics PR #353:** actual multi-goal 3D rotational FK, 6D absolute grip, relative dual-grip bridge, contact-position pins, bounded singularity restarts, temporal active-DOF hysteresis, secondary joint motion and prepared LBS. 128 focused tests pass on 2026-11-06, but the cross-parent example is 8 joints, 5 rotational DOFs and 3 skin vertices. The real Stage47 source-native LBS replay is research evidence, **not source→canonical V7 correspondence**.
- **Compiler's CanonicalMechanicsIR.v1 qualification** checks rig/skin lineage, operator schema, frame validity, rest pose and rest skin. This qualification **does not itself prove all declared edit-consequence behaviors**. The new research solver is optional and has not replaced product default runtime.
- No Graphics/CAA/IRIS/TESSA ownership belongs in AXIS/MIRA or Mechanics.

## P0 — Blockers before any V7 puppet execution/product claims

| ID | Observed technical gap | Consequence | Owning subsystem / proof needed |
| --- | --- | --- | --- |
| **P0.1** | AXIS V7 inherits **8** V6 constraint operators: \`IK, AIM, LIMIT_ROTATION, LIMIT_TRANSLATION, FOLLOW, COPY_TRANSFORM, PATH, STRETCH\`. Canonical execution schema admits only **5**, omitting limits and PATH. The opt-in analytic multi-goal solver supports only a narrower rotational IK subset. | A syntactically legal V7 RigProgram is not necessarily executable; rest-pose compiler success must not masquerade as interactive success. | **Mechanics capability matrix** per operator/mode; reject or explicitly mask unsupported declarations; end-to-end operator-specific behavioral courts; no unqualified auto-simplification |
| **P0.2** | V7's inherited frame language permits **9** DOF labels (\`TX/TY/TZ/RX/RY/RZ/SX/SY/SZ\`), while the new analytic joint-space IK currently represents only \`RX/RY/RZ\`. Angle limits are optional frame metadata, not guaranteed authored physical truth. | Legally expressed translational/scale modes or absent limits cannot be silently approximated as rotational-only kinematics. | Versioned executability and constraint manifolds (rotational, translational, scale); source-evidenced limits, fallback/masked authority |
| **P0.3** | V7's inherited \`ANCHOR\` event carries position/parent but has no *separate typed* grip orientation or cross-parent rigid-attachment bridge. The 6D and two-grip research tests supply these manually. | The model cannot yet independently generate, qualify and persist the full contract needed for sword/prop grip proofs. | **Explicit AttachmentFrame and AttachmentRelation proposals**, source-backed teachers and independent courts, then a versioned V7 grammar change; do not smuggle orientation into unauthoritative event defaults |
| **P0.4** | New analytic multigoal solver holds the world-space root joint location fixed in the FK loop; user interactive control does not yet solve root/hip/torso translation and rotational control hierarchy together with the attachment bridge. | Our intended sword-tip manipulation should recruit wrist/elbow/shoulder/torso/hip/feet but cannot be certified by the current constrained synthetic branch examples. | Generalized 6D root + hierarchical whole-body task-space solver, declarative DOF bounds, multiple contacts and a full-body intervention court |
| **P0.5** | No **qualified source→canonical** binding for native bones, rest frames, DOF manifolds, grip coordinate frames, control roles and the full SkinIR produced by V7. Stage47 verified native source replay and product LBS samples use source-privileged correspondence, not learned canonical binding. | A perfect synthetic Mechanics result says nothing about whether IRIS/TESSA→AXIS/MIRA produces truthful knobs on an unseen character. | Source-to-student teacher reconciliation and held-out family behavior proofs; distinguish source-exposed diagnostics from blind unseen |
| **P0.6** | V7 MIRA 64k FSQ/attention codec and AXIS decoder have research-only gradient, token-parity and forward contracts; no frozen, versioned, admitted Corpus370-trained model/codec. | End-to-end input image→mesh→Rig/Skin→Mechanics remains unproven, even if consumers are implementation-ready. | Model lane/Corpus370 authority, train split and unseen seals; no Mechanics-owned training authority |
| **P0.7** | PR #352 and PR #353 are independent siblings of V6 PR #297. No combined lineaged \`ModelRelease → CanonicalRig → Skin → CanonicalMechanics → Motion → Graphics → Proof → .rss\` artifact/behavior test. | CI for each PR cannot establish their integration or real user-flow editability. | Integration PR only after reproducible compatibility courts, full release product binding and source-backed runtime tests |

**Automated drift detector now provided:** \`tools/mechanics/audit_v7_mechanics_execution_gap_v1.py\` reads *actual repository Python source constants* using AST without importing torch. It records producer operator and DOF inventories, runtime operator inventory and 6D attachment event payload omissions. This is a necessary **static incompatibility receipt**, not a complete dynamic codegen or semantics proof. Its tests fail closed on source drift. Do not interpret a future empty static gap list as product readiness.

## P1 — Generic Mechanics features needing a dedicated experiment and capability gate

| Area | Required product behavior / negative proof |
| --- | --- |
| **Constraint graph and precedence** | Explicit causal DAG for dependent and cyclic constraints, target conflicts, solver ownership, mixing weights, round-trip order, stable priority under animation. Test inconsistent loops and preserve independent users' edit rights; Spine's ordering concepts are reference only. |
| **Real contact state** | Plant/release semantics and foot-floor manifold, no skate, contact normals, friction and support reaction distinct from a 3D position pin; two-foot support and jump airborne phases. No physical source labels are fabricated. |
| **Multiple simultaneous constraint types** | Oriented grip/aim, pole and bend plane, joint-limit manifold, telescopic/stretch chain, multi-parent relationships, optional soft targets without demoting hard constraints; impossible targets remain FAIL. |
| **Whole-body balance / center of mass** | Root and torso translation/rotation, hip adaptation and stability under dragging weapon target, independent contact-support constraint; mass and physical CoM require separate source/capability policy (cannot be inferred from generic skeleton graph alone). |
| **Secondary motion beyond trailing position lag** | Spring/tail/wing/cape/antenna/hair attachments, joint-space and deformation-space inertial effects, angular/velocity limits, optional dynamics vs authored motion, deterministic collision and layering effects. Existing state is a bounded deterministic spring policy, not realistic hair/cloth/soft-body physics. |
| **Self/prop/world collision** | Explicit contact shapes or geometry proxies, broad/narrow collision, interpenetration, constraint consistency with IK, cloth/wing folding, occlusion handoff to Graphics; current solver has no collision authority. |
| **DOF and control choice** | V7 output can include control kinds not executable in the solver; support ROOT/FK/IK_POLE/AIM/SPLINE/SOCKET in a typed operator execution plan. Surface hit→control may be **ambiguous** even with dominant skin influence; abstain instead of inferring controller semantics from skin mass. |
| **Temporal continuity and state lifetime** | Continuous drag, pose-space damping, active-set hysteresis, temporal limit crossings, long clip looping, abrupt target jumps, deterministic reset/seek/replay, variable update rate, clip blending/retarget, and hand switching. No state jump permitted without an explicit source/user event. |
| **Scale/space correctness** | Source Blender/world→canonical normalization; local joint frame handedness, units and transform composition; projection and 2D camera/view transitions; identity and rest/pose frame compatibility, including negative determinant or mirrored samples. |
| **Physics authority separation** | Need distinct \(source\) observed material/kinematic truth, \(capability\) explicit editing permissions, and \(runtime policy\) stiffness/damping, sample rate and solver-space penalties. "Looks plausible" never qualifies physical force, training truth or limb structure. |

## P2 — Product integration, visual consequences and runtime economics

1. **Product performance:** scale from 3–5 DOF and tiny skins to 30–100+ joint mixed morphologies and thousands of vertices, multiple 6D grips, transient high-frequency pointer targets, full LBS, Graphics compositing and GPU/CPU transfer; measure p50/p95/p99 including scene render. The 4.13 ms 8-joint/3-vertex reference benchmark is not a 60-FPS result. Python/SciPy reference is a scientific court, not by itself a low-latency production runtime.
2. **Graphics-consumer interface:** versioned read-only Mechanics world joint/attachment/skin state and consequence/occlusion payload to Graphics; preserve drawable ownership, depth and draw order/visibility under extreme deformation and view changes. Mechanics must not decide artwork fragmentation or alpha/CAA.
3. **Actual interactive editor:** real image/surface hit mapping to canonical mesh and declared control, ambiguity/abstention UI, pointer drag, undo/redo, selection smoothing, pose/clip layers, user-editable rest frame and limits without corrupting hashed lineage.
4. **Persistent puppet:** schema-migrated \`.rss\` containing canonical mesh, rig, skin, controls, attachment relations, Motion clips, policy parameters and independent proof receipt; export to engine-neutral targets with preserved edit semantics and deterministic replay. No invisible re-generation on open.
5. **Target-aware motion:** after manual sword-tip control, a dynamically targeted attack must satisfy the *current* target and contacts. Fixed preauthored \`Slash_01\` alone does not meet the product differentiator.
6. **Proof semantics:** distinguish \`PASS_EXECUTABLE_OPERATOR_SUBSET\`, \`PASS_HARD_WITH_UNMET_ADVISORY\`, \`PASS_SOURCE_MECHANICS_EQUIVALENCE\`, \`PASS_EDIT_CONSEQUENCE\`, \`PASS_GRAPHICS_PRESENTATION\`, \`PASS_PRODUCT_COMPILE\`; no promotion by mere rest-pose CI.
7. **Multi-morphology acceptance:** Knight remains regression, not generic proof. Require ground/walking/running/flying/crawling/tail/wings/props morphology diversity and unseen subject/family gates for reliable product assertions.

## Recommended execution order (large coherent increments)

### M7-A — Producer/executor capability closure (first)

Deliver a versioned **MechanicsCapabilityManifest** (proposed, NOT implemented), explicit enumeration of actually executable V7 operators, supported DOF manifolds, hard/soft constraints, effectors and attachment frames. The teacher/model cannot claim \`Mechanics Executable\` until its **entire** declared interaction contract is either supported or honestly masked. Static coverage is necessary; dynamic negative/positive operator behavior is required. Introduce an explicit **MechanicsExecutionPlan** compiled from the canonical rig, not derived by a runtime name heuristic. Hold all source-to-compiler authority boundaries.

### M7-B — Integrated whole-body interaction and attachment relation grammar

Formalize **AttachmentFrameIR** and **AttachmentRelationIR** (proposed, NOT implemented) so V7 may eventually learn the same 6D relations the compiler actually consumes. Expand root translation and whole-body IK with independently planted foot/feet, grip hand switching and dual-grip rigid object. Check motion continuity with dynamic targets, constraints and deterministic physics-state replay.

### M7-C — Source-truth correspondence and behavioral qualification

Collect native intervention/action truth for joints, contact, attachments and control affordances; run independent source-native response vs compiled Rig/DOF/Skin/Mechanics on **research-exposed** subjects and then protected blind unseen. Close per-family authority, no source-label privileged leakage, held-out morphology. Feed the failure ledger to V7 teacher/model owners; do not retrain from convenience guesses.

### M7-D — Shipping runtime and Graphics/ProductRevision integration

Native/batched solver and prepared skin with full timing, read-only Graphics handoff, one integrated branch of PR #297 / #352 / #353 with independent proof, durable \`.rss\` and real draggable editor. Pass performance and visual tests **before** calling this an editable puppet product.

## Non-negotiable demo gate

On an unseen character carrying a sword: *drag sword tip to a world target, independently observe hand/arm/shoulder/torso/hip adaptations and planted foot/feet; rotate character 45° while holding attachment; retarget a user-directed attack against a newly chosen target; scrub/replay; inspect all deformation, attachments, Graphics draw order, source evidence and latency*. If the system requires per-subject handcrafted grip roles or a hidden fixed slash clip, that is a failure of the desired generic product capability.

**Scope notice:** these future IR names and capability gates are roadmap proposals only. No ModelRelease, training, hidden physical truth or full-product PASS is authorized by this document.
