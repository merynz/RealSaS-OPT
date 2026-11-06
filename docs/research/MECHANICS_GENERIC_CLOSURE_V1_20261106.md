# RealSaS Mechanics Generic Closure V1 — independent research lane

**2026-11-06**  
**Status:** RESEARCH_ONLY / NO MODEL TRAINING AUTHORITY / NO GRAPHICS AUTHORITY  
**Branch:** `research/mechanics-generic-closure-v1`, based on V6 (`axis-v6-rig-program-product`); separate from Corpus370 #327, IRIS/TESSA, and SkinTokens++ #352.

## Product requirement

`compile → interactive puppet`. After compile, mouse-selected surface controls are interpreted through the **already declared** Rig control program. Mechanics owns control-space execution, constraint solving, pose, deformation and typed runtime reports. Motion owns control trajectories. IRIS-connected Graphics Compiler owns appearance, layers, visibility, occlusion, draw order and 2D rendering; none are mechanics targets.

Not every physical pick is automatically an IK target. The user interface may select a declared control through influence/barycentric support, with ambiguity or absence represented explicitly. A handle does not exist merely because the Compiler would like one. The Mechanics solver must never invent parent joints or control programs.

## Concrete V1 finding and fix

Existing `mechanics_control_solver_v1.py` transformed **only directly targeted joint positions** for `GLOBAL/ROOT/FK/SOCKET_CONTROL`. A root translation therefore left children behind (non-rigid segmentation of one chain), even though the root's skin row moved. This is a behavioral bug for any real puppet.

V1 propagates each declared direct-control transform through that target's complete deform hierarchy and processes controls shallow-to-deep. It preserves direct child-control overrides for the child's subtree and then applies declared constraints. This does not solve the full DOF/contact system; it repairs basic FK/ROOT semantics and is tested on a canonical three-joint chain through the actual LBS runtime.

## Mechanics runtime behavioral court

New `mechanics_live_runtime_v1.py`:
- one deterministic executor for standalone edits or Motion-controlled pose with transient mouse overrides;
- verifies exact mesh/rig/skin/Mechanics lineage and constraint schedule;
- accepts only declared control IDs; rejects duplicate edits and invalid motion time binding;
- never mutates original rig, skin, or motion clips;
- computes on **full mesh** via canonical LBS, measures max constraint residual and parent-child segment length strain;
- returns typed PASS/FAIL behavior status. Unreachable IK targets are **not** falsely reported as solved;
- supports monotone real-time input sequences with measured per-vertex speed/acceleration diagnostic, with no FPS claim;
- produces no rendering, graphics or model-training authority.

These are runtime diagnostic courts, not an independent Blender-source-equivalence proof. A numerical residual from the same solver is not enough to establish source-equivalent movement; the next phase must compare independent source interventions.

## Current mechanical executability gaps

1. **DOF**: `CanonicalJointFrameIR.v2` carries discrete DOF labels but no proved numeric angular/translation limits; current FABRIK solver does not enforce joint-space DOF/limits. Do **not** silently infer legal limit values from rotation responsiveness.
2. **Grammar mismatch**: AXIS V6 advertises `IK, AIM, LIMIT_ROTATION, LIMIT_TRANSLATION, FOLLOW, COPY_TRANSFORM, PATH, STRETCH`, while runtime schema supports exactly `IK, AIM, FOLLOW, COPY_TRANSFORM, STRETCH`. A proposed rig containing unsupported operators must be rejected until typed executable semantics and independent courts are provided. No nominal support.
3. **Multi-chain concurrency/contact**: sequential constraints do not yet solve simultaneous targets with planted feet, weighted priorities or closed-chain contact. Local single-effector success does not prove dynamic whole-body control.
4. **Attachments**: stable attachment position, orientation, held-object rigidity and grip are separate tests; correct wrist positioning alone is not grip proof.
5. **Performance**: current Python/NumPy solver and full mesh LBS have no verified 60FPS or P95 bound, and cannot be described as the final low-latency product runtime.
6. **Unseen**: Knight and the canonical three-joint unit fixture are regressions, never genericity evidence.

## Real measured evidence already located (read-only)

- Drive stage47 observed source/calibration workflow: `https://drive.google.com/drive/folders/1R4ROrfPNtkdnSYAn9LTQUF7N7PLJOlop`, including immutable per-subject job + admitted observation records. The sampled job is explicitly `training_allowed=false`.
- Drive `M_EVIDENCE_SUPPORT_V55.json`, `https://drive.google.com/file/d/1bIGdq3azS5JJ5G8TU9sJy9JoWMdkmKCP/view`, includes 3653 lifted M vertices and 14632 nonzero coefficients, but clearly states `EVIDENCE_FEATURE_LIFT_ONLY__NOT_SKIN_FIELD_TRANSPORT`, `source_skin_used=false`; **never feed this as MIRA skin truth**.
- Drive Corpus370 partial authority matrix run 41835338185: `https://drive.google.com/file/d/1zoO19fUZAv-QtiSbqIzdxEB5fKKSvMEJ/view`, OPEN and unsealed, 107 MIRA-eligible **research candidates**, 108 SKIN PASS, 3 MASKED; not train-ready, data family status may change as PR #327 advances.

All evidence is **read-only**. No PR #327 files, authority rosters, producers, geometry operators, upstream model configurations, Drive source bytes or shared training split are modified in this lane. We will not expose globally held-out subjects to research simply to accelerate debugging.

## Next research campaign: real-source mechanical behavior

Use the source-native rig + animation + observed intervention tensors, not just static carrier position. For a bounded pre-registered research-exposed, morphology-diverse TRAIN candidate subset, create an independently derived `SourceIntervention` and `CompiledMechanicsIntervention` pair with exact hashes:

- rest/bind/world-frame and inverse-bind equivalence;
- matched world-space articulated pose under isolated joint perturbations, control target drags, and simultaneous targets;
- parent-child invariants, attachment frame rigidity, global orientation and local pole behavior;
- normalized vertex displacement on valid rows, joint-frame residual and acute fold/inversion;
- abstain on unknown/ambiguous source mapping and unfinished SkinCarrier authority;
- no source-bone names or teacher-only source IDs as student model features;
- independent numerical court, never same helper used for producer and validator;
- clean single-holdout roster frozen **before** any performance-guided tuning.

Store original source intervention bytes by SHA, separate from prediction/artifact. Do not conflate source authored behavior with arbitrary interactive IK target behavior. Random poses improve intervention coverage but do not confer source-authored control intent.

## Planned closure sequence

A. **Baseline repair / dynamic runtime (this PR):** root/FK hierarchy, motion+edit overlay, whole-carrier runtime diagnostics, negative courts and CI.

B. **Generic mechanics constraints:** versioned joint-space DOF/limits representation with known/unknown masks and no fabricated angular bounds, simultaneous IK targets, contact constraints and clearly typed failure/priority rules.

C. **Real-evidence behavioral court:** retrieve a few exact Drive source assets through read-only acquisition, run Blender source-native interventions on GitHub Actions/worker, write SHA-locked result artifacts, compare independently to Mechanics. Retire diagnostics that only check their own implementation.

D. **Motion and latency:** typed interactive control tracks, temporal continuity/velocity/acceleration/jerk, stress on branching/wing/tail/creeping chains, bounded frame times in production runtime. No neural model per animation frame.

E. **Upstream release binding:** once frozen IRIS/TESSA and MIRA/AXIS releases exist, replace oracle mechanical carriers with real product outputs, train/evaluate only against sealed lineage-aware rosters, and preserve Graphics authority.

**Acceptance** requires behavioral correctness with repeatable source-independent evidence across multiple morphologies. Passing a three-joint fixture proves a specific bug fix, not generic closure.

## V1 continuation: simultaneous declared goals and real Stage47 source inspection

The next increment, still **research only**, is in
`compiler/realsas_compiler_core/mechanics_multigoal_solver_v1.py`. It is an opt-in SciPy rotational-FK constrained least-squares *reference solver* rather than a product replacement for FABRIK:

- solves multiple independently declared `IK_EFFECTOR` goals in one nonlinear objective;
- supports exact rest-position pins for declared IK control targets with stronger priority, without synthesizing a foot or contact rig;
- uses `CanonicalJointFrameIRV2.dof_labels` as rotation-axis permissions only; explicit `metadata.rotation_limits_rad` is enforced if present, otherwise ±π is **numerical chart domain only** and no anatomical limit is claimed;
- preserves parent-child segment lengths by forward kinematics; reports every drag/pin residual in source/rig world units;
- fails closed on unsupported pole inputs, translational/scale DOFs, unsupported operators, FK overlays and undeclared handles;
- exposes opt-in `solver_mode="DECLARED_MULTI_GOAL_V1"` through the same live Mechanics and LBS diagnostic path, with active Motion still separately controlled.
- runtime solves have no measured 60 FPS or p95 guarantee; SciPy optimizer is a research-quality comparison baseline.

A real read-only source investigation now exists for **Truth139 `truth139_fff1396de9c37b2bab05`** (Dino.glTF), with exact Google Drive source files and a SHA-verified Stage47 NPZ recorded at
`docs/research/MECHANICS_STAGE47_REAL_SOURCE_PILOT_20261106.json`. This explicitly research-exposes that subject: it must **not** become future blind unseen. The verified source tensor has **3,266 vertices, 129 privileged kinematic control parameters, and 774 source-native interventions** (387 signed pairs); 738 interventions show nonzero geometric response. Source rest, sham and root translation checks match within numerical tolerance.

The independent audit code `tools/mechanics/audit_stage47_source_mechanics_v1.py` validates SHA-bound source evidence and signed perturbation pairs, but **does not compare the RealSaS compiler solver to source Blender results** because the source→canonical joint/weight correspondence is not yet qualified. In particular, these source controls are `KINEMATIC_PARAMETER__NOT_PHYSICAL_DOF_OR_JOINT_LIMIT`, and the teacher explicitly declares `source_weight_lbs_equivalence=NOT_QUALIFIED`. All source-private handles and names remain teacher-only.

**Scientific next barrier:** inventing an IK rest-compatible interaction target is not source-authored IK truth. Before source-vs-compiler comparison, build a separately courted canonical joint/skin mapping that preserves source rest frames and response-space provenance, and compare source-intervention tensors against independent compiled pose/deformation. A numerically perfect optimization result against its own goals is never sufficient for source equivalence.
