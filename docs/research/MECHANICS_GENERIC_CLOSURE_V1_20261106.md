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

## Second increment: measured source FK/LBS closure, controls and warm-start (2026-11-06)

**Actual Stage47 source-world measurements, exact hashes:** `docs/research/MECHANICS_STAGE47_SOURCE_NATIVE_FKLBS_REPLAY_20261106.json`. Source `truth139_fff1396de9c37b2bab05` has **43** native rest bone frames, **129** isolated 3-axis kinematic parameters, **3,266** source vertices, **6,571** sparse source weight coefficients and **774** Blender-measured interventions. Independent analytical replay of each native local bone rotation through the declared source parent forest and descendant influence mass reproduces **all 774** within maximum vertex L2 error **1.2653386341165218e-6 source units**; per-action max error p95 **8.358747185028643e-7**; no action exceeds the preregistered local research threshold of **2.5e-6**. All source input bytes were SHA-verified before running local experiment.

`tools/mechanics/compare_stage47_source_fk_lbs_v1.py` is the reproducible source read-only court. It expects the exact source teacher JSON + NPZ from connected Drive, cross-checks the source hash, source frame/parameter alignment, signed action pairing, armature hierarchy and weight CSR. The optional `--product-probes N` path separately calls the **real** `deform_mesh_lbs_v1` on an explicitly `UNQUALIFIED_SOURCE_ORACLE_RESEARCH_ONLY` carrier/rig/skin. CI covers this same path using a separate synthetic source-bone oracle and measured deformation fixture; it does **not** download the private 24.5MB Stage47 source NPZ during ordinary PR checks.

**Do not conflate:** whole-source analytical source rig deformer equivalence is a PASS for *source-local* mechanical response. It is not a PASS for IRIS/TESSA source→CanonicalMesh fidelity, AXIS controls, MIRA skin output, a learned V7 model, or a product-quality source→compiled Mechanics binding. The actual RealSaS LBS probe has **only** run in the synthetic CI fixture so far; full real-source oracle-to-product-LBS execution remains a research work item. Neither source teacher source names nor source weights can become student model inputs.

**Important control-role distinction:** measured source bones include **5** with zero **direct** skin weight, **2** without **any descendant** skinned vertices and **6** local rotational parameters with exactly zero measured isolated deformation. Those controls have source-private pole-target names, but this stage did not measure their participation in an IK constraint. It is wrong to equate zero direct weight with "no mechanical control effect" or to call pole controls obsolete. The only valid conclusion is no effect under the measured **isolated native rotation experiment**. See `MECHANICS_STAGE47_CONTROL_SKIN_CAUSAL_SEPARATION_20261106.json`.

New `compiler/realsas_compiler_core/mechanics_surface_control_effects_v1.py` models the **declared causal effect graph**: ROOT/FK/SOCKET/GLOBAL control targets propagate to their declared joint descendants, while control inputs to authored constraints gain evidence from explicitly targeted output joints or chain joints. For any hit triangle and barycentrics, the runtime returns ranked control candidates and effect paths, or `AMBIGUOUS` / `ABSTAIN`; no controls or source semantics are invented. CI proves a ROOT with only an unweighted root target can still influence every skinned descendant and an IK pole with no direct skin target can still influence a declared IK chain. This graph is **not** the same as evidence that the resulting model produces a correct native control program on unseen subjects.

The multi-goal reference solver also now accepts *validated, schema-bound previous joint configurations* as warm-start parameters. Live sequences carry a qualified previous solution forward to the next frame; a failed frame clears the warm start. This is for smooth drag research and runtime performance evaluation, not a 60fps claim. Default FABRIK runtime remains unchanged.

## Reproducibility

```bash
python tools/mechanics/compare_stage47_source_fk_lbs_v1.py \
  --teacher /secure/TEACHER_SCENE.json \
  --npz /secure/geometry_and_response.npz \
  --expected-teacher-sha256 87eacdb230f19feae5b0d3e15b67d504fe40908f703254faa4a4d29e9b117b6e \
  --expected-npz-sha256 c13e837029ab80c78c989c01698d31ae10446c18fa8907e1a8840c33ca6c75fb \
  --product-probes 12 \
  --report /secure/source_native_lbs_court_v1.json
```

The source is read-only and remains outside git. The report is reproducible by an authorized executor with exact source bytes; never upload artist source meshes, original bone names or full teacher arrays into public CI artifacts. `--product-probes 12` remains a **planned execution** on actual Stage47 bytes until a corresponding artifact/receipt is recorded.

**Next hard scientific step:** validate the actual product-LBS oracle probe on exact real Stage47 source bytes and then define source→canonical surface and local-frame correspondence evidence from independently qualified geometry; challenge the binding with mismatched source, barycentric support forgery, axis flips and wrong parentage. Validate source-equivalent multi-action deformation without using target/source bone names as student features.

## 2026-11-06 third increment: observable controls + prepared runtime over real source

### Separate a declared relationship from observed mechanical response

The original control-surface graph was only a *potential effect* estimate and had two dangerous shortcuts: it could accept a fabricated triangle containing valid mesh vertex IDs, and it could treat a bare `IK_EFFECTOR` with a target joint but **no executable IK constraint** as a mechanically effective control. Both are now fail-closed.

`mechanics_surface_control_effects_v1.py` now requires a source-consistent `CanonicalMeshIR` hash and membership in **actual canonical triangle topology**; checks unknown surface vertices separately; excludes orphaned IK inputs; and validates supported executable constraint operators. Reachability of an IK chain is limited to declared chain joints rather than assuming all arbitrary subtrees respond.

New `mechanics_control_response_court_v1.py` evaluates actual canonical Mechanics + LBS via small signed translation/rotation interventions to distinguish **graph-potential** from **observed local response**. It provides a typed no-effect/inconclusive result, never automatically deletes an unresponsive pole control. This is a deliberately expensive offline analysis, not the per-frame UI picker and not native-source causal proof.

### Prepared sparse skin execution — real-world numeric comparison

`mechanics_prepared_skin_v1.py` adds a bounded SHA-bound, source-read-only per-character cache of sparse vertex/joint influences and rest offsets. Frame execution applies vectorized world transformations with NumPy einsum + bincount instead of rebuilding Python-side per-vertex/per-weight structures. The returned `DeformedMeshIRV1` remains the same deformation contract; zero Graphics or model ownership. `mechanics_live_runtime_v1.py` accepts `prepared_skin=` as explicit opt-in. The original loop-based `deform_mesh_lbs_v1` stays default and is the independent parity oracle.

An exact-source, real product-kernel experiment was executed on the private Truth139 Dino Stage47 teacher scene and observed Blender NPZ after verifying input SHA-256, the Compiler vendor bundle SHA and the visible optimizer SHA. Both actual RealSaS implementations were called on **128 deterministic action indices spanning all 774 source interventions** with the **same oracle-supplied source-local rig/skin and posed joints**. See `docs/research/MECHANICS_STAGE47_PRODUCT_LBS_PREPARED_128_20261106.json`.

**Numeric comparison on that sample (source-world units):**
- Original actual product LBS vs source Blender: maximum vertex L2 **9.487445122129821e-7**.
- Prepared product LBS vs source Blender: maximum vertex L2 **9.487445126094705e-7**.
- Prepared vs original product LBS: maximum vertex L2 **8.95090418262362e-16**.
- Measured median original deform-only CPU time **66.028 ms**; measured median prepared deform-only CPU time **3.612 ms**, **18.278× faster** on that container/session/hardware. One-time prep **4.246 ms**.
- Source-native analytical reference still spans all **774** interventions, but *actual* product-kernel comparison currently spans **128**; an exhaustive legacy-kernel run exceeded local execution limits and is explicitly **NOT PASS**.

These are **kernel-only timings**. They exclude IK/constraint solve, collision, contacts, Motion sampling, graphics, skinning uploads, drawing, and engine/frame overhead. They are not any 60fps/P95 shipping certification. No user input was stored as student training truth.

### Closure still open

1. Independently qualified source↔canonical mesh, rest-frame and joint/control correspondence, sourced from real IRIS/TESSA and AXIS/MIRA outputs rather than oracle teacher imports.
2. Attachments/orientation, full-body contact, multi-chain priority resolution, temporal stress testing across diverse morphologies.
3. Cached solver state and actual end-to-end interactive frame benchmark including skin, Motion and graphics handoff, not just LBS kernel.
4. Blind morphology-held-out source checks with pre-registered research exposure. This exposed Dino is never later promoted to unseen.

The temporary **code-only** GitHub artifact used to access the exact private repo kernel in the execution container was uploaded with one-day retention during the experiment and its workflow upload step subsequently removed; artist source assets were never uploaded to GitHub.

## CPU Mechanics complete-frame reference — 2026-11-06

The LBS-only 18.28× gain revealed a second bottleneck: the legacy behavioral
court converted and checked each vertex separately in Python. The
`mechanics_live_runtime_v1.py::_behavioral_metrics` implementation now
vectorizes its **full** canonical vertex finite-position/displacement tests and
uses the immutable rest-position cache prepared exactly once by
`mechanics_prepared_skin_v1.py`. The same semantic checks remain present:
joint-set equality, parent-child segment strain, exact vertex-ID coverage,
finite geometry and measured maximum full-vertex displacement. Nonprepared
legacy mode retains an exact ID-reordering fallback, including duplicate-ID,
missing-vertex and NaN failure courts. No sampling approximation.

The full executable frame court is
`tools/mechanics/benchmark_live_mechanics_v1.py`; it uses a seeded
**synthetic 5-joint/3-DOF branched rig**, two IK goals, one stationary control
point, previous qualified joint solution warm-start, full LBS and complete
behavioral diagnostic. It explicitly excludes renderer/Graphics, contact force,
collision, data transfer, frame pacing and any model inference.
GitHub Actions runs it on the `mechanics_generic_closure_v1` job, publishes
JSON evidence artifacts with a short retention and **only gates on
correctness** (not arbitrary shared-runner FPS).

**Verified CI:** commit `dcefe18c5b7efc3b55eef1076ef14c9bbcc2bfb2`,
Actions run `42273795156`, **48 tests PASS**.
Complete report: `MECHANICS_CPU_FULL_FRAME_COURT_20261106.json`.

| Research load | Prepared full Mechanics p50 | Prepared p95 | Legacy sampled p50 | Vertex L2 parity max |
|---|---:|---:|---:|---:|
| 3,266 vertices, 6,532 influences | 5.924 ms | 6.393 ms | 57.807 ms | 4.48e-16 |
| 8,192 vertices, 16,384 influences | 11.351 ms | 12.109 ms | 143.126 ms | 4.48e-16 |

These are **small sample sizes** (16 and 10 actual timed frames) on one
GitHub-hosted CPU. P95 under 16.67 ms on this *fixture* is **not product
60 FPS proof**. Product must also account for graphics, rendering, motion
decoding, input handling and larger articulated systems.

**Next identified bottleneck**: multi-goal SciPy least-squares currently
computes a numerical finite-difference Jacobian. Its cost grows with declared
joint DOF count. The 3-DOF benchmark is intentionally **not** extrapolated to
a 43-bone/129-parameter source skeleton: source-native rotation parameters
are not qualified CanonicalRig physical DOF truth. A separate
DOF/morphology-scaling court and analytically differentiated kinematics
should precede any latency or generic closure claim.

## 2026-11-06 — Analytic rotational FK Jacobian research candidate

The previous SciPy multi-goal solver used an expensive numerical finite-difference Jacobian. An opt-in ANALYTIC_RESEARCH_V1 mode now computes the exact rotational FK positional derivative: dP_tip/dtheta = world_axis × (P_tip - P_pivot) for ancestors of an effector. Drag/pin weights and angle regularization have exact matching Jacobian rows. The default remains NUMERIC_V1.

An independent centered finite-difference differential court (outside timed measurement) verifies the new derivative. Tests cover 3, 6, 9 and 15 rotational DOFs, pinned goal behavior, unsupported solver modes, unreachable targets, explicitly declared limits, and live prepared LBS interaction.

Verified GitHub Actions run 42275027914 at head 315d13edec75e71a009ec689df887684ccb3617e: 55 focused Mechanics tests PASS. Analytic vs numeric solver CPU p50 measured on the same GitHub runner: at 3 DOF 3.635 vs 4.194 ms (1.15x); 6 DOF 4.763 vs 6.917 ms (1.45x); 9 DOF 5.833 vs 9.776 ms (1.68x); 15 DOF 7.576 vs 15.445 ms (2.04x). Maximum independently checked derivative error was about 1.24e-9.

Versioned receipt: docs/research/MECHANICS_ANALYTIC_JACOBIAN_DOF_SCALE_20261106.json. Executable reproducibility: python -m tools.mechanics.benchmark_multigoal_jacobian_v1 --frames 8 --warmup 2 --out /tmp/dof_jacobian_scale.json

Scope: five joints with 3–15 declared rotational parameters, two IK goals and a position pin. This is a numerical/sparsity stress fixture, not 15 proven independently observable source controls. The 15-DOF fixture includes terminal joint rotations that do not move their own joint origins. It is NOT a 43-bone source correspondence court, DOF truth mint, force/contact/collision model, or product 60-FPS certification.

Next scientific barrier: genuinely longer articulated trees, mixed morphologies, source-frame axis correspondence, multi-contact and grip orientation, validated full frame time through source-bound canonical outputs. Keep the numeric fallback until that evidence exists.
