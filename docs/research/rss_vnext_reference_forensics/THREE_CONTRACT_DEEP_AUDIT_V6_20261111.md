# BUSTED / RSS.vNext — V6 Three-Contract Deep Architecture Audit

**Date:** 2026-11-11 | **Repo:** `merynz/RealSaS-OPT` | **Pinned main:** `ded09c49d311ea965acdb61493a624a9bcfe210a`

**Authority:** `AUDIT_OPEN / FREEZE_DENIED / NO_TRAINING_AUTHORIZATION / NO_PRODUCT_QUALIFICATION / CANDIDATE_FIXES_IN_REVIEW`.

## V6 implementation continuation — code and test evidence

The V6 report was originally a read-only forensic snapshot pinned to `ded09c49`. Its **original source observations below describe that baseline**, not a claim that the current PR still has those defects. The following corrections are now committed on short-lived review branch `audit/rss-vnext-three-contract-v6-20261111` in PR #448. They have not been merged into `main` or used to mint product authority.

| Ledger ID | Applied repair | Real regression/CI evidence | Remaining boundary |
|---|---|---|---|
| C01 | `canonical_rig_ir_v2.py`: entity IDs are mesh-and-proposal-local-addressed; qualification hash stays only as provenance | `test_control_ready_rig_v2.py` control-only and deform-edit regressions; `test_v6_typed_rig_skin_main_integration_v1.py` re-mints same skin from unchanged qualified carrier | persistent across new neural recompile/reindex still needs correspondence and revision migration policy |
| C02 | `rig_executor_v2.py`: exact `RIG_CONSTRAINT_OPERATOR_IDS_V1` mask supplied to AXIS, followed by a typed per-operator court before canonical mint | `test_axis_mira_v6_worker_producer_consumer_v1.py` CAS-to-generate probe; worker CI green | release Go provider still v1; P01 remains open |
| M01 | `mechanics_control_solver_v1.py`: zero-weight FOLLOW/AIM leave rotation untouched; fractional rotations interpolate on SO(3) | `test_mechanics_control_solver_v1.py` zero/half-weight, orthonormality and AIM regressions; Mechanics CI green | full DOF compliance and multi-constraint convergence still open |
| M02 | same solver: target residual is measured after blending and reconciled after all later chain constraints | `test_mechanics_control_solver_v1.py` weighted and subsequent overwrite tests; Mechanics CI green | product behavioral-proof envelope and full-body IK still unproved |
| CI integrity | three previously missing PR-concurrency declarations, one unsafe site publish cancellation policy, and aggregate fail-closed violation reporting corrected | `platform-developer-contracts-hosted` green | unrelated deployment readiness is not proven |

The named regression jobs (V6 typed, AXIS/MIRA worker, Mechanics subtree, unified model/Compiler, eight-way teacher and Platform developer contracts) reported **success** on the review branch. Workflow conclusions belong to their exact commit SHA; further edits must be reverified. The existing native authored-puppet E2E job is a genuine native consumer test but does **not** establish learned checkpoint-to-deploy closure. No checkpoint was trained and no real source-to-ProductRevision release was executed.

All other V6 ledger items remain open. Do not relabel the original 24 audit obligations as 24 resolved defects or a PASS report; C01/C02/M01/M02 have candidate source repairs and bounded engineering tests only.

## 1. Decision record

This is the second substantive pass after V4 and V5. It is not an architecture proposal masquerading as an audit: every finding below is split into a source claim, the variable whose truth is needed, the learned/lowered/executed meaning, a falsifier, and the exact acceptance test family. Source files were initially read through the connected GitHub repository at the pinned baseline commit; the full repository was not cloned locally. **This sentence refers to the initial V6 forensic snapshot:** at that point no integrated test was executed. Subsequent fixes and GitHub Actions test outcomes are recorded in the implementation continuation above. Four isolated equation-level examples and eight ledger-validator negatives were also executed locally; these do not promote a product PASS.

The governing contract remains: **observation → learned evidence → finite deterministic Compiler qualification → one canonical puppet → runtime behavior → proof → ProductRevision → Studio.** No preceding stage may infer an unproven value solely because the next stage requires it. Unknown is a legal outcome. Source art wins on observed rays; predicted views are not observations.

### Quantified review scope

| Plane | V6 recorded source-review obligations |
|---|---:|
| LEARNING | 9 |
| COMPILER | 4 |
| MECHANICS | 4 |
| GRAPHICS | 2 |
| MOTION | 2 |
| PLATFORM_STUDIO | 3 |
| **Total** | **24** |

Numbers are ledger records, **not an independently established count of production defects**. V4 had 36 capability rows; V5 found 310 *ledger-schema omissions*, not 310 bugs. V6 focuses on cross-plane falsifiers with code anchors.

## 2. Verified architectural reality — not proposed future behavior

```text
Current research inference:
  FRONT/BACK RGB + 13D cameras
    -> frozen vision -> early image semantics -> IRIS Z/H
    -> TESSA structural memory -> free-run topology proposal
    -> AXIS proposal & SkinTokens/MIRA soft expectation & dynamic candidate
    -> RSSvNextInferenceEvidenceIR.v1 (UNQUALIFIED)
    -> research candidate rig court, NOT a ProductRevision

Current production capability release:
  ProductInput v1 -> Shape v2 [IRIS v3, TESSA v3]
    -> Mesh v1 -> Rig v1 [AXIS v5.4.1] -> Skin v1 [MIRA v5.5]
  independent Python V6 Rig/Skin candidate executors exist
    but Go source registry still selects v1 descriptors

Current product rendering path:
  separate qualified asset + clip frame XYZ arrays
    -> native .rss baked clip -> C++ raster image -> GIF/preview
  not live native controls + skin + constraint + ray-stack execution
```

The research and released paths have individually meaningful components, but a single checkpoint loaded into the former does not select the latter. Any claim that "the model is trained and we merely plug in weights" is presently unproved.

## 3. Learning contract: objective reachability and admissible supervision

**Actual called graph, `RSSvNextStructuralTrainer.objective`:** it calls `encode_observations` then optional trusted-source novel view renderer, conditional FlexiCubes teacher, `_structural_memory`/TESSA, patch BCE, topology CE and topology motion preference. The total objective combines only those six families. It does *not* call `self.model.forward(... rig_sequence=...)`, `self.model.axis(...)`, `self.model.mira(...)`, `dynamic_refine(...)` or direct mesh pose loss. Consequently, the selected trainer cannot alone learn the free-running control/skin task; a head appearing in the model definition is not equivalent to receiving a task-owned gradient.

| Neural family | Direct gradient from active objective? | Indirect gradient / emission | Needed teacher (still admission-gated) |
|---|---|---|---|
| Image-token early semantic logits | Yes, if qualified semantic patch target + mask + positive weight | Also residual changes IRIS input | independent source-render patch axes and mask |
| Image semantic uncertainty | **No visible direct loss** | Emitted; no calibration consumer in trainer | probabilistic error/calibration target |
| IRIS scene planes / H / raw field | Conditional | shared for topology, FC and novel-renderer losses | source-qualified shape, closed-sign FC, view targets |
| IRIS UDF/sign logits (separate head) | **No selected loss** | Output exists; not fed into selected losses | source UDF and global sign with unknown mask |
| Four-way 3D presentation role head | **No role CE call** | gradients via TESSA context, conditional FC extraction | independently visible surface role and 3D address |
| Region membership/attachment affinities | **No matching/relation loss** | gradients may flow through structural memory | source region correspondence and intervention outcome |
| Affordance / control geometry prior | **No direct affordance loss** | influences TESSA topology tokens | motion/attachment measured affordance targets |
| TESSA discrete topology | Yes, qualified tokens or preference | free-run grammar can still diverge | canonical target grammar + motion-negative comparisons |
| AXIS rig/control/skin-code logits | **No selected trainer loss** | free-run only and separate worker | admissible sequential joint/tree/DOF/constraints/FSQ code grammar |
| MIRA FSQ/dense weights | **No selected trainer loss** | unified soft-code expectation; standalone separate codec | exact mesh/joint-bound source W, deform consequence |
| Dynamic vertex/weight refinement | **No selected trainer loss** | free-run when rig generated | posed-source strain/folds/contact/alpha tests |
| Graphics RGB/density/uncertainty | Optional novel-view loss | topology context may provide indirect gradients | source ray RGB/alpha and calibration; unknown depth |

**Important epistemic gap:** `training_engine.py` requires a string `semantic_qualification_key`, while the novel-view path invokes `validate_eight_way_teacher_receipts_v1(...)` with an external trusted registry and exact rendered byte hashes. String-presence is not the same level of teacher admission. FC signed-source guard similarly expects a nonempty key; absent external record proof in that call path, no training claim is licensed from the local check. These protections must be made uniform, not weakened.

**StdGEN-specific decision gate:** `semantic_equivalent_signed_field` uses a 4-role suppression construction over a signed field and has an explicit gate `qualified_signed_domain`. The FlexiCubes research bridge sets that boolean to true *internally for hypothesis computation*; its returned authority is only `SIGNED_HYPOTHESIS_ONLY`. Therefore compile/release must never interpret that internal computational choice as certified source sign. Four-role names in this candidate are presentation strata, not categorical hair/cloth/face truth for arbitrary morphology. The current novel view renderer is post-scene, not StdGEN pre-reconstruction RGB/normal diffusion; reference parity remains open.

**Corpus closure:** post-run evidence covers 370 candidates, 294 execution inputs, 268 computed-unqualified, 26 execution failures, and 76 preparation failures. Prior reported candidate family passes have zero overlap for response/TESSA. Source corrections exist but require new exact Stage49 re-evaluation. No head-wise eligible matrix, final teacher source-rights ledger, lineage-safe split or 16K-compatible materialized corpus is established by this audit. Research teacher data is still real eight-view; student product observation may be two-view as a distinct contract.

## 4. Compiler contract: decisions must be finite, source-bound and testable

**Canonical object split recommended for vNext release:**

```text
SourceObservationIR + QualifiedCameraIR + RightsReceipt
        │ student consumes only FRONT/BACK and allowed camera
        ▼
NeuralEvidenceIR {candidate mesh, anonymous regions, role/ray hypothesis,
                 rig token stream, skin-code stream, dynamic candidates,
                 unknown/provenance; NO canonical authority}
        ▼
Compiler guards: signed/open domain, topology+face address, source fidelity,
    motion consequence, registered operator grammar, joint/skin tuple,
    source appearance/ray admission, mutation+invalidation budget
        ▼
CanonicalMesh + DeformStructure + ControlProgram + CanonicalSkin +
CanonicalGraphics + CanonicalMechanics + CanonicalMotion
        ▼
QualifiedPuppetRevision + domain-specific proof receipt + native runtime ABI
```

**C01 persistent IDs:** current `_cid("J", qh, source_id)` uses `qh` over the whole qualified program. Even if controls alone change, `proposal_lineage_hash` and qualified hash change, so canonical joint identities churn. Fix must separate `DeformStructureIdentity` (mesh, joint coordinates/parent/frame, source address, normalized units) from `ControlProgramIdentity`. A control-only edit may invalidate control/motion/proof, but not skin W or canonical deform joints unless it actually modifies deform structure.

**C02 executable grammar:** the unified research pipeline correctly supplies `RIG_CONSTRAINT_OPERATOR_IDS_V1` for masked AXIS sampling. Separate V6 product rig worker invokes `generate_events` without `allowed_constraint_operators`. Release and model must negotiate one opcode set; unsupported proposed constraint cannot be replaced by an unrelated supported meaning or silently ignored.

**C03 proposal-to-product schema:** `RSSvNextInferenceEvidenceIRV1` can be persisted with exact source hashes and read back; it remains unqualified. Current registered product.mesh consumes `ShapeProposal v2`, not the unified evidence schema. An adapter cannot merely rename a JSON envelope: it must prove numeric normalization/face identity, source-camera/source-surface bindings, mesh qualification and exact descendants.

**C04 rest proof:** `mint_canonical_mechanics_v1` runs rest pose + static residual checks and emits `PASS_MECHANICS_EXECUTABLE`. This does not prove reachable target handling, shared multi-effector solve, DOF limits, collision, contact or secondary dynamics. Separate rest compatibility from interactive dynamic capability receipts, and validate full action traces before the editor presents the corresponding control as supported.

## 5. Runtime execution: behavior is the qualification target

### Mechanics

Current solver makes sequential, per-constraint updates to global joint positions. It is neither a shared unknown-vector solver with coupled contact constraints nor a full-body inverse dynamics program. The source code has support for FK/subtree, FABRIK IK, FOLLOW/COPY, AIM and STRETCH. Source inspection found three precise falsifiers: (1) zero-weight FOLLOW still sets explicit rotation, (2) IK residual is measured *before* applying blend weight, and (3) `dof_labels` are not consulted when applying generated position/rotation updates. A static control-grammar PASS therefore cannot imply behavioral correctness.

**Mathematical witnesses executed locally, not repo tests:**

| Witness | Source-grounded equation | Observed synthetic result |
|---|---|---|
| M01 FOLLOW | `p=(1-w)p+w target` but `explicit_rot=target_r` regardless w | at `w=0`, translation no-op, rotation non-no-op |
| M02 IK | `recorded r=||solved_tip-target||` before `pos=(1-w)pos+w solved` | `w=0`, recorded 0, actual 2.828427 |
| C01 ID | `_cid("J", qualification_hash, source_id)` | changing qualification input changes stable joint ID despite same source ID |
| L02 FSQ | soft expected code `Σp_i code_i` versus hard argmax | `[−1,+1]` with p=0.5 → 0 vs −1 |

These do not show product artifacts were rendered incorrectly; production adversarial tests A07/A10/A12/A20 remain to run on real packages. The central desired behavior remains: sword tip drag with fixed feet/body adaptation, rotate 45°, release and attack, preserving the same mesh/rig/skin identities.

### Secondary motion

There is no activated XPBD/spring integrator in current production main. `solver_registry.py` lists XPBD under historical external implementation authority and its `EXECUTABLE_SOLVERS` is empty. `temporal_response_v1` explicitly stores authored response evidence, not inferred physical parameters. The required lowering is: model proposes region + parent joint/driver + attachment coordinates + candidate dynamic mode + uncertainty; Compiler validates source consequence / support envelope and instantiates a finite dynamic program; Mechanics advances state `s(t+dt)`, evaluates collision/limits and deterministic replay; Graphics samples the new pose. No component may fabricate spring constants as source truth. Hard unknown → reject or declared static presentation, not secret physics.

### Graphics

`CompiledRayGraphicsV2` computes per-ray K-hit source-over consequences from a qualified ray evidence record; `graphics_local_order_v2.hpp` has a real local precedence algorithm and native source tests. However `runtime_v4_reference.cpp::render_rgba` interpolates baked vertex arrays and writes the selected nearest eligible RGBA pixel into a single depth/color sample; it does not call the local order routine or accumulate 3-layer translucent compositing. Until that consumer exists, a compiled graphics evidence artifact does not confer the product ability to preserve crossing orders/stacked transparency during arbitrary edits.

### Motion, native runtime and Studio

`control_motion_runtime_v2` can sample canonical control tracks and re-run Python Mechanics. `playback_v4_control_bridge_v1` produces editable control-keyframe evidence, checks the pose with the Python solver, and exports baked 3D vertices. The native player then renders those frames; it does not execute control graph/skin on user drag. The Platform render tail likewise selects frame collections and invokes native baked player. `studio_persistence.js` references session/history/restore/requalify endpoints; the active BUSTED/LivingCompile route dispatchers do not implement those literal addresses in the reviewed source. An alternative external backend deployment is not proven. Hence neither live-native puppet nor editable persistent Studio E2E is currently a supported released capability.

## 6. Every finding and its concrete falsifier

| ID | Priority | Plane | Claim | Counterexample or red test |
|---|---|---|---|---|
| `L01` | P0 | LEARNING | AXIS causal control head is not optimized by structural trainer | Trainer only invokes encode_observations, TESSA, FC and novel renderer, not self.model.axis |
| `L02` | P0 | LEARNING | MIRA unified expectation and product discrete decoder are non-equivalent training/inference contracts | Unified uses soft expected codebook and local-H features; V2 executor uses discrete IDs with 7D/25D canonical conditioning |
| `L03` | P0 | LEARNING | Four-way 3D presentation-role field has no direct visible-surface teacher loss in active trainer | Two unrelated class logits could agree with topology loss while reversed on unseen source surfaces |
| `L04` | P0 | LEARNING | UDF/sign readouts lack active direct target and are absent from selected structural losses | FC trains IRIS field head, not the separate udf/sign heads |
| `L05` | P1 | LEARNING | Source-semantic uncertainty is emitted without uncertainty calibration objective | Arbitrary log-variance could pass any BCE-only training |
| `L06` | P0 | LEARNING | Region slots and directed attachment affinity lack persistent matching/teacher closure | Slot permutations produce equivalent logits but different owner IDs |
| `L07` | P0 | LEARNING | Dynamic refinement and deformation quality have no active training loss in unified structural trainer | Dynamic output only produced in forward with rig_sequence or free-run controls, not structural trainer |
| `L08` | P0 | LEARNING | Supervision-admission logic differs materially across families | Only eight-way teacher path joins an independently stored trusted record |
| `L09` | P0 | LEARNING | Corpus370 is not a head-wise admitted training set | Prior post-run: 23 response and 14 TESSA passes but zero overlap; repairs unrerun |
| `C01` | P0 | COMPILER | Canonical joint IDs hash entire RigProgram qualification instead of deform-only identity | Control variant changes RigProgram proposal/qualification hash even if joint graph is identical |
| `C02` | P0 | COMPILER | AXIS V6 worker does not provide executable opcode registry when sampling constraints | Axis V6 vocabulary includes operators absent from Mechanics V1 schema |
| `C03` | P0 | COMPILER | Checkpoint evidence path does not mint canonical mesh on registered worker route | Synthetic mesh evidence accepted by candidate court but not production canonical mesh |
| `C04` | P1 | COMPILER | Qualified rest Mechanics status does not prove dynamic constraint behavior | Rest pose at zero control has no stressed multi-chain target |
| `M01` | P0 | MECHANICS | FOLLOW weight zero mutates target rotation | FOLLOW p0=0 and target rotation != identity changes joint orientation |
| `M02` | P0 | MECHANICS | IK residual is recorded from full solver result before weighted pose blend | IK p0=0 with distant target is no-op but recorded residual may be near zero |
| `M03` | P0 | MECHANICS | Rig DOF labels are metadata rather than enforced coordinates in active solver | Output declares only RX yet FK translation changes its location |
| `M04` | P0 | MECHANICS | Secondary dynamic program is not an active current-main runtime operator | Tail_likeness alone does not instantiate a temporal program |
| `G01` | P0 | GRAPHICS | Native runtime V4 flattens closest opaque/cutout sample; not ray-stack source-over | Three alpha=0.5 layers should composite with nonzero contribution from all |
| `G02` | P1 | GRAPHICS | Native pixel-local order implementation is not consumed by RuntimeV4 reference raster | Ray A front A, Ray B front B while view-global cannot satisfy both |
| `R01` | P0 | MOTION | Native V4 playback consumes baked XYZ frames rather than live control+skin program | Move sword-tip between keyframes and request a new target runtime pose |
| `R02` | P1 | MOTION | CanonicalMotion control-track sampling exists in Python, but Platform render tail selects baked frame arrays | Motion clip with control track but no baked frames cannot render on current platform tail |
| `P01` | P0 | PLATFORM_STUDIO | Released product catalogue still names old model interfaces, not unified checkpoint closure | Go release seal resolves to old modules by model architecture ID |
| `P02` | P0 | PLATFORM_STUDIO | Studio persistence JS calls routes not implemented by current BUSTED HTTP dispatcher | Reload saved custom control after restart cannot be proven from current server path |
| `P03` | P0 | PLATFORM_STUDIO | No primary native source-to-product E2E has been executed | Current API rejects /api/compile/start and /api/compile |

## 7. Freeze acceptance matrix: no hollow PASS

A capability is `COMPLETE` **only** if its accepted source evidence, inferred variable, Compiler canonical decision, Runtime state transition, visual result, and independent proof all have bound executing test IDs and receipts. Passing only the neural output schema or a deterministic static court is insufficient.

| A-ID | Real test obligation | Required owner(s) |
|---|---|---|
| `A02` | No teacher-private extra views, source mesh/joint/rig/skin labels in student input; receipt hashes match exact source | Learning/Truth |
| `A03` | Per-head admitted target & mask, gradient test (direct/indirect/zero), strict resume, no-label abstention | Learning/Truth |
| `A04` | 4-way source-visible role extraction and hidden/unknown exclusion; signed vs open source | Learning/Compiler |
| `A05` | Region matching under slot permutation, identical source address across reopen/recompile | Learning/Compiler/Studio |
| `A06` | Free-run topology → candidate qualification → CanonicalMesh; invalid faces fail closed | Learning/Compiler |
| `A07` | Control-only program edit preserves deform IDs and skin bytes; deform changes invalidate precisely | Compiler/Platform |
| `A08` | Release-owned executable constraint vocabulary masks causal AXIS decoding and Compiler validation | Learning/Compiler/Mechanics |
| `A09` | Disallowed DOF remains invariant under direct+IK+contact operations | Mechanics |
| `A10` | Zero-weight FOLLOW/IK no-op, true final residual after blending and sequential/coupled solve | Mechanics/Proof |
| `A11` | Exact LBS/inverse-rest correctness on nonuniform scale and arbitrary hierarchy | Mechanics/Graphics |
| `A12` | Sword drag + locked feet + torso compensation + camera rotate + attack | Compiler/Mechanics/Motion/Graphics |
| `A13` | Held object bind/collision/release, same canonical mesh+attachment lineage | Compiler/Mechanics/Graphics |
| `A14` | Secondary tail/wing dynamic program: fixed-step state, deterministic seed/replay, pose-aware render | Mechanics/Graphics |
| `A15` | Ray-local crossing order switches with camera/pose; no view-global cycle deletion | Graphics/Native |
| `A16` | Three alpha layers exact premultiplied source-over, Python/native pixel parity | Graphics/Native |
| `A17` | Independent art edits on one deform surface, clip/occlusion, persisted edit/reopen | Graphics/Studio |
| `A18` | Control tracks retarget to different morphology, fail closed on unsupported relations | Motion/Mechanics |
| `A19` | Native ABI receives live rig/skin/control commands, emits new pose without new baked file | Native/Platform |
| `A20` | Actual source → checkpoint → worker → Compiler → native → Platform → Studio → edit → reopen | All layers |
| `A21` | Invalid source, absent rights, unknown teacher, hash/camera/lineage/operator drift all fail closed | Truth/Compiler/Platform |
| `A22` | Cold deployment with tenant isolation, saved Studio revision, same-source idempotent retry | Platform/Studio |

## 8. Architectural decisions before code changes

1. **Model ABI:** pick the exact observed-view contract for the first real product release; keep eight native teacher views train-only. Freeze detailed semantic role schema, source-calibrated camera contract and missing-target policy before training.
2. **One neural skin truth:** decide if V6 uses a single causal discrete FSQ+decoder on canonical Mesh/Rig (recommended) or soft expectation with a provable hard-free-run parity contract. Eliminate two non-equivalent skin decoders from the production execution path.
3. **Stable identity:** define `DeformStructureID`, `ControlProgramID`, `PresentationGraphID`, `PuppetRevisionID` and source address mapping, each with dependency-scoped invalidation. Make control edits and requalification explicitly versioned.
4. **Finite mechanics opcodes:** add pose projection for declared DOFs, limit/contact and multi-goal semantics; preserve existing constrained FK/IK algorithms where correct. Non-convergent/incompatible goals must produce typed failure residuals, not a visually plausible final pose.
5. **Graphics compositing:** native renderer must consume the Compiler-qualified ray stack/presentation graph, recompute per changed pose, render source-authorized colors and return unknown on missing source truth. Reuse native local-order code rather than duplicating authority.
6. **Secondary program:** choose finite opcode(s), time-base, deterministic integration, source teacher/mask policy and state serialization; no inferred physics from semantic type alone.
7. **Platform release:** bind unified architecture digest, checkpoint family requirements, executable worker contracts, canonical artifact outputs, native ABI and Studio feature flags to one release. Do not declare success because source contains adapters.

## 9. Exact next actions (implementation work order; not authorization to train)

**Gate 1 — learning ownership:** register all trainable parameter groups and their direct/indirect gradient roots; introduce teacher receipt/mask contracts for each task, define morphological unknown/admissible domains, and run real tiny CPU backward+resume tests. AXIS/MIRA/role/sign/affordance/dynamic must have explicitly admitted objectives or be excluded from freeze.

**Gate 2 — Compiler semantics:** implement control-only identity independence (A07), release operator masks (A08), single candidate-to-canonical mesh adapter (A06), corrected IK/FOLLOW residuals (A10) and DOF enforcement (A09). Test with three or more morphologies and purposely adversarial configurations, not Knight preservation conditions.

**Gate 3 — Runtime behavior:** implement native live control ABI, per-pixel K-hit render, deterministic secondary state and source/presentation addresses; compare Python/native pixel and pose decisions (A11–A19).

**Gate 4 — single product lane:** wire Go/Temporal, CAS, ModelRelease and Studio persistence/requalification; execute synthetic E2E via real product components (A20–A22), then model ABI smoke using an actually loadable unlearned checkpoint without claiming generalization.

Only then consider a freeze vote. Corpus admission → materialization → training → learned unseen qualification → deploy is a subsequent program. No historic Knight fallback may veto generic redesign; historic artifacts remain references and can be reused only when their dependencies remain exact.

## 10. Verifiability and limitations

- Repository source inspection: performed via connected GitHub at pinned main. A source call-site is **not** a runtime execution receipt.
- Numerical/algebraic counterexamples: four passed independent reproductions in `BUSTED_RSSvNext_V6_Algebraic_Proof_2026-11-11.json`; no production module imports.
- Ledger validation: 24 records validated with zero structural violations and eight adversarial validator self-tests. This is a **quality check on the audit record only**.
- No product Python/Go/C++ tests, GPU forward/backward, train job, live-native shader test, Platform smoke or Browser Studio proof were executed in this review. Status stays `FREEZE_DENIED_AUDIT_IN_PROGRESS`.

## 11. Source citations (pinned repo)

Every `source_paths` item in the machine-readable ledger resolves under `https://github.com/merynz/RealSaS-OPT/blob/ded09c49d311ea965acdb61493a624a9bcfe210a/<path>`. The code sections summarized above include `RSSvNextStructuralTrainer.objective`, `RSSvNextUnifiedCharacterModel._structural_memory/propose_teacher_free`, `RSSvNextSkinCoupler.forward`, `mint_canonical_rig_v2`, `solve_control_pose_v1`, `compile_ray_graphics_v2`, `ReferenceRuntime::render_rgba`, `execute_rig_capability_v2`, `execute_skin_capability_v2`, `ProductProviders`, `studio_persistence.js` and `LivingCompileApplication.dispatch`. Each is linked to at least one ledger finding. Upstream StdGEN/SkinTokens/AniGen forensics from V5 remain open and should not be treated as licensed code imports or complete training recipes.
