# RealSaS — Historical Product-Power Rebind Manifest V1

**Date:** 2026-09-13  
**Branch:** `audit/historical-product-power-rebind-20260913`  
**Base:** `main` at audit start (`bbdde6ec851717dddfba52107a28343e91864d68`)  
**Status:** `AUDIT_MANIFEST__NO_PROMOTION__NO_SGW_AUTHORITY_CHANGE`  
**Purpose:** recover still-useful historical Compiler/runtime/product machinery behind the current typed authority without restoring obsolete ownership, parallel truth, or old front-brain semantics.

## 0. Decision

This is a **second-pass product-power restoration audit**, not a rollback of the 2026-09-03/04 restoration.

The first restoration correctly promoted the minimum current execution closure, proof/repair contracts, exact runtime/export interlock and native runtime while intentionally leaving the full historical Compiler chassis external. That decision was correct for authority cleanup. The present product state now exposes consumers that were not yet ready at that time: a real corrected S/G/W/M/B pipeline, professional motion, one-click compile transaction, exact runtime/export identity, and the Living Compile editor shell.

Therefore the new rule is:

> **Do not rewrite a historical mechanism that solves a current product gap until its old source has been source-diffed against the current implementation and classified here. Do not restore its old authority. Restore only reusable behavior behind current typed ownership.**

The current `IRIS/H1 -> S -> Geppetto proposal -> Compiler G -> Arachne proposal -> Compiler W -> M/B -> components -> motion -> proof -> runtime/export` authority remains binding.

## 1. Scheduling decision relative to Mesh and Arachne

The architecture/reuse decision in this manifest should be completed **before freezing the final Arachne/product-consumer contract and before productizing the corrected mesh into the final compile transaction**.

It does **not** cancel, modify or reinterpret the already-preregistered active mesh scientific experiment. That experiment may continue independently under its frozen contract. No threshold, treatment, mesh gate, Geppetto gate or Arachne scientific criterion is changed by this audit.

Practical order:

```text
ACTIVE SCIENCE (unchanged)
  current mesh experiment / current Geppetto evidence as already authorized

IN PARALLEL NOW
  P0 historical product-power rebind audit + transaction/identity contract

THEN
  fresh/corrected G and W authority
  -> final corrected M/B product integration
  -> motion product compiler
  -> exact runtime/export + Living Compile
```

Why P0 comes now: the Compiler/runtime consumer boundary decides the exact artifact IDs, hashes, stage-result envelopes and failure semantics that Mesh/G/W must bind into. Freezing those downstream interfaces after Arachne would create avoidable rework.

## 2. Binding authorities that this audit may not override

1. `CURRENT_STATE.md` remains repository-wide continuation authority.
2. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md` remains architecture authority.
3. `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md` remains the ownership firewall.
4. Current Compiler owns qualified/canonical state and runtime-package authority.
5. Learned models emit proposals/evidence only.
6. Runtime/editor/export may consume/repack qualified state but may not mint S/G/W/M/B or hidden topology.
7. The exact qualified mesh is the runtime/export mesh. Hidden retriangulation or replacement alpha-clipped topology is forbidden.
8. Historical code is evidence/implementation reserve until an explicit typed rebind promotion gate passes.
9. Historical numerical kernels do not become current merely because they once achieved stronger metrics.
10. `PRODUCT_PASS` is not implied by this manifest.

## 3. Historical authority basis

### H0 — Aug-9 v0.5 chassis

Historical source authority:

`RealSaS_M4_v0_5_CANONICAL_MECHANICAL_MEANING_SOURCE.zip`

SHA-256:

`03a819f01d3cc39e806cc30ae291912718d114ca3ff6b75dc2b854d1bbfbf130`

The historical maximum audit records this family as the filesystem/Python Compiler chassis and the primary proof/repair/orchestration source. The current repository already carries selected byte-exact leaves and semantic rebinds; the full monolithic chassis is intentionally not mirrored.

### H1 — R5_3 authority discipline

Preserve semantics only:

- Compiler-minted canonical IDs;
- one graph/product owner;
- deterministic reducer/DAG discipline;
- no second authority at the Compiler boundary.

### H2 — v97.39 single-truth invariant

Preserve as hard invariant:

> Proof, repair, final package, runtime export and editor/runtime inspection must bind the same exact canonical product state and lineage hashes.

Historical implementation shape is not authoritative; the invariant is.

### H3 — late-May numerical maximum

CDT/cotangent, BBW/KKT, ARAP and XPBD/contact remain source-diff reserves. They are **not** bulk-restored in P0. They may be reopened only by a demonstrated current product blocker and an explicit typed promotion gate.

### H4 — v93 architecture-diet inventory

The v93 audit is used as a source-discovery map. Its central finding remains binding for this audit: preserve algorithms/mechanisms, eliminate fragmented authority. `PARK`, `MERGE` and `REWRITE` are not equivalent to `useless`.

## 4. Disposition vocabulary

| Disposition | Meaning |
|---|---|
| `KEEP_CURRENT` | Current implementation already owns the responsibility; historical code is only a regression/reference source. |
| `REUSE_CORE_REBIND_SHELL` | Recover algorithm/state-machine behavior but retype all inputs/outputs to current IRs and remove old ownership/side effects. |
| `REUSE_AS_SUBORDINATE_SERVICE` | Historical mechanism may execute, but only behind current Compiler ownership and explicit registration. |
| `PORT_TESTS_AND_INVARIANTS` | Do not port implementation yet; recover its tests, failure cases and invariants. |
| `REOPEN_SOURCE_DIFF` | Need exact source bytes and file-level dependency closure before deciding implementation reuse. |
| `REFERENCE_ONLY` | Useful design/behavior reference, not executable current code. |
| `DEFER_OPTIONAL_BACKEND` | Real capability but not required for core product closure. |
| `REJECT_OBSOLETE_AUTHORITY` | Do not restore; incompatible with current S/G/W/product ownership. |

## 5. P0 — execute now, before final Arachne/product-consumer contract freeze

P0 is architecture and reuse closure. It must not change active scientific gates.

| ID | Historical mechanism/source | Old responsibility | Current home / equivalent | Current gap solved | Disposition | Promotion gate |
|---|---|---|---|---|---|---|
| P0-01 | v0.5 `realsas_orchestrator/pipeline.py` (~5k LOC historical monolith) | end-to-end stage orchestration, retry, feedback, qualification, product progression | **no equivalent full transaction owner**; current responsibilities are distributed across `compiler/realsas_compiler_core`, services, experiments and product shell | one deterministic compile transaction from qualified stage outputs to product/runtime | `REUSE_CORE_REBIND_SHELL` | exact source SHA verified; extract stage-state/retry/feedback behavior only; zero old front-brain imports; current typed artifact-only I/O; deterministic replay tests |
| P0-02 | v0.5 `meaningful_proof_loop.py` + proof-loop helpers | repeated probe -> measure -> diagnose -> repair/re-proof control | current proof/attribution/repair services exist, but orchestration is intentionally thin/fail-closed | operational proof loop with bounded attempts and exact child-lineage re-proof | `REOPEN_SOURCE_DIFF` then `REUSE_CORE_REBIND_SHELL` | all loops terminate; attempt budget explicit; no implicit repair credit; same-probe re-proof mandatory |
| P0-03 | historical retry/feedback/qualification modules adjacent to orchestrator | retry policy, residual routing, fail/abstain propagation | current typed proof reports + repair directives | standardized stage failure semantics instead of ad-hoc runner logic | `REOPEN_SOURCE_DIFF` | typed failure taxonomy; no exception-swallow PASS; deterministic retry decisions |
| P0-04 | v93 `SaSCanonicalProductionPipeline` (715 nonblank LOC) | stage interface/context/trace plus a broad single-spine pipeline | no direct current equivalent; current branch runners are experiment-specific | reusable stage interface, trace, result envelope, stop/rollback behavior | `PORT_TESTS_AND_INVARIANTS` + `REFERENCE_ONLY` implementation | recover stage/result semantics without Unity assets, AuthoringTruth, file-exists gates or secondary truth |
| P0-05 | v93 `SaSAgentGoldenPathPipeline` (645 LOC) | external Build -> Bake -> Proof golden-path API with hashes | `product/living_compile` consumes finished bundles but does not own compile orchestration | user/agent-facing `Compile` command wrapper | `REUSE_CORE_REBIND_SHELL` | wrapper calls one canonical transaction only; no build/proof ownership; exact hashes returned |
| P0-06 | `SaSProductionPipelineContracts.cs` + runtime `SaSPipelineContracts.cs` | request/result, stage issues, blockers, artifacts, clip coverage, policy/guard types | current V4 IRs + proof bundle + bundle routes, but no single compile-job envelope | typed `CompileRequest`, `CompileStageResult`, `CompileIssue`, `CompileArtifactRef`, terminal status | `REUSE_CORE_REBIND_SHELL` | vocabulary mapped to current IR names; duplicate stage enums removed; hashes mandatory |
| P0-07 | historical `SaSCanonicalDataGraph` / builder | stage dependency graph, residuals, correction authority | current authority/experiment ledgers + typed product graph | machine-readable execution DAG + residual routing diagnostics | `REUSE_AS_SUBORDINATE_SERVICE` | diagnostic only; cannot PASS/FAIL product independently; all nodes point to exact artifact IDs/hashes |
| P0-08 | v97.39 final-package single-truth rebind | force proof and final package to observe same canonical graph snapshot | current `ProductProofBundleIR`, bundle hashing, current runtime-v2 projection | end-to-end binding firewall | `KEEP_CURRENT` + `PORT_TESTS_AND_INVARIANTS` | add regression asserting S/G/W/M/B/components/motion/proof/export hashes are equal to transaction-selected identities |
| P0-09 | v93 `SaSProductionPathLockdownPipeline` | non-empty track/clip coverage and production-manifest checks | current proof/runtime contracts do not yet express full professional clip coverage | product-ready motion/runtime coverage gate | `REUSE_CORE_REBIND_SHELL` | gate consumes `QualifiedMotionIR`/runtime manifest only; no second acceptance authority; fail-closed |
| P0-10 | historical `SaSProductionOperations` facade | editor-facing production operations | Living Compile/UI | thin public operations API | `REFERENCE_ONLY` | UI operations call canonical transaction API; no direct artifact mutation |
| P0-11 | historical `SaSCharacterCompiler.SourceToRuntimeProduct` spine | source-to-runtime ownership | current typed Compiler split | naming/lifecycle precedent for one production spine | `REFERENCE_ONLY` | no old source/decomposition/Unity ownership revived |
| P0-12 | `SaSPartIntakePipeline` / `SaSBuildCharacterPipeline` | stage assembly and thin build facade | current S/G/W/M/B qualification modules | facade and transaction decomposition patterns | `PORT_TESTS_AND_INVARIANTS` | current IR-only; no stable Generated-path mutation |
| P0-13 | current `bundle_routes.py`, `hashing.py`, `v4_types.py`, proof bundle | artifact routing, identities, hashes | current canonical | already the correct authority basis for the new transaction | `KEEP_CURRENT` | new orchestrator must consume these, never replace them |
| P0-14 | current `product/living_compile/` | editor/product shell, user edit layer, preview | current canonical consumer | destination UI for compile transaction and editable puppet | `KEEP_CURRENT` | compile orchestration remains outside UI; user edits remain sidecar + requalification required |

### P0 required new current contract

The historical code must not dictate the schema. P0 should result in a small current typed transaction layer, conceptually:

```text
CompileRequestIR
  source/evidence refs
  requested product capabilities
  deterministic policy/version IDs

CompileTransactionIR
  run_id
  exact input hashes
  stage DAG
  selected artifact identities
  per-stage status {PENDING,RUNNING,PASS,FAIL,ABSTAIN,SKIPPED}
  bounded attempts
  residual/failure signatures
  proof lineage

CompileResultIR
  exact S/G/W/M/B/component/motion IDs + hashes
  ProductProofBundleIR
  RuntimePackageRef (only if proof/export gates pass)
  editor bundle refs
  terminal status
```

This is a **current schema** informed by historical orchestration behavior, not a restoration of old classes.

## 6. P1 — bind corrected G/W/M/B into the transaction after their scientific/product gates close

| ID | Mechanism | Current gap | Disposition | Gate |
|---|---|---|---|---|
| P1-01 | current Compiler skeleton qualification | transaction-selected exact G | `KEEP_CURRENT` | selected G hash recorded once and used downstream |
| P1-02 | current Compiler skin qualification | transaction-selected exact W | `KEEP_CURRENT` | selected W hash recorded once and used downstream |
| P1-03 | current directional product mesh qualifier | exact current M product authority | `KEEP_CURRENT` | active frozen mesh policy passes on real corrected inputs; transaction cannot substitute candidate coverage |
| P1-04 | current mesh-skin binding | exact B = current M + current W | `KEEP_CURRENT` | deterministic support-bound transfer; B binds exact M/W/S IDs |
| P1-05 | current component/mechanical assembly | explicit visible part ownership/attachments | `KEEP_CURRENT` | no dropped qualified visible component |
| P1-06 | v97.39-style continuity assertions | prevent branch-local/result-local identity drift | `PORT_TESTS_AND_INVARIANTS` | all stage consumers prove exact selected upstream hashes |

No historical CDT, BBW/KKT or old rig solver becomes current authority in P1. If the active corrected mesh experiment identifies a specific numerical blocker, open a separate preregistered source-diff lane; do not mutate this manifest post-hoc to force a historical backend.

## 7. P2 — professional Motion / deformation / secondary response restoration

These mechanisms become meaningful only after exact corrected S/G/W/M/B exists.

| ID | Historical mechanism/source | Capability worth recovering | Current equivalent/gap | Disposition | Required rebind / gate |
|---|---|---|---|---|---|
| P2-01 | `SaSClosedLoopMotionPipeline2D` (419 LOC) | iterative motion retune around measured failures | current motion lane is proof-oriented and rotation-only | `REUSE_CORE_REBIND_SHELL` | input `MotionProposalIR` + exact product; output proposed correction/retune only; acceptance remains Compiler proof |
| P2-02 | `SaSMotionAutoRepair2D` | local motion repair candidates | current repair execution intentionally has 0 promoted executors | `REOPEN_SOURCE_DIFF` | separate promotion per operation; child lineage + same-probe re-proof mandatory |
| P2-03 | `SaSPhaseAwareMotionGenerator` | phase-aware authored/procedural clip shaping | professional preset pipeline open | `REUSE_AS_SUBORDINATE_SERVICE` | emits proposal curves only; no canonical motion authority |
| P2-04 | `SaSPhysicalMotionCalibration2D` | physically coherent calibration | no full professional calibrator | `REUSE_AS_SUBORDINATE_SERVICE` | deterministic calibration report + proposed edits, bound to exact clip/product |
| P2-05 | `SaSAdaptivePhaseTiming2D` | timing adaptation | professional quality passes open | `REUSE_AS_SUBORDINATE_SERVICE` | proposal/pass with explicit before/after metrics |
| P2-06 | `SaSMinimumJerkMotionPass2D` | trajectory smoothing/minimum jerk | professional clip polish open | `REUSE_AS_SUBORDINATE_SERVICE` | never allowed to move constrained contacts/attachments outside tolerance |
| P2-07 | `SaSLoopSeamRepairPass2D` | loop closure/idle-run seam repair | loop-quality product gate open | `REUSE_AS_SUBORDINATE_SERVICE` | exact endpoint/velocity continuity metrics; child clip lineage |
| P2-08 | `SaSImpactSpringDamperMotionPass2D` | impact/settle response | action polish/secondary response open | `REUSE_AS_SUBORDINATE_SERVICE` | proposal only; deterministic contact/limit proof remains authoritative |
| P2-09 | `SaSOrganicFluidMotionPass2D`, `SaSPerceptualFluidityPass2D` | style/readability/fluidity shaping | current motion proof is mechanical, not artist-quality | `REOPEN_SOURCE_DIFF` | objective metrics recovered as diagnostics first; no aesthetic score alone can PASS product |
| P2-10 | motion quality analyzers/scorers (transition matrix, readability, quality evaluator) | transition/readability diagnostics | product motion quality gate incomplete | `PORT_TESTS_AND_INVARIANTS` | metrics feed evidence; final acceptance remains frozen current policy |
| P2-11 | `SaSBakeActionPipeline` | clip-to-runtime bake lifecycle | current motion-bake path handles qualification-owned frames but not full professional authoring lifecycle | `REUSE_CORE_REBIND_SHELL` | consumes `QualifiedMotionIR`; exact frames remain proof-owned; no export-time solver replay |
| P2-12 | `SaSMotionGraphAsset` | action definitions, envelopes, transitions, motion metadata | no single modern professional motion-program IR | `REUSE_CORE_REBIND_SHELL` | split product program from proof/evidence; derive current `MotionProgramIR`/clip graph, not Unity asset authority |
| P2-13 | `SaSPhysicalResponseCompiler` (historical ~1230 LOC) | dash/knockback/wind/cloth/secondary response bus | secondary/physical response product layer open | `REOPEN_SOURCE_DIFF` then `REUSE_CORE_REBIND_SHELL` | split signal/response proposal from proof; cannot write baked product/proof itself |
| P2-14 | historical draw-order / visibility tracks (`SaSPartDrawOrderKey` etc.) | animated sorting/order/visibility | current Living Compile authoring needs professional non-bone tracks | `REUSE_CORE_REBIND_SHELL` | current component IDs + exact view scope; validate against qualified components; no topology mutation |
| P2-15 | integrated motion proof pipeline | dynamic validation composition | current proof exists but product-quality evidence is incomplete | `PORT_TESTS_AND_INVARIANTS` | reuse failure cases/metrics, not old acceptance owner |

### P2 target contract

```text
intent / artist clip / learned proposal
  -> MotionProposalIR
  -> deterministic current Motion Compiler
       retarget
       root trajectory
       contacts
       joint limits
       attachments
       draw order / visibility legality
       optional promoted polish passes
  -> QualifiedMotionIR
  -> exact product-state motion bake
  -> V0..V7 dynamic proof
  -> editable curves + live puppet / sprite bake projection
```

Historical motion modules are subordinate proposal/repair services only.

## 8. P3 — runtime, state machine, export and editor product power

| ID | Historical/current mechanism | Capability | Disposition | Gate |
|---|---|---|---|---|
| P3-01 | current `compiler/realsas_compiler_services/export/current_v4_runtime_v2.py` | exact proof/bake -> runtime package projection | `KEEP_CURRENT` | exact selected S/G/W/M/B/component/motion/proof hashes; no solver replay |
| P3-02 | current `runtime/realsas_cpp` | native `.rss/.rsr` consumer | `KEEP_CURRENT` | CMake/ABI/CTEST + exact package-open/render interlock |
| P3-03 | historical `SaSAnimationStateMachineController` | runtime action/state transition conductor | `REUSE_CORE_REBIND_SHELL` | consumes qualified motion program only; no compile/proof authority |
| P3-04 | historical `SaSBakedCharacterRuntimeController` | game-facing runtime conductor | `REUSE_CORE_REBIND_SHELL` | adapter over current runtime package; must not reconstruct mesh/skin/rig |
| P3-05 | historical baked runtime consumption report | runtime consumption evidence | `PORT_TESTS_AND_INVARIANTS` | proves exact package hash, selected clip/state, mesh identity and frames actually consumed |
| P3-06 | historical `.rss/.realsas/.rsr` product boundary | engine-neutral export contract | `KEEP_CURRENT` + source-diff only if fields missing | backward-compatible additions only when current product requires them |
| P3-07 | Living Compile recovered editor | rig/mesh/weight/clip edit + preview | `KEEP_CURRENT` | compile button delegates to one canonical transaction; edits trigger requalification |
| P3-08 | parked clip/recipe/editor surfaces (`SaSClipPreviewWindow`, recipe/compiler menus, authoring-program editor) | UX/workflow ideas | `REFERENCE_ONLY` | mine interaction patterns only; no old serialized truth/Unity generated-path authority |
| P3-09 | GoldenPath/handoff/report layers | one-click result surfacing | `REFERENCE_ONLY`/thin wrapper | final reports aggregate canonical transaction facts; they cannot define product truth |

## 9. P4 — optional high-power backends, explicitly deferred from core closure

These may become valuable after the core product works, but they should not block fresh G/W/M/B or professional motion closure unless a measured blocker points to them.

| ID | Historical backend | Disposition now | Reopen condition |
|---|---|---|---|
| P4-01 | late-May CDT/cotangent mesh maximum | `DEFER_OPTIONAL_BACKEND` | current typed mesh solver fails a preregistered product-quality/capability requirement that historical backend plausibly addresses; exact support-lineage adapter possible |
| P4-02 | BBW / active-set / KKT weights | `DEFER_OPTIONAL_BACKEND` | fresh Arachne/qualified W demonstrates a specific mechanics/regularization blocker; must remain proposal/fallback, never second W owner |
| P4-03 | ARAP corrective deformation | `DEFER_OPTIONAL_BACKEND` | LBS/exact current deformation passes legality but fails a declared high-quality deformation target; exact product identity and proof-owned correction contract available |
| P4-04 | XPBD/contact/SDF secondary dynamics | `DEFER_OPTIONAL_BACKEND` | secondary dynamics becomes a declared product capability after core motion/product closure |
| P4-05 | historical rig assembly/graph optimization stronger variants | `DEFER_OPTIONAL_BACKEND` | fresh Geppetto + current graph qualifier exposes a measured graph/assembly blocker not solvable within current authority |

## 10. Explicit rejects — do not restore

| Historical behavior | Decision | Reason |
|---|---|---|
| old front-brain image decomposition / teacher-exact owner ontology | `REJECT_OBSOLETE_AUTHORITY` | IRIS/H1 + current substrate own observation/evidence boundary |
| `SaSAuthoringTruthAsset` god-root semantics | `REJECT_OBSOLETE_AUTHORITY` | creates parallel source/product/proof truth |
| RealCompiler cache/R-assets as direct runtime truth | `REJECT_OBSOLETE_AUTHORITY` | proposal/cache must be promoted through current typed qualification |
| proof/handoff code mutating runtime product | `REJECT_OBSOLETE_AUTHORITY` | violates single-truth and evidence separation |
| stable `Generated/` path as identity | `REJECT_OBSOLETE_AUTHORITY` | identity must be content/run/hash based |
| file-presence-only PASS gates | `REJECT_OBSOLETE_AUTHORITY` | product proof must inspect exact content/behavior |
| hidden runtime retriangulation / alpha-clipped replacement topology | `REJECT_OBSOLETE_AUTHORITY` | exact M is runtime product authority |
| export-time evaluator/solver replay | `REJECT_OBSOLETE_AUTHORITY` | export projects proof-owned state only |
| monolithic v0.5 orchestrator wholesale import | `REJECT_OBSOLETE_AUTHORITY` | old dependencies/owners would recreate the architecture problem; only reusable control behavior is extracted |

## 11. Dependency-closure protocol for every historical executable candidate

Before any historical implementation is copied/retyped:

1. identify exact source archive + SHA;
2. record exact historical path and file SHA if available;
3. materialize only the required source subtree;
4. compute imports/callers/side effects and test dependencies;
5. classify every dependency as `CURRENT_EQUIVALENT`, `REBIND_REQUIRED`, `REFERENCE_ONLY`, or `REJECTED`;
6. prove zero import/reachability to old front-brain/teacher truth/AuthoringTruth product ownership;
7. define current typed input/output contract;
8. port/restore tests before promotion;
9. run current repository invariant suite + subsystem contract tests on local self-hosted runner;
10. add exact provenance to the promotion record;
11. only then permit current execution registration.

If exact historical bytes cannot be obtained/verified, implementation-level reuse remains blocked; design/test invariants may still be recovered from audited reports.

## 12. Transaction identity firewall required before downstream product freeze

The new current orchestrator must bind at minimum:

```text
run_id
source_observation_set_hash
surface_id / surface_hash                 # S
skeleton_id / skeleton_hash               # G
skin_id / skin_hash                       # W
mesh_id[8] / mesh_hash[8]                 # M
mesh_skin_binding_id[8] / hash[8]         # B
component_assembly_id / hash
motion_program_id / hash
qualified_motion_id / hash
proof_bundle_hash
runtime_package_hash
compiler_semantic_version
policy_bundle_hash
```

Every downstream stage receives these identities explicitly. No stage may discover a "latest" asset by path and silently substitute it.

A stage that receives an identity mismatch returns `FAIL`/`ABSTAIN`; it does not repair identity by remapping onto another topology/product.

## 13. One canonical compile transaction — target spine

```text
8 RGBA + exact cameras
  -> H1 / observation evidence
  -> Compiler-qualified S
  -> Geppetto proposal -> Compiler-qualified G
  -> Arachne proposal -> Compiler-qualified W
  -> directional mesh candidate -> strict Compiler-qualified M
  -> exact M<-W binding B
  -> component/mechanical assembly
  -> motion proposal / artist preset
  -> deterministic Motion Compiler -> QualifiedMotionIR
  -> dynamic proof + failure signatures
  -> optional bounded promoted repair -> child state -> same-probe re-proof
  -> ProductProofBundleIR PASS
  -> exact runtime package projection
  -> native runtime interlock
  -> Living Compile / export / editable puppet
```

This transaction is a coordinator, not a new semantic owner.

## 14. Promotion waves and stop/go criteria

### Wave P0A — source/index closure

Deliver:

- exact historical source map for P0-01..P0-12;
- file SHA / archive SHA where retrievable;
- dependency graph;
- current equivalent map;
- no-source-rewrite decision per item.

PASS only when no P0 item is `UNKNOWN`.

### Wave P0B — current transaction contract

Deliver current typed transaction/result schemas and contract tests.

PASS only if:

- no current learned/product authority changes;
- exact artifact identities/hashes mandatory;
- deterministic stage replay;
- FAIL/ABSTAIN propagation tested;
- no path/latest lookup authority;
- v97.39-style single-truth regression tested.

### Wave P0C — historical control behavior rebind

Rebind only accepted orchestration/retry/feedback behavior.

PASS only if old front-brain import count = 0 and current transaction tests + repository invariants pass.

### Wave P1 — real corrected product integration

Starts after corrected G/W/M/B closure. The transaction must consume exact already-qualified artifacts rather than regenerating or replacing them.

### Wave P2 — professional motion

Starts after exact product state is available. Motion restoration may propose/repair curves but must pass current deterministic qualification and dynamic proof.

### Wave P3 — runtime/editor productization

Starts after exact qualified motion. Runtime/export must consume the exact proof-selected product identities. Living Compile becomes the user-facing surface.

### Wave P4 — optional numerical power

Only measured blockers may reopen these backends.

## 15. Immediate next engineering action after this manifest

Do **not** write a new 5k-line orchestrator.

Next task is P0A:

1. reacquire/materialize exact v0.5 orchestrator/proof/retry/feedback source by recorded SHA where accessible;
2. source-diff it against current `realsas_compiler_core`, proof/repair services, runtime/export and Living Compile;
3. produce a file-level dependency closure with `KEEP_CURRENT / REBIND / TEST_ONLY / REJECT` decisions;
4. then implement the smallest current typed transaction shell that reuses proven control behavior.

In parallel, do not alter the frozen active mesh experiment. Do not start/freeze a new Arachne product-consumer contract until P0B has fixed exact transaction identity semantics.

## 16. Claim boundary

This manifest establishes a **reuse/rebind plan**, not restored product capability.

It authorizes forensic source-diff and current-contract design only. It does not authorize:

- changing current model checkpoints or scientific gates;
- restoring old front-brain authority;
- promoting historical numerical kernels;
- executing unqualified repair operations;
- claiming professional motion closure;
- claiming runtime/export reclosure;
- claiming `PRODUCT_PASS` or unseen generalization.

The end condition is not "all historical code restored". The end condition is:

> **one current typed RealSaS production spine that reaches full product capability while every useful historical mechanism is either rebound behind current authority, explicitly deferred, or explicitly rejected with evidence.**
