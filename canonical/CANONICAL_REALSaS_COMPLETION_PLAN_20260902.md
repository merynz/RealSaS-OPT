# RealSaS — Canonical Completion Audit & Execution Plan — 2026-09-02

**Status:** `FORENSIC_AUDIT_CLOSED__ZERO_ARCHITECTURE_UNKNOWN__IMPLEMENTATION_GAPS_TYPED__FIT_BLOCKED_UNTIL_GENERIC_SOURCE_COMPLETION`

**Audited base main:** `5fa4bf7883283ac0ae4f6b9cb8357cb1b13d6adf`

**Machine matrix:** `canonical/CANONICAL_REALSaS_COMPLETION_MATRIX_20260902.json`

## 0. What this closes

This document closes the unfinished forensic audit that preceded Architecture V4. It does **not** claim that the implementation or scientific fitting is complete. It restores the gate that was accidentally skipped: **generic source completion must precede real-family optimization**.

The audit closure criterion is now satisfied: every remaining item has an explicit disposition and dependency; there is no unresolved architecture decision. A row is not DONE merely because a class compiles. Its matrix source gate must pass.

Binding distinctions:

- `Architecture V4 source closure` = product/type/authority topology is closed.
- `This completion audit closure` = remaining implementation gaps are classified and ordered.
- `Source-complete` = all blocking BUILD/FIX/PORT rows pass their source gates.
- `Scientific PASS` = a real-family rung passes its preregistered consumer/proof criterion.

No real optimizer, FIT8 reproduction, or generalization is authorized by audit closure alone.

## 1. Binding product ontology

`3D_EQUIVALENT_MECHANICS != FULL_3D_RECONSTRUCTION`.

The learner/compiler may use depth, analytic P, local geometry, exact cameras and correspondence as 3D-equivalent mechanical evidence. The shipping authority is the directional 2D/2.5D puppet. Shared S/G/W mechanical semantics feed exactly eight direction-local renderables. Canonical motion is puppet-local `translation_xy`, `rotation_deg`, `scale_xy`, `depth_offset`; quaternion/3D-rigid canonical motion remains forbidden.

Teacher skeleton/skin/source mesh remain training/eval-only. Hidden completion may not enter S/G/W. Compiler owns canonical IDs, legal tree, legal skin, product state, proof and export authority.

## 2. Critical corrections found by completing the audit

### 2.1 Architecture closure was not completion-audit closure

Architecture V4 and PR #22 are real useful work, but they did not discharge the earlier promised `CANONICAL_REALSaS_COMPLETION_PLAN`. The old audit ended with remaining IRIS, Geppetto/Arachne, MWB, proof/export/runtime and integration source work. That missing execution authority is what this document and the machine matrix now provide.

### 2.2 Current Geppetto has a blocking cardinality defect

The audited FIT skeleton projection corpus passed `2914/2914` hard contracts and contains a maximum of **328 deform controls**. The audit explicitly rejected fixed product K.

Current `GeppettoCandidateV1`, however, uses `max_joint_queries=64`; it creates a learned 64-query bank, a 65-class count head and a 64x64 parent bias, clips proposal count to 64, while `GeppettoLossV1` raises when teacher `J > K`. This is a source defect, not an optimizer problem.

**Binding fix:** `GEP-03 = FIX_BLOCKING`. The v2 proposer must use resource-bounded variable-cardinality allocation that covers at least the frozen admitted corpus envelope (`>=328`) without making a small semantic product cap part of the architecture. Source tests must include counts above 64 and the 328-control envelope.

### 2.3 Current Geppetto lost audited position uncertainty

Current candidate emits continuous positions/root/parent/support/existence/count/abstain evidence, but no heteroscedastic position uncertainty. The audited R6 mechanism included continuous location uncertainty and an uncertainty-sensitive objective.

**Binding fix:** restore calibrated proposal-local location uncertainty. Compiler still owns final topology and canonical IDs.

### 2.4 Current Arachne is still only a candidate

Current Arachne correctly consumes current S + Compiler-qualified G, carries parent context, cross-attends joints to surface, predicts per-joint codec latents and reuses the exact SkinFieldCodec decoder. Those pieces are KEEP.

But the frozen information contract also requires explicit point↔control geometry, bone-segment geometry and latent uncertainty. These are absent from `ArachneCandidateV1`.

**Binding fixes:** `AR-05 = FIX_BLOCKING`, `AR-06 = FIX`. Add generic pair/segment geometry and latent uncertainty before Arachne fitting. No source-bone semantic identity may be introduced.

### 2.5 Codec static loss is not the scientific gate

`SkinFieldCodecV1` is a useful dynamic-N/dynamic-J continuous representation and shared decoder. Its current loss is weighted cross-entropy + L1. That does **not** satisfy R6-A0, whose PASS criterion is field reconstruction **plus verified deformation consequence**.

Build a verified-LBS evaluator/loss using authoritative deformation probes, show teacher-W parity, and show controlled W mutations causally worsen the deformation metric. Arachne optimizer remains forbidden until explicit R6-A0 PASS.

## 3. Exact historical source recovered

DTB-ND1 is no longer an unknown-source item.

- Frozen robust local-plane operator: `robust_local_plane_v1.py`, historical Git blob `4e612978c3f70a537dcf3189948dc19a38b6e582`.
- Frozen carrier runtime: `dtb_nd1_runtime_v1.py`, historical Git blob `329fb839bb94310ab8d28f2789849f893fcea6dd`.
- Historical branch: `geppetto-arachne-v0-1-20260830`.
- Scientific evidence: robust local-plane clean non-inferiority `6/6 PASS`; audited current-historical proxy bracket `0.00250 <= epsilon_critical < 0.00275 RMS`.

Therefore DTB-ND1 is `PORT_QUALIFY`, not a new algorithm design. The promoted operator must be behaviorally parity-tested and its source/version hash must enter downstream conditioning identity.

Historical `bridge_persistence_v1.py` and E0 surface-builder machinery are also reusable teacher-free correspondence/persistence apparatus. They are `PORT_REBIND` into the V4 `ObservationEvidenceIR -> persistence -> RiggingSurfaceIR` route; unsupported evidence must remain fail-closed.

## 4. Final implementation authority by subsystem

The exact row dispositions, target paths, dependencies and source gates are machine-readable in the matrix. This section is the human execution summary.

### DATA

KEEP Master source-textured 8x1024 RGBA + exact cameras as observation authority. Skeleton/skin truth stays training/eval-only. Build a finalized-S / Compiler-qualified-G truth adapter and a fresh-process truth firewall. FIT8 is a validation panel, not architecture.

### IRIS

PORT/ADAPT exact observation/camera algebra, hull/q-domain, frozen DINO/cache machinery and useful native-resolution sampler mechanics. BUILD the final V2 pipeline:

1. q-point foundation/native descriptor sampler integration;
2. per-q/per-view evidence preserving view tokens;
3. compact learned `C(Z,X,Y)` evidence field;
4. approximately isotropic world/common-frame regularizer;
5. supported/ambiguous mode scorer;
6. multimodal ray-mode extraction, not forced global soft-argmin;
7. local continuous depth refinement;
8. forward d validity/support/uncertainty head;
9. typed `ObservationEvidenceIR` emitter;
10. deterministic train/checkpoint/eval apparatus and synthetic capacity suite.

REJECT learned P/N product heads, P/N truth losses, hidden occupancy completion and any direct-depth learner used as a substitute for q evidence.

### SURFACE / LOCAL GEOMETRY

KEEP current `GeometricSubstrateAssembler` and V4 support=False exclusion. PORT/REBIND reciprocal persistence. PORT/QUALIFY the exact DTB-ND1 operator as hash-bound deterministic `N_d = PlaneFit(P)` derived geometry; teacher normals are not inference authority.

### GEPPETTO

KEEP R6 anonymous teacher projection. FIX/QUALIFY conditioning information contract. FIX BLOCKING cardinality (`64 -> resource-bounded variable cardinality >= current 328 envelope`). FIX calibrated location uncertainty. BUILD/FIX loss, trainer, checkpoint, permutation/sibling-serialization robustness, dynamic-cardinality synthetic capacity and real Compiler qualification runner.

Required product behavior remains: global surface conditioning, variable cardinality, continuous/multimodal loci, root/parent evidence, endogenous count/STOP/unsupported, proposal-only ownership. Compiler owns canonical tree/IDs.

### SKIN FIELD / ARACHNE

BUILD finalized-S/qualified-G target rebind. KEEP/QUALIFY current SkinFieldCodec representation and exact shared decoder. BUILD verified-LBS consequence evaluator/loss and R6-A0 trainer/evaluator. Only after A0 PASS: FIX Arachne pair/segment geometry and latent uncertainty, then BUILD A1 trainer/evaluator/Compiler/deformation rung.

Arachne must emit `SkinProposalIR` only. Compiler owns legal skin qualification/sparsification.

### MWB / DIRECTIONAL PRODUCT

KEEP V4 exact-eight directional product schema. BUILD the actual MWB2 producer:

`S -> useful view-local discretization/topology -> local-convex S binding -> QualifiedEditableMeshIR -> W interpolation -> QualifiedMeshSkinIR`.

Direct source-mesh copying is forbidden. UNKNOWN crossing is forbidden. CDT may be promoted only if a simpler deterministic producer fails a typed quality gate.

### APPEARANCE

KEEP V4 per-face-corner authority classes and completion firewall. BUILD the generic exact-Master-frame appearance producer using local raster authority first, then exact cross-view transport, else UNKNOWN. The old Appearance Sibling pilot is evidence/mechanism, not corpus-wide primary observation authority. V1R6 ProductAligned renders remain useful for visual candidate discovery, not Master training-pair camera authority.

### MOTION

KEEP V4 puppet-local motion representation. BUILD a generic deterministic structural target resolver/preset builder without authored `FootL`/`WingL`-style IDs or game-specific action enums. At least one preset must actually mutate canonical joint state.

### PROOF

KEEP V4 exact-state proof envelope and product-hash invalidation. BUILD real measurement producers for all required domains: mechanical structure, mesh quality, deformation, directional visual, motion and runtime consumption. Every metric family must have a causal mutation test. Missing required measurement remains ABSTAIN; required-domain failure remains FAIL.

Historical ARAP/XPBD/correctives/repair loops are conditional typed promotions only after a demonstrated base failure.

### EXPORT / RUNTIME

KEEP PASS-only `RuntimePackageIR` and direction/component-safe routing. BUILD a current V4 reference runtime consumer/export bridge. Historical native C++ remains external byte authority (`realsas_cpp_v05_source.zip`, SHA256 `1af741c9a3d30456a6703809e067a9c3a61220da51a6a1a9cbda2b8a4755e8b0`) and is materialized only after the current V4 Python/reference boundary passes.

### INTEGRATION

BUILD a fresh-process image-only firewall and one canonical generic `run_complete_e2e_v1.py` orchestrator:

`images/cameras -> IRIS -> S -> Geppetto proposal -> Compiler G -> codec/Arachne -> Compiler W -> 8 M/B -> appearance -> motion -> proof -> runtime`.

Before any real optimizer, a synthetic/mock-learned proposal E2E test must traverse the real Compiler/proof/runtime route with no family hardcoding and with truth paths poisoned/removed in inference mode.

## 5. Canonical dependency order

1. Exact DTB-ND1 port/qualification + persistence bridge.
2. Generic IRIS V2 source + synthetic/source-gate apparatus.
3. Geppetto information contract + dynamic cardinality + uncertainty + trainer/Compiler rung.
4. Finalized-S / qualified-G truth rebind.
5. SkinFieldCodec verified-LBS R6-A0 apparatus.
6. Arachne pair/segment geometry + uncertainty + trainer/Compiler rung.
7. MWB2 production directional mesh/skin producer.
8. Generic observation-derived appearance producer.
9. Deterministic puppet-local preset motion builder.
10. Real proof measurement engine.
11. Current V4 runtime bundle/reference consumer.
12. Image-only firewall + COMPLETE_E2E runner.
13. Only then final FIT8 freeze/visual review + first witness.
14. Only then real one-family optimizer ladder.
15. Only after COMPLETE_E2E_FIT PASS: remaining FIT8 reproduction, then explicit held-out/generalization.

## 6. Mandatory component-component test seams

- Observation -> IRIS: exact 8x1024/camera hashes; no resize/camera drift; image-only imports.
- IRIS -> persistence: d/support/uncertainty lineage; unsupported cannot enter carrier.
- Persistence -> S: reciprocal support/raster provenance; support=False excluded.
- S -> N_d: exact DTB parity + operator hash in conditioning identity.
- S/N_d -> Geppetto: order invariance; required-information contract; dynamic count >=328; uncertainty causal.
- Geppetto -> Compiler G: no hard learned canonical tree/IDs; deform-root/assembly-root semantics preserved.
- S/G + teacher W -> Codec: exact target rebind; dynamic N/J; simplex; shared decoder identity.
- Codec -> deformation: teacher baseline parity; controlled W mutation worsens verified-LBS metric.
- S/G -> Arachne: exact qualified-G lineage; point/control + bone-segment geometry; latent uncertainty.
- Arachne -> Compiler W: dense coverage; legal qualification; no teacher identity leakage.
- S/W -> M/B: no source mesh; no UNKNOWN crossing; local-convex binding; skin simplex.
- M/B -> appearance: exact per-corner coverage and raster provenance; completion visual-only.
- Product -> motion: puppet-local 2D/2.5D only; actual canonical state change.
- Product -> proof: six required domains; causal mutation sensitivity; exact product hash.
- Proof -> runtime: PASS-only; stale proof rejected.
- Fresh images -> E2E: truth-path poison test + full lineage + reproducible bundle/render/animation.

## 7. Branch reconciliation

`main` remains the only continuation authority. Side branches are historical/evidence donors unless explicitly promoted with provenance and parity.

**Merged/main-reached lineage:**
`architecture/compiler-ir-solver-canonical-20260825`, `architecture-v3-svg-20260901`, `architecture-v4-single-family-e2e-20260902`, `compiler-runtime-migration-audit-20260901`, `compiler-runtime-migration-closure-20260901`, `consumer-interlock-v0-20260829`, `geometric-substrate-rename-20260901`, `geppetto-r6-teacher-projection-port-20260901`, `integration/compiler-runtime-canonical-20260828`, `mwb0-closure-backlog-20260831`, `mwb1-identity-baseline-20260901`, `mwbo-typed-seam-20260831`, `single-family-e2e-models-v1-20260902`.

**Historical scientific evidence:**
`agent/n1d-global-mechanical-solver-research-20260819`, `agent/n1d-observable-functional-audit-v2-20260820`, `agent/n1d-observable-functional-quotient-rebuild-20260819`, `agent/n1d-v2-source-parity-recovery-20260820`, `audit/iris-architecture-discipline-20260824`, `dino-controlled-ladder-prereg-20260829`, `dino-zero-step-preflight-20260829`, `g0-g1/single-pose-geometry`, `geppetto-arachne-v0-1-20260830`, `legacy-geppetto-arachne-reconcile-20260901`, `m4-closure-20260829`, `m4-execution-seal-20260829`, `m4r-closure-20260829`, `next/depth-bridge-route-correction-20260829`, `next/m4-grid-prereg-20260829`, `next/proxy-evaluator-restoration-20260829`, `next/structured-depth-bridge-20260829`, `ops/temp-trigger-n1d-v2-recovery-20260820`, `ops/trigger-n1d-v2-recovery-v2-verify-20260820`.

**Separate experiment, not promoted:** `iris/mapanything-ortho-apache` — preflight evidence only; no hidden fitted final IRIS.

**Superseded/demo-only:** `demo/investor-single-specimen-e2e`, `integration/compiler-runtime-heavy-promotion-20260901`.

**Temporary/no source authority:** `tmp/appearance-witness-fetch-20260826`.

**Continuation pointer only:** `single-family-e2e-fit-v1-20260902` currently equals audited main and contains no additional authority.

No audited side branch contains a hidden source-complete final IRIS/Arachne/MWB2/proof/runtime implementation that supersedes this plan. PR20/demo source may be selectively adapted; wholesale merge remains forbidden.

## 8. Scientific authorization

- Generic source completion: **OPEN and next**.
- Synthetic/source capacity tests: **AUTHORIZED**.
- Real one-family optimizer: **BLOCKED** until every blocking matrix row through integration passes its source gate and final FIT membership is frozen.
- Geppetto real fit: additionally blocked by `GEP-03/GEP-04/GEP-05`.
- SkinFieldCodec R6-A0: blocked until `AR-01..04` source gates pass.
- Arachne optimizer: **FORBIDDEN** until explicit R6-A0 PASS.
- FIT8 reproduction/generalization: **FORBIDDEN** until one family reaches COMPLETE_E2E_FIT PASS.

## 9. Anti-drift rules

1. Newer files do not override this authority unless a mainline amendment names the contradiction and evidence.
2. A compiling source class is not a completed row; tests and source gate must pass.
3. A model candidate is not architecture-complete if a required information channel is absent.
4. Family-specific fixes are forbidden before generic source gates close.
5. Teacher truth may support training/eval only; fresh-process inference must succeed with truth paths poisoned/removed.
6. Hidden completion, source mesh, teacher root/tree/IDs, hidden BBW or historical runtime cannot become canonical by convenience.
7. Every future closure must update the machine matrix and validator in the same commit.

## 10. Closure verdict

The forensic audit is now **closed at the architecture/implementation-planning level**: no architecture decision remains UNKNOWN. The repository is **not source-complete** and **not fit-authorized**. The next canonical phase is the ordered generic source-completion program above. Only after those blocking source gates pass may the one-family scientific ladder begin.
