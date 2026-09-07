# RealSaS Structural System Index

> **STRUCTURAL/NAVIGATION INDEX — NOT CONTINUATION AUTHORITY.**
> For the current scientific gate, active experiment, branch, next decision, and current FIT authorization, read `canonical/REHYDRATION_PACKET.md` and then `CURRENT_STATE.md`.
> Historical gate language must never be inferred from this file as current program state.

This file answers: **"What subsystems exist on current main and where do they live?"** It deliberately does **not** answer "what experiment/gate are we currently running?".

## Learned stack

| Subsystem | Current mainline | State |
|---|---|---|
| IRIS V2 | `models/iris/v2/` | **CURRENT — inference + foundation/apparatus + checkpoint + train/eval visible** |
| Geppetto V2 | `models/geppetto/v2/` | **CURRENT SOURCE HOME — architecture refreeze/promotion status is governed by `CURRENT_STATE.md`** |
| SkinFieldCodec V1 | `models/skin_field_codec/v1/` | **CURRENT — learned codec/checkpoint/config + A0 train/eval visible** |
| Arachne V2 | `models/arachne/v2/` | **CURRENT — inference/conditioning/geometry + base A1 train/eval visible** |

No fifth learned subsystem is currently authorized by V4 architecture/current source composition. Older neural implementations remain research/provenance. Models emit evidence/proposals only; Compiler qualification remains authoritative.

## Compiler mainline

| Layer | Canonical home | State |
|---|---|---|
| Typed IR / hashing / V4 contracts | `compiler/realsas_compiler_core/{types.py,v4_types.py,hashing.py,v4.py}` | CURRENT |
| Surface/substrate/local geometry | `compiler/realsas_compiler_core/substrate/{surface.py,local_geometry.py,iris_v2.py}` | **CURRENT / PHYSICALLY NORMALIZED** |
| Skeleton qualification | `compiler/realsas_compiler_core/rig.py` | CURRENT |
| Skin qualification | `compiler/realsas_compiler_core/skin.py` | CURRENT |
| MWB2 mesh + skin binding | `compiler/realsas_compiler_core/mesh/{mwb2.py,mwb2_skin.py,mesh_binding.py}` | **CURRENT / PHYSICALLY NORMALIZED / BEHAVIORALLY HARDENED** |
| Appearance | `compiler/realsas_compiler_core/appearance.py` | CURRENT / BEHAVIORALLY HARDENED |
| Motion construction | `compiler/realsas_compiler_core/motion.py` | CURRENT / BEHAVIORALLY HARDENED |
| Compiler-owned directional joint/view binding | `compiler/realsas_compiler_core/directional_binding.py` | **CURRENT / RESTORATION CLOSURE PASS** |
| Deformation measurement | `compiler/realsas_compiler_core/deformation.py` + services numerics | CURRENT |
| Product composition | `compiler/realsas_compiler_core/{api.py,product.py,bundle_routes.py}` | CURRENT |
| Proof binding | `compiler/realsas_compiler_core/proof_engine.py` | **CURRENT / TYPED FAIL-CLOSED DIRECTIONAL PROVIDER / CLOSURE PASS** |
| Qualification-owned motion-frame binding | `compiler/realsas_compiler_services/proof/motion_bake.py` | **CURRENT — BINDS FRAMES; DOES NOT INVENT AUTHORITY** |
| Qualified directional motion evaluator | `compiler/realsas_compiler_services/proof/directional_motion_evaluator.py` | **CURRENT — ROTATION-ONLY PRESET; OTHER SEMANTICS FAIL CLOSED** |
| Typed directional motion provider | `compiler/realsas_compiler_services/proof/directional_motion_provider.py` | **CURRENT — HASH-BOUND PRODUCT/BINDING/POLICY/EVALUATOR AUTHORITY** |
| Dynamic motion consequence metrics | `compiler/realsas_compiler_services/proof/motion_frame_metrics.py` | **CURRENT / EVALUATOR-INDEPENDENT** |
| Direct mechanical-joint -> directional `P.xy` probe | retracted historical restoration attempt | **RETRACTED / NOT CURRENT** |
| Failure signatures | `compiler/realsas_compiler_services/proof/failure_signatures.py` | PROMOTED; DIAGNOSTIC ONLY |
| Controlled causal owner attribution | `compiler/realsas_compiler_services/proof/causal_attribution.py` | **PROMOTED / CONTROLLED-INTERVENTION ONLY / REGRESSION PASS** |
| Bounded repair directive + re-proof contract | `compiler/realsas_compiler_services/proof/repair_loop.py` | **PROMOTED / NO IN-PLACE MUTATION / REGRESSION PASS** |
| Repair operation authority registry | `compiler/realsas_compiler_core/repair_registry.py` | **CURRENT / FAIL-CLOSED / 0 EXECUTORS / FIRST-FIT ACCEPTABLE** |
| Real child-product semantic audit | `compiler/realsas_compiler_core/repair_attempt.py` | **CURRENT / PARENT-LINEAGE + ACTUAL-DELTA GATE** |
| Proof-to-deploy bake handoff | `compiler/realsas_compiler_services/export/qualification_bake.py` | **CURRENT / SAME-BAKE / NO REPLAY** |
| Runtime deploy bake codec | `compiler/realsas_compiler_services/export/runtime_deploy_bake.py` | PROMOTED |
| Pure native runtime-v2 writer | `compiler/realsas_compiler_services/export/runtime_v2.py` | **VALID PURE WRITER** |
| Current V4 proof/bake -> native-v2 projection | `compiler/realsas_compiler_services/export/current_v4_runtime_v2.py` | **CURRENT / CLOSURE PASS / PROOF-OWNED XY / EXACT LOCAL-RASTER UV / NO REPLAY** |
| Numerical LBS probe | `compiler/realsas_compiler_services/numerics/lbs.py` | **PROMOTED / CURRENT NUMERICAL DEPENDENCY** |
| Historical CDT | v0.5 source-diff reserve | **NOT CURRENT — future typed mesh-candidate producer only** |
| Historical BBW/KKT | v0.5 source-diff reserve | **NOT CURRENT — future SkinProposal/fallback candidate only** |
| Historical ARAP | v0.5 source-diff reserve | **NOT CURRENT — future typed corrective-deformer extension only** |
| Historical XPBD + SDF contact | v0.5 source-diff reserve | **NOT CURRENT — future secondary-dynamics/contact extension only** |
| Rig parent repair executor | historical candidate | **NOT CURRENT — future proposal/requalification operation only** |
| Retained weight-candidate repair executor | historical candidate | **NOT CURRENT — future retained-candidate/requalification operation only** |

Historical restoration evidence remains valid evidence, but not current continuation state:
- `canonical/AUTHORED_MOTION_PROOF_RETRACTION_V1_20260903.json`
- `canonical/P0_DIRECTIONAL_BINDING_RUNTIME_INTERLOCK_CLOSURE_V1_20260904.json`
- `canonical/HISTORICAL_NUMERICS_SOURCE_DIFF_DISPOSITION_V1_20260904.json`
- `canonical/REPAIR_EXECUTION_AUTHORITY_DISPOSITION_V1_20260904.json`
- `canonical/CANONICAL_MAIN_BEFORE_FIT_GATE_V1_20260904.json`
- `canonical/RESTORATION_CLOSURE_VERDICT_V1_20260904.json`

## Runtime

| Runtime | Current home | State |
|---|---|---|
| Native C++17 runtime | `runtime/realsas_cpp/` | **RESTORED EXACT CONSUMER; SEALED SUBTREE; CLOSURE PASS** |
| Python V4 reference consumer | `runtime/reference_v4/` | CURRENT CONFORMANCE REFERENCE |
| Current V4 `.rss/.rsr` projection/materialization | Compiler export services | **CURRENT / CLOSURE PASS** |
| Exact `.rss/.rsr` -> sealed C++ open/sample/render probe | `tests/runtime/current_v4_native_package_probe.cpp` | **CI PASS — EXACT CURRENT-V4 PACKAGE OPEN/SAMPLE/RENDER VERIFIED** |

`RUNTIME_CONSUMPTION` inside the product proof is a **pre-export compatibility** proof domain, not evidence that the C++ package was executed. Post-export native execution is a separate interlock; historical closure evidence records the exact successful interlock run.

## Repair behavior

Automatic repair is **not** a required current product capability or proof domain. Current architecture policy remains:

- `ProductProofBundleIR.overall_status == PASS` -> runtime export may proceed;
- `FAIL` or `ABSTAIN` -> stop fail-closed;
- no historical repair executor receives automatic authority;
- future executor promotion must preserve the current distinct-child + bounded-owner-local + same-probe re-proof contract.

## Historical restoration gate — superseded for continuation

The former sequence

`restoration closure PASS -> canonical main ref equality -> post-merge integrity -> first-fit-base freeze`

is preserved as historical provenance only. **Do not use its old `FIT NOT AUTHORIZED` wording as current state.** Later work has already moved beyond that gate. Current FIT/Geppetto authorization is exclusively defined by `CURRENT_STATE.md` and the live continuity views.

## Research / evidence zones

| Area | Purpose |
|---|---|
| `experiments/` | active/historical research, ablations, capacity/overfit diagnostics and upgrade candidates |
| `canonical/` | architecture, preregistration, closure, promotion/retraction and authority evidence |
| `historical/` | provenance/source-diff reserve; never an alternate executable mainline |

## Promotion rule

```text
experiment / historical evidence
  -> audit / canonical decision
  -> promote, rebind, or retract code in models/compiler/runtime
  -> mainline regression + E2E
  -> reconcile CURRENT_STATE + machine authority/experiment registry + scientific journal
```

If this file and the actual tree disagree, treat that disagreement as a repository-integrity bug. If this file and `CURRENT_STATE.md` disagree about **program state**, `CURRENT_STATE.md` wins and this file must be repaired.