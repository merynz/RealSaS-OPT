# RealSaS Current System Index

This is the shortest answer to: **"What do we currently have, where is it, and what is its status?"**

## Learned stack

| Subsystem | Current mainline | State |
|---|---|---|
| IRIS V2 | `models/iris/v2/` | **CURRENT — inference + foundation/apparatus + checkpoint + train/eval visible** |
| Geppetto V2 | `models/geppetto/v2/` | **CURRENT — inference/conditioning/checkpoint + V2 target/loss/train/eval visible** |
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
| Compiler-owned directional joint/view binding | `compiler/realsas_compiler_core/directional_binding.py` | **CURRENT CANDIDATE / SOURCE+MATH CLOSURE READY** |
| Deformation measurement | `compiler/realsas_compiler_core/deformation.py` + services numerics | CURRENT |
| Product composition | `compiler/realsas_compiler_core/{api.py,product.py,bundle_routes.py}` | CURRENT |
| Proof binding | `compiler/realsas_compiler_core/proof_engine.py` | **CURRENT / TYPED FAIL-CLOSED DIRECTIONAL PROVIDER** |
| Qualification-owned motion-frame binding | `compiler/realsas_compiler_services/proof/motion_bake.py` | **CURRENT — BINDS FRAMES; DOES NOT INVENT AUTHORITY** |
| Qualified directional motion evaluator | `compiler/realsas_compiler_services/proof/directional_motion_evaluator.py` | **CURRENT CANDIDATE — ROTATION-ONLY PRESET; OTHER SEMANTICS FAIL CLOSED** |
| Typed directional motion provider | `compiler/realsas_compiler_services/proof/directional_motion_provider.py` | **CURRENT CANDIDATE — HASH-BOUND PRODUCT/BINDING/POLICY/EVALUATOR AUTHORITY** |
| Dynamic motion consequence metrics | `compiler/realsas_compiler_services/proof/motion_frame_metrics.py` | **CURRENT / EVALUATOR-INDEPENDENT** |
| Direct mechanical-joint -> directional `P.xy` probe | retracted historical restoration attempt | **RETRACTED / NOT CURRENT** |
| Failure signatures | `compiler/realsas_compiler_services/proof/failure_signatures.py` | PROMOTED; DIAGNOSTIC ONLY |
| Controlled causal owner attribution | `compiler/realsas_compiler_services/proof/causal_attribution.py` | **PROMOTED / CONTROLLED-INTERVENTION ONLY** |
| Bounded repair directive + re-proof contract | `compiler/realsas_compiler_services/proof/repair_loop.py` | **PROMOTED / NO IN-PLACE MUTATION** |
| Repair operation authority registry | `compiler/realsas_compiler_core/repair_registry.py` | **CURRENT / FAIL-CLOSED / 0 EXECUTABLE OPERATIONS** |
| Real child-product semantic audit | `compiler/realsas_compiler_core/repair_attempt.py` | **CURRENT / PARENT-LINEAGE + ACTUAL-DELTA GATE** |
| Proof-to-deploy bake handoff | `compiler/realsas_compiler_services/export/qualification_bake.py` | **CURRENT / SAME-BAKE / NO REPLAY** |
| Runtime deploy bake codec | `compiler/realsas_compiler_services/export/runtime_deploy_bake.py` | PROMOTED |
| Pure native runtime-v2 writer | `compiler/realsas_compiler_services/export/runtime_v2.py` | **VALID PURE WRITER** |
| Current V4 proof/bake -> native-v2 projection | `compiler/realsas_compiler_services/export/current_v4_runtime_v2.py` | **SOURCE CLOSURE READY / PROOF-OWNED XY / EXACT LOCAL-RASTER UV / NO REPLAY** |
| Numerical LBS probe | `compiler/realsas_compiler_services/numerics/lbs.py` | PROMOTED |
| Rig parent repair executor | historical candidate | **BLOCKED — must re-enter current proposal/qualification seam** |
| Retained weight-candidate repair executor | historical candidate | **BLOCKED — current retained-candidate portfolio absent** |
| CDT / BBW-KKT / ARAP / XPBD/contact | competing historical authorities | SOURCE-DIFF REQUIRED |

Retraction evidence: `canonical/AUTHORED_MOTION_PROOF_RETRACTION_V1_20260903.json`.  
P0/runtime source-closure evidence: `canonical/P0_DIRECTIONAL_BINDING_RUNTIME_INTERLOCK_CLOSURE_V1_20260904.json`.

## Runtime

| Runtime | Current home | State |
|---|---|---|
| Native C++17 runtime | `runtime/realsas_cpp/` | **RESTORED EXACT CONSUMER; SEALED SUBTREE UNCHANGED** |
| Python V4 reference consumer | `runtime/reference_v4/` | CURRENT CONFORMANCE REFERENCE |
| Current V4 `.rss/.rsr` projection/materialization | Compiler export services | **SOURCE CLOSURE READY** |
| Exact `.rss/.rsr` -> sealed C++ open/sample/render probe | `tests/runtime/current_v4_native_package_probe.cpp` | **HARNESS READY; CURRENT-SOURCE EXECUTION PENDING RUNNER** |

`RUNTIME_CONSUMPTION` inside the product proof is a **pre-export compatibility** proof domain, not evidence that the C++ package was executed. Post-export native execution is a separate required interlock.

## Research / evidence zones

| Area | Purpose |
|---|---|
| `experiments/` | active/historical research, ablations, capacity/overfit diagnostics and upgrade candidates |
| `canonical/` | architecture, preregistration, closure, promotion/retraction and authority evidence |
| `historical/` | provenance/source-diff reserve; never an alternate executable mainline |

## Current program gate

- Global architecture refreeze: **NOT PERFORMED**
- Formal Family-1 selection: **BLOCKED**
- Real-family FIT: **NOT AUTHORIZED**
- Directional P0 source/math closure: **READY**
- Native current-source interlock: **PENDING RUNNER EXECUTION**
- Current work: **numerical source-diff -> native milestone gate when runner available -> restoration-wide closure -> refreeze/FIT decision**

## Promotion rule

```text
experiment / historical evidence
  -> audit / canonical decision
  -> promote, rebind, or retract code in models/compiler/runtime
  -> mainline regression + E2E
  -> update this index + state ledger
```

If this file and the actual tree disagree, treat that disagreement as a repository-integrity bug.
