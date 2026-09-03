# RealSaS Current System Index

This is the shortest answer to: **"What do we currently have, where is it, and what is its status?"**

## Learned stack

| Subsystem | Current mainline | State |
|---|---|---|
| IRIS V2 | `models/iris/v2/` | **CURRENT MAINLINE — inference + current train/eval/checkpoint package promoted** |
| Geppetto V2 | `models/geppetto/v2/` | **CURRENT MAINLINE INFERENCE — training lane audit pending** |
| SkinFieldCodec V1 | `models/skin_field_codec/v1/` | **CURRENT MAINLINE INFERENCE/CODEC — training lane audit pending** |
| Arachne V2 | `models/arachne/v2/` | **CURRENT MAINLINE INFERENCE — training/remediation lane audit pending** |

Models emit evidence/proposals only. Compiler qualification remains authoritative.

The original dated experiment trees remain intact as provenance/research labs. Current mainline Python is being guarded against imports from `experiments.*`.

## Compiler mainline

| Layer | Current home | State |
|---|---|---|
| Typed IR / hashing / V4 contracts | `compiler/realsas_compiler_core/{types.py,v4_types.py,hashing.py,v4.py}` | CURRENT |
| Surface/substrate/local geometry | `compiler/realsas_compiler_core/{surface.py,local_geometry.py}` | CURRENT |
| Skeleton qualification | `compiler/realsas_compiler_core/rig.py` | CURRENT |
| Skin qualification | `compiler/realsas_compiler_core/skin.py` | CURRENT |
| MWB2 mesh + skin binding | `compiler/realsas_compiler_core/{mwb2.py,mwb2_skin.py,mesh_binding.py}` | CURRENT / BEHAVIORALLY HARDENED |
| Appearance | `compiler/realsas_compiler_core/appearance.py` | CURRENT / BEHAVIORALLY HARDENED |
| Motion | `compiler/realsas_compiler_core/motion.py` | CURRENT / BEHAVIORALLY HARDENED |
| Deformation measurement | `compiler/realsas_compiler_core/deformation.py` + services numerics | CURRENT |
| Product composition | `compiler/realsas_compiler_core/{api.py,product.py,bundle_routes.py}` | CURRENT |
| Proof binding | `compiler/realsas_compiler_core/proof_engine.py` | CURRENT BUT RESTORATION-INCOMPLETE |
| Failure signatures | `compiler/realsas_compiler_services/proof/failure_signatures.py` | PROMOTED |
| Causal mutation helpers | `compiler/realsas_compiler_services/proof/causal_mutations.py` | PROMOTED SUPPORT |
| Export/deploy bake | `compiler/realsas_compiler_services/export/runtime_deploy_bake.py` | PROMOTED |
| Numerical LBS probe | `compiler/realsas_compiler_services/numerics/lbs.py` | PROMOTED |
| IRIS persistence/substrate adapter | source classified Compiler-owned | NEXT SOURCE-OWNERSHIP PROMOTION |
| Motion probe / playback measurement | historical authority under audit | PENDING AFTER SOURCE-OWNERSHIP CLOSURE |
| Owner attribution + bounded repair/re-proof | historical authority under audit | PENDING |
| CDT / BBW-KKT / ARAP / XPBD/contact | competing historical authorities | SOURCE-DIFF REQUIRED |

See `compiler/README.md` for the logical layer map.

## Runtime

| Runtime | Current home | State |
|---|---|---|
| Native C++17 runtime | `runtime/realsas_cpp/` | RESTORED EXACT CONSUMER / source/build gate previously qualified |
| Python V4 reference consumer | `runtime/reference_v4/` | CURRENT CONFORMANCE REFERENCE |
| Exact current V4 proof -> `.rss/.rsr` -> native runtime interlock | Compiler services + runtime | PENDING RESTORATION CLOSURE |

## Research / evidence zones

| Area | Purpose |
|---|---|
| `experiments/` | active/historical research, training apparatus, falsification and upgrade candidates |
| `canonical/` | architecture, preregistration, closure, promotion and authority evidence |
| `historical/` | provenance/source-diff reserve; never an alternate executable mainline |

## Current program gate

- Global architecture refreeze: **NOT PERFORMED**
- Formal Family-1 selection: **BLOCKED**
- Real-family FIT: **NOT AUTHORIZED**
- Current work: **model training/source normalization + Compiler/runtime historical promotion**

## Promotion rule

```text
experiment evidence
  -> audit / canonical decision
  -> promote or replace code in models/compiler/runtime
  -> mainline regression + E2E
  -> update this index + state ledger
```

If this file and the actual tree disagree, treat that disagreement as a repository-integrity bug.
