# RealSaS Current System Index

This is the shortest answer to: **"What do we currently have, where is it, and what is its status?"**

## Learned stack

| Subsystem | Current mainline | State |
|---|---|---|
| IRIS V2 | `models/iris/v2/` | **CURRENT — inference + foundation/apparatus + checkpoint + train/eval visible** |
| Geppetto V2 | `models/geppetto/v2/` | **CURRENT — inference/conditioning/checkpoint + V2 target/loss/train/eval visible** |
| SkinFieldCodec V1 | `models/skin_field_codec/v1/` | **CURRENT — learned codec/checkpoint/config + A0 consequence-sensitive train/eval visible** |
| Arachne V2 | `models/arachne/v2/` | **CURRENT — inference/conditioning/geometry + base A1 hard-tail train/eval visible** |

No fifth learned subsystem is currently authorized by V4 architecture/current source composition. Older IRIS and Geppetto/Arachne V1 neural implementations remain research/provenance.

Models emit evidence/proposals only. Compiler qualification remains authoritative.

## Compiler mainline

| Layer | Canonical home | State |
|---|---|---|
| Typed IR / hashing / V4 contracts | `compiler/realsas_compiler_core/{types.py,v4_types.py,hashing.py,v4.py}` | CURRENT |
| Surface/substrate/local geometry | `compiler/realsas_compiler_core/substrate/{surface.py,local_geometry.py,iris_v2.py}` | **CURRENT / PHYSICALLY NORMALIZED** |
| Legacy surface/local-geometry imports | `compiler/realsas_compiler_core/{surface.py,local_geometry.py}` | COMPATIBILITY FACADES ONLY |
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
| Motion probe / playback measurement | historical authority under audit | **NEXT HISTORICAL SEMANTIC PROMOTION** |
| Owner attribution + bounded repair/re-proof | historical authority under audit | PENDING |
| CDT / BBW-KKT / ARAP / XPBD/contact | competing historical authorities | SOURCE-DIFF REQUIRED |

See `compiler/README.md` for the physical/logical layer map and `canonical/COMPILER_CORE_SUBSTRATE_LAYOUT_SEAL_V1_20260903.json` for the first core relocation seal.

## Runtime

| Runtime | Current home | State |
|---|---|---|
| Native C++17 runtime | `runtime/realsas_cpp/` | RESTORED EXACT CONSUMER; source/build gate previously qualified |
| Python V4 reference consumer | `runtime/reference_v4/` | CURRENT CONFORMANCE REFERENCE |
| Exact current V4 proof -> `.rss/.rsr` -> native runtime interlock | Compiler services + runtime | PENDING RESTORATION CLOSURE |

## Research / evidence zones

| Area | Purpose |
|---|---|
| `experiments/` | active/historical research, ablations, capacity/overfit diagnostics, post-failure remediation and upgrade candidates |
| `canonical/` | architecture, preregistration, closure, promotion and authority evidence |
| `historical/` | provenance/source-diff reserve; never an alternate executable mainline |

## Current program gate

- Global architecture refreeze: **NOT PERFORMED**
- Formal Family-1 selection: **BLOCKED**
- Real-family FIT: **NOT AUTHORIZED**
- Current work: **Compiler physical source normalization + historical proof/runtime promotion**

## Promotion rule

```text
experiment evidence
  -> audit / canonical decision
  -> promote or replace code in models/compiler/runtime
  -> mainline regression + E2E
  -> update this index + state ledger
```

If this file and the actual tree disagree, treat that disagreement as a repository-integrity bug.
