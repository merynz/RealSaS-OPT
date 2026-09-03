# RealSaS Current System Index

This is the shortest answer to: **"What do we currently have, where is it, and what is its status?"**

It is an index, not a substitute for `CURRENT_STATE.md`, `RESTORATION_STATE.md` or canonical architecture contracts.

## Learned stack

| Subsystem | Current semantic role | Current executable/source location | Mainline status |
|---|---|---|---|
| IRIS | 8-view RGB observation evidence; learned geometric authority ends at depth/support/uncertainty | current V2 candidate still under `experiments/iris_reprojection_v2_20260831/`; production owner reserved at `models/iris/` | **SOURCE NORMALIZATION IN PROGRESS** |
| Geppetto | anonymous multimodal skeleton/control proposal | current V2 candidate mixed under `experiments/geppetto_arachne_r6_20260901/`; production owner reserved at `models/geppetto/` | **SOURCE NORMALIZATION IN PROGRESS** |
| SkinFieldCodec | continuous per-joint influence-field latent + shared decoder | current candidate `experiments/geppetto_arachne_r6_20260901/skin_field_codec_v1.py`; production owner reserved at `models/skin_field_codec/` | **SOURCE NORMALIZATION IN PROGRESS** |
| Arachne | qualified-skeleton-conditioned dense skin proposal | current V2 candidate mixed under `experiments/geppetto_arachne_r6_20260901/`; production owner reserved at `models/arachne/` | **SOURCE NORMALIZATION IN PROGRESS** |

Models emit evidence/proposals only. Compiler qualification remains authoritative.

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
| Failure signatures | `compiler/realsas_compiler_services/proof/failure_signatures.py` | PROMOTED / CI PASS |
| Causal mutation helpers | `compiler/realsas_compiler_services/proof/causal_mutations.py` | PROMOTED SUPPORT |
| Export/deploy bake | `compiler/realsas_compiler_services/export/runtime_deploy_bake.py` | PROMOTED |
| Numerical LBS probe | `compiler/realsas_compiler_services/numerics/lbs.py` | PROMOTED |
| Motion probe / playback measurement | historical authority under audit | NEXT RESTORATION ITEM |
| Owner attribution + bounded repair/re-proof | historical authority under audit | PENDING |
| CDT / BBW-KKT / ARAP / XPBD/contact | competing historical authorities | SOURCE-DIFF REQUIRED |

See `compiler/README.md` for the logical layer map.

## Runtime

| Runtime | Current home | State |
|---|---|---|
| Native C++17 runtime | `runtime/realsas_cpp/` | RESTORED EXACT CONSUMER / CI PASS |
| Python V4 reference consumer | `runtime/reference_v4/` | CURRENT CONFORMANCE REFERENCE |
| Exact current V4 proof -> `.rss/.rsr` -> native runtime interlock | Compiler services + runtime | PENDING RESTORATION CLOSURE |

## Research / evidence zones

| Area | Purpose |
|---|---|
| `experiments/` | active and historical research questions, training/eval apparatus, falsification and upgrade candidates |
| `canonical/` | architecture, preregistration, closure, promotion and authority evidence |
| `historical/` | provenance/source-diff reserve; never an alternate executable mainline |

## Current program gate

- Global architecture refreeze: **NOT PERFORMED**
- Formal Family-1 selection: **BLOCKED**
- Real-family FIT: **NOT AUTHORIZED**
- Current work: **source ownership normalization + Compiler/runtime historical promotion**

## Promotion rule

A successful experiment does not remain the de facto current implementation inside `experiments/`.

```text
experiment evidence
  -> audit / canonical decision
  -> promote or replace code in models/compiler/runtime
  -> mainline regression + E2E
  -> update this index + state ledger
```

If this file and the actual tree disagree, treat that disagreement as a repository-integrity bug and repair the index or implementation before making new architecture claims.