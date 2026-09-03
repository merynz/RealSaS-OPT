# RealSaS Compiler

## Canonical authority

`realsas_compiler_core/` is the single current semantic owner. It qualifies proposals, mints canonical IDs, binds product lineage, proof state and runtime eligibility.

`realsas_compiler_services/` contains subordinate promoted mechanisms consumed through current typed authority. Services may measure, diagnose, export or execute bounded numerics; they may not create alternate product truth.

## Current logical layer index

The core package is still physically flatter than the intended library-quality layout. Until dependency-safe normalization is complete, use this table as the authoritative navigation map.

| Logical layer | Current files | Responsibility |
|---|---|---|
| Public entry / composition | `api.py`, `v4.py`, `product.py`, `bundle_routes.py` | Compiler entry surfaces, V4 assembly, product composition and bundle routing |
| Typed contracts / identity | `types.py`, `v4_types.py`, `hashing.py` | IR datatypes, schema contracts and content/lineage identity |
| Observation -> substrate | `surface.py`, `local_geometry.py` | admitted mechanical substrate construction and deterministic local geometry |
| Rig qualification | `rig.py` | proposal -> qualified skeleton, canonical joint authority |
| Skin qualification | `skin.py` | proposal -> qualified skin legality/normalization |
| Mesh / binding | `mwb2.py`, `mwb2_skin.py`, `mesh_binding.py` | directional editable mesh construction and mesh/skin binding |
| Appearance | `appearance.py` | observed/cross-view/completion appearance binding |
| Motion | `motion.py` | current typed preset motion construction/qualification |
| Deformation measurements | `deformation.py` | mesh/skin measurement adapters using promoted numerical services |
| Proof binding | `proof_engine.py` | V4 proof-plan/domain/product proof binding; richer historical mechanisms are being promoted behind this boundary |
| Solver policy | `solver_registry.py` | fail-closed solver capability registry / provenance policy |

## Promoted service index

| Service path | Current role | Product authority? |
|---|---|---:|
| `realsas_compiler_services/proof/` | failure signatures and controlled causal mutation/proof helpers | No |
| `realsas_compiler_services/numerics/` | promoted bounded numerical kernels such as LBS probe | No |
| `realsas_compiler_services/export/` | proof-gated runtime package/deploy bake mechanisms | No |

Historical motion probes, owner attribution, repair/re-proof and additional numerical mechanisms are promoted only after source audit; they must land in the corresponding service or core semantic owner rather than recreate an old monolith.

## Intended normalization

After dependency audit, the physical core tree should converge toward obvious semantic packages equivalent to:

```text
contracts/
substrate/
rig/
skin/
mesh/
appearance/
motion/
product/
proof/
```

This is a **navigation/ownership normalization**, not an architecture rewrite. Existing public imports will be preserved through controlled re-exports while source files move. No file moves solely for aesthetics.

## Research upgrade rule

Compiler experiments may live under `experiments/`, but an experimental mechanism cannot become a current Compiler dependency while still living in a dated experiment package. Successful mechanisms are audited, promoted into `realsas_compiler_core/` or `realsas_compiler_services/`, and then covered by mainline regressions.

## Historical execution closure

The narrow v0.5 vendor closure remains byte-guarded for existing dependencies. It is not a license to import the full old compiler graph.

## Forbidden

- old front-brain/truth ownership;
- teacher-exact shipping authority;
- second canonical graph/ID producer;
- hidden historical solver execution;
- permanent mainline imports from dated experiment packages;
- repair without exact-state re-proof.
