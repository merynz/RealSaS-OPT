# RealSaS Compiler

## Canonical authority

`realsas_compiler_core/` is the single current semantic owner. It qualifies proposals, mints canonical IDs, binds product lineage, proof state and runtime eligibility.

`realsas_compiler_services/` contains subordinate promoted mechanisms consumed through current typed authority. Services may measure, diagnose, export or execute bounded numerics; they may not create alternate product truth.

## Current physical + logical layer index

Physical normalization is active. A compatibility file at an older flat path is not a second implementation: it is a facade whose canonical source is the semantic package named below.

| Logical layer | Canonical implementation | Compatibility / composition | Responsibility |
|---|---|---|---|
| Public entry / composition | `api.py`, `v4.py`, `product.py`, `bundle_routes.py` | root package exports | Compiler entry surfaces, V4 assembly, product composition and bundle routing |
| Typed contracts / identity | `types.py`, `v4_types.py`, `hashing.py` | — | IR datatypes, schema contracts and content/lineage identity |
| Observation -> substrate | `substrate/{surface.py,local_geometry.py,iris_v2.py}` | `surface.py`, `local_geometry.py` facades | evidence handoff, mechanical substrate construction and deterministic local geometry |
| Rig qualification | `rig.py` | — | proposal -> qualified skeleton, canonical joint authority |
| Skin qualification | `skin.py` | — | proposal -> qualified skin legality/normalization |
| Mesh / binding | `mesh/{mwb2.py,mwb2_skin.py,mesh_binding.py}` | `mwb2.py`, `mwb2_skin.py`, `mesh_binding.py` facades | directional editable mesh construction, qualification and mesh/skin binding |
| Appearance | `appearance.py` | — | observed/cross-view/completion appearance binding |
| Motion | `motion.py` | proof services | current typed preset motion construction/qualification plus subordinate dynamic consequence measurement |
| Deformation measurements | `deformation.py` | services numerics | mesh/skin measurement adapters |
| Proof binding | `proof_engine.py` | `realsas_compiler_services/proof/` | V4 proof-plan/domain/product proof binding; current MOTION domain includes dynamic authored-motion proof |
| Solver policy | `solver_registry.py` | — | fail-closed solver capability registry / provenance policy |

## Normalized core packages

### `substrate/`

- `substrate/surface.py` is byte-identical to the former flat `surface.py` implementation.
- `substrate/local_geometry.py` is byte-identical to the former flat `local_geometry.py` implementation.
- `substrate/iris_v2.py` is byte-identical to the audited former experiment `persistence_adapter_v2.py`.
- old flat `surface.py` and `local_geometry.py` remain compatibility facades.

Seal: `canonical/COMPILER_CORE_SUBSTRATE_LAYOUT_SEAL_V1_20260903.json`.

### `mesh/`

- `mesh/mwb2.py`, `mesh/mesh_binding.py` and `mesh/mwb2_skin.py` are byte-identical relocations of the current flat implementations.
- old flat mesh paths remain compatibility facades, preserving current consumer imports while making the canonical source obvious.

Seal: `canonical/COMPILER_CORE_MESH_LAYOUT_SEAL_V1_20260903.json`.

## Promoted service index

| Service path | Current role | Product authority? |
|---|---|---:|
| `realsas_compiler_services/proof/failure_signatures.py` | deterministic failed-invariant localization without causal-owner invention | No |
| `realsas_compiler_services/proof/{motion_probe.py,motion_probe_geometry.py}` | exact current authored-motion dynamic deformation measurement | No |
| `realsas_compiler_services/proof/causal_mutations.py` | controlled mutation helpers for later causal tests | No |
| `realsas_compiler_services/numerics/` | promoted bounded numerical kernels such as LBS probe | No |
| `realsas_compiler_services/export/` | proof-gated runtime package/deploy bake mechanisms | No |

The current motion probe is a semantic rebind of the valuable historical rule that authored/requested motion must be exercised and its consequences measured. It consumes current V4 product state only and reports measurements; `proof_engine.py` remains the sole proof-status owner.

Historical causal owner attribution, bounded repair/re-proof and additional numerical mechanisms are promoted only after source audit; they must land in the corresponding service or core semantic owner rather than recreate an old monolith.

## Intended normalization

The remaining flat core should converge dependency-safely toward semantic packages equivalent to:

```text
contracts/
substrate/        # normalized
rig/
skin/
mesh/             # normalized
appearance/
motion/
product/
proof/
```

This is a **navigation/ownership normalization**, not an architecture rewrite. Existing public imports are preserved through controlled re-exports while canonical source moves. No file moves solely for aesthetics.

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
- causal owner inference from a failure label alone;
- repair without exact-state re-proof.
