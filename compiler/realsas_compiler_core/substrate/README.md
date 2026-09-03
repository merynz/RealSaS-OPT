# Compiler Substrate

`substrate/` is the deterministic Compiler-owned seam between learned observation evidence and qualified mechanical substrate state.

## Canonical source

The physical implementation home is now this package:

- `surface.py` — generic persistence/evidence -> `RiggingSurfaceIR` construction;
- `local_geometry.py` — DTB-ND1 robust local-plane geometry;
- `iris_v2.py` — IRIS V2 evidence-to-substrate adapter;
- `types.py` / `hashing.py` — internal dependency bridges used only so the two pre-existing core implementations can remain byte-preserved during relocation.

`surface.py` and `local_geometry.py` are the exact former flat-core blobs. Their old paths, `realsas_compiler_core.surface` and `realsas_compiler_core.local_geometry`, are compatibility facades that reflect these canonical modules so existing tests, experiments and public imports remain valid.

`iris_v2.py` is likewise promoted byte-identically from:

`experiments/iris_reprojection_v2_20260831/persistence_adapter_v2.py`

## Authority boundary

Learned perception stops at typed observation evidence. Code in this package may construct persistence groups, observed local relations, local differential geometry and `RiggingSurfaceIR` state from admitted evidence, but it may not consume teacher rig/skin truth, source meshes, family-specific constants or create canonical skeleton/skin/product authority.

The IRIS V2 adapter provides:

- `build_persistence_groups_v2`
- `attach_observed_local_relations_v2`
- `compile_surface_v2`
- `attach_dtb_nd1_from_evidence`

## Import policy

`substrate.__init__` intentionally does not eagerly import `iris_v2`. During the compatibility window the byte-preserved adapter imports the old flat surface/local-geometry paths; eager import would create a package-init cycle. Consumers should import IRIS-specific assembly from `realsas_compiler_core.substrate.iris_v2` explicitly.

## Forbidden

- learned model parameters or inference ownership;
- source-mesh topology as hidden substrate truth;
- teacher skeleton/skin identity;
- canonical graph/ID minting;
- family-specific fit constants;
- imports from dated experiment packages.
