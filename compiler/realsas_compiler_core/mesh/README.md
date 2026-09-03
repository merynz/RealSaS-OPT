# Compiler Mesh

`mesh/` is the canonical physical home for editable mesh construction, mesh qualification and qualified skin transfer.

## Canonical implementations

- `mesh_binding.py` — mesh candidate/qualified mesh/mesh-skin validation and lineage authority;
- `mwb2.py` — current MWB2 directional editable-mesh candidate and qualification implementation;
- `mwb2_skin.py` — current surface-support convex skin transfer into qualified mesh skin.

All three implementation files are byte-identical to the former flat-core source blobs. `types.py` and `hashing.py` are internal dependency bridges that preserve the original relative imports without rewriting algorithm source.

## Compatibility

The former public paths remain valid:

- `realsas_compiler_core.mesh_binding`
- `realsas_compiler_core.mwb2`
- `realsas_compiler_core.mwb2_skin`

Those files are now compatibility facades reflecting the canonical modules under `realsas_compiler_core.mesh`.

## Authority boundary

This layer may construct and qualify editable mesh state from admitted substrate support, transfer already-qualified skin to mesh vertices, and bind exact lineage. It does not own learned geometry, canonical skeleton identity, skin proposal inference, appearance, motion or proof authority.

## Invariants

- mesh rest positions derive only from admitted surface support;
- topology references must be internally valid;
- support coefficients are finite/nonnegative simplex values;
- mesh/skin lineage must bind the exact current surface/skeleton/skin/mesh state;
- no hidden source mesh is admitted as shipping truth.
