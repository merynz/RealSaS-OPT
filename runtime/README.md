# RealSaS Runtime

Runtime is a **consumer/projection**, never product authority.

## Layout

- `realsas_cpp/` — byte-exact promoted native C++17 deployment consumer from the verified v0.5 source archive.
- `reference_v4/` — current Python V4 reference/conformance consumer.

## Native source provenance

Restoration authority used for the promoted subtree:

- `RealSaS_M4_v0_5_CANONICAL_MECHANICAL_MEANING_SOURCE.zip`
- SHA-256 `03a819f01d3cc39e806cc30ae291912718d114ca3ff6b75dc2b854d1bbfbf130`
- archive subtree `runtime/realsas_cpp/`

Historical records also mention a standalone `realsas_cpp_v05_source.zip`. It is retained as provenance only; it was **not** used as byte authority for this restoration.

The promoted subtree is sealed file-by-file in `canonical/COMPILER_RUNTIME_PROMOTION_SOURCE_SEAL_V1_20260903.json`.

## Qualification

The sealed C++ subtree configures and builds cleanly under CMake and its `realsas_runtime_abi_smoke` CTest passes. `.github/workflows/native_runtime_source_gate.yml` re-verifies every sealed file SHA-256 before building and testing.

The native runtime consumes compiler-authored deployment products. It may load, validate, sample, blend already-baked clip frames, and provide a reference renderer; it may not infer or reinterpret rigging, weights, deformation truth, proof qualification, or compiler hypotheses.

The next integration gate is current V4 exact-PASS proof -> production package -> native open/clip lookup/render.
