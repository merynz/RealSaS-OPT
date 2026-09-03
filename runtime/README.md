# RealSaS Runtime

Runtime is a **consumer/projection**, never product authority.

## Layout

- `realsas_cpp/` — exact promoted native C++17 runtime source from the verified v0.5 source archive.
- `reference_v4/` — current Python V4 reference/conformance consumer.

## Native source provenance

Parent archive:
- `RealSaS_M4_v0_5_CANONICAL_MECHANICAL_MEANING_SOURCE.zip`
- SHA-256 `03a819f01d3cc39e806cc30ae291912718d114ca3ff6b75dc2b854d1bbfbf130`

Historical standalone runtime source authority:
- `realsas_cpp_v05_source.zip`
- SHA-256 `1af741c9a3d30456a6703809e067a9c3a61220da51a6a1a9cbda2b8a4755e8b0`

The promoted subtree is file-hash sealed in `canonical/COMPILER_RUNTIME_PROMOTION_SOURCE_SEAL_V1_20260903.json`.

## Qualification

The exact promoted C++ subtree configures and builds cleanly under CMake and its `realsas_runtime_abi_smoke` CTest passes. CI enforces this on every change to native runtime source.

The next integration gate is current V4 exact-PASS proof -> production package -> native open/clip lookup/render.
