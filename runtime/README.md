# RealSaS Runtime

Canonical runtime boundary: engine-neutral `.rss/.rsr` projection of an exact proven `CanonicalPuppetGraph` lineage into the historical C++17 runtime ABI.

Historical source authority:
- `realsas_cpp_v05_source.zip`
- SHA-256 `1af741c9a3d30456a6703809e067a9c3a61220da51a6a1a9cbda2b8a4755e8b0`
- byte authority: Google Drive historical compiler/runtime source archive

Qualification performed during restoration:
- clean CMake configure/build: PASS
- CTest runtime ABI smoke: 1/1 PASS
- Python exporter `.rss/.rsr` -> native open -> clip lookup -> 8x8 render: PASS
- ABI baseline: 3

The runtime is a projection/consumer, never canonical product authority. Source bytes remain heavy-byte authority in Drive; this repo records the exact source SHA and qualification contract so runtime can be re-materialized without ambiguity.
