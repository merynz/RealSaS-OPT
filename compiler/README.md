# RealSaS Compiler

Canonical entry surface: `realsas_compiler_core`.

Current typed authority code is ordinary source under `realsas_compiler_core/`. Its only historical execution dependency is the narrow v0.5 closure under `vendor/realsas_v05_current_execution_closure.b64/`.

On import the entrypoint verifies every transport part, reconstructs raw ZIP SHA-256 `3a6076b30e0a23807f952365d39d81ddf5d4b1dba734c0bdba47567bced26850`, safely extracts it to a filesystem cache, verifies every declared restored entry, then exposes the historical graph/artifact dependencies. The current closure is 9 byte-exact v0.5 leaf modules + 4 controlled namespace rebinds. Old perception/training/truth/front-brain packages are excluded.

Geppetto hierarchy proposals are consumed by the existing `CanonicalGraphOptimizationRequest -> optimize_canonical_graph_v18_98 -> CanonicalGraphOptimizationResult` layer; Compiler then mints product-canonical joint IDs and emits `QualifiedSkeletonIR`.

Heavy proof/repair and late-May numerical authorities remain SHA-bound Drive authorities until a current consumer requires their exact executable closure. They are not silently replaced by weaker ports.
