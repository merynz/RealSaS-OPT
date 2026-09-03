# Repository Map

This file is the human navigation contract for RealSaS-OPT.

| Path | Meaning | May own product truth? |
|---|---|---|
| `compiler/realsas_compiler_core/` | Current typed Compiler authority: qualification, canonical IDs, product lineage, proof binding | **Yes — only here** |
| `compiler/realsas_compiler_services/` | Promoted proof/repair/export/numerical services consumed through current Compiler contracts | No |
| `runtime/realsas_cpp/` | Native C++17 runtime consumer of proof-gated export packages | No |
| `runtime/reference_v4/` | Python reference/conformance consumer | No |
| `canonical/` | Decision records, seals, preregistrations, closures, authority index | No executable ownership |
| `docs/` | Human documentation and repository policies | No |
| `experiments/` | Research/training/diagnostic apparatus | No |
| `tests/` | Regression/causal/source-contract tests | No |
| `historical/` | Provenance pointers and superseded-source maps; not an executable shadow tree | No |
| `.github/workflows/` | CI enforcement | No |

## Reading order for a new engineer

`README.md -> CURRENT_STATE.md / RESTORATION_STATE.md -> canonical/README.md -> compiler/README.md -> runtime/README.md`

## Rule of one obvious home

A new production concept gets one semantic owner and one obvious directory. If an implementation is experimental, historical, generated evidence, or a runtime consumer, its path must say so. Compatibility aliases may exist temporarily, but they cannot create a second owner.
