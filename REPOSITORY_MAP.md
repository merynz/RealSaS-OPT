# Repository Map

This file is the human navigation contract for RealSaS-OPT.

| Path | Meaning | May own product truth? | Local index |
|---|---|---:|---|
| `compiler/realsas_compiler_core/` | Current typed Compiler authority: qualification, canonical IDs, product lineage, proof binding | **Yes — only here** | `compiler/README.md` |
| `compiler/realsas_compiler_services/` | Promoted proof/repair/export/numerical services consumed through current Compiler contracts | No | `compiler/realsas_compiler_services/README.md` |
| `runtime/realsas_cpp/` | Native C++17 runtime consumer of proof-gated export packages | No | `runtime/README.md` |
| `runtime/reference_v4/` | Python reference/conformance consumer | No | `runtime/README.md` |
| `canonical/` | Decision records, seals, preregistrations, closures, authority index | No executable ownership | `canonical/README.md` |
| `docs/` | Human documentation and repository policies | No | `docs/repository/` |
| `experiments/` | Research/training/diagnostic apparatus | No | `experiments/README.md` |
| `tests/` | Cross-cutting regression/causal/source-contract tests | No | `tests/README.md` when present; compiler-local tests remain beside compiler package |
| `historical/` | Provenance pointers and superseded-source maps; not an executable shadow tree | No | `historical/README.md` |
| `.github/workflows/` | CI enforcement only; a green workflow proves only its named contract | No | `.github/workflows/README.md` |

## Reading order for a new engineer

1. `README.md`
2. `CURRENT_STATE.md` — last canonical-main scientific state
3. `RESTORATION_STATE.md` — active restoration branch ledger
4. `canonical/README.md`
5. `compiler/README.md`
6. `compiler/realsas_compiler_services/README.md`
7. `runtime/README.md`
8. experiment/workflow indexes only when investigating a specific gate

## Rule of one obvious home

A new production concept gets one semantic owner and one obvious directory. If an implementation is experimental, historical, generated evidence, or a runtime consumer, its path must say so. Compatibility aliases may exist temporarily, but they cannot create a second owner.

## No directory-by-date authority

Dates are useful for experiment/provenance names, never for authority. A newer dated folder does not supersede another folder unless `canonical/` or the current state ledger says so.

## No workflow-by-existence authority

The repository contains many historical and narrow CI workflows. Existence or a green check does not make a workflow a product-level gate. `.github/workflows/README.md` defines how workflow scope is interpreted.
