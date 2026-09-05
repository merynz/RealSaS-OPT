# Repository Map

This file is the human navigation contract for RealSaS-OPT.

RealSaS-OPT is a **research library with a continuously upgraded executable mainline**. The mainline tells you what the system currently is; `experiments/` tells you what we are testing as a possible improvement or falsification.

| Path | Meaning | May own product truth? | Local index |
|---|---|---:|---|
| `models/` | Current learned model implementations: IRIS, Geppetto, SkinFieldCodec, Arachne | **Evidence/proposals only** | `models/README.md` |
| `compiler/realsas_compiler_core/` | Current typed Compiler authority: qualification, canonical IDs, product lineage, proof binding | **Yes — only here** | `compiler/README.md` |
| `compiler/realsas_compiler_services/` | Promoted proof/repair/export/numerical services consumed through current Compiler contracts | No | `compiler/realsas_compiler_services/README.md` |
| `product/` | User-facing product surfaces; Living Compile inspector/editor consumes current Compiler/proof/runtime authority | **No — consumer only** | `product/README.md` |
| `runtime/realsas_cpp/` | Native C++17 runtime consumer of proof-gated export packages | No | `runtime/README.md` |
| `runtime/reference_v4/` | Python reference/conformance consumer | No | `runtime/README.md` |
| `experiments/` | Research/training/diagnostic apparatus and upgrade candidates | No | `experiments/README.md` |
| `canonical/` | Decision records, seals, preregistrations, closures, authority index | No executable ownership | `canonical/README.md` |
| `historical/` | Provenance pointers and superseded-source maps; not an executable shadow tree | No | `historical/README.md` |
| `tests/` | Cross-cutting regression/causal/source-contract tests | No | `tests/README.md` when present; compiler-local tests remain beside compiler package |
| `docs/` | Human documentation and repository policies | No | `docs/repository/` |
| `.github/workflows/` | CI enforcement only; a green workflow proves only its named contract | No | `.github/workflows/README.md` |

## Reading order for a new engineer

1. `README.md`
2. `REPOSITORY_MAP.md`
3. `CURRENT_STATE.md` — last canonical-main scientific state
4. `RESTORATION_STATE.md` — active restoration branch ledger
5. `models/README.md` — current learned stack and its boundaries
6. `compiler/README.md`
7. `compiler/realsas_compiler_services/README.md`
8. `product/README.md`
9. `runtime/README.md`
10. `canonical/README.md`
11. experiment/workflow indexes only when investigating a specific gate

## Research-library lifecycle

```text
mainline = models/ + compiler/ + product/ + runtime/
labs     = experiments/
decision = canonical/
reserve  = historical/

experiment -> evidence/closure -> canonical decision -> audited mainline promotion/replacement -> regression
```

Mainline means **best currently authorized implementation**, not scientifically final. Later experiments may replace it when they provide stronger evidence.

See `docs/repository/RESEARCH_LIBRARY_MODEL.md` for the promotion and replacement rules.

## Rule of one obvious home

A current executable concept gets one semantic owner and one obvious mainline directory. If an implementation is experimental, historical, generated evidence, or a runtime consumer, its path must say so. Compatibility aliases may exist temporarily during migration, but they cannot create a second authority.

Production/mainline code should not permanently import dated experiment implementations. Experiments may import mainline code; successful experimental mechanisms are promoted into their semantic home before they become current dependencies.

## Product-surface authority rule

`product/` may visualize, edit, and stage user intent, but it may not mint `CanonicalPuppetGraph`, proof, qualification, or deployable-runtime truth. Living Compile user edits remain sibling authoring layers until Compiler requalification and fresh dynamic proof accept them.

## No directory-by-date authority

Dates are useful for experiment/provenance names, never for authority. A newer dated folder does not supersede another folder unless `canonical/` or the current state ledger says so.

## No workflow-by-existence authority

The repository contains many historical and narrow CI workflows. Existence or a green check does not make a workflow a product-level gate. `.github/workflows/README.md` defines how workflow scope is interpreted.
