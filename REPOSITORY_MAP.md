# Repository Map

This file is the human navigation contract for RealSaS-OPT.

RealSaS-OPT is a **research library with a continuously upgraded executable mainline**. The mainline tells you what the system currently is; `experiments/` tells you what we are testing as a possible improvement or falsification.

## Continuity spine — read this before browsing the library

Current context is intentionally centralized so a new chat/agent does not have to reconstruct truth from hundreds of reports:

1. `canonical/REHYDRATION_PACKET.md` — compact generated current-context view.
2. `canonical/FIT1_EVIDENCE_INDEX_20260909.md` — current technical/investor proof path.
3. `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl` — current append-only causal decision chronology.
4. `CURRENT_STATE.md` — canonical continuation / stop-go authority.
5. `canonical/LIVE_AUTHORITY_MAP.md` — generated live branch + active-experiment navigation.
6. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md` — mechanism implementation/test/canonical-state map.
7. `canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md` — exact experiment semantics and explicit non-claims.
8. `canonical/EXPERIMENT_REGISTRY_V2.json` — current machine experiment registry.

Machine state behind those views:

- `canonical/CONTEXT_STATE_V1.json`
- `canonical/AUTHORITY_MAP_V1.json`

`canonical/EXPERIMENT_REGISTRY_V1.json` and `canonical/SCIENTIFIC_JOURNAL_V1.jsonl` are historical continuity artifacts after the 2026-09-09 Geppetto promotion transaction. Do **not** reconstruct current authority from old report wording, filename dates, branch recency, or workflow color.

| Path | Meaning | May own product truth? | Local index |
|---|---|---:|---|
| `models/` | Current learned model implementations: IRIS, Geppetto, SkinFieldCodec, Arachne | **Evidence/proposals only** | `models/README.md` |
| `compiler/realsas_compiler_core/` | Current typed Compiler authority: qualification, canonical IDs, product lineage, proof binding | **Yes — only here** | `compiler/README.md` |
| `compiler/realsas_compiler_services/` | Promoted proof/repair/export/numerical services consumed through current Compiler contracts | No | `compiler/realsas_compiler_services/README.md` |
| `product/` | User-facing product surfaces; Living Compile inspector/editor consumes current Compiler/proof/runtime authority | **No — consumer only** | `product/README.md` |
| `runtime/realsas_cpp/` | Native C++17 runtime consumer of proof-gated export packages | No | `runtime/README.md` |
| `runtime/reference_v4/` | Python reference/conformance consumer | No | `runtime/README.md` |
| `experiments/` | Research/training/diagnostic apparatus and upgrade candidates | No | `experiments/README.md` |
| `canonical/` | Decision records, seals, preregistrations, closures, live continuity state/indexes | No executable ownership | `canonical/README.md` |
| `historical/` | Provenance pointers and superseded-source maps; not an executable shadow tree | No | `historical/README.md` |
| `tests/` | Cross-cutting regression/causal/source-contract tests | No | `tests/README.md` when present; compiler-local tests remain beside compiler package |
| `docs/` | Human documentation and repository policies | No | `docs/repository/` |
| `.github/workflows/` | CI enforcement only; a green workflow proves only its named contract | No | `.github/workflows/README.md` |

## Current learned-source landmarks

- `models/iris/v2/` — promoted IRIS observation/foundation/evidence base package.
- `models/iris/v3/` — promoted scene-first signed-geometry head and frozen Mage FIT1 upstream witness.
- `models/geppetto/reference_strength_v1/` — current FIT1-frozen Geppetto source.
- `models/geppetto/v2/` — prior Geppetto source/provenance, superseded for the Mage FIT1 frozen formulation.
- `models/skin_field_codec/v1/` — base mainline codec source; current 278M V7 A0 apparatus remains research-only.
- `models/arachne/v2/` — prior A1 scaffold; no V7-native A1 is promoted.

IRIS V2 and V3 are layers of the same subsystem ownership envelope; they do not create a fifth learned subsystem.

## Reading order for a new engineer / new chat / new agent

1. `canonical/REHYDRATION_PACKET.md`
2. `canonical/FIT1_EVIDENCE_INDEX_20260909.md`
3. recent tail of `canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl`
4. `CURRENT_STATE.md`
5. `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md`
6. `canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md`
7. `canonical/LIVE_AUTHORITY_MAP.md`
8. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md`
9. `canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md`
10. `canonical/EXPERIMENT_REGISTRY_V2.json`
11. `models/README.md`
12. `compiler/README.md`
13. `compiler/realsas_compiler_services/README.md`
14. `product/README.md`
15. `runtime/README.md`
16. `canonical/README.md`
17. only then specific experiment/workflow/report indexes when a current ledger points there

`RESTORATION_STATE.md` is preserved restoration-era evidence and is explicitly superseded for current continuation. Read it only when investigating restoration lineage.

## Research-library lifecycle

```text
mainline = models/ + compiler/ + product/ + runtime/
labs     = experiments/
decision + continuity = canonical/
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

Routine scientific/mainline Actions execution uses the local self-hosted RealSaS runner. Avoid high-volume workflow fan-out; hosted Actions previously triggered a usage/quota warning and are not the routine execution authority.
