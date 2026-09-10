# Canonical Authority Index

`canonical/` contains sealed decisions/evidence **and** the compact continuity state used to reconstruct current RealSaS context. It is intentionally not an executable package.

## Start here

1. `REHYDRATION_PACKET.md` — generated compact current-context packet for new chats/agents/sessions.
2. `FIT1_EVIDENCE_INDEX_20260909.md` — short technical/investor audit path for the current Mage FIT1 chain.
3. `GEPPETTO_FIT1_EVIDENCE_MANIFEST_V1.json` — machine-readable Geppetto source/result/hash/external-artifact binding.
4. `SCIENTIFIC_JOURNAL_V2_20260909.jsonl` — current append-only decision chronology; V1 journal is historical.
5. `../CURRENT_STATE.md` — continuation and stop/go authority on `main`.
6. `SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md` — learned/deterministic responsibility boundary.
7. `FIT1_SCIENTIFIC_LINEAGE_V1.md` — semantic FIT1-to-now chronology.
8. `LIVE_AUTHORITY_MAP.md` — generated live branch/active-experiment navigation.
9. `ARCHITECTURE_AUTHORITY_LEDGER_V1.md` — mechanism implementation vs test vs canonical-status ledger.
10. `EXPERIMENT_AUTHORITY_LEDGER_V1.md` — experiment meaning, result scope and explicit non-claims.
11. `EXPERIMENT_REGISTRY_V2.json` — current structured experiment registry; V1 registry remains historical.
12. `BRANCH_AUTHORITY_V1.md` — branch/promotion law.

Machine state behind the generated views:

- `CONTEXT_STATE_V1.json`
- `AUTHORITY_MAP_V1.json`

Do not infer which registry/journal version is current from filenames alone. `CONTEXT_STATE_V1.json` and `AUTHORITY_MAP_V1.json` carry the live pointers; after the 2026-09-09 Geppetto promotion they point to registry V2 and journal V2 while preserving V1 as historical continuity.

Generated views are navigation/cache only. If they conflict with `../CURRENT_STATE.md`, `CURRENT_STATE.md` wins.

## Current promoted FIT1 evidence

- IRIS Mage signed witness: `../models/iris/v3/PROMOTED_MAGE_FIT_WITNESS_V1.json`.
- Geppetto scientific closure: `GEPPETTO_REFERENCE_STRENGTH_FIT1_CLOSURE_20260908.md`.
- Geppetto separate promotion/refreeze: `GEPPETTO_REFERENCE_STRENGTH_MAINLINE_PROMOTION_20260909.md`.
- Geppetto machine evidence manifest: `GEPPETTO_FIT1_EVIDENCE_MANIFEST_V1.json`.
- Arachne remains research-only at A0; no V7-native A1 model is currently promoted.

## Current top-level architecture lineage

- `SYSTEM_ARCHITECTURE_V4_20260902.md` — product/type boundary.
- `CANONICAL_REALSaS_COMPLETION_PLAN_20260902.md` — completion/execution lineage except where explicitly superseded by current state.
- `CANONICAL_REALSaS_COMPLETION_MATRIX_20260902.json` — machine dependency lineage except where explicitly superseded.
- `BEHAVIORAL_HARDENING_LEDGER_20260903.md` — preserved behavioral-hardening evidence only.

## Historical restoration lineage

Compiler/runtime restoration closed and remains valuable evidence, but it is **not the active continuation program**. `../RESTORATION_STATE.md` is explicitly marked superseded for continuation. Its restoration-era stop/go statements must not be used to answer current FIT/Geppetto authorization questions.

Relevant preserved restoration records include:

- `COMPILER_RUNTIME_PROMOTION_PLAN_V1_20260903.md`
- `COMPILER_RUNTIME_PROMOTION_MANIFEST_V1_20260903.json`
- `COMPILER_RUNTIME_PROMOTION_SOURCE_SEAL_V1_20260903.json`
- `RESTORATION_CLOSURE_VERDICT_V1_20260904.json`

## Supersession rule

Closed dated records are not silently rewritten. A later decision supersedes continuation authority by explicit `CURRENT_STATE` / authority-ledger entry while preserving the old record as historical evidence.

A report existing, being newer, or being larger does not make it authoritative. A mechanism is reconstructable only when the architecture/experiment ledgers identify its implementation, evidence status and canonical disposition.
