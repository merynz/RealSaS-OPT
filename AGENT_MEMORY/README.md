# RealSaS Agent Memory

> **NON-AUTHORITATIVE DERIVED MEMORY.**
> Persistent, queryable architecture/recovery memory for agents. It never overrides exact current source, canonical policy, tests, CI evidence, explicit user decisions, or promoted product state.

## Purpose

This layer prevents RealSaS from requiring conversational memory or repeated Git archaeology to reconstruct:
- the exact 46-stage DAG;
- semantic stage contracts and authority boundaries;
- typed IR/schema ownership;
- symbol/data-flow relationships;
- current versus superseded/falsified decisions;
- branch/ref lineage and recovery evidence;
- promotion slices, gates and known open seams;
- source hashes and staleness boundaries.

## Rehydration order

1. `AGENT_MEMORY/CURRENT_TRUTH_SNAPSHOT.md`
2. `AGENT_MEMORY/RECOVERY_OVERLAY_20261001.json`
3. `AGENT_MEMORY/DECISION_SUPERSESSION_LEDGER_V1.json`
4. `AGENT_MEMORY/STAGE_GRAPH_V2.json`
5. `AGENT_MEMORY/STAGE_SEMANTIC_CONTRACTS_V1.json`
6. `AGENT_MEMORY/SCHEMA_REGISTRY_V1.json`
7. `AGENT_MEMORY/SYMBOL_DATAFLOW_GRAPH_V1.json`
8. `AGENT_MEMORY/SUBSYSTEM_GRAPH_NORMALIZED_V2.json`
9. `AGENT_MEMORY/PROMOTION_SLICES_20260930.json` and `RECOVERY_CANDIDATE_MATRIX.md`
10. `AGENT_MEMORY/MAIN_REF_TIMELINE_20260930.json` and `BRANCH_LINEAGE_20260930.json`
11. `AGENT_MEMORY/GATE_MATRIX_20260930.md`, `NORMALIZED_DELTA_INVENTORY.json`, `PRODUCT_RESEARCH_FRONTIER_1941.*`
12. `AGENT_MEMORY/SOURCE_MANIFEST.json`
13. exact current source/tests/policies for the subsystem being touched
14. only then broader branch/commit archaeology

## Current reconstruction fact

The normalized V2 lineage was genuinely merged to `main` on 2026-09-30, then intentionally safety-rolled back during recovery-polling/context-loss instability. Rollback is not engineering falsification.

Recovery work on `recovery/canonical-main-20261001` has since advanced beyond the original memory baseline:
- Stage35 G3B envelope binding is repaired and subject-free tested;
- Stage37 owns qualified final visual presentation;
- typed source-owned visual transport now reaches Stage42, RSS v2, native playback, Stage45 and Stage46 gate scope;
- native source extension seal V16 passes;
- external Quaternius motion qualification remains fail-closed on non-root translation semantics.

Read `RECOVERY_OVERLAY_20261001.json` before treating older open-seam documents as current.

## Safety/staleness rule

If a relevant ref or source blob differs from `SOURCE_MANIFEST.json`, mark affected memory **STALE UNTIL RECONCILED**. Never patch production solely from derived agent memory.

## Mutation rule

- Do not force-move `main` during reconstruction.
- Do not wholesale merge research/ops branches.
- Promote qualified semantic slices, not chronological commit piles.
- A green gate proves only its executed scope.
- Preserve uncertainty explicitly.
- Product/runtime code must not be modified merely to make this memory internally convenient.

## Platform transition

This memory is the specification bridge to the professional backend:
- stage graph -> dependency resolver contracts;
- schema registry -> artifact types;
- evidence/gates -> qualification records;
- recovery attempts -> immutable Attempts;
- promoted compatible artifact set -> ProductRevision.

The future backend, not Git history, must answer normal product-state questions. Exact Git/hash provenance remains expandable forensic evidence.

## Seal status

`MEMORY_SEAL_V1.json` records whether the derived map is complete enough for agent rehydration. “Sealed” means map coverage is complete, **not** that RealSaS product engineering is complete.
