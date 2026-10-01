# RealSaS Agent Memory

> **NON-AUTHORITATIVE DERIVED MEMORY.**
> This directory exists to give an agent a persistent, queryable mental model of RealSaS.
> It never overrides current canonical policy, exact source code, tests, CI evidence, explicit user decisions, or run-local ledgers.

## Why this exists

RealSaS is too large and lineage-heavy to reconstruct safely from conversational context alone. This layer externalizes the working model an agent would otherwise try to keep in a context window:

- current truth versus historical green evidence;
- the exact 46-stage dependency graph;
- subsystem and authority boundaries;
- branch/ref lineage and timeout-era recovery history;
- promotion slices and exact gate scope;
- known open seams and repair owners;
- source hashes used to build the map.

## Rehydration order for Alfred

1. `AGENT_MEMORY/CURRENT_TRUTH_SNAPSHOT.md`
2. `AGENT_MEMORY/MAIN_REF_TIMELINE_20260930.json`
3. `AGENT_MEMORY/BRANCH_LINEAGE_20260930.json`
4. `AGENT_MEMORY/SUBSYSTEM_GRAPH_NORMALIZED_V2.json`
5. `AGENT_MEMORY/VISUAL_RUNTIME_SEAM_MAP.json`
6. `AGENT_MEMORY/PROMOTION_SLICES_20260930.json`
7. `AGENT_MEMORY/GATE_MATRIX_20260930.md`
8. `AGENT_MEMORY/NORMALIZED_DELTA_INVENTORY.json`
9. `AGENT_MEMORY/SOURCE_MANIFEST.json` and `STAGE_GRAPH_V2.json`
10. `AGENT_MEMORY/EVIDENCE_AND_LINEAGE_RULES.md`
11. current canonical sources and exact source/tests for the subsystem being touched
12. only then broader branch/commit archaeology

## Central reconstruction fact

The normalized V2 lineage was genuinely merged to `main` on 2026-09-30, then intentionally safety-rolled back because recovery-polling/context-loss made further agent-driven main mutation unsafe. The rollback is **not** an engineering rejection. The preserved normalized backup and current rolled-back main are both evidence states.

## Safety rule

If authority/source blobs or relevant branch heads differ from the hashes recorded by this memory, treat affected sections as **STALE UNTIL REBUILT**. Never repair production code from stale agent memory.

## Mutation rule

During reconstruction:
- do not force-move `main`;
- do not wholesale merge research/ops branches;
- validate candidate slices against exact current contracts;
- distinguish a green gate from the scope it did not execute;
- record uncertainty instead of synthesizing a false “current truth”.

## Branch/workflow side effect

Creating `alfred/repo-memory-map-20261001` triggered the repository's branch-create Live Authority Context workflow, which indirectly refreshed generated authority/context files on `main`. No compiler/runtime production source changed. Avoid creating additional branches during reconstruction unless necessary.
