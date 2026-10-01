# RealSaS Agent Memory

> **NON-AUTHORITATIVE DERIVED MEMORY.**
> This directory exists to give an agent a persistent, queryable mental model of RealSaS.
> It never overrides `CURRENT_STATE.md`, canonical policy, exact source code, tests, CI evidence, or run-local ledgers.

## Why this exists

RealSaS is too large and lineage-heavy to reconstruct safely from conversational context alone. This layer externalizes the working model that an agent would otherwise try to keep in a context window:

- current truth versus historical green evidence;
- the exact 46-stage dependency graph;
- authority and ownership boundaries;
- evidence strength and supersession rules;
- known open blockers and uncertainty;
- source hashes used to build the map.

## Rehydration order for Alfred

1. `AGENT_MEMORY/CURRENT_TRUTH_SNAPSHOT.md`
2. `AGENT_MEMORY/SOURCE_MANIFEST.json`
3. `AGENT_MEMORY/STAGE_GRAPH_V2.json`
4. `AGENT_MEMORY/EVIDENCE_AND_LINEAGE_RULES.md`
5. canonical sources listed by the source manifest, starting with readiness and mainline plan
6. relevant source/tests for the subsystem being changed
7. only then branch/commit archaeology

## Safety rule

If any authority source blob SHA differs from `SOURCE_MANIFEST.json`, this memory is **STALE UNTIL REBUILT**. Do not repair production code from stale memory.

## Branch rule

This directory is being built on `alfred/repo-memory-map-20261001`. It is intentionally not promoted to `main` while repo-state reconstruction is still in progress.
