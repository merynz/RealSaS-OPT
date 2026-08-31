# RealSaS — Compiler / Runtime Drive-to-GitHub Migration Completeness Audit Prereg — 2026-09-01

**Status:** `FROZEN_BEFORE_FINAL_ARCHITECTURE_SVG__AUDIT_REQUIRED__NO_ASSUMED_COMPLETENESS`

## Purpose

Before producing the requested final end-to-end architecture SVG, reconstruct exactly what the historical Drive-to-GitHub Compiler/runtime transfer intended to migrate, verify what is actually present and reachable in the current GitHub repository, and distinguish intentionally external/historical authority from accidental omission or silent architectural sidelining.

The audit must not assume that "recorded by SHA" means "migrated into GitHub", nor that "not imported by the happy-path entrypoint" means "dead".

## Gate relationship

This audit is a mandatory prerequisite of master backlog item 67, the final detailed architecture SVG.

The SVG may not be finalized until this audit reaches either:

- `PASS_INTENDED_MIGRATION_COMPLETE__ARCHITECTURE_INVENTORY_RECONCILED`, or
- an explicit fail/repair closure identifying and correcting missing intended migration.

## A. Historical transfer authority discovery

First locate the detailed Drive report that describes the Compiler + runtime transfer/migration. Do not substitute a newer summary merely because it is easier to find.

Record:

- Drive file ID/URL/title;
- file/revision metadata where available;
- report content hash/export hash where possible;
- exact transfer scope and exclusions;
- every source archive/package/module/runtime component the report says should exist in GitHub;
- every component the report explicitly says remains Drive-only/historical/external.

If multiple reports or amendments exist, establish their order and final authority before comparing bytes.

## B. Supporting Drive authority inventory

Use supporting Drive evidence where needed, including historical compiler/runtime archives, module-lifecycle inventories, restoration logs/reports, and source SHA authorities.

Known SHA anchors already recorded in GitHub include:

- Aug-v0.5 full source: `03a819f01d3cc39e806cc30ae291912718d114ca3ff6b75dc2b854d1bbfbf130`;
- Aug-6 R5_3 identity/reducer/DAG authority: `6224661cb4323f78a9b808af10f68dd584431a422e28d26a69e87816a3b0ef80`;
- May v97.39 single-truth rebind: `ac755b4ba9319de963e9f68f962d7a1839e40c640928ac0c035a6af1758673e1`;
- late-May v97_43 stronger numerical/rig authority: `09a94871f938b069ba5c8219f203355e724f2f58afa6e10dc5c6148d98b43efb`;
- Aug-v0.5 C++17 runtime source: `1af741c9a3d30456a6703809e067a9c3a61220da51a6a1a9cbda2b8a4755e8b0`.

These anchors are provenance evidence only; they do not themselves establish migration completeness.

## C. GitHub inventory to reconcile

At minimum inspect and inventory:

1. current `compiler/realsas_compiler_core/**` typed source;
2. `compiler/vendor/realsas_v05_current_execution_closure.b64/**` including decoded record inventory and hashes;
3. current `compiler/realsas_compiler_core/__init__.py` import/entrypoint reachability;
4. `compiler/realsas_compiler_core/solver_registry.py` and every referenced package/module;
5. `restoration/**`, including historical text authority and source pointers;
6. `runtime/**` and any runtime source/vendor material elsewhere in the repository;
7. `canonical/COMPILER_RUNTIME_RESTORATION_CLOSURE_20260828.md` and later amendments;
8. `canonical/SOLVER_AUTHORITY_MATRIX_V1.md` and ownership-overlap audit;
9. all branches/commits that participated in restoration/migration where current `main` may have omitted a file;
10. CI workflows/results that claim source parity, build parity, ABI smoke, or package-to-native render.

## D. Required classification per component

Every discovered Compiler/runtime component must receive exactly one primary classification:

- `CANONICAL_MAINLINE_EXECUTABLE` — directly reachable current product authority;
- `CANONICAL_TYPED_BUT_NOT_YET_EXECUTED` — current typed contract/code awaiting later gate;
- `VENDORED_CURRENT_EXECUTION_CLOSURE` — byte-restored dependency reachable by current code;
- `OPTIONAL_OR_FALLBACK_AUTHORITY` — valid but not normal happy path;
- `HISTORICAL_EXTERNAL_BYTE_AUTHORITY_INTENTIONAL` — deliberately kept outside GitHub/current execution;
- `RESTORATION_REFERENCE_ONLY` — report/reference retained but source not intended for direct current execution;
- `ORPHANED_OR_UNREACHABLE_REQUIRES_DECISION` — present but no current consumer/authority mapping;
- `MISSING_INTENDED_MIGRATION_BLOCKER` — transfer report says it should be present/recoverable in GitHub but it is not;
- `DEAD_SUPERSEDED_CONFIRMED` — explicitly superseded with no remaining architectural obligation.

Do not classify a component as dead merely because an import is absent. Check reports, callers, solver ownership, runtime ABI and historical capability first.

## E. Byte/provenance checks

For every component intended to be migrated or vendored:

- verify exact file/path presence;
- verify recorded SHA when a source hash is available;
- for chunked/base64 transport, verify chunk set, chunk hashes, decoded size and decoded SHA;
- verify namespace rebinds are explicitly identified rather than mistaken for byte-exact source;
- verify no manual transcription copy silently replaced a stronger byte authority;
- verify source archive identity against the authoritative Drive object when accessible.

## F. Reachability and architectural-use checks

For each present component determine:

- imported by canonical entrypoint?;
- invoked by current product route?;
- referenced only by registry/documentation?;
- used only by experiment/test/restoration tooling?;
- expected future promotion gate?;
- duplicates another semantic owner?;
- retains a capability no current replacement provides?

This is the step that detects components that were physically migrated but silently pushed out of the architecture.

## G. Known current facts that must not be misinterpreted

Current GitHub evidence already states:

- the vendored current Compiler execution closure is deliberately narrow;
- its manifest contains 9 byte-exact v0.5 leaf modules plus 4 namespace rebinds;
- heavy proof/repair/numerical authority is pointed back to Drive SHA authority;
- `solver_registry.py` references CDT/BBW/ARAP/XPBD packages that are not currently present in the main GitHub tree/entrypoint;
- `runtime/` currently exposes a README and records the C++17 runtime source as Drive byte authority.

These facts may represent correct intentional scope **or** reveal mismatch with the separate historical migration report. Only the report reconciliation decides which.

## H. Required outputs

1. `COMPILER_RUNTIME_MIGRATION_COMPLETENESS_AUDIT_V1.json`
   - source authorities;
   - complete component inventory;
   - intended-transfer status;
   - GitHub presence/path/hash;
   - entrypoint/reachability status;
   - architectural classification;
   - discrepancies.

2. `COMPILER_RUNTIME_MIGRATION_COMPLETENESS_AUDIT_CLOSURE_20260901.md`
   - concise human-readable verdict;
   - blockers/repairs if any;
   - architecture-SVG authorization status.

3. If missing intended migration exists, repair commits + parity evidence before closure.

## I. Final SVG authorization rule

The final architecture SVG may begin its **final authoritative rendering** only after migration reconciliation is closed.

Preparatory architecture notes/inventory may be collected before then, but no diagram may imply that current GitHub happy-path reachability equals the full intended RealSaS architecture.
