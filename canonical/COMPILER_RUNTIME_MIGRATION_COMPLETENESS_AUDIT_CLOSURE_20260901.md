# RealSaS — Compiler / Runtime Migration Completeness Audit Closure — 2026-09-01

**Verdict:** `PASS_INTENDED_MIGRATION_COMPLETE__ARCHITECTURE_INVENTORY_RECONCILED`  
**Final architecture SVG gate:** `AUTHORIZED_AFTER_THIS_CLOSURE_AND_REGRESSION_REPAIR_MERGE`  
**Full historical-source mirror claim:** `FALSE`

## Executive conclusion

The 25-Aug historical-maximum report and the 28-Aug GitHub restoration did **not** define the same scope.

The 25-Aug report is the pre-migration restoration authority. It defines a composite historical maximum:

- Aug-9 v0.5 Python compiler chassis;
- Aug-6 R5_3 authority discipline;
- May v97.39 single-truth invariant;
- late-May v95/v96/v97 rig/numerical maximum;
- Aug engine-neutral export and C++17 runtime.

It requires preservation and subsystem-by-subsystem reconciliation; it ends by asking for an exact file-level restoration manifest and dependency closure. It does **not** claim that all of those bytes were already destined for an immediate GitHub mirror.

The 28-Aug restoration then deliberately chose a smaller **current IRIS-to-Compiler execution closure** for GitHub. That intended GitHub payload is complete: 9 byte-exact v0.5 leaf modules plus 4 controlled namespace rebinds, encoded by 9 base64 transport fragments and a manifest. The canonical compiler entrypoint checks every fragment size/SHA, decoded ZIP size/SHA, safe extraction path, and every restored record size/SHA before the historical dependency closure is made importable.

Therefore the correct statement is:

> The intended 28-Aug GitHub current-execution migration is complete and byte-guarded. The broader historical compiler/numerical/proof/runtime maximum was intentionally retained as external SHA-bound authority for later typed promotion, not accidentally omitted from that migration.

## Source authority

Historical audit authority:

`RealSaS_Compiler_Runtime_Historical_Maximum_Audit_Final_Verdict_2026-08-25.docx`

- Library file id: `file_000000007bf4820aa97807ca9564f3bb`
- version: `1`
- raw DOCX SHA-256: `82698e86ba0454fd48848c69c10f6de6d8a78132d2c1b52ede5cb03e7e72aece`
- direct Drive URL/object was not resolved by the available connector; none is invented here.

Supporting full-v0.5 archive inventory authority:

`RealSaS_M4_v0_5_ZIP_TEST.txt`

- Library file id: `file_000000004abc81f4bc1064756dac6017`
- confirms the historical archive contained the expected mesh, weight, rig, deformation, proof, orchestrator, export, reference-runtime and native C++ runtime trees.

## Intended GitHub payload reconciliation

Current vendor authority:

`compiler/vendor/realsas_v05_current_execution_closure.b64/manifest.json`

Frozen decoded closure:

- raw SHA-256: `3a6076b30e0a23807f952365d39d81ddf5d4b1dba734c0bdba47567bced26850`;
- raw size: `40998` bytes;
- transport fragments: `9 / 9` present;
- byte-exact v0.5 records: `9 / 9` declared and import-verified;
- namespace rebind records: `4 / 4` declared and import-verified;
- total declared restored records: `13 / 13`;
- old front-brain/truth imports: `0`.

The canonical package entrypoint performs verification before import. Current compiler regression suites exercise that import path, so successful CI is downstream evidence that the transport and declared restored records remain valid.

### Branch-loss check

Restoration source branch:

`integration/compiler-runtime-canonical-20260828 @ e423cd46200220f95452890f74640c0afb57fd28`

Compared with current `main`, it is a strict ancestor:

- ahead of current main: `0` commits;
- unique files relative to current main: `0`;
- therefore no restoration file remains stranded on that branch.

The older `architecture/compiler-ir-solver-canonical-20260825` branch is a divergent research/architecture lineage with hundreds of unrelated experiment commits. It is not the 28-Aug migration source and is not wholesale merge authority.

## Historical maximum components intentionally outside current GitHub execution

These are not missing-migration blockers. They remain architectural obligations and historical byte/numerical authority until a current typed consumer justifies exact promotion.

### Full v0.5 Python chassis

Historical archive includes, among other packages:

- `realsas_mesh`;
- `realsas_weight`;
- `realsas_deformation`;
- `realsas_orchestrator`;
- `realsas_proof`;
- `realsas_export`;
- `realsas_reference_runtime`;
- `realsas_rig`.

Current `compiler/` intentionally does not mirror those package trees. Its executable surface is the new typed `realsas_compiler_core` plus the narrow vendor closure.

### Numerical maximum

Historical external authority remains:

- CDT / constrained triangulation / cotangent-topology quality;
- BBW / active-set QP / simplex / ADMM / coupled and sparse KKT / Schur-style solves;
- ARAP / corrective deformation;
- XPBD / contact / SDF contact.

Promotion is not timestamp-based and may not create a second semantic owner. Each future promotion must satisfy the current solver-authority matrix and typed product lineage.

### Motion proof / attribution / repair

The historical v0.5 tree contains richer motion/deformation probe, failure-signature, owner-attribution, retry/feedback and repair machinery than current main presently executes. Those semantics remain required architecture inventory; current main already enforces exact product/proof/runtime hash binding, while heavy execution is restored only when required by a product gate.

### Native C++17 runtime

Historical runtime authority remains the full `runtime/realsas_cpp` SDK, including:

- `CMakeLists.txt`;
- format/README;
- CMake package config;
- C ABI header;
- C++ wrapper header;
- implementation;
- demo;
- ABI smoke test.

Current GitHub `runtime/` intentionally contains the boundary/authority README rather than a source mirror. Restoration evidence records clean build, ABI smoke `1/1 PASS`, and package-to-native-render PASS.

## Repaired audit finding — solver registry

One present-but-sidelined component required repair.

Before this audit, `compiler/realsas_compiler_core/solver_registry.py` eagerly imported:

- `realsas_mesh.cdt_production`;
- `realsas_weight.production_bbw`;
- `realsas_deformation.production_arap`;
- `realsas_deformation.secondary_xpbd`.

Those packages are intentionally absent from current main. The registry was itself not imported by the canonical package entrypoint, so normal execution worked while a direct registry import would fail and its `EXECUTABLE_SOLVERS` name could misleadingly imply a hidden current solver path.

Repair in this closure:

- remove eager historical imports;
- classify all four heavy solvers explicitly as `HISTORICAL_EXTERNAL_BYTE_AUTHORITY_INTENTIONAL`;
- keep `EXECUTABLE_SOLVERS = {}` on current main;
- add `resolve_promoted_solver(...)` which fails closed until an authority record is explicitly promoted to `CANONICAL_MAINLINE_EXECUTABLE`;
- add regression tests ensuring registry import requires no missing historical package and no historical solver can execute by provenance alone.

This is an authority/reachability hygiene repair, **not** a solver promotion.

## Explicitly superseded historical components

The historical audit explicitly forbids restoring the old image-decomposition/front-brain ontology and authored-owner/teacher-exact product ownership where those responsibilities conflict with the current observable-substrate architecture.

They are `DEAD_SUPERSEDED_CONFIRMED` for the canonical product path. Historical reports/results remain archival evidence, not executable architecture.

## Missing intended migration blockers

`NONE`

No component that the 28-Aug current-execution migration intended to embed remains missing from current main.

## Final SVG consequence

The migration prerequisite for the requested detailed architecture SVG is closed after this branch's CI passes and the closure is merged.

The SVG must **not** draw only the happy path. It must visibly distinguish:

1. current canonical executable path;
2. learned proposal/training layers;
3. Compiler qualification and canonical ownership;
4. typed mesh/mesh-skin/product lineage;
5. current exact-state proof/runtime-package binding;
6. historical heavy motion-proof/attribution/repair authority awaiting typed promotion;
7. historical CDT/BBW-KKT/ARAP/XPBD authorities and their future promotion gates;
8. native C++17 runtime authority;
9. corpus/training/evaluator-only apparatus;
10. explicit dead/superseded historical front-brain components;
11. any optional/fallback path separately from canonical mainline.

No absent import may be interpreted as architectural nonexistence, and no historical SHA pointer may be drawn as a current executable path.

## Authority status after closure

- intended GitHub migration completeness: **PASS**;
- branch-loss check: **PASS**;
- vendored byte closure: **PASS / fail-closed verified**;
- hidden heavy solver execution: **NONE**;
- solver registry orphan/reachability defect: **REPAIRED, pending CI/merge**;
- full historical-source mirror: **NOT INTENDED / NOT CLAIMED**;
- final architecture SVG: **UNBLOCKED AFTER CI + MERGE**;
- learned optimizer authorization: **UNCHANGED / FALSE**.
