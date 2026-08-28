# RealSaS — Canonical System Architecture V2

**Date:** 2026-08-28  
**Status:** `CANONICAL_RESPONSIBILITY_AND_IR_ARCHITECTURE__COMPILER_RUNTIME_RESTORED`

`PRODUCT_CONTRACT_V1.md` owns the product responsibility contract. `OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md` and sealed E0/N-B3 evidence own the scientific observation-substrate claims. This document owns the current typed system composition.

## Canonical system

```text
8 ordered neutral views + known orthographic cameras
        ↓
IRIS — learned forward depth d
        ↓ analytic P = O + dF
SurfaceBuilder + teacher-free persistence
        ↓ RiggingSurfaceIR S
Geppetto
        ↓ SkeletonProposalIR G*
Compiler graph qualification
        ↓ QualifiedSkeletonIR G
Arachne
        ↓ SkinProposalIR W*
Compiler skin qualification
        ↓ QualifiedSkinIR W
Compiler Core
        ↓ CanonicalPuppetGraph Y
motion/deformation proof → attribution → bounded repair → mandatory re-proof
        ↓ only proven exact Y
engine-neutral .rss/.rsr
        ↓
C++17 runtime ABI / host adapters
```

Responsibility boundaries do not imply permanently separate neural checkpoints or processes.

## Authority invariants

- IRIS/Geppetto/Arachne produce evidence or proposals, never final product truth.
- Compiler owns canonical product IDs and the single `CanonicalPuppetGraph` lineage.
- Proof, repair, serialization, export and runtime projection bind to the exact same product state.
- Diagnostics and proof artifacts are derived/non-owning.
- Runtime is a projection, not a second canonical graph.
- Stale proposal/proof bindings fail closed.

## IR layers

The executable IR is frozen in `IR_TYPE_SYSTEM_V1.md` and `compiler/realsas_compiler_core/types.py`.

Current hierarchy qualification deliberately reuses the restored historical graph optimizer. There is no duplicate topology/hierarchy authority:

```text
G* -> CanonicalGraphOptimizationRequest/Result -> QualifiedSkeletonIR G
```

`ShapeSkeletonGraph` remains morphology/surface evidence only.

## SurfaceBuilder boundary

Current learned geometric authority is forward depth `d`; common-frame point geometry is analytic from known rays. Support/provenance and teacher-free persistence remain deterministic/observation-grounded. Explicit learned N is not required by closed N-B3 under its frozen regime.

SurfaceBuilder may later expose additional deterministic relations/derived geometry when a real consumer requires them, but may not silently absorb Geppetto/Arachne mechanical responsibilities.

## Geppetto boundary

`RiggingSurfaceIR S -> SkeletonProposalIR G*`.

Geppetto proposes joint/control geometry and graph evidence. Compiler validates surface binding, solves the admitted root/parent graph with the existing canonical graph optimizer, and mints new canonical product joint IDs only after qualification.

## Arachne boundary

`RiggingSurfaceIR S + QualifiedSkeletonIR G -> SkinProposalIR W*`.

Arachne proposes editable influence weights. Compiler owns admissibility, solver-backed projection where justified, residual reporting and fail-closed behavior.

## Compiler Core / proof / repair

Compiler is the only canonical product owner after proposal stages. The proof loop is:

```text
Y -> ResolvedMotionProbePlan -> measurements -> proof
PASS -> export allowed
FAIL -> failure signatures -> attribution -> bounded RepairDirective -> Y' -> re-proof
```

No repair against stale Y may authorize export of another state.

## Runtime boundary

The restored v0.5 C++17 SDK is the engine-neutral baseline: `.rss/.realsas` package + `.rsr` payload -> stable C ABI/C++ wrapper -> Unity/Unreal/Godot/custom/WASM adapters. Host engines do not own canonical product truth.

## Historical composition and current selection

Restoration is a controlled composition, not “newest wins”:

- Aug-9 v0.5: executable compiler/proof/export/runtime chassis;
- Aug-6 R5_3: compiler-owned identity/reducer/DAG authority discipline;
- May v97.39: single product/proof/repair/export truth invariant;
- late-May v97.43: stronger numerical reference for CDT/cotangent, BBW/QP/KKT, ARAP and XPBD/contact where parity with later Python ports is unproven.

Executable v0.5 solver ports remain baselines. A historical numerical implementation becomes current production authority only after typed-contract, residual/invariant and downstream parity evidence; timestamp alone is forbidden.

## Restoration status

File-level source/provenance closure, typed adapters, graph/mesh/weight/deformation regressions, proof/repair tests, runtime ABI test and package-to-native-runtime fixture are closed in `COMPILER_RUNTIME_RESTORATION_CLOSURE_20260828.md`.

Open work is no longer “restore compiler”; it is using the restored compiler in the active IRIS predicted-depth qualification program and promoting stronger numerical implementations only through solver-specific parity gates.
