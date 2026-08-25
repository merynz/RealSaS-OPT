# RealSaS — Compiler & Runtime Restoration Plan V1

**Date:** 2026-08-25  
**Status:** `RESTORATION_PLAN_FROZEN_FROM_HISTORICAL_MAXIMUM_AUDIT__FILE_LEVEL_EXTRACTION_PENDING`

## 1. Source authority

This plan records the frozen verdict of the user-provided report:

**RealSaS Compiler & Runtime Historical Maximum Audit — Final Verdict and Restoration Baseline — 25 August 2026**  
Decision ID: `REALSAS_COMPILER_HISTORICAL_MAXIMUM_20260825`

The report is restoration authority, **not proof that the code has already been migrated into this repository**.

## 2. Composite restoration target

There is no single historical version to copy wholesale.

```text
Aug-9 v0.5
  source-tree/chassis
  orchestrator / proof / repair / export spine

+ Aug-6 R5_3
  compiler-owned canonical IDs / reducer / DAG
  no second authority at compiler boundary

+ May v97.39
  final-package single-truth invariant
  proof, repair and final product use the same canonical graph

+ late-May v95/v96/v97 numerical maximum
  CDT / cotangent
  BBW / active-set / KKT family
  ARAP
  XPBD / contact
  rig assembly / graph optimization where stronger

+ Aug runtime line
  .rss / .realsas / .rsr
  C++17 runtime SDK
```

## 3. Preservation buckets

| Bucket | Action | Historical source family |
|---|---|---|
| Base chassis | extract/freeze | Aug-9 v0.5 |
| IR/graph authority | preserve semantics + tests | R5_3 + v0.5 reconciliation |
| Single-truth lineage | hard invariant | v97.39 |
| Rig graph | source-diff/select | late-May vs v0.5 |
| Mesh/CDT/cotangent | source-diff/select | v95.x vs v0.5 |
| Weights/BBW/KKT | source-diff/select | v96.x vs v0.5 |
| ARAP/correctives | source-diff/select | late-May/July vs v0.5 |
| XPBD/contact/SDF | source-diff/select | late-May/July vs v0.5 |
| Motion proof/failure signatures | restore core | Aug v0.5 |
| Attribution/repair | restore core | Aug v0.5 |
| Export | restore | Aug v0.5 |
| Runtime | freeze baseline | Aug v0.5 C++17 SDK |

## 4. Prohibited restoration behavior

Do not:

- choose one version wholesale;
- choose by timestamp alone;
- rewrite proven numerical kernels before source-diff comparison;
- revive parallel production truths;
- force the old authored-owner/teacher-exact front-end ontology onto current IRIS/SurfaceBuilder/Geppetto/Arachne;
- make Unity the canonical product definition;
- promote newer IRIS research solvers merely because they are newer.

## 5. Current target boundary

The current observable-geometry product architecture remains:

```text
Images
  ↓
IRIS → ObservationEvidenceIR
  ↓
SurfaceBuilder → RiggingSurfaceIR S
  ↓
Geppetto → SkeletonProposalIR G*
  ↓
Compiler qualification → QualifiedSkeletonIR G
  ↓
Arachne → SkinProposalIR W*
  ↓
Compiler qualification → QualifiedSkinIR W
  ↓
Compiler Core → CanonicalPuppetGraph Y
  ↓
Motion Proof / Attribution / Repair
  ↓
verified editable puppet
  ↓
engine-neutral package / C++ runtime
```

Historical interfaces must adapt to these boundaries; these boundaries must not be distorted merely to match old types.

## 6. Forensic restoration sequence

### R0 — locate exact authorities

Locate/archive the exact historical artifacts referenced by the audit, including the v0.5 canonical source archive, R5_3 authority material, v97.39 rebind material and late-May numerical source families.

### R1 — byte/hash registry

For every candidate archive/file record:

```text
source artifact
archive SHA-256
internal path
file SHA-256
size
historical date/version
claimed subsystem
known tests/reports
```

No code promotion before provenance is known.

### R2 — v0.5 clean extraction branch

Extract the v0.5 chassis **without edits** into a clean restoration branch/tree. Preserve original paths and hashes where practical.

### R3 — dependency closure

Build an import/include/call dependency graph for:

- IR/graph authority;
- orchestrator;
- proof;
- failure signatures;
- attribution;
- repair;
- export;
- runtime;
- numerical subsystems.

Identify god-object dependencies and old front-end ontology dependencies separately rather than carrying them forward invisibly.

### R4 — source-diff selection

Subsystem-by-subsystem compare late-May/July numerical maxima to v0.5 ports. Record a decision per file/function/class, not merely per archive.

### R5 — authority rebind

Recreate R5_3/v97.39 semantics against the new typed IR:

```text
proposal IDs != canonical IDs
one Compiler owner
one canonical product lineage
proof == repair == exported product state
```

### R6 — solver regression fixtures

Before S/G/W integration, freeze mathematical fixtures for CDT, rig graph, BBW/KKT, ARAP and XPBD/contact as applicable.

### R7 — proof/repair restoration

Restore motion proof, failure signatures, attribution, repair and orchestrator retries around synthetic/current typed product fixtures.

### R8 — runtime restoration

Restore `.rss/.realsas/.rsr` package path and `runtime/realsas_cpp`; reproduce build + ABI smoke; then add package-to-runtime fixture.

### R9 — narrow S/G/W adapters

Only after S/G/W schemas are sufficiently frozen, write narrow adapters into the restored Compiler Core. Do not resurrect old perception ontology.

## 7. Branch discipline

Recommended repository flow:

```text
architecture/compiler-ir-solver-canonical-20260825
        ↓ architecture + restoration contracts only

restoration/compiler-v05-forensic-base
        ↓ byte-exact historical extraction; no semantic edits

restoration/compiler-historical-maximum
        ↓ subsystem source-diff selections + regression fixtures

integration/sgw-compiler-rebind
        ↓ new typed adapters after S/G/W freeze
```

Do not mix forensic extraction, architecture changes and learned-stage experiments into one branch.

## 8. Completion criteria for restoration baseline

The historical compiler/runtime is considered restored only when:

1. exact file-level provenance registry exists;
2. selected source tree builds/imports;
3. canonical authority/single-truth regression tests pass;
4. selected numerical kernels pass residual/invariant fixtures;
5. motion proof → failure → attribution → repair → re-proof fixture passes;
6. final product proof binds to the exact exported product hash;
7. C++ runtime builds and ABI smoke passes;
8. package-to-runtime fixture passes;
9. old front-end ontology is not a hidden dependency of current S/G/W adapters.

Until then, historical audit confidence is not equivalent to migrated-production proof.
