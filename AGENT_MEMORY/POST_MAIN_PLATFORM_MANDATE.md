# Post-Main Platform Mandate

> User-authorized program of work. This is a planning/continuity record, not current production authority.

## Order of operations

### Phase A — Recover and prove canonical main

Do not begin a broad platform rewrite until the repository's canonical engineering state is reconstructed and a recovery head is proven.

Required outcome:
- identify the latest justified product-research frontier from the 1941-commit Knight trunk and later normalization lineages;
- distinguish latest research frontier from scientifically proven/final truth;
- preserve only living/superseding decisions and generic fixes;
- repair known open seams needed for a coherent main;
- run exact current gates;
- produce one canonical mainline;
- preserve evidence/provenance without leaving multiple ambiguous active product states.

### Phase B — Turn RealSaS into a professional system

Once canonical main is restored, platform engineering becomes a first-class product workstream.

This phase is explicitly authorized to:
- choose the implementation language(s);
- introduce mature third-party infrastructure where appropriate;
- add a database, artifact store, workflow/orchestration engine, model registry or other platform software;
- replace ad-hoc repository/hash choreography with professional abstractions;
- refactor orchestration and artifact lifecycle substantially when justified;
- avoid reimplementing commodity infrastructure merely for ownership purity.

Technology choices must be evidence-driven and should prefer proven components over bespoke reinvention.

## Product target

RealSaS must behave as a real compiler/product system:

```text
Input Character
      ↓
Compile
      ↓
Editable Puppet Ready
      ↓
Animation Preset
      ↓
Immediate Scene Preview / Render
```

The internal complexity of IRIS, Geppetto, Arachne, Compiler, CAA, mechanics, motion, proof and runtime must not leak into ordinary product operation.

## Required platform separation

```text
RealSaS Engine
    ├── Models / inference
    ├── Compiler
    ├── Artifact Registry
    ├── Dependency Resolver
    ├── Orchestrator
    ├── Runtime
    └── Proof / qualification

RealSaS Forge
    └── Developer / research interface

RealSaS Studio
    └── Product / artist interface
```

Forge and Studio must use the same Engine and artifact truth. There must not be a second product pipeline.

## Core platform requirements

1. **Typed artifact registry**
   - Human/agent-facing logical artifact identities.
   - Exact hashes retained underneath for verification/provenance.
   - No normal workflow should require manually discovering dozens of SHAs.

2. **Atomic ProductRevision**
   - One promoted compatible set of geometry, mesh, skeleton, skin, appearance, visual presentation, motion and runtime artifacts.
   - Product-current may change only through an explicit promotion transaction.

3. **Artifact identity separate from execution identity**
   - Run IDs/timestamps/workers belong to execution provenance.
   - Semantically identical artifacts must remain reusable across runs.

4. **Content-addressed immutable storage**
   - Large checkpoints, meshes, textures, CAA assets, VisualMeshSet, RSS, motion, proofs and renders stored by immutable content identity.
   - Metadata lives separately from bytes.

5. **Minimal semantic invalidation**
   - A subsystem change invalidates only its true dependent artifact subgraph.
   - Full Stage01–46 replay is exceptional, not normal.

6. **Training is Developer-only**
   - Product compile/render may use promoted model checkpoints and model inference.
   - Product render may never silently fit IRIS/Geppetto/Arachne, recalibrate policy or promote research state.

7. **Stable render semantics**
   - Repeating the same render request against the same ProductRevision and settings resolves the same lineage.
   - `render again` means reuse + minimal downstream execution, not rebuild-the-world.

8. **Transactional research attempts**
   - Repairs produce child attempts.
   - Unaffected artifacts are inherited/reused by exact identity.
   - Research attempts cannot silently become product current.

9. **Durable resumability**
   - Process/runner failure must not require reconstructing valid completed work.
   - Execution state should be durable and queryable.

10. **Forensic expansion on demand**
    - Normal operation presents logical typed identities.
    - Exact hashes, source versions, policy versions, CI/gate evidence and execution provenance remain expandable for audit/debug.

## Build-vs-buy principle

Do not rediscover America.

Commodity capabilities should use mature infrastructure when it provides the right semantics, for example:
- relational metadata database;
- object/CAS storage;
- model registry;
- durable workflow execution;
- observability;
- migrations/backups.

Custom RealSaS code should concentrate on what is unique:
- typed artifact semantics;
- compiler dependency contracts;
- qualification/promotion rules;
- repair ownership;
- product revision resolution;
- RealSaS-specific minimal invalidation.

## Acceptance bar

The platform is not complete merely because it can execute all stages.

It is complete when:

- a developer can ask why an artifact changed without Git archaeology;
- an agent can resolve the current product state without inspecting dozens of hashes/files;
- a user can import one character and press Compile;
- an already compiled puppet can preview another preset without model retraining;
- a local subsystem change causes bounded recomputation;
- research and product state cannot accidentally contaminate each other;
- the system remains reproducible and forensically exact.

## Authorization

After canonical main is established, platform architecture and implementation choices are delegated to Alfred, including language and infrastructure choices, with the constraint that established reliable systems should be preferred over unnecessary bespoke reimplementation.
