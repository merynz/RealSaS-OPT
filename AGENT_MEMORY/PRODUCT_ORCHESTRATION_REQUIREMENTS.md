# Product Orchestration Requirements

> NON-AUTHORITATIVE TARGET CONTRACT. Promote only after main recovery and explicit implementation review.

## Why this is P0

Two observed product-facing failures expose the same orchestration defect:

1. Repeating a render request produced a different render because the effective artifact lineage changed.
2. A later render request attempted to fit models again instead of reusing the already-qualified character state.

These are not renderer UX quirks. They mean the system does not yet have a stable product-level definition of **render current character**.

## Product invariant: deterministic current lineage

For a promoted subject state:

```text
(subject_current_artifact_set,
 motion_preset,
 view/camera,
 render_settings)
        -> deterministic render identity
```

Repeated render requests MUST resolve to the same sealed lineage unless the user explicitly promotes or selects a different lineage.

A render request may not silently:
- switch mesh/skeleton/skin/appearance lineage;
- select a newer research attempt;
- refit IRIS, Geppetto or Arachne;
- recalibrate policies;
- rerun unrelated upstream stages;
- mutate product-current pointers.

## Product invariant: render is not compile

Default product operations are distinct:

```text
COMPILE
  Resolve desired puppet state.
  Reuse compatible artifacts.
  Execute only missing/invalidated graph.
  May run model inference.
  May use already-promoted model checkpoints.
  Must NOT train/fit models in product mode.

RENDER
  Resolve one exact promoted puppet lineage.
  Reuse sealed puppet + motion + appearance artifacts.
  Execute only required projection/package/native/render tail.
  Never fit/train/calibrate/promote.

RENDER_AGAIN
  Same lineage and settings as previous render unless caller changes an explicit input.
```

Training/fitting belongs to Developer/Forge mode only.

## Product-mode forbidden operations

Unless an explicit developer/research command is used:

```text
FIT_IRIS                FORBIDDEN
FIT_GEPPETTO            FORBIDDEN
FIT_ARACHNE             FORBIDDEN
CALIBRATE_POLICY        FORBIDDEN
PROMOTE_RESEARCH_STATE  FORBIDDEN
AUTO_SWITCH_LINEAGE     FORBIDDEN
```

## Stable promoted subject pointer

Product UI should resolve a subject through a typed promoted state, conceptually:

```text
SubjectCurrent:
  observation
  geometry
  mesh
  skeleton
  skin
  appearance
  visual_presentation
  motion_library
  runtime_compatibility
```

Each entry is a typed immutable artifact reference. SHA/content identities remain underneath the abstraction and are used for verification, not as the normal human/agent interface.

## Two identities, not one

Execution provenance and artifact reuse identity must be separated.

```text
ArtifactIdentity =
  semantic_stage_version
  + implementation identity
  + policy identity
  + exact input artifact identities

ExecutionIdentity =
  ArtifactIdentity
  + run/attempt/machine/timing provenance
```

`run_id` must not make byte/semantic-equivalent artifacts unusable across runs.

## Minimal invalidation

A code or subsystem change invalidates only the artifact subgraph that semantically depends on it.

Examples:

- Runtime renderer implementation changes:
  reuse model checkpoints, geometry, mesh, skeleton, skin, CAA, visual mesh, motion;
  rerun runtime/render tail only.

- Stage35 mechanics changes:
  reuse unrelated upstream model fits and observations;
  rerun the true mechanics descendants and any presentation/runtime artifacts that bind to them.

- Appearance completion changes:
  do not refit mechanics models;
  invalidate appearance/presentation/runtime descendants only.

- IRIS model/checkpoint changes:
  invalidate the graph that consumes IRIS evidence, but do not retrain Geppetto/Arachne merely because a new execution run was opened; reuse remains governed by compatibility and artifact identity.

## Research repair model

Repairs should use versioned attempts, not in-place graph cycles:

```text
Attempt N
  immutable artifact graph
      |
      | typed RepairDirective(owner=Stage X)
      v
Attempt N+1
  inherit unaffected artifacts by identity
  recompute owner + true descendants only
```

This preserves exact provenance and avoids full Stage01–46 reruns.

## Developer vs Product interface

### Developer / Forge

May:
- inspect raw stage graph and lineage;
- fit/train models;
- run single stages/courts;
- invalidate subgraphs;
- create candidate attempts;
- run counterfactuals;
- inspect hashes and source seals;
- explicitly promote a qualified attempt.

### Product / Studio

May:
- import character;
- compile character;
- select/edit puppet;
- select animation preset;
- preview/render;
- export.

Product UI does not expose or invoke fit/training/research operations.

Both interfaces use the same Engine and artifact registry. There is no separate product pipeline.

## Target user experience

```text
Import character
      ↓
Compile
      ↓
Puppet Ready
      ↓
Idle / Run / Slash / Jump
      ↓
Immediate preview from the exact promoted puppet
```

Once a puppet is compiled, a simple **render again** request should normally take seconds to minutes, not retrigger hours of model fitting or a full pipeline replay.

## Acceptance tests for professional orchestration

1. **Render determinism**
   - Two identical render requests resolve identical artifact lineage and output hash, modulo explicitly declared nondeterministic rendering fields (ideally none).

2. **No training on render**
   - Product render path mechanically proves zero fit/train adapters executed.

3. **Cross-run reuse**
   - A new run ID with unchanged semantic inputs reuses qualified artifacts.

4. **Minimal invalidation**
   - A runtime-only code change does not invalidate model fits, mesh, rig, skin or appearance.

5. **Stable product pointer**
   - Research attempts cannot silently become the product-current subject.

6. **Explicit promotion**
   - Product-current changes only through a qualified promotion transaction.

7. **Render-only resume**
   - Existing sealed subject can generate a new preset/view/render without Stage09–33 training/compile replay.

8. **Forensic traceability**
   - Any render can still expand to exact artifact hashes, producer versions, policies and execution provenance when needed.

## Existing evidence to preserve

The 1941-commit product-research trunk already contains useful pieces rather than starting from zero:

- DAG scheduler and descendant invalidation logic;
- per-stage input / implementation / policy fingerprints;
- stale PASS/FAIL reopening semantics;
- content-addressed cache primitives and tests;
- historical import verification;
- render-only Knight resume path with explicit `stage17_25_recomputed=False`;
- orchestration audit recommending transactional versioned DAG attempts.

The platform task is therefore largely **normalization and completion of existing capabilities into one professional execution contract**, not replacement of the compiler research.
