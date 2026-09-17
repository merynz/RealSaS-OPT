# Runtime-v4 compact playback preregistration

Status: PREREGISTERED DESIGN — IMPLEMENTATION NOT YET CLAIMED

Parent design input: `experiments/playback_stack_v1/SPINE_CLEANROOM_V1.md`

## Motivation

Runtime-v3 is scientifically useful but is not an acceptable shipping representation for dense BODY. The current TEST_SUBJECT_001 path constructs a separate BODY mesh for each of 8 views and serializes every mesh's posed XYZ for every sampled frame.

For the current dense BODY vertex count (257,505), posed XYZ alone costs approximately:

`257505 vertices * 8 views * 3 float32 * 4 bytes = 24,720,480 bytes/frame`

This is about 23.6 MiB/frame before composition, topology, appearance, textures and container overhead.

At 18 total frames (9 idle + 9 run), posed XYZ alone is ~424 MiB. This explains the observed ~469 MiB BODY-only package order of magnitude. Increasing temporal sampling without changing representation would make package size and export cost pathological.

Runtime-v4 must remove that multiplicative duplication before higher-FPS product rendering is used as the normal path.

## Non-negotiable authority boundary

- compiler owns topology discovery, rig interpretation, skin qualification, motion qualification, appearance provenance, component ownership and draw order;
- runtime does not infer any of those;
- dense scientific BODY remains a sealed compiler/proof authority;
- runtime-v4 may consume a dense representation for diagnostics, but shipping output must support a separately qualified compact runtime mesh;
- no runtime optimization may silently change scientific authority or upgrade an unqualified artifact.

## Core representation

### 1. Immutable attachment assets

Each unique attachment asset is stored once and identified by `attachment_asset_id`.

A deformable asset contains, as applicable:

- rest/canonical 3D vertices once;
- topology once;
- skin/binding identity or a baked-deformation policy identity;
- provenance to sealed compiler truth;
- per-view appearance bindings/UVs/provenance as view overlays, not duplicated geometry.

A rigid asset contains:

- local geometry once;
- canonical parent-joint binding;
- bind transform authority;
- per-view appearance binding/atlas reference.

### 2. View overlays

Each view stores only view-specific state:

- `view_id` / `view_index`;
- exact qualified camera/projection contract;
- texture/atlas references;
- per-attachment UV mapping or equivalent source appearance mapping;
- per-face appearance provenance where required;
- static view visibility policy where qualified.

The same canonical deformable geometry must not be duplicated solely because eight source views exist.

### 3. Runtime slots

A compact slot table stores:

- stable slot ID;
- canonical parent joint or deformable owner;
- setup order;
- default attachment asset ID;
- allowed attachment IDs;
- optional clip relation metadata.

Current Mage minimum slots:

1. BODY_UNDERLAY
2. CAPE_FOREGROUND
3. HAT_FOREGROUND
4. BOOK_FOREGROUND
5. STAFF_FOREGROUND

### 4. Frame state

A frame stores only changing state.

For deformable BODY in v4 initial mode:

- canonical posed XYZ is stored **once per deformable attachment per frame**, not once per view;
- runtime applies only the exact precompiled view projection to obtain screen XYZ/depth;
- projection is not scientific re-solving and may not alter deformation semantics.

For rigid attachments:

- prefer compact canonical parent-joint transform/carry state over writing every rigid vertex every frame;
- local rigid geometry remains immutable.

Per-view frame composition stores only:

- ordered slot indices or a compact draw-order delta;
- active attachment index per slot when different from setup state;
- visibility overrides;
- clip intervals when present.

### 5. No repeated static payload

The following are forbidden per-frame unless an explicit topology-changing feature requires them:

- topology;
- UV arrays;
- atlas payloads;
- source provenance tables;
- rest geometry;
- static rigid geometry;
- full component metadata;
- sealed proof graphs.

## Initial compatibility mode

Runtime-v4 shall first support an exact diagnostic mode that preserves current dense D1 posed geometry semantics while removing 8-view XYZ duplication.

This mode is intended to prove representation equivalence before runtime mesh reduction is introduced.

Required equivalence check:

For every tested clip/frame/view and every canonical BODY vertex:

`project_v4(canonical_posed_xyz_once, qualified_camera[view])`

must match the Runtime-v3 per-view posed XYZ used by the reference renderer within a preregistered numerical tolerance.

No visual/product pass follows from this parity alone.

## Compact runtime mesh mode

A later/parallel runtime-mesh derivation stage reduces the dense scientific substrate to a platform-appropriate runtime mesh.

The reduction must preserve and report:

- 8-view silhouette/coverage error;
- deformation error across qualified motion samples;
- attachment seam error;
- triangle-flip / invalid topology checks;
- appearance reprojection error;
- provenance from each runtime vertex/triangle to the sealed dense authority;
- exact source authority hashes.

The dense scientific mesh remains available for proof and regression comparison but is not shipped by default.

## Product-closure performance contract

Runtime-v4 work does not replace compiler optimization. Both are required.

Before the next full Mage render, the product path must additionally obey:

- already sealed dense BODY/derivation/assembly artifacts are not recursively revalidated at every composition boundary;
- parent hashes compose sealed child identities instead of serializing entire descendants again;
- identical component/direction/set validations are cached only under exact immutable object identity + mechanical authority;
- exact coverage validation is O(n) time and O(1) auxiliary memory where uniqueness is already proven;
- V4 does not assemble an intermediate product that it immediately discards;
- progress telemetry is emitted by long-running stages;
- prior sealed stages survive interruption and are directly reusable.

## Measurement gates

The corrected implementation must report at minimum:

- wall-clock duration by stage;
- peak RSS by stage where practical;
- bytes in static asset tables;
- bytes in per-frame deform state;
- bytes in per-view state;
- package total bytes;
- frame count and nominal FPS;
- dense vs compact runtime vertex/face counts;
- Runtime-v3 versus Runtime-v4 parity result for diagnostic mode.

No performance success is claimed without measured results on the reference workstation.

## Current expected size effect before mesh reduction

Holding the current dense vertex count fixed, storing canonical posed XYZ once rather than separately for all 8 views should reduce the dominant BODY per-frame XYZ payload by approximately 8x, before compression and before considering additional rigid-component savings.

This is an expectation from data layout arithmetic, not a measured implementation result.

## Fail-closed rules

- no guessed camera/projection;
- no runtime appearance completion;
- no inferred draw order;
- no filename-derived component semantics;
- no runtime weight synthesis;
- no runtime topology repair;
- no promotion from diagnostic parity to PRODUCT_PASS;
- if exact view projection equivalence cannot be demonstrated, Runtime-v4 diagnostic mode remains unqualified.
