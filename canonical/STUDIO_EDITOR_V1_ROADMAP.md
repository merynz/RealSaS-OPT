# RealSaS Studio Editor V1 Roadmap

Status: ACTIVE PRODUCTIZATION CONTRACT

## 1. Product boundary

Studio edits an already proof-qualified RealSaS puppet. It is not the Awaken scene and it is not product-pass authority.

Canonical rule:

`ProductRevision (immutable) + StudioAuthoringRevision (sibling overlay) -> compiler requalification -> optional new ProductRevision`

Studio MUST NOT mutate the sealed ProductRevision or its canonical bundle in place. Preview state is never authority. A Studio edit becomes canonical only after the affected product capabilities re-execute and terminal `product.proof` passes.

## 2. Stable editor state names

These names are canonical for Studio V1 and should remain stable across backend, browser state and tests.

- `StudioProject`: persistent editing project anchored to one immutable ProductRevision.
- `StudioAuthoringRevision`: immutable persisted snapshot of the user-authored overlay.
- `StudioProjectHead`: current persisted revision pointer + optimistic `lock_version`.
- `StudioDocument`: in-memory editor document = canonical base + mutable working overlay.
- `StudioOperation`: one reversible editor command.
- `StudioOperationJournal`: undo/redo stack of StudioOperations since the last persisted revision.
- `StudioSelection`: current bones, controls, mesh vertices/faces, attachments, slots, keys or curves.
- `StudioToolContext`: active tool, snapping, transform mode and interaction state.
- `StudioTimeline`: clip, playhead, loop range, tracks, keys and curve state.
- `StudioDirtyState`: clean/dirty/saving/conflict/requalification-required state.

## 3. Current V1 editor status

The target is a professional 2D skeletal animation editor in the Spine class of workflow quality. The current implementation is a professional foundation, not a claim of complete Spine feature parity.

### A. Project/document system — P0

DONE:

- proof-qualified immutable base puppet
- persisted `StudioProject`
- append-only `StudioAuthoringRevision` history
- optimistic concurrency using client-held lock version
- reopen restores latest authoring head
- explicit conflict state; no silent last-write-wins
- debounced autosave plus explicit Save and keyboard Save
- `StudioDocument` sibling-authoring boundary
- `StudioOperationJournal` undo/redo
- `StudioDirtyState`
- browser revision-history drawer
- audit-safe restore: restore appends a new revision instead of rewinding history
- Postgres courts for stale editors, idempotent save/restore and append-only revision triggers

REMAINING:

- user-facing named snapshots/bookmarks on top of immutable revision history
- history diff/compare presentation

### B. Viewport and transforms — P0

DONE / FOUNDATION:

- pan, zoom, fit and high-DPI interaction overlay
- per-view multi-selection and primary selection
- translate/rotate/scale transforms
- snapping + modifier bypass
- custom pivot data model
- box selection
- art/mesh/rig/weight overlays
- per-view selection preservation
- local/world mode state and UI

REMAINING:

- local-axis transform math completion
- lasso selection
- explicit draggable pivot gizmo
- viewport performance profiling on large puppets

### C. Rig editor — P0

DONE / PROFESSIONAL FOUNDATION:

- deform-joint vs animator-control semantic split
- control transforms
- add/delete/reparent authoring operations
- hierarchy editing foundation
- bone position/length/orientation editing foundation
- mirrored edit support foundation
- IK authoring list/form and target editing
- compiler/Mechanics remains true constraint authority

REMAINING:

- richer hierarchy reorder UX
- richer direct bone-handle interaction
- complete constraint inspector / constraint-specific gizmos
- additional mirror policies where structural mapping is valid

### D. Mesh/topology editor — P0

DONE / STRUCTURAL P0:

- vertex/edge/face selection
- stable projected canonical IDs plus user-authoring IDs
- add/move/delete vertex
- add/delete face
- split edge
- weld vertices
- prune unreferenced vertices
- structural subtool and overlay
- browser-side deterministic replay validation
- server-side independent topology validation
- duplicate-face, degenerate-area, manifold-edge and split/weld safety courts
- canonical mesh remains immutable; structural edits live in `mesh_topology_edits`

REMAINING:

- local remesh/repair command with explicit user intent
- UV/source-address inspection where available
- richer topology diagnostics and visual invalidity markers

### E. Skin/weight editor — P0

DONE / PROFESSIONAL FOUNDATION:

- per-vertex deform-joint weight authoring
- add/subtract/replace/smooth brush modes
- radius/strength/falloff
- auto-normalization
- influence locks
- mirror workflow
- numeric stats/inspection foundation
- weight visualization
- weighted deformation preview during user animation
- undo/autosave integration

REMAINING:

- richer influence table editing and pruning controls
- copy/paste weight sets
- more explicit symmetry-map diagnostics
- larger-project performance profiling

### F. Slots, attachments and draw order — P0

DONE / P0 FOUNDATION:

- Graphics-derived qualified per-view back-to-front order as default
- slot reorder front/back
- hide/show per view
- reset to qualified Graphics presentation
- explicit Graphics evidence departure state
- attachment/socket binding editor
- deform-joint reassignment
- per-view translation, rotation, scale and visibility overrides
- socket gizmo overlay
- V0/V1 isolation keyed by `view_id + binding_id`
- user presentation/binding edits remain sibling authoring data and require requalification

AUTHORITY RULE:

Studio does not invent canonical draw-order truth. Qualified Graphics presentation is the default. User overrides are explicit departures and must be compiler-requalified before becoming authoritative.

REMAINING:

- expose/link true compiler attachment anchors when available instead of relying only on generic binding rows
- slot/draw-order keying on the animation timeline
- richer visibility/occlusion evidence inspection

### G. Animation editor — P0

DONE / STRONG FOUNDATION:

- multiple user clips
- create/duplicate/delete/rename
- duration/FPS/loop
- play/pause/stop/scrub
- animator-control transform tracks
- translation/rotation/scale keys
- key add/delete/select/multi-select/move/scale
- key copy/paste and frame nudge
- step/linear/bezier editor interpolation
- curve visualization and draggable Bezier tangent handles
- timeline zoom/pan, fit-time and loop-range interaction foundation
- dopesheet drag integrated with `StudioOperationJournal`
- proof-owned runtime clips remain read-only
- user-clip weighted deformation preview

REMAINING:

- pose copy/paste
- onion/ghost comparison
- richer tangent modes and tangent policy UX
- multi-clip organization polish
- slot/draw-order keys

IMPORTANT SHIPPING LIMIT:

The first compiler requalification slice intentionally accepts only one saved Studio animation clip in animator-control space with linear keys. The editor may preview richer interpolation, but Bezier/step execution is not allowed to mint canonical Motion until its runtime semantics receive separate qualification courts.

### H. Proof/requalification — P1

DONE / FIRST SHIPPING SLICE:

- dirty edits mark requalification-required state
- deterministic invalidated-domain planning
- fail-closed unsupported edit sets
- immutable `RealSaS.StudioAuthoringRevision/v1` capture with authority class `STUDIO_AUTHORING_INPUT`
- distinct `PRODUCT_REQUALIFICATION` execution goal
- exact base EngineRelease/model snapshot preserved
- base ProductRevision never mutated
- animation-only canonical cut-set inherits qualified Mesh/Rig/Skin/Mechanics/Graphics authority
- requalification executes only `product.motion -> product.proof`
- Motion consumes Studio animation through a typed proposal adapter and mints canonical `CanonicalMotionIR.v2`
- terminal Proof must pass before a new ProductRevision is sealed
- successful requalification creates a new opaque `cmp_*` session and new proof-gated workspace
- browser polls opaque `srq_*` state through the product-shell boundary
- Studio automatically hands off to the new clean workspace after PASS
- Postgres court verifies no fake execution/PASS state is created at admission, exact model lineage is retained, and inherited authority remains qualified

CURRENT SHIPPING SLICE:

`ANIMATION_LINEAR_V1`

Admission requirements:

- current persisted Studio HEAD
- exactly one user animation clip
- animator-control track space
- linear curve execution only
- no mesh, topology, rig, weight, binding, IK or presentation-evidence departure edits in the same requalification request

REMAINING REQUALIFICATION SLICES:

1. Rig/IK edits -> Rig/Mechanics/Motion/Proof as required by typed authoring adapters.
2. Weight edits -> Skin/Mechanics/Motion/Proof.
3. Mesh/topology edits -> Mesh and all dependent downstream authorities through Proof.
4. Binding/attachment edits -> typed Rig/Mechanics/Graphics downstream path.
5. Presentation/draw-order edits -> Graphics downstream path.
6. Qualified Bezier/step animation execution semantics.
7. proof-failure UX that links directly to implicated edit/domain and affected evidence.

No slice may be enabled merely because Studio can preview the edit. Each slice needs a typed compiler adapter, dependency cut-set, lineage rules and execution/proof courts.

### I. Export — P1

CURRENT RULE:

- `EXPORT` is authoritative only for the current proof-qualified base ProductRevision.
- when Studio edits exist, UI explicitly says `EXPORT QUALIFIED BASE`; current edits are excluded.
- after requalification PASS and handoff to the newly sealed ProductRevision, that new base may be exported normally.

REMAINING:

- edited-runtime export convenience once all relevant requalification slices are supported
- engine adapter packaging/polish while keeping `.rss` as engine-agnostic authority

## 4. Backend/API contract

Public browser-visible identifiers:

- compile: `cmp_<32 hex>`
- Studio project: `stp_<32 hex>`
- requalification: `srq_<32 hex>`
- revision number + lock version

Never expose:

- ProductRevision UUID
- Attempt UUID
- EngineRelease UUID
- model release UUIDs
- artifact UUIDs
- internal authoring revision UUIDs
- product gateway service token

Public product-shell routes include:

- `GET /api/studio/session?bundle=<public bundle>`
- `GET /api/studio/history?id=<stp_...>&limit=N`
- `POST /api/studio/restore`
- `POST /api/living/edit-layer`
- Studio requalification request/status proxy routes
- existing compile status/workspace handoff routes

Platform service routes include owner-scoped Studio project/revision/history/restore and requalification endpoints. Browser traffic reaches them through the product shell; service credentials stay server-side.

## 5. Implementation order from current checkpoint

1. Keep animation requalification `ANIMATION_LINEAR_V1` green across Motion, Platform and Postgres courts. **DONE at current checkpoint.**
2. Remove temporary productization-only CI helpers and keep only stable shipping workflows. **NEXT CLEANUP.**
3. Pose copy/paste + onion/ghost + tangent-mode polish.
4. Add typed Rig/IK requalification adapter and dependency cut-set.
5. Add typed Skin/weight requalification adapter.
6. Add typed Graphics presentation/binding requalification adapters.
7. Add Mesh/topology requalification adapter last because it invalidates the widest downstream authority set.
8. Expose true compiler attachment anchors and richer evidence inspection.
9. Finish named snapshots/history compare.
10. Performance/profile pass and deployment polish.

## 6. Performance budgets

Interactive editor operations MUST remain local and frame-rate safe; network persistence is never on the pointer-move hot path.

Initial V1 budgets:

- viewport target: 60 fps on normal project sizes
- pointer interaction response: <16 ms target per frame
- undo/redo: <50 ms for ordinary operations
- autosave serialization: off pointer hot path
- persisted revision write: asynchronous, optimistic-lock protected
- runtime proof-bake preview requests may be asynchronous and must never block editing tools

## 7. Non-goals for this phase

- replacing Awaken with Studio controls
- mutating ProductRevision directly from editor actions
- promoting Studio preview state directly to product authority
- hiding proof failures by silently repairing user edits
- engine-specific editor forks
- training/fitting models inside Studio interactive tools
- coupling editor progress to availability of final IRIS/TESSA/AXIS/MIRA checkpoints
