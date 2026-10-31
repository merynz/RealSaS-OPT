# RealSaS Studio Editor V1 Roadmap

Status: ACTIVE PRODUCTIZATION CONTRACT

## 1. Product boundary

Studio edits an already proof-qualified RealSaS puppet. It is not the Awaken scene and it is not product-pass authority.

Canonical rule:

`ProductRevision (immutable) + StudioAuthoringRevision (sibling overlay) -> requalification -> optional new ProductRevision`

Studio MUST NOT mutate the sealed ProductRevision or its canonical bundle in place.

## 2. Canonical editor state names

These names are reserved for Studio V1 implementation and should stay stable across backend, browser state and tests.

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

## 3. V1 parity target

The target is a professional 2D skeletal animation editor in the Spine class of workflow quality. It does not need to clone Spine's UI, but the core editing loop must be equivalently practical.

### A. Project/document system — P0

Required:

- proof-qualified immutable base puppet
- persisted StudioProject
- append-only StudioAuthoringRevision history
- optimistic concurrency using client-held lock version
- reopen restores latest authoring head
- explicit conflict state; no silent last-write-wins
- autosave debounce plus explicit Save
- revision history and named snapshots
- local undo/redo independent of persisted revisions

Current status:

- persistent project/revision/head schema: DONE
- append-only revision writes: DONE
- client-held optimistic lock bridge: DONE
- head/history APIs: DONE
- reopen applies latest authoring payload to StudioDocument: NEXT
- operation journal / undo / redo: NEXT
- autosave: NEXT

### B. Viewport and transforms — P0

Required:

- pan/zoom/fit
- high-DPI canvas
- select/add/remove/multi-select
- translate/rotate/scale gizmos
- local/world transform modes
- pivot editing
- snapping and modifier keys
- box/lasso selection
- art/mesh/rig/weight overlays
- direction/view switching without losing selection semantics

### C. Rig editor — P0

Required:

- create/delete/reparent bones and controls
- hierarchy tree reorder
- bone length/orientation editing
- deform vs control distinction
- IK constraints and targets
- constraint inspector
- mirrored creation/editing where valid
- canonical IDs remain stable; user-created IDs live in authoring namespace

### D. Mesh/topology editor — P0

Required:

- vertex/edge/face selection
- add/delete/split/weld vertices and edges
- face creation/deletion
- triangulation visualization
- topology validity checks while editing
- local remesh/repair command without silent canonical mutation
- UV/source-address inspection where available

### E. Skin/weight editor — P0

Required:

- weight paint brush
- add/subtract/replace/smooth modes
- brush radius/strength/falloff
- normalize and prune
- lock influences
- numeric per-vertex influence inspector
- mirror/copy weights when mapping is valid
- heat/gradient visualization
- deformation preview while painting

### F. Slots, attachments and draw order — P0

Required:

- attachment list and transforms
- bind attachment to control/bone
- slot/draw-order timeline
- per-view presentation inspection
- source-supported visibility/occlusion evidence display
- explicit warning when user override departs from qualified Graphics evidence

### G. Animation editor — P0

Required:

- multiple clips
- play/pause/stop/scrub/loop range
- dopesheet with multi-track selection
- key add/delete/duplicate/move/scale
- transform channels
- interpolation: step/linear/bezier
- curve editor with tangent handles
- copy/paste keys and poses
- timeline zoom/pan
- ghost/onion comparison where useful
- canonical motion bake remains read-only; user clips/edits are sibling authoring data

### H. Proof/requalification — P1

Required:

- dirty edits clearly mark `requires_requalification`
- show which proof domains an edit invalidates
- requalification request from Studio
- compare last qualified ProductRevision vs working authoring overlay
- failed proof links user back to implicated edit/domain instead of generic failure

### I. Export — P1

Required:

- export only qualified native `.rss` as authoritative runtime package
- explicit draft preview/export distinction for unqualified Studio edits
- later engine adapters consume `.rss`; engine-specific state must not become Studio authority

## 4. Backend/API contract

Public browser-visible identifiers:

- compile: `cmp_<32 hex>`
- Studio project: `stp_<32 hex>`
- revision number + lock version

Never expose:

- ProductRevision UUID
- Attempt UUID
- EngineRelease UUID
- model release UUIDs
- artifact UUIDs
- internal authoring revision UUIDs
- product gateway service token

Current public edge routes:

- `GET /api/studio/session?bundle=<public bundle>`
- `GET /api/studio/history?id=<stp_...>&limit=N`
- existing `POST /api/living/edit-layer` carries public Studio project id + expected lock version in production mode

Platform service routes:

- `POST /v1/product/compiles/{cmp}/studio-project`
- `GET /v1/product/studio/projects/{stp}`
- `GET /v1/product/studio/projects/{stp}/head`
- `GET /v1/product/studio/projects/{stp}/revisions`
- `POST /v1/product/studio/projects/{stp}/revisions`

## 5. Implementation order

1. Close Studio persistence: reopen head, local operation journal, autosave, conflict UX.
2. Professional selection/transform foundation shared by Rig/Mesh/Attachment tools.
3. Rig + IK creation/editing.
4. Mesh/topology tools.
5. Weight paint and deformation preview.
6. Dopesheet and key editing.
7. Bezier curve editor.
8. Slots/attachments/draw order and Graphics evidence inspection.
9. Proof invalidation/requalification UX.
10. Export polish and performance profiling.

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
- hiding proof failures by silently repairing user edits
- engine-specific editor forks
- training/fitting models inside Studio interactive tools
- coupling editor progress to availability of final IRIS/TESSA/AXIS/MIRA checkpoints
