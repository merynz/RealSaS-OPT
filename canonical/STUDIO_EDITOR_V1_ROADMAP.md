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

A deployable Studio is not merely a puppet correction surface. A user must be able to open a compiled puppet, author or repair the puppet, create animation clips, key transforms, preview weighted deformation, save authoring revisions and continue later without mutating the sealed ProductRevision.

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
- reopen applies latest authoring payload to StudioDocument: DONE
- StudioDocument authoring overlay boundary: DONE
- StudioOperationJournal / undo / redo: DONE
- StudioDirtyState clean/dirty/saving/conflict/requalification: DONE
- debounced autosave + explicit Save + keyboard Save: DONE
- playback isolated from authoring rig edits: DONE
- Studio editor-core Node contract court: DONE / EXPANDED
- real Postgres Studio repository court: IN CI
- revision history browser surface: API READY, UI DEFERRED
- named snapshots: TODO

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

Current status:

- zoom/fit and art/mesh/rig/weight overlays: EXISTING
- `StudioSelection` per-view multi-select + primary selection: FOUNDATION DONE
- `StudioToolContext`: FOUNDATION DONE
- high-DPI interaction overlay: FOUNDATION DONE
- translate/rotate/scale selection transforms: FOUNDATION DONE
- pan + cursor-centered zoom: FOUNDATION DONE
- translate/rotation/scale snapping + bypass modifier: FOUNDATION DONE
- selection/primary/custom pivot data model: FOUNDATION DONE
- box selection: FOUNDATION DONE
- direction switching preserves per-view StudioSelection state: FOUNDATION DONE
- local/world mode state and UI: FOUNDATION DONE; local-axis transform math still TODO
- lasso selection: TODO
- explicit draggable pivot gizmo: TODO

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

Current status:

- control transform editing: EXISTING + PROFESSIONAL TRANSFORM FOUNDATION
- add/delete/reparent authoring operations: EXISTING FOUNDATION
- basic IK authoring operation: EXISTING FOUNDATION
- hierarchy reorder, bone length/orientation handles, mirrored authoring and complete constraint inspector: TODO

### D. Mesh/topology editor — P0

Required:

- vertex/edge/face selection
- add/delete/split/weld vertices and edges
- face creation/deletion
- triangulation visualization
- topology validity checks while editing
- local remesh/repair command without silent canonical mutation
- UV/source-address inspection where available

Current status:

- vertex selection + multi-select transform: FOUNDATION DONE
- existing triangulation display: EXISTING
- edge/face selection and structural topology tools: TODO
- topology validity/live repair courts: TODO

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

Current status:

- per-vertex weight delta authoring: EXISTING FOUNDATION
- weight visualization: EXISTING FOUNDATION
- Studio user-clip weighted deformation preview: FOUNDATION DONE
- professional brush modes/radius/falloff/normalization/locks/mirror: TODO

### F. Slots, attachments and draw order — P0

Required:

- attachment list and transforms
- bind attachment to control/bone
- slot/draw-order timeline
- per-view presentation inspection
- source-supported visibility/occlusion evidence display
- explicit warning when user override departs from qualified Graphics evidence

Current status:

- attachment reassignment operation: EXISTING FOUNDATION
- attachment transform/slot/draw-order editor and Graphics-evidence inspection: TODO

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

Current status:

- `StudioTimeline` authoring core: FOUNDATION DONE
- user clip create/duplicate/delete/rename: FOUNDATION DONE
- duration/FPS/loop authoring: FOUNDATION DONE
- transform tracks and translation/rotation/scale keys: FOUNDATION DONE
- key add/delete/select/multi-select/move/scale core: FOUNDATION DONE
- key copy/paste and frame nudge: FOUNDATION DONE
- step/linear/bezier interpolation: FOUNDATION DONE
- Bezier numeric control + curve visualization: FOUNDATION DONE
- local user-clip playback with weight-driven mesh deformation preview: FOUNDATION DONE
- proof-owned runtime clips remain read-only: PRESERVED
- dopesheet key drag integration with StudioOperationJournal: FOUNDATION DONE
- draggable Bezier tangent handles: TODO
- UI for selected-key time scaling: TODO
- timeline zoom/pan + loop-range handles: TODO
- pose copy/paste: TODO
- onion/ghost comparison: TODO

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

The deploy target requires a real authoring loop, so animation is intentionally advanced in parallel instead of being deferred behind every puppet tool.

1. Close Studio persistence: reopen head, local operation journal, autosave, conflict UX. **DONE except named snapshot/history presentation.**
2. Professional selection/transform foundation shared by Rig/Mesh/Attachment tools. **FOUNDATION DONE.**
3. Real animation-authoring foundation: StudioTimeline, clips, dopesheet keys, curves, weighted deformation preview. **FOUNDATION DONE.**
4. Finish Rig + IK professional creation/editing and hierarchy UX.
5. Finish Mesh/topology tools and live validity courts.
6. Finish Weight Paint professional brush/deformation workflow.
7. Finish animation parity: tangent handles, timeline zoom/pan, loop range, key scaling UI, pose copy/paste.
8. Slots/attachments/draw order and Graphics evidence inspection.
9. Proof invalidation/requalification UX.
10. Revision-history/named snapshots, export polish and performance profiling.

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
