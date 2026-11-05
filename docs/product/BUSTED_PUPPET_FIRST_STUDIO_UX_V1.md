# BUSTED — Puppet-first product & Studio UX contract v1

**Decision:** The paid deliverable is a proof-qualified, editable puppet **with a real Idle**. Motion/authoring builds on the same persistent puppet identity. The engine is RealSaS; the consumer product is BUSTED.

## Product flow

SOURCE ART → AWAKEN / REAL COMPILER → PROOF-QUALIFIED PUPPET + IDLE → ALIVE REVEAL → PUPPET SETUP / MOTION AUTHORING / EXPORT

- **AWAKEN:** Source intake and actual compiler events. Show explicitly reported overall percent when available. Otherwise present the discrete fraction of completed stages, clearly labeled *stages sealed*, and never invent elapsed-time progress.
- **REVEAL:** Proof PASS seals a ProductRevision. Show **QUALIFIED** while waiting for real Idle evidence. Show **ALIVE** only after a genuine Idle runtime clip and first valid deformable mesh frame.
- **PUPPET:** Same identity, setup-time mesh, rig, weights, graphics, attachment, constraint authoring, visibility and ownership.
- **MOTION:** Same identity, animate-time pose controls, direct gizmos, IK targets, clips, keyframes, timeline, dope sheet, curves, and preview.
- **EXPORT:** Proof-qualified base or properly requalified derivative. Do not silently export unqualified Studio edits.

## Product state vocabulary

| Name | Proposed role |
| --- | --- |
| CompileSession | Server-bound execution and evidence events |
| ProductRevision | Immutable proof-qualified compiler output |
| PuppetArtifact | Mesh/rig/skin/graphics/mechanics owned by the revision |
| StudioWorkspace | User-owned, persisted sibling authoring revision |
| PoseControl | Directly selectable, poseable rig/animation handle |
| IKTarget | End-effector target and constraints; live solving not yet shipped |
| MotionClip | Qualified baked or user-authored animation |
| RequalificationJob | Authoring-to-new-ProductRevision controlled compiler lane |

These are architecture/UX names, not claims that every schema already ships.

## Studio interface

Studio is **one qualified character with two linked internal workspaces**, rather than two unconnected editors. The separate Motion Lab page may remain a library/playback experience for baked clips.

### PUPPET workspace

Viewport has priority. Left: views, hierarchy, slots, attachments. Right: selection-aware inspector and proof status. Tools: Rig, Mesh, Skin, Attach, IK definition, transforms. Timeline closed by default. Structural edits remain sibling authoring state.

### MOTION workspace

Same viewport, view, product bundle, and selection identity. Timeline opens, user clips and animator controls become primary. Move/Rotate/Scale acts on real controls, inserts user keys, and feeds actual animation preview. Never silently convert a read-only baked clip to editable.

### Global acceptance criteria

A real puppet is selectable, movable, rotatable, keyable, undoable, replayable, reloadable, and exportable without losing identity or violating qualified source authority. No source image replacement, wireframe fake, or placeholder animation counts as delivered capability.

## Development milestones

1. **S0 Linked workspaces (this PR):** Distinct Setup/Motion controls, real editor action routing, fail-closed qualification state, timeline context, responsive UX.
2. **S1 Interaction quality:** Control hit-testing, selection overlays, pivots, coordinate spaces, undo history, multi-view camera behavior, accessibility/shortcuts.
3. **S2 Real live IK:** Drag a weapon tip/hand target; reach solutions propagate through validated joints, shoulder, torso, pelvis, foot contacts and attachments. Typed solver and runtime parity tests required.
4. **S3 Spine-class motion:** Pose library, mirrored poses, timeline grouping, onion skin, copy/paste, blend and keyframe/curve editing, persistent motion assets.
5. **S4 View/identity continuity:** Multi-view turns with correct draw order, visibility, occlusion and mechanics/graphics continuity.
6. **S5 Production closure:** Persistent revisions, requalification, signed editable native bundle (.rss), runtime-equivalent export, and paid compile+Idle acceptance gate.

## Commercial architecture

Charge primarily for **compilation of a persistent qualified puppet with verified first Idle**. Make later animation generation/authoring inexpensive or free within a measured resource budget. Billing must not treat an unsuccessful puppet+Idle closure as a delivered paid compile. Specific pricing and entitlement design require a separate decision.

## Explicit not-yet-shipped capabilities

Live end-effector IK solving, universal controller coverage, automatic multi-view mechanics under arbitrary turns, and full Spine parity remain open workstreams. The Studio selector does not pretend these capabilities exist; existing edit/animation tools are surfaced faithfully.
