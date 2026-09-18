# RealSaS Mainline VNext Contract — 2026-09-18

## Mainline rule

`main` is the only continuation pointer for current architecture and implementation.
Historical experiment branches and Drive artifacts remain immutable evidence; they do
not define current execution merely because they are newer or contain a local PASS.

The current controlled product target is an automatic 8-direction, Spine-class 2D
puppet compiler. The implementation may use 3D internally, but the product appearance
must remain artist-authored 2D.

## Canonical pipeline

```
8 qualified artist observations + calibrated cameras
    -> IRIS / learned signed geometry evidence
    -> Compiler-qualified dense surface lineage
    -> RiggingSurfaceIR (mechanical/control substrate)
    -> Geppetto -> Compiler-qualified SkeletonIR
    -> Arachne -> Compiler-qualified SkinIR on RiggingSurfaceIR
    -> DrawableSurfaceIR derived from the SAME dense surface lineage
    -> DrawableSupportBindingIR (explicit drawable <-> rigging seam)
    -> transferred qualified skin on drawable vertices
    -> canonical 3D FK/LBS motion
    -> shared 3D posed geometry
    -> per-view projection + reference z-buffer visibility
    -> view-conditioned source-provenance appearance
    -> slot / attachment / draw-order / clipping / secondary-motion composition
    -> native runtime / bake
    -> 2D output
```

RiggingSurfaceIR is not the drawable mesh. DrawableSurfaceIR is not a second rigging
truth. The support binding is first-class and hash-bound so runtime/export cannot
silently invent a second mechanical identity.

## Observation contract comes first

For controlled 8-view FIT subjects, all eight inputs must contain the complete admitted
subject with a nonzero safety margin. Alpha touching any image border is a qualification
failure, not something a downstream recall threshold may override.

Observation evidence is tri-state:

- in-frame foreground = POSITIVE evidence;
- in-frame background = NEGATIVE evidence;
- out-of-frame = UNKNOWN.

UNKNOWN may not be converted into negative geometry evidence. A view that does not
observe a locus cannot veto another view that positively observes it.

## Geometry and appearance have different authority

Canonical 3D surface geometry is the deformation and visibility carrier. It is never
RGB appearance authority.

RGB comes from artist source observations with explicit provenance. Mainline forbids
lighting, material relighting, normal-based shading, rim light, bloom, tone mapping or
other mechanisms that turn the puppet into a generic 3D render.

The preferred initial appearance rule is one donor view per face. Cross-view color
blending is not part of the product path. Bilinear sampling and padding inside the same
donor texture remain allowed.

Direction changes may switch the authored 8-direction appearance discretely. That is
normal 8-direction 2D behavior, not a defect.

## Seen vs truly unseen

A drawable locus seen in at least one qualified source observation is SEEN_REQUIRED.
It may not disappear because another view is out of frame, because confidence is low,
or because donor provenance is inconvenient.

A locus not visible in any of the eight qualified source observations is TRULY_UNSEEN.
No generative completion is authorized in the current product path. Truly unseen
regions remain explicitly marked by policy until a separately qualified completion
mechanism exists.

## Rest-before-motion rule

Before motion work may authorize a subject:

1. condition the drawable surface;
2. establish its support binding and transferred skin;
3. render each rest direction with true shared 3D geometry and depth;
4. use DIRECT_SOURCE appearance only;
5. compare against the source observation and separately report silhouette-edge error.

Motion cannot hide a failed rest representation.

## Spine-class runtime semantics

RealSaS keeps Spine-class concepts where they are semantically useful:

- slots and active attachments;
- authored/compiled draw order;
- clipping attachments;
- component-local secondary motion;
- rigid replaceable equipment;
- deterministic playback.

True 3D depth answers physical front/back visibility. Slot ordering answers semantic
composition and ties; neither replaces the other.

## Genericity

No current mainline implementation may branch on Mage, a Mage component name, Mage
statistics or Mage topology. Subject-specific facts belong only in fixtures or external
artifacts.

## Historical Mage decision

Mage FIT results, checkpoints and reports remain preserved as scoped scientific evidence.
The Mage full-subject product lineage is reopened because the observation apparatus used
body-only-derived framing for a broader subject. Those artifacts are not deleted, but
they cannot authorize the next product subject.

Subject-2 starts from the observation contract, not from a downstream mesh patch.
