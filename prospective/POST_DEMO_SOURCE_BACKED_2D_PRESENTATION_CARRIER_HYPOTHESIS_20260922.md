# Post-demo research hypothesis: source-backed 2D presentation carriers

Status: **PARKED — DO NOT PIVOT KNIGHT DEMO ARCHITECTURE**
Date: 2026-09-22

## Why this is parked

The Knight investor demo is close. The current demo-critical architecture remains IRIS V3 -> canonical 3D mesh -> rig/skin/motion -> posed XYZ -> canonical-depth runtime -> CAA. We will not replace this path with a 2D carrier runtime before the demo.

The current geometry/source-fidelity failure must be understood to owner level before opening a new presentation substrate. A premature pivot would trade known failures for new, unmeasured failures and would invalidate a large amount of already-qualified runtime/motion work.

## Hypothesis to revisit after demo/P0 closure

A future RealSaS presentation substrate may keep 3D as mechanics/correspondence/depth/visibility authority while moving final pixel authority to source-backed 2D presentation carriers. Candidate inputs for anonymous presentation segmentation include skin-weight structure, topology, connected components, source appearance boundaries, multi-view visibility and Stage37 presentation partition evidence. Semantic names such as arm/cape/sword are optional.

Possible advantages to investigate later:

- source pixels remain direct appearance authority;
- 3D no longer needs to reproduce source silhouette at subpixel precision merely to repaint known pixels;
- per-pixel depth can still come from canonical posed 3D correspondence, preserving cyclic occlusion handling;
- draw order can remain 3D-derived rather than becoming a coarse Spine slot order;
- hidden-region completion can use other source views through canonical correspondence.

The hard research questions are motion-time deformation, newly exposed hidden regions, view transitions, presentation segmentation boundaries, incomplete 3D correspondence and editability/export semantics.

## Reminder trigger

Revisit this hypothesis **after Knight demo-critical P0 closure / investor demo**, not before. Compare it against the measured limit of the frozen 3D presentation path rather than using it as an escape from unresolved V3 geometry failures.
