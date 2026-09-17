# Directional render-mesh clean-room contract v1

Status: PREREGISTERED DESIGN — IMPLEMENTATION NOT YET CLAIMED

Parent contracts:

- `SPINE_CLEANROOM_V1.md`
- `RUNTIME_V4_COMPACT_PREREG.md`

## Problem statement

The dense CharacterGen-style zero-surface is valuable as 3D scientific geometry/support authority, but direct projection of that dense closed surface is not necessarily the correct 2D product representation for an artist-authored directional puppet.

The current BODY-only Runtime-v3 diagnostic exposed two product problems:

1. dense geometry is far too large for a normal mobile runtime representation;
2. strict same-view visible-face appearance authority can leave black holes when geometry becomes visible under motion but had no direct same-view source appearance claim in the rest observation.

Rigid component restoration can cover some large gaps, but it cannot by itself solve residual holes inside BODY.

## Clean-room behavioural reference

Public Spine documentation describes mesh attachments as textured polygons whose vertices may be weighted to bones. The rendered mesh is an attachment representation, not a scientific closed 3D reconstruction. Public guidance also recommends minimizing mesh vertex count because vertex transforms accumulate runtime cost.

RealSaS adopts only this public behavioural principle: **render representation may be a compact textured attachment mesh driven by qualified mechanics, while scientific geometry remains a separate authority.**

No Spine source code or private implementation detail is used.

## DR-1 — Dense scientific BODY remains authoritative support

The dense zero-surface continues to own:

- geometric evidence/proof;
- canonical support locations;
- 3D correspondence;
- normal/surface evidence where needed;
- mapping into canonical S/G/W mechanics;
- regression truth for runtime-mesh derivation.

It is not automatically the runtime drawable.

## DR-2 — Runtime BODY is directional render support

For each qualified source view V0..V7, build a compact render mesh whose primary product-space objective is to represent the source BODY raster/alpha coverage for that view.

The runtime mesh is a distinct representation class and must never be relabelled as canonical scientific S.

Each runtime vertex must carry an exact support mapping to sealed 3D/mechanical truth, for example one of:

- exact dense vertex identity;
- dense face + barycentric coordinates;
- exact convex support row over sealed surface nodes;
- another explicitly qualified deterministic support representation.

No nearest-neighbour or filename heuristic may be silently used as authority.

## DR-3 — Source coverage is an admission gate

At rest/setup pose, the candidate render mesh must be rasterized with the exact qualified camera/view contract and compared against the source BODY owner mask.

Required measurements include:

- source BODY alpha recall;
- precision / off-owner spill;
- deep-interior miss fraction;
- edge miss fraction;
- owner-boundary miss fraction;
- connected-component sizes of residual misses.

A mesh is not admitted merely because its 3D source lineage is correct.

The previous D0/ABC localization remains a regression reference, not a product pass.

## DR-4 — Appearance comes from the source raster, not hidden 3D faces

Directional runtime mesh UV/appearance authority is view-local source art.

The setup/rest mesh must map to source pixels/regions belonging to the qualified BODY owner mask for that view.

The runtime path must not require every drawable triangle to correspond to a 3D face that was independently visible in the source z-buffer. That requirement caused representation-level holes when used as a product drawable constraint.

This does not authorize hallucinated pixels. All admitted appearance remains source-derived or separately qualified completion authority.

## DR-5 — Motion deformation comes from exact mechanical support

Runtime render vertices are driven by canonical mechanics through their sealed support mapping.

For a support row over dense/scientific nodes:

1. resolve the exact support points/weights;
2. consume the already-qualified S/G/W mechanical deformation;
3. produce the runtime vertex pose deterministically;
4. retain provenance hashes binding the runtime result to the source support and motion authority.

Historical weight transfer is not permitted merely to make the mesh convenient.

## DR-6 — Directional meshes may differ by view

The eight source views are artist-authored directional render representations. They do not need identical 2D topology.

They must, however, share canonical mechanical semantics and exact view-specific provenance.

This is deliberate: RealSaS represents **3D-equivalent mechanics with directional 2D/2.5D renderables**, not a requirement that all directional artwork be one projected render topology.

## DR-7 — Component layering is explicit

BODY and rigid components are assembled through runtime slots/components with qualification-owned draw order.

Current Mage components:

- BODY_UNDERLAY — deformable directional mesh;
- CAPE_FOREGROUND — rigid/qualified attachment representation;
- HAT_FOREGROUND — rigid/qualified attachment representation;
- BOOK_FOREGROUND — rigid/qualified attachment representation;
- STAFF_FOREGROUND — rigid/qualified attachment representation.

If later evidence shows a component itself needs deformable treatment, it must receive its own qualified deformable render mesh rather than an implicit fallback.

## DR-8 — Motion exposure must be tested, not assumed

A rest-perfect render mesh can still expose holes under animation.

Every candidate must therefore be evaluated over preregistered idle/run samples for all 8 views using:

- source-supported continuity where directly measurable;
- triangle inversion/degeneracy;
- seam opening;
- owner/component boundary drift;
- newly exposed unsupported regions;
- draw-order correctness;
- texture stretch metrics.

A dynamic hole is a render-representation failure unless it is explicitly covered by another qualified component or separately qualified source authority.

## DR-9 — Compactness is constrained by visual/mechanical error

Do not choose a fixed triangle count blindly.

Generate/evaluate progressively smaller candidates and admit the smallest representation that satisfies frozen gates.

Candidate sequence may be, for example:

- 20k tris;
- 12k tris;
- 8k tris;
- 5k tris;

but exact counts are not preregistered as pass thresholds here.

The current mobile-oriented target is low tens of thousands or less per character, not hundreds of thousands.

## DR-10 — Full product test order

Before claiming the new path fixed the visual problem:

1. build directional BODY render meshes from exact source owner rasters;
2. seal exact 3D/mechanical support mapping;
3. assemble BODY + four rigid components through slots;
4. render rest/setup pose for 8 views;
5. compare against source owner/alpha coverage;
6. run idle and run with dense temporal sampling;
7. measure residual black/unsupported regions;
8. compare residual localization against old ABC/D0 evidence;
9. only then request founder visual approval.

## Performance requirements

The directional runtime meshes must also solve the representation-cost problem:

- topology stored once per view/attachment asset, never once per frame;
- source texture/UV/provenance stored once;
- frame state contains only changing deform state;
- product closure consumes sealed support hashes rather than recursively serializing dense truth;
- dense scientific mesh is not embedded in the normal shipping package unless explicitly requested for diagnostics.

## Fail-closed non-claims

This contract does not yet claim:

- the current black-hole bug is fixed;
- any particular triangle budget passes;
- cross-view appearance completion is qualified;
- arbitrary/unseen character generalization;
- semantic animation quality;
- PRODUCT_PASS.

It defines the next implementation path and the evidence required to accept it.
